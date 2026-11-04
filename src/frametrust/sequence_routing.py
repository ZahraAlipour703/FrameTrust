"""Sequence-level dynamic-programming selection over pose-repair candidates.

The unary cost is the existing inference-time predicted localization risk. A
margin is added to repair candidates so that zero transition cost exactly
recovers the current raw-referenced selective rule. Pairwise costs regularize
candidate identity changes without using ground-truth labels.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .selective import select_candidates


def _transition_cost(
    previous: int,
    current: int,
    frame: int,
    candidates: Mapping[str, np.ndarray],
    candidate_names: list[str],
    torso_scale: np.ndarray | None,
    penalty: float,
    mode: str,
) -> float:
    if previous == current or penalty == 0.0:
        return 0.0
    if mode == "constant":
        return float(penalty)
    if mode != "disagreement":
        raise ValueError("mode must be 'constant' or 'disagreement'")

    old = np.asarray(candidates[candidate_names[previous]], dtype=float)
    new = np.asarray(candidates[candidate_names[current]], dtype=float)
    # The disagreement at the destination frame measures the coordinate jump
    # introduced by switching to `current` instead of continuing `previous`.
    # If either candidate is unavailable there, try the source frame. If the
    # pair is unavailable at both endpoints, impose no disagreement cost; this
    # avoids blocking a repair transition through a genuine missing interval.
    pair_frame = frame
    both = np.isfinite(old[pair_frame]).all() and np.isfinite(new[pair_frame]).all()
    if not both:
        pair_frame = frame - 1
        both = (
            np.isfinite(old[pair_frame]).all()
            and np.isfinite(new[pair_frame]).all()
        )
    if not both:
        return 0.0
    scale = 1.0
    if torso_scale is not None:
        value = float(torso_scale[pair_frame])
        if np.isfinite(value) and value > 1e-9:
            scale = value
    disagreement = float(np.linalg.norm(old[pair_frame] - new[pair_frame]) / scale)
    return float(penalty * disagreement)


def route_candidate_sequence(
    predicted_errors: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    candidate_names: list[str] | tuple[str, ...],
    min_predicted_gain: float = 0.01,
    switch_penalty: float = 0.0,
    mode: str = "constant",
    torso_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Choose one candidate identity per joint over a complete sequence.

    Args:
        predicted_errors: [frames, joints, candidates] predicted torso-normalized
            localization errors. Candidate 0 must be raw.
        candidates: candidate coordinate arrays, each [frames, joints, 2].
        candidate_names: names in the same order as the prediction candidate axis.
        min_predicted_gain: risk margin applied to every non-raw candidate.
        switch_penalty: non-negative pairwise cost. In ``constant`` mode this is
            charged for every identity change. In ``disagreement`` mode it is
            multiplied by cross-candidate coordinate disagreement, normalized
            by the torso scale at the boundary.
        mode: ``constant`` or ``disagreement``.
        torso_scale: optional [frames] scale for disagreement normalization.

    Returns:
        Integer array [frames, joints] with candidate indices.

    With switch_penalty=0, this is exactly equivalent to the independent
    ``select_candidates`` rule, including its raw-first tie breaking.
    Ground truth is neither accepted nor consulted by this function.
    """
    predicted = np.asarray(predicted_errors, dtype=float)
    names = list(candidate_names)
    if predicted.ndim != 3 or predicted.shape[-1] < 2:
        raise ValueError("predicted_errors must have shape [frames, joints, candidates]")
    if not names or names[0] != "raw" or len(names) != predicted.shape[-1]:
        raise ValueError("candidate_names must match the candidate axis and begin with 'raw'")
    if min_predicted_gain < 0 or switch_penalty < 0:
        raise ValueError("min_predicted_gain and switch_penalty must be non-negative")
    if mode not in {"constant", "disagreement"}:
        raise ValueError("mode must be 'constant' or 'disagreement'")
    if not np.isfinite(min_predicted_gain) or not np.isfinite(switch_penalty):
        raise ValueError("min_predicted_gain and switch_penalty must be finite")
    # Preserve the exact >= threshold and raw-first tie semantics of the
    # established selector rather than relying on DP argmin tie-breaking.
    if switch_penalty == 0.0:
        return select_candidates(predicted, min_predicted_gain)

    n_frames, n_joints, n_candidates = predicted.shape
    if n_frames == 0 or n_joints == 0:
        raise ValueError("predicted_errors must contain at least one frame and joint")
    arrays = [np.asarray(candidates[name], dtype=float) for name in names if name in candidates]
    if len(arrays) != len(names):
        missing = [name for name in names if name not in candidates]
        raise ValueError(f"missing candidate array(s): {missing}")
    if any(a.shape != (n_frames, n_joints, 2) for a in arrays):
        raise ValueError("each candidate must have shape [frames, joints, 2]")
    if torso_scale is not None:
        scale = np.asarray(torso_scale, dtype=float)
        if scale.shape != (n_frames,):
            raise ValueError("torso_scale must have shape [frames]")
    else:
        scale = np.ones(n_frames, dtype=float)

    unary = predicted.copy()
    unary[..., 1:] += float(min_predicted_gain)
    unary = np.nan_to_num(unary, nan=1.0, posinf=1.0, neginf=0.0)
    choices = np.zeros((n_frames, n_joints), dtype=np.int64)
    joint_ids = np.arange(n_joints)
    identity_change = ~np.eye(n_candidates, dtype=bool)

    def disagreement_matrix(frame: int) -> tuple[np.ndarray, np.ndarray]:
        # Return [joint, previous_candidate, current_candidate] disagreement and
        # the corresponding pairwise availability mask.
        coords = np.stack([a[frame] for a in arrays], axis=0)  # [K,J,2]
        finite = np.isfinite(coords).all(axis=-1)  # [K,J]
        delta = coords[:, None, :, :] - coords[None, :, :, :]
        distance = np.linalg.norm(delta, axis=-1).transpose(2, 0, 1)
        available = (finite[:, None, :] & finite[None, :, :]).transpose(2, 0, 1)
        denom = scale[frame]
        if not np.isfinite(denom) or denom <= 1e-9:
            denom = 1.0
        return distance / float(denom), available

    dp = unary[0].copy()  # [joint,candidate]
    back = np.zeros((n_frames, n_joints, n_candidates), dtype=np.int64)
    for frame in range(1, n_frames):
        if mode == "constant":
            transition = np.broadcast_to(
                identity_change[None, :, :] * float(switch_penalty),
                (n_joints, n_candidates, n_candidates),
            )
        elif switch_penalty == 0.0:
            transition = np.zeros((n_joints, n_candidates, n_candidates), dtype=float)
        else:
            dest_distance, dest_available = disagreement_matrix(frame)
            src_distance, src_available = disagreement_matrix(frame - 1)
            distance = np.where(
                dest_available,
                dest_distance,
                np.where(src_available, src_distance, 0.0),
            )
            transition = distance * float(switch_penalty)
            transition *= identity_change[None, :, :]

        totals = dp[:, :, None] + transition  # [J,previous,current]
        best_previous = np.argmin(totals, axis=1)  # [J,current]
        best_total = np.take_along_axis(
            totals, best_previous[:, None, :], axis=1
        )[:, 0, :]
        dp = unary[frame] + best_total
        back[frame] = best_previous

    state = np.argmin(dp, axis=1)
    choices[-1] = state
    for frame in range(n_frames - 1, 0, -1):
        state = back[frame, joint_ids, state]
        choices[frame - 1] = state
    return choices
