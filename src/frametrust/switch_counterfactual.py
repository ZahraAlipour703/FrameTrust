"""Counterfactual diagnostics for per-joint candidate switching boundaries."""
from __future__ import annotations

from typing import Mapping, Sequence, Any

import numpy as np


def _velocity_error(
    p0: np.ndarray,
    p1: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    frame_diagonal: float,
    fps: float,
) -> float:
    values = np.concatenate([p0, p1, g0, g1])
    if not np.isfinite(values).all():
        return float("nan")
    pred_delta = p1 - p0
    gt_delta = g1 - g0
    return float(np.linalg.norm(pred_delta - gt_delta) * fps / frame_diagonal)


def switch_counterfactual_events(
    candidates: Mapping[str, np.ndarray],
    choices: np.ndarray,
    gt_xy: np.ndarray,
    valid_gt: np.ndarray,
    candidate_names: Sequence[str],
    frame_diagonal: float,
    fps: float = 30.0,
    sequence: str = "",
    condition: str = "",
    action: str = "",
) -> list[dict[str, Any]]:
    """Evaluate hybrid switches against same-candidate continuation counterfactuals.

    For a switch old->new between frames t-1 and t, the observed hybrid interval
    uses old(t-1) and new(t). Two diagnostic counterfactuals instead use old at
    both endpoints or new at both endpoints. GT is used only to score these
    alternatives, never to construct choices. The lower-error continuation is
    an oracle diagnostic and is NOT a deployable selector.
    """
    choice = np.asarray(choices, dtype=int)
    gt = np.asarray(gt_xy, dtype=float)
    valid = np.asarray(valid_gt, dtype=bool)
    names = list(candidate_names)
    if choice.ndim != 2:
        raise ValueError("choices must have shape [frames, joints]")
    if gt.shape != (*choice.shape, 2) or valid.shape != choice.shape:
        raise ValueError("gt_xy/valid_gt shapes do not match choices")
    if not names or names[0] != "raw":
        raise ValueError("candidate_names must begin with 'raw'")
    if ((choice < 0) | (choice >= len(names))).any():
        raise ValueError("choices contain an invalid candidate index")
    if frame_diagonal <= 0 or fps <= 0:
        raise ValueError("frame_diagonal and fps must be positive")

    arrays: dict[str, np.ndarray] = {}
    for name in names:
        if name not in candidates:
            raise ValueError(f"missing candidate {name!r}")
        values = np.asarray(candidates[name], dtype=float)
        if values.shape != gt.shape:
            raise ValueError(f"candidate {name!r} shape does not match gt_xy")
        arrays[name] = values

    selected = np.full_like(gt, np.nan, dtype=float)
    for idx, name in enumerate(names):
        selected[choice == idx] = arrays[name][choice == idx]

    rows: list[dict[str, Any]] = []
    n_frames, n_joints = choice.shape
    for t in range(1, n_frames):
        for joint in np.flatnonzero(choice[t] != choice[t - 1]):
            old_idx = int(choice[t - 1, joint])
            new_idx = int(choice[t, joint])
            old_name, new_name = names[old_idx], names[new_idx]
            if old_idx == 0 and new_idx > 0:
                transition_type = "repair_entry"
            elif old_idx > 0 and new_idx == 0:
                transition_type = "repair_exit"
            else:
                transition_type = "repair_method_switch"

            gt_pair_valid = bool(valid[t - 1, joint] and valid[t, joint])
            if gt_pair_valid:
                g0, g1 = gt[t - 1, joint], gt[t, joint]
                hybrid_error = _velocity_error(
                    selected[t - 1, joint], selected[t, joint],
                    g0, g1, frame_diagonal, fps,
                )
                old_error = _velocity_error(
                    arrays[old_name][t - 1, joint], arrays[old_name][t, joint],
                    g0, g1, frame_diagonal, fps,
                )
                new_error = _velocity_error(
                    arrays[new_name][t - 1, joint], arrays[new_name][t, joint],
                    g0, g1, frame_diagonal, fps,
                )
            else:
                hybrid_error = old_error = new_error = float("nan")

            finite_continuations = [v for v in (old_error, new_error) if np.isfinite(v)]
            best_continuation = min(finite_continuations) if finite_continuations else float("nan")
            both_continuations = np.isfinite(old_error) and np.isfinite(new_error)
            hybrid_supported = np.isfinite(hybrid_error)
            rows.append({
                "sequence": sequence,
                "action": action,
                "condition": condition,
                "transition_at_frame": int(t),
                "joint_index": int(joint),
                "from_candidate": old_name,
                "to_candidate": new_name,
                "transition_type": transition_type,
                "gt_pair_valid": gt_pair_valid,
                "hybrid_supported": bool(hybrid_supported),
                "old_continuation_supported": bool(np.isfinite(old_error)),
                "new_continuation_supported": bool(np.isfinite(new_error)),
                "both_continuations_supported": bool(both_continuations),
                "hybrid_velocity_error_norm_s": hybrid_error,
                "old_continuation_velocity_error_norm_s": old_error,
                "new_continuation_velocity_error_norm_s": new_error,
                "best_continuation_velocity_error_norm_s": best_continuation,
                "hybrid_minus_old_continuation": (
                    hybrid_error - old_error
                    if np.isfinite(hybrid_error) and np.isfinite(old_error) else float("nan")
                ),
                "hybrid_minus_new_continuation": (
                    hybrid_error - new_error
                    if np.isfinite(hybrid_error) and np.isfinite(new_error) else float("nan")
                ),
                "hybrid_minus_best_continuation": (
                    hybrid_error - best_continuation
                    if np.isfinite(hybrid_error) and np.isfinite(best_continuation) else float("nan")
                ),
                "hybrid_worse_than_both_continuations": (
                    bool(hybrid_error > old_error and hybrid_error > new_error)
                    if hybrid_supported and both_continuations else None
                ),
                "frame_diagonal_px": float(frame_diagonal),
                "fps": float(fps),
            })
    return rows
