"""Pure utilities for auditing per-joint candidate-selection transitions."""
from __future__ import annotations

from typing import Any

import numpy as np


def transition_masks(choices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return adjacent-change mask, transition-containing triples, stable triples.

    choices has shape [frames, joints]. Triple masks have shape [frames-2, joints]
    and are aligned with centered second-difference windows (t-1,t,t+1).
    """
    values = np.asarray(choices, dtype=int)
    if values.ndim != 2:
        raise ValueError("choices must have shape [frames, joints]")
    if len(values) >= 2:
        adjacent = values[1:] != values[:-1]
    else:
        adjacent = np.zeros((0, values.shape[1]), dtype=bool)
    if len(values) >= 3:
        transition_triples = adjacent[:-1] | adjacent[1:]
        stable_triples = ~transition_triples
    else:
        transition_triples = np.zeros((0, values.shape[1]), dtype=bool)
        stable_triples = np.zeros_like(transition_triples)
    return adjacent, transition_triples, stable_triples


def summarize_choice_sequence(
    choices: np.ndarray, candidate_names: list[str], scoreable: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize selection frequencies and switch types for one sequence."""
    values = np.asarray(choices, dtype=int)
    if values.ndim != 2:
        raise ValueError("choices must have shape [frames, joints]")
    if not candidate_names:
        raise ValueError("candidate_names must not be empty")
    if values.size and ((values < 0).any() or (values >= len(candidate_names)).any()):
        raise ValueError("choices contain candidate indices outside candidate_names")
    if scoreable is not None and np.asarray(scoreable).shape != values.shape:
        raise ValueError("scoreable must have the same [frames, joints] shape as choices")

    adjacent, _, _ = transition_masks(values)
    previous = values[:-1]
    current = values[1:]
    n_steps = int(adjacent.size)
    n_transitions = int(adjacent.sum())
    n_entries = int(((previous == 0) & (current > 0)).sum())
    n_exits = int(((previous > 0) & (current == 0)).sum())
    n_repair_switches = int(((previous > 0) & (current > 0) & (previous != current)).sum())

    result: dict[str, Any] = {
        "n_frames": int(values.shape[0]),
        "n_joints": int(values.shape[1]),
        "n_adjacent_joint_steps": n_steps,
        "n_candidate_transitions": n_transitions,
        "transition_rate": float(n_transitions / n_steps) if n_steps else float("nan"),
        "transitions_per_1000_joint_steps": float(1000.0 * n_transitions / n_steps) if n_steps else float("nan"),
        "repair_entry_count": n_entries,
        "repair_exit_count": n_exits,
        "repair_method_switch_count": n_repair_switches,
        "repair_transition_count": n_entries + n_exits + n_repair_switches,
        "repair_fraction": float((values != 0).mean()) if values.size else float("nan"),
    }
    for idx, name in enumerate(candidate_names):
        result[f"selected_fraction_{name}"] = float((values == idx).mean()) if values.size else float("nan")

    if scoreable is not None:
        valid = np.asarray(scoreable, dtype=bool)
        pair_valid = valid[:-1] & valid[1:]
        scored_transitions = adjacent & pair_valid
        n_valid_pairs = int(pair_valid.sum())
        result["n_torso_scoreable_adjacent_pairs"] = n_valid_pairs
        result["n_torso_scoreable_transitions"] = int(scored_transitions.sum())
        result["torso_scoreable_transition_rate"] = (
            float(scored_transitions.sum() / n_valid_pairs) if n_valid_pairs else float("nan")
        )
    return result


def candidate_run_lengths(choices: np.ndarray, candidate_names: list[str]) -> dict[str, list[int]]:
    """Collect contiguous dwell lengths, in frames, for each candidate."""
    values = np.asarray(choices, dtype=int)
    if values.ndim != 2:
        raise ValueError("choices must have shape [frames, joints]")
    if values.size and ((values < 0).any() or (values >= len(candidate_names)).any()):
        raise ValueError("choices contain candidate indices outside candidate_names")
    runs: dict[str, list[int]] = {name: [] for name in candidate_names}
    n_frames, n_joints = values.shape
    for joint in range(n_joints):
        if n_frames == 0:
            continue
        starts = np.r_[0, np.flatnonzero(values[1:, joint] != values[:-1, joint]) + 1]
        stops = np.r_[starts[1:], n_frames]
        for start, stop in zip(starts, stops):
            idx = int(values[start, joint])
            runs[candidate_names[idx]].append(int(stop - start))
    return runs


def transition_matrix_rows(
    choices: np.ndarray, candidate_names: list[str], sequence: str, condition: str,
) -> list[dict[str, Any]]:
    values = np.asarray(choices, dtype=int)
    if len(values) < 2:
        return []
    previous, current = values[:-1], values[1:]
    rows: list[dict[str, Any]] = []
    for src in range(len(candidate_names)):
        for dst in range(len(candidate_names)):
            if src == dst:
                continue
            count = int(((previous == src) & (current == dst)).sum())
            if count:
                rows.append({
                    "sequence": sequence,
                    "condition": condition,
                    "from_candidate": candidate_names[src],
                    "to_candidate": candidate_names[dst],
                    "transition_count": count,
                })
    return rows


def conditional_velocity_metrics(
    pose: np.ndarray,
    gt_xy: np.ndarray,
    valid_gt: np.ndarray,
    frame_diagonal: float,
    fps: float,
    group_support: np.ndarray,
) -> tuple[dict[str, float], int]:
    """Measure one-step vector-velocity error on selected transition intervals."""
    pose = np.asarray(pose, dtype=float)
    gt = np.asarray(gt_xy, dtype=float)
    valid = np.asarray(valid_gt, dtype=bool)
    support = np.asarray(group_support, dtype=bool)
    if pose.ndim != 3 or pose.shape[-1] != 2 or gt.shape != pose.shape:
        raise ValueError("pose and gt_xy must have matching [frames, joints, 2] shapes")
    expected_shape = (max(len(pose) - 1, 0), pose.shape[1])
    if valid.shape != pose.shape[:2] or support.shape != expected_shape:
        raise ValueError("valid_gt or interval support shape does not match pose")
    if frame_diagonal <= 0 or fps <= 0:
        raise ValueError("frame_diagonal and fps must be positive")
    if len(pose) < 2:
        return {
            "one_step_velocity_error_norm_s": float("nan"),
            "predicted_displacement_norm_mean": float("nan"),
            "ground_truth_displacement_norm_mean": float("nan"),
        }, 0
    eligible = (
        valid[:-1] & valid[1:]
        & np.isfinite(pose[:-1]).all(axis=-1)
        & np.isfinite(pose[1:]).all(axis=-1)
        & support
    )
    pred_delta = pose[1:] - pose[:-1]
    gt_delta = gt[1:] - gt[:-1]
    velocity_error = np.linalg.norm(pred_delta - gt_delta, axis=-1) * (fps / frame_diagonal)
    pred_displacement = np.linalg.norm(pred_delta, axis=-1) / frame_diagonal
    gt_displacement = np.linalg.norm(gt_delta, axis=-1) / frame_diagonal
    return {
        "one_step_velocity_error_norm_s": float(np.mean(velocity_error[eligible])) if eligible.any() else float("nan"),
        "predicted_displacement_norm_mean": float(np.mean(pred_displacement[eligible])) if eligible.any() else float("nan"),
        "ground_truth_displacement_norm_mean": float(np.mean(gt_displacement[eligible])) if eligible.any() else float("nan"),
    }, int(eligible.sum())
