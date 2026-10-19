"""Sequence-cross-fitted selective pose repair.

The selector predicts each candidate method's localization error from inputs
available at inference time. It applies a non-raw candidate only when its
predicted error improves on raw by a configurable margin. Ground-truth labels
are used only to fit the error predictor on other sequences, never to build
features or choose repairs on the held-out sequence.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

DEFAULT_CANDIDATES = (
    "raw",
    "linear_interpolation",
    "linear_missing_only",
    "confidence_aware",
    "one_euro",
    "savgol",
)


def build_selective_features(
    coords: np.ndarray,
    confidence: np.ndarray,
    sizes: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    confidence_threshold: float = 0.35,
    max_gap_feature: int = 10,
) -> np.ndarray:
    """Build finite per-frame/per-joint features without ground-truth access."""
    raw = np.asarray(coords, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    frame_sizes = np.asarray(sizes, dtype=float)
    if raw.ndim != 3 or raw.shape[-1] != 2:
        raise ValueError("coords must have shape [frames, joints, 2]")
    if conf.shape != raw.shape[:2]:
        raise ValueError("confidence must have shape [frames, joints]")
    if frame_sizes.shape != (raw.shape[0], 2):
        raise ValueError("sizes must have shape [frames, 2] as width,height")
    if not np.isfinite(frame_sizes).all() or (frame_sizes <= 0).any():
        raise ValueError("sizes must be finite and positive")
    if not (0.0 < confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold must be in (0, 1]")
    if max_gap_feature < 1:
        raise ValueError("max_gap_feature must be >= 1")
    for name, values in candidates.items():
        if np.asarray(values).shape != raw.shape:
            raise ValueError(f"candidate {name!r} does not match coords shape")

    n_frames, n_joints, _ = raw.shape
    w = frame_sizes[:, 0, None]
    h = frame_sizes[:, 1, None]
    diag = np.sqrt(w * w + h * h)
    raw_finite = np.isfinite(raw).all(axis=-1)
    conf_finite = np.isfinite(conf)
    conf_clean = np.where(conf_finite, np.clip(conf, 0.0, 1.0), 0.0)
    reliable = raw_finite & conf_finite & (conf >= confidence_threshold)
    unreliable = ~reliable
    features: list[np.ndarray] = []

    def add(name: str, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=float)
        if arr.shape != (n_frames, n_joints):
            raise ValueError(f"feature {name!r} has shape {arr.shape}")
        features.append(arr)

    add("confidence", conf_clean)
    add("confidence_missing", (~conf_finite).astype(float))
    add("raw_finite", raw_finite.astype(float))
    add("raw_unreliable", unreliable.astype(float))
    add("raw_x_normalized", np.where(raw_finite, raw[:, :, 0] / w, 0.0))
    add("raw_y_normalized", np.where(raw_finite, raw[:, :, 1] / h, 0.0))

    prev_conf = np.zeros_like(conf_clean)
    next_conf = np.zeros_like(conf_clean)
    if n_frames > 1:
        prev_conf[1:] = conf_clean[:-1]
        next_conf[:-1] = conf_clean[1:]
    add("previous_confidence", prev_conf)
    add("next_confidence", next_conf)
    add("neighbor_confidence_min", np.minimum(prev_conf, next_conf))
    add("neighbor_confidence_mean", 0.5 * (prev_conf + next_conf))

    prev_step = np.zeros((n_frames, n_joints), dtype=float)
    next_step = np.zeros_like(prev_step)
    local_innovation = np.zeros_like(prev_step)
    has_prev = np.zeros_like(prev_step, dtype=bool)
    has_next = np.zeros_like(prev_step, dtype=bool)
    if n_frames > 1:
        has_pair = raw_finite[1:] & raw_finite[:-1]
        delta = np.linalg.norm(raw[1:] - raw[:-1], axis=-1)
        step = np.zeros_like(delta)
        scale = diag[1:, :]
        step[has_pair] = (delta / scale)[has_pair]
        prev_step[1:] = step
        next_step[:-1] = step
        has_prev[1:] = has_pair
        has_next[:-1] = has_pair
    if n_frames > 2:
        triple = raw_finite[:-2] & raw_finite[1:-1] & raw_finite[2:]
        midpoint_error = np.linalg.norm(
            raw[1:-1] - 0.5 * (raw[:-2] + raw[2:]), axis=-1
        ) / diag[1:-1, :]
        local_innovation[1:-1] = np.where(triple, midpoint_error, 0.0)
    add("previous_step_norm", prev_step)
    add("next_step_norm", next_step)
    add("local_acceleration_residual", local_innovation)
    add("has_previous_raw", has_prev.astype(float))
    add("has_next_raw", has_next.astype(float))

    # Two-sided gap features are suitable for the offline protocol used here.
    gap_length = np.zeros((n_frames, n_joints), dtype=float)
    distance_left = np.zeros_like(gap_length)
    distance_right = np.zeros_like(gap_length)
    cap = float(max_gap_feature)
    for j in range(n_joints):
        good_ids = np.flatnonzero(reliable[:, j])
        bad_ids = np.flatnonzero(unreliable[:, j])
        if not len(bad_ids):
            continue
        for t in bad_ids:
            left = good_ids[good_ids < t]
            right = good_ids[good_ids > t]
            dl = (t - int(left[-1])) if len(left) else max_gap_feature + 1
            dr = (int(right[0]) - t) if len(right) else max_gap_feature + 1
            distance_left[t, j] = min(float(dl), cap) / cap
            distance_right[t, j] = min(float(dr), cap) / cap
            start, stop = t, t
            while start > 0 and unreliable[start - 1, j]:
                start -= 1
            while stop + 1 < n_frames and unreliable[stop + 1, j]:
                stop += 1
            gap_length[t, j] = min(float(stop - start + 1), cap) / cap
    add("unreliable_run_length", gap_length)
    add("distance_to_left_anchor", distance_left)
    add("distance_to_right_anchor", distance_right)

    candidate_names = list(candidates)
    reference_name = "linear_interpolation" if "linear_interpolation" in candidates else candidate_names[0]
    reference = np.asarray(candidates[reference_name], dtype=float)
    reference_finite = np.isfinite(reference).all(axis=-1)
    for name in candidate_names:
        candidate = np.asarray(candidates[name], dtype=float)
        candidate_finite = np.isfinite(candidate).all(axis=-1)
        add(f"{name}_finite", candidate_finite.astype(float))
        delta_raw = np.zeros((n_frames, n_joints), dtype=float)
        shared_raw = raw_finite & candidate_finite
        if shared_raw.any():
            dist = np.linalg.norm(candidate - raw, axis=-1) / diag
            delta_raw[shared_raw] = dist[shared_raw]
        add(f"{name}_delta_from_raw", delta_raw)
        delta_reference = np.zeros((n_frames, n_joints), dtype=float)
        shared_ref = candidate_finite & reference_finite
        if shared_ref.any():
            dist_ref = np.linalg.norm(candidate - reference, axis=-1) / diag
            delta_reference[shared_ref] = dist_ref[shared_ref]
        add(f"{name}_delta_from_interpolation", delta_reference)

    stacked = np.stack(features, axis=-1)
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=10.0, neginf=0.0)
    return np.clip(stacked, -10.0, 10.0).astype(np.float32)


def candidate_error_targets(
    candidates: Mapping[str, np.ndarray],
    gt_xy: np.ndarray,
    valid_gt: np.ndarray,
    torso_scale: np.ndarray,
    candidate_names: Sequence[str] = DEFAULT_CANDIDATES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized candidate errors and a mask of GT-valid training points."""
    gt = np.asarray(gt_xy, dtype=float)
    valid = np.asarray(valid_gt, dtype=bool)
    scale = np.asarray(torso_scale, dtype=float)
    if gt.ndim != 3 or gt.shape[-1] != 2:
        raise ValueError("gt_xy must have shape [frames, joints, 2]")
    if valid.shape != gt.shape[:2] or scale.shape != (gt.shape[0],):
        raise ValueError("valid_gt/torso_scale shapes do not match gt_xy")
    scoreable = valid & np.isfinite(scale[:, None]) & (scale[:, None] > 1e-9)
    errors = np.empty((*gt.shape[:2], len(candidate_names)), dtype=float)
    for k, name in enumerate(candidate_names):
        if name not in candidates:
            raise ValueError(f"missing candidate {name!r}")
        pred = np.asarray(candidates[name], dtype=float)
        if pred.shape != gt.shape:
            raise ValueError(f"candidate {name!r} shape does not match gt_xy")
        finite = np.isfinite(pred).all(axis=-1)
        distance = np.linalg.norm(pred - gt, axis=-1)
        normalized = distance / np.maximum(scale[:, None], 1e-9)
        errors[:, :, k] = np.where(
            finite & np.isfinite(normalized),
            np.clip(normalized, 0.0, 1.0),
            1.0,
        )
    return errors.astype(np.float32), scoreable


def fit_error_predictor(
    features: np.ndarray,
    candidate_errors: np.ndarray,
    ridge: float = 1.0,
) -> dict[str, np.ndarray | float | int]:
    """Fit multi-output ridge regression to candidate localization errors."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(candidate_errors, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("features and candidate_errors must be 2D with matching rows")
    if x.shape[0] < 2 or x.shape[1] < 1 or y.shape[1] < 2:
        raise ValueError("not enough rows/features/candidates to fit selector")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    good = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x, y = x[good], y[good]
    if x.shape[0] < 2:
        raise ValueError("fewer than two finite training samples")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    xz = (x - mean) / std
    design = np.column_stack([np.ones(xz.shape[0]), xz])
    penalty = np.eye(design.shape[1], dtype=float) * float(ridge)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(lhs) @ rhs
    return {
        "mean": mean,
        "std": std,
        "coefficients": coefficients,
        "ridge": float(ridge),
        "training_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_candidates": int(y.shape[1]),
    }


def predict_candidate_errors(
    model: Mapping[str, np.ndarray | float | int],
    features: np.ndarray,
) -> np.ndarray:
    """Predict each candidate's torso-normalized error for each frame/joint."""
    x = np.asarray(features, dtype=float)
    if x.ndim < 2:
        raise ValueError("features must have at least two dimensions")
    feature_count = x.shape[-1]
    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    coef = np.asarray(model["coefficients"], dtype=float)
    if mean.shape != (feature_count,) or std.shape != mean.shape:
        raise ValueError("feature count does not match fitted model")
    flat = x.reshape(-1, feature_count)
    design = np.column_stack([np.ones(flat.shape[0]), (flat - mean) / std])
    prediction = design @ coef
    prediction = np.clip(prediction, 0.0, 1.0)
    return prediction.reshape(*x.shape[:-1], coef.shape[1]).astype(np.float32)


def select_candidates(
    predicted_errors: np.ndarray,
    min_predicted_gain: float = 0.01,
) -> np.ndarray:
    """Select a repair only when predicted error improves enough over raw.

    Candidate index 0 must be raw. Otherwise, the selector abstains and preserves
    the raw coordinate, even if it is missing.
    """
    expected = np.asarray(predicted_errors, dtype=float)
    if expected.ndim < 2 or expected.shape[-1] < 2:
        raise ValueError("predicted_errors needs candidate axis with raw plus repair")
    if min_predicted_gain < 0:
        raise ValueError("min_predicted_gain must be non-negative")
    flat = expected.reshape(-1, expected.shape[-1])
    choice = np.zeros(flat.shape[0], dtype=np.int64)
    best_repair = 1 + np.argmin(flat[:, 1:], axis=1)
    raw_error = flat[:, 0]
    best_error = flat[np.arange(len(flat)), best_repair]
    repair = (raw_error - best_error) >= min_predicted_gain
    choice[repair] = best_repair[repair]
    return choice.reshape(expected.shape[:-1])


def apply_candidate_choices(
    candidates: Mapping[str, np.ndarray],
    choices: np.ndarray,
    candidate_names: Sequence[str] = DEFAULT_CANDIDATES,
) -> np.ndarray:
    """Compose a per-point hybrid trajectory from candidate outputs."""
    names = list(candidate_names)
    choice = np.asarray(choices, dtype=int)
    if not names or names[0] != "raw":
        raise ValueError("candidate_names must begin with 'raw'")
    arrays = [np.asarray(candidates[name], dtype=float) for name in names]
    shape = arrays[0].shape
    if len(shape) != 3 or choice.shape != shape[:2]:
        raise ValueError("choices must have shape [frames, joints]")
    if any(a.shape != shape for a in arrays):
        raise ValueError("all candidate arrays must have the same shape")
    if ((choice < 0) | (choice >= len(names))).any():
        raise ValueError("choices contain an invalid candidate index")
    out = np.full(shape, np.nan, dtype=float)
    for k, arr in enumerate(arrays):
        mask = choice == k
        out[mask] = arr[mask]
    return out
