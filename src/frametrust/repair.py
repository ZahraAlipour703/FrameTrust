"""Simple, inspectable temporal repair baselines for landmark trajectories."""
from __future__ import annotations

import numpy as np


def interpolate_missing(coords: np.ndarray, confidence: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    """Linearly interpolate low-confidence/missing 2D/3D joint coordinates.

    Args:
        coords: float array [frames, joints, dimensions].
        confidence: float array [frames, joints] in [0, 1].
        threshold: observations below this confidence are considered unreliable.

    Long gaps are interpolated too; the caller should report gap length because
    interpolation over long occlusions can be physically implausible.
    """
    c = np.asarray(coords, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    _validate_shapes(c, conf)
    out = c.copy()
    n_frames, n_joints, n_dims = out.shape
    x = np.arange(n_frames)
    for j in range(n_joints):
        good = np.isfinite(out[:, j, :]).all(axis=1) & np.isfinite(conf[:, j]) & (conf[:, j] >= threshold)
        if not good.any():
            continue
        for d in range(n_dims):
            out[:, j, d] = np.interp(x, x[good], out[good, j, d])
    return out


def confidence_aware_repair(
    coords: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.35,
    alpha_low: float = 0.20,
    alpha_high: float = 0.75,
) -> np.ndarray:
    """Interpolate unreliable points, then adaptively smooth each joint.

    High-confidence observations receive a larger update weight; low-confidence
    observations receive a smaller weight. This is a transparent baseline, not
    a learned denoiser and not guaranteed to improve every motion.
    """
    if not (0 < alpha_low <= alpha_high <= 1):
        raise ValueError("Require 0 < alpha_low <= alpha_high <= 1")
    c = np.asarray(coords, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    filled = interpolate_missing(c, conf, threshold=threshold)
    out = filled.copy()
    if out.shape[0] < 2:
        return out
    valid_conf = np.where(np.isfinite(conf), np.clip(conf, 0.0, 1.0), 0.0)
    for t in range(1, out.shape[0]):
        alpha = alpha_low + (alpha_high - alpha_low) * valid_conf[t, :, None]
        # Do not smooth joints that had no interpolable observation.
        finite = np.isfinite(out[t]) & np.isfinite(out[t - 1])
        smoothed = alpha * out[t] + (1.0 - alpha) * out[t - 1]
        out[t] = np.where(finite, smoothed, out[t])
    return out


def _validate_shapes(coords: np.ndarray, confidence: np.ndarray) -> None:
    if coords.ndim != 3:
        raise ValueError("coords must have shape [frames, joints, dimensions]")
    if confidence.ndim != 2 or confidence.shape != coords.shape[:2]:
        raise ValueError("confidence must have shape [frames, joints]")
    if coords.shape[0] == 0 or coords.shape[1] == 0:
        raise ValueError("coords must contain at least one frame and one joint")
