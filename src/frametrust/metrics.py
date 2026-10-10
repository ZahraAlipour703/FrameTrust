"""Metrics for model-output agreement, visibility and temporal jitter."""
from __future__ import annotations

import numpy as np


def mean_joint_displacement(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Mean XY displacement in normalized image coordinates (not ground truth)."""
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    if ref.shape != est.shape or ref.ndim != 3 or ref.shape[-1] < 2:
        raise ValueError("reference and estimate must have equal [frames, joints, >=2] shapes")
    valid = np.isfinite(ref[..., :2]).all(axis=-1) & np.isfinite(est[..., :2]).all(axis=-1)
    if not valid.any():
        return float("nan")
    dist = np.linalg.norm(ref[..., :2] - est[..., :2], axis=-1)
    return float(np.mean(dist[valid]))


def low_confidence_rate(confidence: np.ndarray, threshold: float = 0.35) -> float:
    conf = np.asarray(confidence, dtype=float)
    if conf.size == 0:
        return float("nan")
    return float(np.mean(~np.isfinite(conf) | (conf < threshold)))


def temporal_jitter(coords: np.ndarray) -> float:
    """RMS norm of second differences in normalized image coordinates/frame²."""
    c = np.asarray(coords, dtype=float)
    if c.ndim != 3 or c.shape[-1] < 2:
        raise ValueError("coords must have shape [frames, joints, >=2]")
    if c.shape[0] < 3:
        return float("nan")
    second = c[2:, :, :2] - 2.0 * c[1:-1, :, :2] + c[:-2, :, :2]
    valid = np.isfinite(second).all(axis=-1)
    if not valid.any():
        return float("nan")
    norms = np.linalg.norm(second, axis=-1)
    return float(np.sqrt(np.mean(norms[valid] ** 2)))
