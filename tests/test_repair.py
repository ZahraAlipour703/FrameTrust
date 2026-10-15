import numpy as np
import pytest

from frametrust.metrics import low_confidence_rate, mean_joint_displacement, temporal_jitter
from frametrust.repair import confidence_aware_repair, interpolate_missing


def test_interpolation_fills_a_low_confidence_gap():
    coords = np.array([[[0.0, 0.0, 0.0]], [[np.nan, np.nan, np.nan]], [[2.0, 0.0, 0.0]]])
    confidence = np.array([[0.9], [0.1], [0.9]])
    repaired = interpolate_missing(coords, confidence, threshold=0.35)
    assert np.allclose(repaired[1, 0], [1.0, 0.0, 0.0])


def test_unobservable_joint_remains_nan():
    coords = np.full((4, 2, 3), np.nan)
    confidence = np.zeros((4, 2))
    repaired = interpolate_missing(coords, confidence)
    assert np.isnan(repaired).all()


def test_confidence_aware_repair_shape_and_finite_output():
    coords = np.array([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
    confidence = np.array([[0.9], [0.1], [0.9]])
    repaired = confidence_aware_repair(coords, confidence)
    assert repaired.shape == coords.shape
    assert np.isfinite(repaired).all()


def test_metrics_behaviour():
    ref = np.zeros((4, 2, 3))
    est = ref.copy()
    assert mean_joint_displacement(ref, est) == 0.0
    assert low_confidence_rate(np.array([[0.9, 0.1]]), 0.35) == 0.5
    assert temporal_jitter(ref) == 0.0


def test_shape_validation():
    with pytest.raises(ValueError):
        interpolate_missing(np.zeros((3, 2)), np.zeros((3, 2)))


def test_symmetric_smoother_preserves_linear_motion():
    from frametrust.repair import confidence_aware_symmetric_repair

    coords = np.array([
        [[0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
        [[2.0, 0.0, 0.0]],
    ])
    confidence = np.ones((3, 1))
    smoothed = confidence_aware_symmetric_repair(
        coords, confidence, threshold=0.35, alpha_low=0.2, alpha_high=0.2
    )
    # At the middle frame, the centered neighbor average equals the true
    # coordinate on a constant-velocity trajectory (no one-sided phase lag).
    assert np.allclose(smoothed[1, 0], coords[1, 0])


def test_symmetric_smoother_reduces_single_frame_spike():
    from frametrust.repair import confidence_aware_symmetric_repair

    coords = np.array([
        [[0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
    ])
    confidence = np.ones((3, 1))
    smoothed = confidence_aware_symmetric_repair(
        coords, confidence, threshold=0.35, alpha_low=0.2, alpha_high=0.2
    )
    assert smoothed[1, 0, 0] == pytest.approx(0.2)
