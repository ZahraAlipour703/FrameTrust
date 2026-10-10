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
