import numpy as np
import pytest

from frametrust.selective import (
    apply_candidate_choices,
    build_selective_features,
    candidate_error_targets,
    fit_error_predictor,
    predict_candidate_errors,
    select_candidates,
)


def _toy_candidates():
    raw = np.array([
        [[0.0, 0.0]],
        [[np.nan, np.nan]],
        [[2.0, 0.0]],
    ])
    interp = np.array([
        [[0.0, 0.0]],
        [[1.0, 0.0]],
        [[2.0, 0.0]],
    ])
    return {"raw": raw, "linear_interpolation": interp}


def test_selective_features_are_finite_and_have_expected_shape():
    candidates = _toy_candidates()
    raw = candidates["raw"]
    confidence = np.array([[0.9], [0.0], [0.9]])
    sizes = np.array([[100, 100], [100, 100], [100, 100]])
    features = build_selective_features(raw, confidence, sizes, candidates)
    assert features.shape[:2] == raw.shape[:2]
    assert features.ndim == 3
    assert np.isfinite(features).all()


def test_candidate_targets_penalize_missing_and_mask_invalid_ground_truth():
    candidates = _toy_candidates()
    gt = np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]])
    valid = np.array([[True], [True], [False]])
    scale = np.array([10.0, 10.0, 10.0])
    errors, scoreable = candidate_error_targets(
        candidates, gt, valid, scale,
        ["raw", "linear_interpolation"],
    )
    assert errors[1, 0, 0] == pytest.approx(1.0)
    assert errors[1, 0, 1] == pytest.approx(0.0)
    assert scoreable[:, 0].tolist() == [True, True, False]


def test_selector_abstains_unless_predicted_gain_clears_margin():
    predicted = np.array([
        [[0.10, 0.095, 0.08]],
        [[0.10, 0.07, 0.09]],
    ])
    choices = select_candidates(predicted, min_predicted_gain=0.02)
    assert choices.tolist() == [[0, 1]]


def test_fit_and_predict_error_model_shapes():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(80, 5))
    y = np.column_stack([
        np.clip(0.4 + 0.1 * x[:, 0], 0, 1),
        np.clip(0.2 + 0.05 * x[:, 1], 0, 1),
        np.clip(0.3 + 0.1 * x[:, 2], 0, 1),
    ])
    model = fit_error_predictor(x, y, ridge=1.0)
    predicted = predict_candidate_errors(model, x[:4])
    assert predicted.shape == (4, 3)
    assert np.isfinite(predicted).all()
    assert ((predicted >= 0) & (predicted <= 1)).all()


def test_apply_choices_composes_jointwise_hybrid():
    candidates = {
        "raw": np.zeros((2, 1, 2)),
        "repair": np.ones((2, 1, 2)),
    }
    out = apply_candidate_choices(candidates, np.array([[0], [1]]), ["raw", "repair"])
    assert np.allclose(out[:, 0], [[0.0, 0.0], [1.0, 1.0]])
