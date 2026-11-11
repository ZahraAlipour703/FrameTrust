import numpy as np

from frametrust.sequence_routing import route_candidate_sequence


def _independent(predicted_errors, min_predicted_gain=0.01):
    expected = np.asarray(predicted_errors, dtype=float)
    flat = expected.reshape(-1, expected.shape[-1])
    choice = np.zeros(flat.shape[0], dtype=np.int64)
    best_repair = 1 + np.argmin(flat[:, 1:], axis=1)
    raw_error = flat[:, 0]
    best_error = flat[np.arange(len(flat)), best_repair]
    repair = (raw_error - best_error) >= min_predicted_gain
    choice[repair] = best_repair[repair]
    return choice.reshape(expected.shape[:-1])


def _candidates(n_frames=3):
    return {
        "raw": np.zeros((n_frames, 1, 2), dtype=float),
        "repair": np.ones((n_frames, 1, 2), dtype=float),
    }


def test_zero_penalty_exactly_matches_independent_selector():
    predicted = np.array([
        [[0.10, 0.08, 0.09]],
        [[0.10, 0.095, 0.07]],
        [[0.10, 0.06, 0.09]],
    ])
    candidates = {
        "raw": np.zeros((3, 1, 2)),
        "a": np.ones((3, 1, 2)),
        "b": np.full((3, 1, 2), 2.0),
    }
    names = ["raw", "a", "b"]
    actual = route_candidate_sequence(
        predicted, candidates, names, min_predicted_gain=0.01,
        switch_penalty=0.0, mode="constant",
    )
    expected = _independent(predicted, min_predicted_gain=0.01)
    assert np.array_equal(actual, expected)


def test_constant_switch_cost_suppresses_short_candidate_flips():
    predicted = np.array([
        [[0.10, 0.09]],
        [[0.10, 0.11]],
        [[0.10, 0.09]],
    ])
    candidates = _candidates()
    independent = _independent(predicted, min_predicted_gain=0.0)
    routed = route_candidate_sequence(
        predicted, candidates, ["raw", "repair"], min_predicted_gain=0.0,
        switch_penalty=0.03, mode="constant",
    )
    assert independent[:, 0].tolist() == [1, 0, 1]
    assert len(np.unique(routed[:, 0])) == 1


def test_disagreement_cost_is_zero_for_identical_candidates():
    predicted = np.array([
        [[0.10, 0.09]],
        [[0.10, 0.11]],
        [[0.10, 0.09]],
    ])
    candidates = {
        "raw": np.zeros((3, 1, 2)),
        "repair": np.zeros((3, 1, 2)),
    }
    routed = route_candidate_sequence(
        predicted, candidates, ["raw", "repair"], min_predicted_gain=0.0,
        switch_penalty=1.0, mode="disagreement",
    )
    assert np.array_equal(routed, _independent(predicted, 0.0))


def test_disagreement_cost_uses_torso_normalized_candidate_distance():
    predicted = np.array([
        [[0.10, 0.09]],
        [[0.10, 0.11]],
        [[0.10, 0.09]],
    ])
    near = {
        "raw": np.zeros((3, 1, 2)),
        "repair": np.full((3, 1, 2), 0.01),
    }
    far = {
        "raw": np.zeros((3, 1, 2)),
        "repair": np.full((3, 1, 2), 10.0),
    }
    kwargs = dict(
        candidate_names=["raw", "repair"], min_predicted_gain=0.0,
        switch_penalty=0.02, mode="disagreement", torso_scale=np.ones(3),
    )
    near_choice = route_candidate_sequence(predicted, near, **kwargs)
    far_choice = route_candidate_sequence(predicted, far, **kwargs)
    assert near_choice[:, 0].tolist() == [1, 0, 1]
    assert len(np.unique(far_choice[:, 0])) == 1


def test_validation_rejects_bad_shapes_and_negative_penalty():
    predicted = np.zeros((3, 1, 2))
    candidates = _candidates()
    try:
        route_candidate_sequence(
            predicted, candidates, ["raw", "repair"], switch_penalty=-1.0
        )
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative penalty must fail")
