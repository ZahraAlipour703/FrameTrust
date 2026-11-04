import numpy as np
import pytest

from frametrust.switch_counterfactual import switch_counterfactual_events


def test_switch_boundary_compared_with_same_candidate_continuations():
    # The hybrid switches from raw to repair at t=1. The raw trajectory would
    # make a large one-step error; continuing the repair is exact.
    raw = np.array([[[4.0, 0.0]], [[5.0, 0.0]], [[6.0, 0.0]]])
    repair = np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]])
    choices = np.array([[0], [1], [1]])
    gt = np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]])
    valid = np.ones((3, 1), dtype=bool)

    rows = switch_counterfactual_events(
        {"raw": raw, "repair": repair}, choices, gt, valid,
        ["raw", "repair"], frame_diagonal=1.0, fps=1.0,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["transition_type"] == "repair_entry"
    assert row["hybrid_velocity_error_norm_s"] == pytest.approx(4.0)
    assert row["old_continuation_velocity_error_norm_s"] == pytest.approx(0.0)
    assert row["new_continuation_velocity_error_norm_s"] == pytest.approx(0.0)
    assert row["hybrid_minus_old_continuation"] == pytest.approx(4.0)
    assert row["hybrid_minus_new_continuation"] == pytest.approx(4.0)
    assert row["hybrid_worse_than_both_continuations"] is True


def test_switch_between_two_finite_repairs_compares_both_counterfactuals():
    raw = np.zeros((2, 1, 2))
    a = np.array([[[0.0, 0.0]], [[0.0, 0.0]]])
    b = np.array([[[0.0, 0.0]], [[10.0, 0.0]]])
    choices = np.array([[1], [2]])
    gt = np.zeros((2, 1, 2))
    valid = np.ones((2, 1), dtype=bool)

    rows = switch_counterfactual_events(
        {"raw": raw, "a": a, "b": b}, choices, gt, valid,
        ["raw", "a", "b"], frame_diagonal=10.0, fps=1.0,
    )
    row = rows[0]
    assert row["transition_type"] == "repair_method_switch"
    assert row["hybrid_velocity_error_norm_s"] == pytest.approx(1.0)
    assert row["old_continuation_velocity_error_norm_s"] == pytest.approx(0.0)
    assert row["new_continuation_velocity_error_norm_s"] == pytest.approx(1.0)
    assert row["best_continuation_velocity_error_norm_s"] == pytest.approx(0.0)
    assert row["hybrid_worse_than_both_continuations"] is False


def test_invalid_ground_truth_pair_keeps_event_but_marks_metrics_missing():
    raw = np.zeros((2, 1, 2))
    repair = np.ones((2, 1, 2))
    choices = np.array([[0], [1]])
    gt = np.zeros((2, 1, 2))
    valid = np.array([[True], [False]])
    rows = switch_counterfactual_events(
        {"raw": raw, "repair": repair}, choices, gt, valid,
        ["raw", "repair"], frame_diagonal=10.0,
    )
    assert len(rows) == 1
    assert rows[0]["gt_pair_valid"] is False
    assert np.isnan(rows[0]["hybrid_velocity_error_norm_s"])


def test_no_switches_returns_empty_events():
    raw = np.zeros((3, 2, 2))
    choices = np.zeros((3, 2), dtype=int)
    gt = np.zeros((3, 2, 2))
    valid = np.ones((3, 2), dtype=bool)
    rows = switch_counterfactual_events(
        {"raw": raw}, choices, gt, valid, ["raw"], frame_diagonal=10.0,
    )
    assert rows == []
