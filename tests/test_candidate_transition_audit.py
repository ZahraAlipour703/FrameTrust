import numpy as np
import pytest

from frametrust.transition_audit import (
    candidate_run_lengths,
    conditional_velocity_metrics,
    summarize_choice_sequence,
    transition_masks,
    transition_matrix_rows,
)


def test_transition_masks_mark_only_triples_touching_a_switch():
    choices = np.array([[0], [0], [0], [1], [1], [1]])
    adjacent, transition, stable = transition_masks(choices)
    assert adjacent[:, 0].tolist() == [False, False, True, False, False]
    assert transition[:, 0].tolist() == [False, True, True, False]
    assert stable[:, 0].tolist() == [True, False, False, True]


def test_summarize_choice_sequence_separates_entry_exit_and_repair_switch():
    # raw -> repair-A -> repair-B -> raw
    choices = np.array([[0], [1], [2], [0]])
    result = summarize_choice_sequence(choices, ["raw", "repair_a", "repair_b"])
    assert result["n_candidate_transitions"] == 3
    assert result["repair_entry_count"] == 1
    assert result["repair_exit_count"] == 1
    assert result["repair_method_switch_count"] == 1
    assert result["repair_fraction"] == pytest.approx(0.5)
    assert result["transitions_per_1000_joint_steps"] == pytest.approx(1000.0)


def test_candidate_run_lengths_preserve_jointwise_contiguous_dwell():
    choices = np.array([
        [0, 1],
        [0, 1],
        [2, 0],
        [2, 0],
        [2, 1],
    ])
    runs = candidate_run_lengths(choices, ["raw", "repair_a", "repair_b"])
    assert sorted(runs["raw"]) == [2, 2]
    assert sorted(runs["repair_a"]) == [1, 2]
    assert sorted(runs["repair_b"]) == [3]


def test_transition_matrix_counts_only_actual_changes():
    choices = np.array([[0], [1], [1], [2], [0]])
    rows = transition_matrix_rows(choices, ["raw", "a", "b"], "0001", "blur")
    observed = {(r["from_candidate"], r["to_candidate"]): r["transition_count"] for r in rows}
    assert observed == {("raw", "a"): 1, ("a", "b"): 1, ("b", "raw"): 1}


def test_torso_scoreable_transition_rate_uses_pairs_valid_on_both_frames():
    choices = np.array([[0], [1], [1], [0]])
    scoreable = np.array([[True], [True], [False], [True]])
    result = summarize_choice_sequence(choices, ["raw", "repair"], scoreable)
    assert result["n_torso_scoreable_adjacent_pairs"] == 1
    assert result["n_torso_scoreable_transitions"] == 1
    assert result["torso_scoreable_transition_rate"] == pytest.approx(1.0)


def test_conditional_velocity_metrics_detects_a_prediction_step_jump():
    gt = np.zeros((3, 1, 2), dtype=float)
    pred = gt.copy()
    pred[1, 0, 0] = 10.0
    valid = np.ones((3, 1), dtype=bool)
    support = np.array([[True], [False]])
    metrics, n = conditional_velocity_metrics(
        pred, gt, valid, frame_diagonal=10.0, fps=30.0, group_support=support,
    )
    assert n == 1
    assert metrics["one_step_velocity_error_norm_s"] == pytest.approx(30.0)
