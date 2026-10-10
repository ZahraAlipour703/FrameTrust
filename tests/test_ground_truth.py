import numpy as np
import pandas as pd
import pytest

from frametrust.ground_truth import evaluate_ground_truth, validate_ground_truth


def sample_annotations():
    return pd.DataFrame(
        {
            "frame_index": [0, 0],
            "joint_index": [11, 12],
            "joint_name": ["left_shoulder", "right_shoulder"],
            "x": [0.2, 0.8],
            "y": [0.2, 0.8],
            "visible": [1, 1],
        }
    )


def test_perfect_predictions_have_zero_error_and_full_pck():
    annotations = sample_annotations()
    estimate = np.full((1, 33, 3), np.nan)
    estimate[0, 11, :2] = [0.2, 0.2]
    estimate[0, 12, :2] = [0.8, 0.8]

    result = evaluate_ground_truth(annotations, [0], estimate)

    assert result["gt_evaluable_points"] == 2
    assert result["gt_prediction_coverage"] == 1.0
    assert result["gt_mean_error_normalized_image"] == 0.0
    assert result["gt_mean_error_body_diagonal"] == 0.0
    assert result["gt_pck"] == 1.0


def test_missing_predictions_reduce_coverage_and_pck():
    annotations = sample_annotations()
    estimate = np.full((1, 33, 3), np.nan)
    estimate[0, 11, :2] = [0.2, 0.2]

    result = evaluate_ground_truth(annotations, [0], estimate)

    assert result["gt_evaluable_points"] == 2
    assert result["gt_prediction_coverage"] == 0.5
    assert result["gt_pck"] == 0.5


def test_duplicate_annotations_are_rejected():
    annotations = sample_annotations()
    annotations = pd.concat([annotations, annotations.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        validate_ground_truth(annotations)


def test_visible_points_require_normalized_coordinates():
    annotations = sample_annotations()
    annotations.loc[0, "x"] = 1.2

    with pytest.raises(ValueError, match=r"normalized to"):
        validate_ground_truth(annotations)


def test_annotations_outside_current_run_are_reported():
    annotations = sample_annotations()
    estimate = np.full((1, 33, 3), np.nan)

    result = evaluate_ground_truth(annotations, [5], estimate)

    assert result["gt_points_outside_run"] == 2
    assert result["gt_evaluable_points"] == 0
    assert np.isnan(result["gt_pck"])
