import numpy as np
import pandas as pd
import pytest

from scripts.sweep_repair import evaluate_common_support_metrics


def test_common_support_uses_only_points_detected_by_raw():
    annotations = pd.DataFrame([
        {"frame_index": 0, "joint_index": 0, "x": 0.0, "y": 0.0, "visible": 1},
        {"frame_index": 0, "joint_index": 1, "x": 1.0, "y": 0.0, "visible": 1},
        {"frame_index": 1, "joint_index": 0, "x": 0.0, "y": 0.0, "visible": 1},
        {"frame_index": 1, "joint_index": 1, "x": 1.0, "y": 0.0, "visible": 1},
    ])
    raw = np.array([
        [[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
        [[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
    ])
    estimate = raw.copy()
    estimate[:, 0, 0] = 0.1

    result = evaluate_common_support_metrics(
        annotations, np.array([0, 1]), raw, estimate, pck_fraction=0.15
    )
    assert result["gt_common_support_points"] == 2
    assert result["gt_common_support_coverage"] == pytest.approx(1.0)
    assert result["gt_common_support_mean_error_body_diagonal"] == pytest.approx(0.1)
    assert result["gt_common_support_pck"] == pytest.approx(1.0)


def test_common_support_counts_missing_method_output_as_failure():
    annotations = pd.DataFrame([
        {"frame_index": 0, "joint_index": 0, "x": 0.0, "y": 0.0, "visible": 1},
        {"frame_index": 0, "joint_index": 1, "x": 1.0, "y": 0.0, "visible": 1},
    ])
    raw = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    estimate = np.array([[[np.nan, np.nan, np.nan], [1.0, 0.0, 0.0]]])

    result = evaluate_common_support_metrics(
        annotations, np.array([0]), raw, estimate, pck_fraction=0.1
    )
    assert result["gt_common_support_points"] == 2
    assert result["gt_common_support_coverage"] == pytest.approx(0.5)
    assert result["gt_common_support_pck"] == pytest.approx(0.5)
