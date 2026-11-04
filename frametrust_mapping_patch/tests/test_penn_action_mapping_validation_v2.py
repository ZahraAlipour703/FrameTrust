from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from penn_action_mapping_validation_v2 import (  # noqa: E402
    evaluate_mapping,
    frame_masks_and_scales,
    make_padded_crop,
)


def test_make_padded_crop_translates_roi_coordinates_and_clips_to_image():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    result = make_padded_crop(image, np.array([20, 10, 60, 70]), padding=0.10)
    assert result is not None
    crop, (left, top, right, bottom) = result
    assert (left, top, right, bottom) == (16, 4, 64, 76)
    assert crop.shape[:2] == (72, 48)


def test_make_padded_crop_rejects_invalid_boxes():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    assert make_padded_crop(image, np.array([10, 10, 10, 30]), 0.15) is None
    assert make_padded_crop(image, np.array([np.nan, 10, 20, 30]), 0.15) is None


def test_bbox_normalized_metrics_count_missing_predictions_as_pck_failures():
    gt = np.array([
        [[10.0, 10.0], [20.0, 20.0]],
        [[10.0, 10.0], [20.0, 20.0]],
    ])
    visible = np.ones((2, 2), dtype=bool)
    boxes = np.array([[0.0, 0.0, 3.0, 4.0], [0.0, 0.0, 3.0, 4.0]])
    sizes = np.array([[40, 40], [40, 40]])
    score_mask, bbox_valid, diagonal = frame_masks_and_scales(gt, visible, boxes, sizes)
    assert score_mask.all()
    assert bbox_valid.all()
    assert np.allclose(diagonal, 5.0)

    pred = np.array([
        [[10.0, 10.0], [21.0, 20.0]],
        [[np.nan, np.nan], [20.0, 20.0]],
    ])
    metrics = evaluate_mapping(pred, gt, score_mask, diagonal)
    assert metrics["gt_points"] == 4
    assert metrics["detected_points"] == 3
    assert np.isclose(metrics["prediction_coverage"], 0.75)
    assert np.isclose(metrics["pck_bbox_0.20"], 0.75)
    assert np.isclose(metrics["mean_capped_error_bbox_norm_all_gt"], 0.30)


def test_bbox_diagonal_scoring_does_not_require_shoulder_hip_anchor():
    gt = np.array([[[5.0, 5.0], [15.0, 15.0]]])
    visible = np.array([[True, True]])
    boxes = np.array([[0.0, 0.0, 20.0, 30.0]])
    sizes = np.array([[40, 40]])
    mask, valid, diagonal = frame_masks_and_scales(gt, visible, boxes, sizes)
    assert valid[0]
    assert mask.all()
    assert np.isclose(diagonal[0], np.hypot(20.0, 30.0))
