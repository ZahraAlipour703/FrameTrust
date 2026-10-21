import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_temporal_fidelity.py"
spec = importlib.util.spec_from_file_location("evaluate_temporal_fidelity_test_module", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_temporal_metrics_zero_for_identical_constant_velocity_trajectories():
    frames, joints = 8, 3
    t = np.arange(frames, dtype=float)
    gt = np.zeros((frames, joints, 2), dtype=float)
    gt[:, :, 0] = t[:, None] * 2.0
    gt[:, :, 1] = t[:, None] * 0.5
    valid = np.ones((frames, joints), dtype=bool)

    metrics, n_supported, n_possible = module.temporal_metric_values(
        gt.copy(), gt, valid, frame_diagonal=100.0, fps=30.0
    )

    assert n_supported == (frames - 2) * joints
    assert n_possible == (frames - 2) * joints
    assert np.isclose(metrics["temporal_jitter_norm_s2"], 0.0, atol=1e-12)
    assert np.isclose(metrics["velocity_error_norm_s"], 0.0, atol=1e-12)
    assert np.isclose(metrics["acceleration_error_norm_s2"], 0.0, atol=1e-12)


def test_temporal_metrics_count_only_valid_finite_triples():
    frames, joints = 6, 2
    gt = np.zeros((frames, joints, 2), dtype=float)
    pred = gt.copy()
    valid = np.ones((frames, joints), dtype=bool)
    pred[2, 0] = np.nan
    valid[3, 1] = False

    metrics, n_supported, n_possible = module.temporal_metric_values(
        pred, gt, valid, frame_diagonal=100.0, fps=30.0
    )

    # Invalid GT at frame 3 removes three possible triples for joint 1.
    # A missing prediction at frame 2 removes three supported triples for joint 0.
    assert n_possible == 2 * (frames - 2) - 3
    assert n_supported == 2
    assert np.isfinite(metrics["temporal_jitter_norm_s2"])


def test_temporal_mask_keeps_visible_points_without_torso_scale():
    gt = np.array([
        [[10.0, 10.0], [20.0, 20.0]],
        [[11.0, 10.0], [21.0, 20.0]],
        [[12.0, 10.0], [22.0, 20.0]],
    ])
    visible = np.ones((3, 2), dtype=bool)
    # The first landmark is invisible on the middle frame, but that must not
    # invalidate the second landmark or require any torso-scale anchors.
    visible[1, 0] = False

    mask = module.temporal_ground_truth_mask(
        gt, visible, width=100, height=100
    )

    assert mask[1, 0] is np.False_ or not mask[1, 0]
    assert mask[1, 1]
    assert mask.sum() == 5


def test_temporal_mask_excludes_out_of_frame_annotations():
    gt = np.array([[[10.0, 10.0], [101.0, 20.0]]])
    visible = np.ones((1, 2), dtype=bool)

    mask = module.temporal_ground_truth_mask(
        gt, visible, width=100, height=100
    )

    assert mask.tolist() == [[True, False]]
