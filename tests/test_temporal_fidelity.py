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
    assert metrics["temporal_jitter_norm_s2"] == 0.0
    assert metrics["velocity_error_norm_s"] == 0.0
    assert metrics["acceleration_error_norm_s2"] == 0.0


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

    # Joint 0 loses triples centred around frames 1, 2, 3. Joint 1 loses
    # triples containing GT-invalid frame 3 (triples [1,2,3], [2,3,4]).
    assert n_possible == 2 * (frames - 2) - 2
    assert n_supported < n_possible
    assert np.isfinite(metrics["temporal_jitter_norm_s2"])
