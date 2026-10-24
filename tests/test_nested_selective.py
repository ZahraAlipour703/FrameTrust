import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "evaluate_selective_repair_test_module",
    SCRIPTS / "evaluate_selective_repair.py",
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _toy_cache():
    rng = np.random.default_rng(24)
    names = list(module.DEFAULT_CANDIDATES)
    selected = [f"{i:04d}" for i in range(8)]
    cache = {}
    for i, seq in enumerate(selected):
        frames, joints, dims = 20, 6, 4
        features = rng.normal(size=(frames, joints, dims)).astype(np.float32)
        raw_error = np.clip(0.18 + 0.05 * features[..., 0], 0.0, 1.0)
        errors = np.empty((frames, joints, len(names)), dtype=np.float32)
        errors[..., 0] = raw_error
        for k in range(1, len(names)):
            improvement = 0.025 * k + 0.02 * features[..., min(k - 1, dims - 1)]
            errors[..., k] = np.clip(raw_error - improvement, 0.0, 1.0)

        raw = np.zeros((frames, joints, 2), dtype=float)
        candidates = {}
        for k, name in enumerate(names):
            candidates[name] = raw + (0.01 * k)
        cache[(seq, "clean")] = {
            "action": f"action_{i // 2}",
            "features": features,
            "errors": errors,
            "scoreable": np.ones((frames, joints), dtype=bool),
            "valid_gt": np.ones((frames, joints), dtype=bool),
            "torso_scale": np.ones(frames, dtype=float),
            "gt_xy": np.zeros((frames, joints, 2), dtype=float),
            "raw": raw,
            "candidates": candidates,
        }
    return cache, selected, names


def test_nested_threshold_selection_excludes_outer_heldout_and_returns_one_margin_each():
    cache, selected, names = _toy_cache()
    thresholds = [0.0, 0.01, 0.05, 0.10]
    margins, selected_rows, sweep_rows = module.nested_threshold_selection(
        cache=cache,
        selected=selected,
        conditions=["clean"],
        names=names,
        thresholds=thresholds,
        ridge=1.0,
        seed=123,
        n_inner_folds=3,
        pck_tolerance=0.005,
        min_coverage=0.90,
        max_false_correction=1.0,
    )

    assert len(margins) == len(selected)
    assert set(margins) == {("clean", seq) for seq in selected}
    assert all(value in thresholds for value in margins.values())
    assert len(selected_rows) == len(selected)
    assert len(sweep_rows) == len(selected) * (len(selected) - 1) * len(thresholds)
    for row in sweep_rows:
        assert row["outer_heldout_sequence"] != row["inner_validation_sequence"]
    for row in selected_rows:
        assert row["selected_inner_pck_gap_from_best"] <= 0.005 + 1e-9



def test_feature_cache_round_trip_and_signature_guard(tmp_path):
    cache, selected, names = _toy_cache()
    source = cache[(selected[0], "clean")]
    path = tmp_path / "feature_cache" / "toy.npz"
    module._save_feature_cache(path, source, names, signature="test-signature")

    loaded = module._load_feature_cache(path, names, expected_signature="test-signature")
    assert loaded["action"] == source["action"]
    assert np.array_equal(loaded["scoreable"], source["scoreable"])
    assert np.allclose(loaded["features"], source["features"])
    assert np.allclose(loaded["errors"], source["errors"])
    for name in names:
        assert np.allclose(loaded["candidates"][name], source["candidates"][name])

    with pytest.raises(ValueError, match="settings do not match"):
        module._load_feature_cache(path, names, expected_signature="wrong-signature")
