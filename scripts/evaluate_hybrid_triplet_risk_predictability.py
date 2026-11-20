#!/usr/bin/env python3
"""Cross-fitted predictability of temporal error on spatially selected hybrid pose paths.

For each outer held-out Penn Action sequence:
  1. Generate training-sequence spatial choices with action-balanced inner cross-fitting.
  2. Compose the actual hybrid paths selected by the spatial selector.
  3. Fit a scalar Ridge model to predict acceleration error of those hybrid paths.
  4. Evaluate on the outer held-out sequence, whose spatial selector is trained on
     all other outer-training sequences.

This is deliberately a predictability pilot, not a new routing algorithm. It asks
whether inference-time features can predict the temporal error of the *actual mixed-
candidate trajectory* better than a global or candidate-triplet-pattern mean.

All target scaling and model fitting use outer-training sequences only. The official
Penn Action test split is not read. MediaPipe is not rerun; existing feature caches
are reused. Unsupported hybrid triples receive the maximum normalized target (1.0),
while temporal support is reported separately.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from frametrust.selective import (  # noqa: E402
    apply_candidate_choices,
    predict_candidate_errors,
    select_candidates,
)
from evaluate_selective_repair import _fit_selector_for_sequences  # noqa: E402
from evaluate_sequence_routing import (  # noqa: E402
    _action_balanced_folds,
    _load_cache_item,
)
from evaluate_temporal_risk_predictability import _load_sequence_info  # noqa: E402


def _safe_vector(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    out[mask] = values[mask].astype(np.float32)
    return out


def _build_hybrid_triplet_data(
    item: dict[str, Any],
    choices: np.ndarray,
    pose: np.ndarray,
    sequence_info: dict[str, Any],
    candidate_names: list[str],
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build inference features and unscaled temporal targets for centered triples.

    Returns:
      X: [T-2, J, F] inference-only features for the selected hybrid triple
      y: [T-2, J] acceleration error; NaN if hybrid prediction lacks support
      gt_mask: [T-2, J] three-frame GT validity
      pred_support: [T-2, J] finite hybrid-pose support
      pattern: [T-2, J] integer code for (choice[t-1], choice[t], choice[t+1])
    """
    features = np.asarray(item["features"], dtype=np.float32)
    raw = np.asarray(item["raw"], dtype=float)
    gt = np.asarray(sequence_info["gt_xy"], dtype=float)
    valid = np.asarray(sequence_info["valid_gt"], dtype=bool)
    selected = np.asarray(choices, dtype=np.int64)
    hybrid = np.asarray(pose, dtype=float)
    diagonal = float(sequence_info["diagonal"])
    k_count = len(candidate_names)

    if features.ndim != 3 or raw.ndim != 3 or hybrid.shape != raw.shape:
        raise ValueError("Unexpected feature/raw/hybrid dimensions")
    t_count, joint_count = features.shape[:2]
    if gt.shape != (t_count, joint_count, 2) or valid.shape != (t_count, joint_count):
        raise ValueError("Ground-truth shape does not match cached predictions")
    if selected.shape != (t_count, joint_count) or t_count < 3:
        raise ValueError("Choice shape mismatch or sequence shorter than three frames")
    if not np.isfinite(diagonal) or diagonal <= 0 or fps <= 0:
        raise ValueError("Invalid image diagonal or FPS")
    if ((selected < 0) | (selected >= k_count)).any():
        raise ValueError("Spatial choices contain an invalid candidate index")

    # Center-frame base features already include confidence, local motion,
    # neighboring confidence, missingness, and candidate-vs-raw descriptors.
    base = features[1:-1]

    prev_choice = selected[:-2]
    center_choice = selected[1:-1]
    next_choice = selected[2:]
    ids = np.stack([prev_choice, center_choice, next_choice], axis=-1)
    identity = np.concatenate(
        [np.eye(k_count, dtype=np.float32)[ids[..., q]] for q in range(3)],
        axis=-1,
    )
    switches = np.stack(
        [prev_choice != center_choice, center_choice != next_choice], axis=-1
    ).astype(np.float32)
    pattern = (prev_choice * k_count * k_count + center_choice * k_count + next_choice).astype(np.int32)

    hybrid_prev = hybrid[:-2]
    hybrid_center = hybrid[1:-1]
    hybrid_next = hybrid[2:]
    raw_prev = raw[:-2]
    raw_center = raw[1:-1]
    raw_next = raw[2:]

    hybrid_finite = (
        np.isfinite(hybrid_prev).all(axis=-1)
        & np.isfinite(hybrid_center).all(axis=-1)
        & np.isfinite(hybrid_next).all(axis=-1)
    )
    raw_finite = (
        np.isfinite(raw_prev).all(axis=-1)
        & np.isfinite(raw_center).all(axis=-1)
        & np.isfinite(raw_next).all(axis=-1)
    )

    hybrid_velocity = (hybrid_next - hybrid_prev) * (fps / (2.0 * diagonal))
    hybrid_acceleration = (
        hybrid_next - 2.0 * hybrid_center + hybrid_prev
    ) * (fps * fps / diagonal)
    raw_velocity = (raw_next - raw_prev) * (fps / (2.0 * diagonal))
    raw_acceleration = (
        raw_next - 2.0 * raw_center + raw_prev
    ) * (fps * fps / diagonal)

    hybrid_velocity = _safe_vector(hybrid_velocity, hybrid_finite)
    hybrid_acceleration = _safe_vector(hybrid_acceleration, hybrid_finite)
    raw_velocity = _safe_vector(raw_velocity, raw_finite)
    raw_acceleration = _safe_vector(raw_acceleration, raw_finite)
    delta_acceleration = hybrid_acceleration - raw_acceleration

    # Per-frame repair displacement relative to raw coordinates.
    offsets = []
    for hybrid_part, raw_part in (
        (hybrid_prev, raw_prev),
        (hybrid_center, raw_center),
        (hybrid_next, raw_next),
    ):
        both = np.isfinite(hybrid_part).all(axis=-1) & np.isfinite(raw_part).all(axis=-1)
        diff = (hybrid_part - raw_part) / diagonal
        offsets.append(_safe_vector(diff, both))
    offsets_flat = np.concatenate(offsets, axis=-1)

    step_prev = np.linalg.norm(hybrid_center - hybrid_prev, axis=-1) / diagonal
    step_next = np.linalg.norm(hybrid_next - hybrid_center, axis=-1) / diagonal
    step_prev = np.where(
        np.isfinite(hybrid_center).all(axis=-1) & np.isfinite(hybrid_prev).all(axis=-1),
        step_prev, 0.0,
    )
    step_next = np.where(
        np.isfinite(hybrid_next).all(axis=-1) & np.isfinite(hybrid_center).all(axis=-1),
        step_next, 0.0,
    )

    geometry = np.concatenate(
        [
            hybrid_velocity,
            hybrid_acceleration,
            raw_velocity,
            raw_acceleration,
            delta_acceleration,
            offsets_flat,
            step_prev[..., None].astype(np.float32),
            step_next[..., None].astype(np.float32),
            hybrid_finite[..., None].astype(np.float32),
            raw_finite[..., None].astype(np.float32),
        ],
        axis=-1,
    )

    X = np.concatenate([base, identity, switches, geometry], axis=-1)
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    X = np.clip(X, -10.0, 10.0).astype(np.float32)

    gt_mask = valid[:-2] & valid[1:-1] & valid[2:]
    gt_acceleration = (
        gt[2:] - 2.0 * gt[1:-1] + gt[:-2]
    ) * (fps * fps / diagonal)
    raw_error = np.linalg.norm(hybrid_acceleration - gt_acceleration, axis=-1)
    y = np.full((t_count - 2, joint_count), np.nan, dtype=np.float32)
    score = gt_mask & hybrid_finite & np.isfinite(raw_error)
    y[score] = raw_error[score].astype(np.float32)

    return X, y, gt_mask, hybrid_finite, pattern


def _fit_scalar_ridge(
    X: np.ndarray,
    y: np.ndarray,
    ridge: float,
) -> dict[str, np.ndarray | float | int]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X and y must be 2D/1D with matching samples")
    good = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[good], y[good]
    if X.shape[0] < 3 or X.shape[1] < 1:
        raise ValueError("Not enough finite training rows for Ridge")
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xz = (X - mean) / std
    design = np.column_stack([np.ones(Xz.shape[0]), Xz])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(lhs) @ rhs
    return {
        "mean": mean,
        "std": std,
        "coef": coef,
        "ridge": float(ridge),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }


def _predict_scalar_ridge(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    flat = X.reshape(-1, X.shape[-1])
    if flat.shape[1] != mean.size:
        raise ValueError("Model feature dimension mismatch")
    design = np.column_stack([np.ones(len(flat)), (flat - mean) / std])
    return np.clip(design @ coef, 0.0, 1.0).reshape(X.shape[:-1]).astype(np.float32)


def _pattern_baseline(
    train_patterns: np.ndarray,
    train_targets: np.ndarray,
    test_patterns: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    p = np.asarray(train_patterns, dtype=np.int64).reshape(-1)
    y = np.asarray(train_targets, dtype=float).reshape(-1)
    good = np.isfinite(y)
    p, y = p[good], y[good]
    if not len(y):
        raise RuntimeError("No valid rows for pattern-mean baseline")
    global_mean = float(y.mean())
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for key, value in zip(p.tolist(), y.tolist()):
        sums[key] = sums.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1
    means = {key: sums[key] / counts[key] for key in sums}
    test = np.asarray(test_patterns, dtype=np.int64)
    pred = np.fromiter(
        (means.get(int(key), global_mean) for key in test.reshape(-1)),
        dtype=np.float32,
        count=test.size,
    ).reshape(test.shape)
    return pred, global_mean, len(means)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return float("nan")
    return float(pd.Series(x[good]).corr(pd.Series(y[good]), method="spearman"))


def _bootstrap_ci(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[draws].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out", type=Path, default=Path("outputs/selective_repair_crossfit"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--out", type=Path, default=Path("outputs/hybrid_triplet_risk_predictability"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--max-train-points-per-sequence", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()

    if args.fps <= 0 or args.ridge < 0 or args.inner_folds < 2:
        parser.error("FPS must be positive, ridge non-negative, and inner-folds >=2")
    if args.max_train_points_per_sequence < 0 or args.bootstrap_samples < 100:
        parser.error("max train points must be non-negative and bootstrap samples >=100")

    start_time = time.time()
    source_out = args.source_out.resolve()
    dataset_root = args.dataset_root.resolve()
    metadata_path = source_out / "run_metadata.json"
    margin_path = source_out / "nested_threshold_selection.csv"
    cache_dir = source_out / "feature_cache"
    for path in (metadata_path, margin_path, cache_dir):
        if not path.exists():
            raise FileNotFoundError(f"Missing required source artifact: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = [str(s).zfill(4) for s in metadata["selected_sequences"]]
    conditions = list(metadata["conditions"])
    candidate_names = list(metadata["candidate_order"])
    signature = str(metadata["feature_cache"]["signature"])
    spatial_ridge = float(metadata.get("ridge", 10.0))
    if not candidate_names or candidate_names[0] != "raw" or len(selected) < 3:
        raise ValueError("Metadata must define raw-first candidates and >=3 sequences")

    margin_df = pd.read_csv(margin_path, dtype={"outer_heldout_sequence": str})
    margin_df["outer_heldout_sequence"] = margin_df["outer_heldout_sequence"].str.zfill(4)
    margin_map = {
        (str(row.condition), str(row.outer_heldout_sequence)): float(row.selected_margin)
        for row in margin_df.itertuples(index=False)
    }

    print("Loading compatible caches and validating ground truth...", flush=True)
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    sequence_info: dict[str, dict[str, Any]] = {}
    for seq in selected:
        for condition in conditions:
            cache_path = cache_dir / f"{seq}_{condition}.npz"
            cache[(seq, condition)] = _load_cache_item(cache_path, candidate_names, signature)
        sequence_info[seq] = _load_sequence_info(
            dataset_root, seq, cache[(seq, conditions[0])]["gt_xy"]
        )
        for condition in conditions:
            cached_gt = cache[(seq, condition)]["gt_xy"]
            expected_gt = sequence_info[seq]["gt_xy"]
            if cached_gt.shape != expected_gt.shape or not np.allclose(
                cached_gt, expected_gt, equal_nan=True
            ):
                raise ValueError(f"Cache/GT mismatch for {seq}/{condition}")

    result_rows: list[dict[str, Any]] = []
    sequences_count = len(selected)

    for cond_idx, condition in enumerate(conditions):
        print(f"\n[{condition}] cross-fitting spatial paths", flush=True)
        for outer_idx, heldout in enumerate(selected):
            outer_training = [seq for seq in selected if seq != heldout]
            key = (condition, heldout)
            if key not in margin_map:
                raise KeyError(f"No nested spatial margin for {condition}/{heldout}")
            spatial_margin = margin_map[key]

            folds = _action_balanced_folds(
                cache,
                condition,
                outer_training,
                args.inner_folds,
                args.seed + 1009 * outer_idx + 9176 * cond_idx,
            )

            train_feature_blocks: list[np.ndarray] = []
            train_raw_target_blocks: list[np.ndarray] = []
            train_pattern_blocks: list[np.ndarray] = []
            train_seq_raw_errors: list[np.ndarray] = []
            train_seq_masks: list[np.ndarray] = []
            train_seq_data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

            # Cross-fitted spatial choices on all outer-training sequences.
            for fold_idx, validation_sequences in enumerate(folds):
                validation_set = set(validation_sequences)
                inner_training = [seq for seq in outer_training if seq not in validation_set]
                spatial_model = _fit_selector_for_sequences(
                    cache,
                    condition,
                    inner_training,
                    spatial_ridge,
                    f"triplet-risk inner spatial fold={fold_idx + 1}; outer={heldout}; {condition}",
                )
                for val_seq in validation_sequences:
                    val_item = cache[(val_seq, condition)]
                    pred = predict_candidate_errors(spatial_model, val_item["features"])
                    choices = select_candidates(pred, spatial_margin)
                    hybrid_pose = apply_candidate_choices(val_item["candidates"], choices, candidate_names)
                    X, y_raw, gt_mask, pred_support, pattern = _build_hybrid_triplet_data(
                        val_item, choices, hybrid_pose, sequence_info[val_seq],
                        candidate_names, args.fps,
                    )
                    train_seq_data.append((val_seq, X, y_raw, pattern, gt_mask, pred_support))
                    eligible = gt_mask.reshape(-1)
                    finite_targets = y_raw.reshape(-1)
                    finite_targets = finite_targets[eligible & np.isfinite(finite_targets)]
                    if finite_targets.size:
                        train_seq_raw_errors.append(finite_targets)

            if not train_seq_raw_errors:
                raise RuntimeError(f"No finite OOF hybrid temporal targets for {condition}/{heldout}")
            train_scale = float(np.quantile(np.concatenate(train_seq_raw_errors), 0.95))
            if not np.isfinite(train_scale) or train_scale <= 1e-9:
                train_scale = 1.0

            # Normalize targets; unsupported OOF triples receive the maximum value.
            sampled_X: list[np.ndarray] = []
            sampled_y: list[np.ndarray] = []
            sampled_patterns: list[np.ndarray] = []
            rng = np.random.default_rng(args.seed + 7919 * outer_idx + 104729 * cond_idx)
            for seq, X, y_raw, pattern, gt_mask, pred_support in train_seq_data:
                y_norm = np.ones(y_raw.shape, dtype=np.float32)
                supported = gt_mask & pred_support & np.isfinite(y_raw)
                y_norm[supported] = np.clip(y_raw[supported] / train_scale, 0.0, 1.0)
                eligible_idx = np.flatnonzero(gt_mask.reshape(-1))
                if args.max_train_points_per_sequence > 0 and len(eligible_idx) > args.max_train_points_per_sequence:
                    eligible_idx = np.sort(rng.choice(
                        eligible_idx, size=args.max_train_points_per_sequence, replace=False
                    ))
                x_flat = X.reshape(-1, X.shape[-1])
                y_flat = y_norm.reshape(-1)
                p_flat = pattern.reshape(-1)
                sampled_X.append(x_flat[eligible_idx])
                sampled_y.append(y_flat[eligible_idx])
                sampled_patterns.append(p_flat[eligible_idx])

            X_train = np.concatenate(sampled_X, axis=0)
            y_train = np.concatenate(sampled_y, axis=0)
            p_train = np.concatenate(sampled_patterns, axis=0)
            path_model = _fit_scalar_ridge(X_train, y_train, args.ridge)
            train_global_mean = float(y_train.mean())

            # Outer spatial prediction and hybrid path on the held-out sequence.
            outer_spatial_model = _fit_selector_for_sequences(
                cache,
                condition,
                outer_training,
                spatial_ridge,
                f"triplet-risk outer spatial; heldout={heldout}; {condition}",
            )
            test_item = cache[(heldout, condition)]
            test_pred = predict_candidate_errors(outer_spatial_model, test_item["features"])
            test_choices = select_candidates(test_pred, spatial_margin)
            test_pose = apply_candidate_choices(test_item["candidates"], test_choices, candidate_names)
            X_test, y_test_raw, test_gt_mask, test_support, test_pattern = _build_hybrid_triplet_data(
                test_item, test_choices, test_pose, sequence_info[heldout],
                candidate_names, args.fps,
            )

            y_test = np.ones(y_test_raw.shape, dtype=np.float32)
            supported_test = test_gt_mask & test_support & np.isfinite(y_test_raw)
            y_test[supported_test] = np.clip(y_test_raw[supported_test] / train_scale, 0.0, 1.0)
            pred_model = _predict_scalar_ridge(path_model, X_test)
            pred_pattern, _, n_train_patterns = _pattern_baseline(
                p_train, y_train, test_pattern
            )
            pred_global = np.full(y_test.shape, train_global_mean, dtype=np.float32)

            mask = test_gt_mask
            if not mask.any():
                raise RuntimeError(f"No GT-valid hybrid triples for {heldout}/{condition}")
            true_values = y_test[mask]
            model_values = pred_model[mask]
            pattern_values = pred_pattern[mask]
            global_values = pred_global[mask]

            model_mae = float(np.mean(np.abs(model_values - true_values)))
            pattern_mae = float(np.mean(np.abs(pattern_values - true_values)))
            global_mae = float(np.mean(np.abs(global_values - true_values)))
            support = test_support[mask]
            supported_mae_model = (
                float(np.mean(np.abs(model_values[support] - true_values[support])))
                if support.any() else float("nan")
            )
            supported_mae_pattern = (
                float(np.mean(np.abs(pattern_values[support] - true_values[support])))
                if support.any() else float("nan")
            )
            result_rows.append({
                "condition": condition,
                "sequence": heldout,
                "action": test_item["action"],
                "n_gt_valid_triples": int(mask.sum()),
                "temporal_support_coverage": float(support.mean()),
                "spatial_margin_used": float(spatial_margin),
                "train_target_scale_q95": train_scale,
                "n_train_sequences": len(outer_training),
                "n_train_rows": int(len(y_train)),
                "n_train_patterns": int(n_train_patterns),
                "n_test_patterns": int(np.unique(test_pattern[mask]).size),
                "mae_model": model_mae,
                "mae_pattern_mean_baseline": pattern_mae,
                "mae_global_mean_baseline": global_mae,
                "mae_improvement_vs_pattern": pattern_mae - model_mae,
                "mae_improvement_vs_global": global_mae - model_mae,
                "mae_model_supported_only": supported_mae_model,
                "mae_pattern_supported_only": supported_mae_pattern,
                "spearman_model": _spearman(model_values, true_values),
                "spearman_pattern_baseline": _spearman(pattern_values, true_values),
                "n_spatial_repairs": int((test_choices > 0).sum()),
                "n_spatial_points": int(test_choices.size),
                "model_ridge": float(args.ridge),
                "model_training_samples": int(path_model["n_samples"]),
            })

            if (outer_idx + 1) % 5 == 0 or outer_idx == sequences_count - 1:
                print(
                    f"  outer {outer_idx + 1}/{sequences_count} ({heldout}): "
                    f"MAE model={model_mae:.4f}, pattern={pattern_mae:.4f}, "
                    f"support={support.mean():.3f}",
                    flush=True,
                )

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    per_sequence = pd.DataFrame(result_rows)
    per_sequence_path = out / "hybrid_triplet_risk_by_sequence.csv"
    per_sequence.to_csv(per_sequence_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    for cond_idx, (condition, group) in enumerate(per_sequence.groupby("condition")):
        improvement_pattern = group["mae_improvement_vs_pattern"].to_numpy(float)
        improvement_global = group["mae_improvement_vs_global"].to_numpy(float)
        ci_pattern = _bootstrap_ci(improvement_pattern, args.seed + cond_idx, args.bootstrap_samples)
        ci_global = _bootstrap_ci(improvement_global, args.seed + 1000 + cond_idx, args.bootstrap_samples)
        summary_rows.append({
            "condition": condition,
            "n_sequences": int(group["sequence"].nunique()),
            "mae_model_sequence_mean": float(group["mae_model"].mean()),
            "mae_pattern_mean_baseline_sequence_mean": float(group["mae_pattern_mean_baseline"].mean()),
            "mae_global_mean_baseline_sequence_mean": float(group["mae_global_mean_baseline"].mean()),
            "mae_improvement_vs_pattern_sequence_mean": float(improvement_pattern.mean()),
            "mae_improvement_vs_pattern_ci95_low": ci_pattern[0],
            "mae_improvement_vs_pattern_ci95_high": ci_pattern[1],
            "mae_improvement_vs_global_sequence_mean": float(improvement_global.mean()),
            "mae_improvement_vs_global_ci95_low": ci_global[0],
            "mae_improvement_vs_global_ci95_high": ci_global[1],
            "spearman_model_sequence_mean": float(group["spearman_model"].mean()),
            "spearman_pattern_baseline_sequence_mean": float(group["spearman_pattern_baseline"].mean()),
            "temporal_support_coverage_sequence_mean": float(group["temporal_support_coverage"].mean()),
            "n_train_rows_mean": float(group["n_train_rows"].mean()),
            "n_train_patterns_mean": float(group["n_train_patterns"].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = out / "hybrid_triplet_risk_summary.csv"
    summary.to_csv(summary_path, index=False)

    protocol = {
        "experiment": "cross_fitted_hybrid_triplet_temporal_risk_predictability",
        "dataset": "Penn Action",
        "official_split": "selected official training sequences only; load_sequence verifies train membership",
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_order": candidate_names,
        "outer_protocol": "leave-one-sequence-out; held-out sequence is excluded from spatial model fitting, temporal model fitting, and temporal target scaling",
        "inner_protocol": "action-balanced sequence folds within each outer-training set; spatial predictions on temporal-model training sequences are generated by models that exclude the corresponding validation fold",
        "spatial_policy": "existing cross-fitted spatial Ridge selector; the spatial margin is read from nested_threshold_selection.csv for the outer fold and applied consistently to its inner OOF training paths and outer held-out path",
        "target": "image-diagonal-normalized centered three-frame acceleration error of the actual framewise hybrid pose selected by the spatial selector; GT-valid triples with incomplete hybrid prediction receive normalized target 1.0",
        "target_scaling": "95th percentile of finite OOF hybrid-path temporal errors from outer-training sequences only; normalized target clipped to [0,1]",
        "model": "scalar Ridge regression using center-frame inference features, previous/center/next selected-candidate one-hot identities, switch indicators, and hybrid/raw three-frame velocity, acceleration, displacement, and finite-support descriptors",
        "baselines": [
            "training-only mean target per selected candidate-identity triplet (fallback to global training mean for unseen patterns)",
            "training-only global mean target",
        ],
        "training_sampling": f"up to {args.max_train_points_per_sequence} GT-valid temporal points per outer-training sequence; deterministic seed; zero means use all",
        "metrics": ["sequence-macro MAE", "sequence-macro Spearman", "temporal support coverage", "paired sequence bootstrap CI for MAE improvement"],
        "fps_assumed": args.fps,
        "ridge": args.ridge,
        "inner_folds": args.inner_folds,
        "bootstrap": {"samples": args.bootstrap_samples, "unit": "sequence", "interval": "two-sided percentile 95% CI"},
        "official_test_split_read": False,
        "media_pipe_rerun": False,
        "runtime_seconds": round(time.time() - start_time, 2),
        "interpretation_guardrail": "This tests predictability of the temporal error of the selected hybrid path. It does not test alternative triplet routing, prove a routing benefit, or establish official test performance.",
    }
    metadata_out = out / "hybrid_triplet_risk_metadata.json"
    metadata_out.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("\n=== Hybrid triplet temporal-risk predictability (sequence macro) ===")
    print(summary.round(5).to_string(index=False))
    print(f"\nSaved: {per_sequence_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {metadata_out}")
    print("Cache-only development pilot; official test split was not read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
