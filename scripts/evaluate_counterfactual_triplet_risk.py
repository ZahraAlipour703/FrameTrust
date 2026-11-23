#!/usr/bin/env python3
"""LOSO predictability of temporal risk for every counterfactual candidate triplet.

For each GT-valid centered triple (t-1,t,t+1), evaluate all K^3 combinations
(a,b,c), where candidate a supplies t-1, b supplies t, and c supplies t+1.
A scalar Ridge model predicts the image-diagonal-normalized acceleration error
of each mixed-candidate triplet from inference-time geometry and cached features.

Baselines are (1) a training-only mean for each exact candidate identity triplet
and (2) a training-only global mean. Training centers are sampled equally per
sequence to control CPU/RAM. Outer evaluation is leave-one-sequence-out; target
scaling is fitted on outer-training sequences only. No MediaPipe rerun and no
official test split access.

This is a counterfactual triplet-risk predictability pilot. It does NOT yet
perform second-order DP or demonstrate improvement of a routed trajectory.
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

from evaluate_sequence_routing import _load_cache_item  # noqa: E402
from evaluate_temporal_risk_predictability import _load_sequence_info  # noqa: E402


def _safe_vec(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.float32)
    if mask.any():
        out[mask] = values[mask].astype(np.float32)
    return out


def _triplet_features_target(
    item: dict[str, Any],
    info: dict[str, Any],
    candidate_names: list[str],
    combo: tuple[int, int, int],
    flat_indices: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature rows, unscaled acceleration errors, and prediction support."""
    a, b, c = combo
    t_count, joint_count, _ = item["raw"].shape
    features = np.asarray(item["features"], dtype=np.float32)
    raw = np.asarray(item["raw"], dtype=float)
    gt = np.asarray(info["gt_xy"], dtype=float)
    diagonal = float(info["diagonal"])
    k_count = len(candidate_names)

    # Flat indices refer to [T-2,J] centered triples.
    frame_idx = flat_indices // joint_count
    joint_idx = flat_indices % joint_count
    center_features = features[1:-1].reshape(-1, features.shape[-1])[flat_indices]

    cand_a = np.asarray(item["candidates"][candidate_names[a]], dtype=float)
    cand_b = np.asarray(item["candidates"][candidate_names[b]], dtype=float)
    cand_c = np.asarray(item["candidates"][candidate_names[c]], dtype=float)
    prev = cand_a[frame_idx, joint_idx]
    mid = cand_b[frame_idx + 1, joint_idx]
    nxt = cand_c[frame_idx + 2, joint_idx]
    raw_prev = raw[frame_idx, joint_idx]
    raw_mid = raw[frame_idx + 1, joint_idx]
    raw_next = raw[frame_idx + 2, joint_idx]

    prev_ok = np.isfinite(prev).all(axis=-1)
    mid_ok = np.isfinite(mid).all(axis=-1)
    next_ok = np.isfinite(nxt).all(axis=-1)
    raw_prev_ok = np.isfinite(raw_prev).all(axis=-1)
    raw_mid_ok = np.isfinite(raw_mid).all(axis=-1)
    raw_next_ok = np.isfinite(raw_next).all(axis=-1)
    hybrid_ok = prev_ok & mid_ok & next_ok
    raw_ok = raw_prev_ok & raw_mid_ok & raw_next_ok

    # Coordinates and candidate-vs-raw offsets are normalized by image diagonal.
    pos_prev = _safe_vec(prev / diagonal, prev_ok)
    pos_mid = _safe_vec(mid / diagonal, mid_ok)
    pos_next = _safe_vec(nxt / diagonal, next_ok)
    off_prev = _safe_vec((prev - raw_prev) / diagonal, prev_ok & raw_prev_ok)
    off_mid = _safe_vec((mid - raw_mid) / diagonal, mid_ok & raw_mid_ok)
    off_next = _safe_vec((nxt - raw_next) / diagonal, next_ok & raw_next_ok)
    pos_block = np.concatenate([pos_prev, pos_mid, pos_next], axis=-1)
    offset_block = np.concatenate([off_prev, off_mid, off_next], axis=-1)

    hybrid_velocity = (nxt - prev) * (fps / (2.0 * diagonal))
    hybrid_acceleration = (nxt - 2.0 * mid + prev) * (fps * fps / diagonal)
    raw_velocity = (raw_next - raw_prev) * (fps / (2.0 * diagonal))
    raw_acceleration = (raw_next - 2.0 * raw_mid + raw_prev) * (fps * fps / diagonal)
    hv = _safe_vec(hybrid_velocity, hybrid_ok)
    ha = _safe_vec(hybrid_acceleration, hybrid_ok)
    rv = _safe_vec(raw_velocity, raw_ok)
    ra = _safe_vec(raw_acceleration, raw_ok)
    delta_acc = ha - ra

    step_prev = np.linalg.norm(mid - prev, axis=-1) / diagonal
    step_next = np.linalg.norm(nxt - mid, axis=-1) / diagonal
    step_prev = np.where(prev_ok & mid_ok, step_prev, 0.0).astype(np.float32)
    step_next = np.where(mid_ok & next_ok, step_next, 0.0).astype(np.float32)

    identity = np.concatenate(
        [
            np.broadcast_to(np.eye(k_count, dtype=np.float32)[a], (len(flat_indices), k_count)),
            np.broadcast_to(np.eye(k_count, dtype=np.float32)[b], (len(flat_indices), k_count)),
            np.broadcast_to(np.eye(k_count, dtype=np.float32)[c], (len(flat_indices), k_count)),
        ],
        axis=-1,
    )
    switches = np.broadcast_to(
        np.asarray([float(a != b), float(b != c)], dtype=np.float32),
        (len(flat_indices), 2),
    )
    geometry = np.concatenate(
        [
            pos_block,
            offset_block,
            hv,
            ha,
            rv,
            ra,
            delta_acc,
            step_prev[:, None],
            step_next[:, None],
            prev_ok[:, None].astype(np.float32),
            mid_ok[:, None].astype(np.float32),
            next_ok[:, None].astype(np.float32),
            raw_ok[:, None].astype(np.float32),
        ],
        axis=-1,
    )
    X = np.concatenate([center_features, identity, switches, geometry], axis=-1)
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    X = np.clip(X, -10.0, 10.0).astype(np.float32)

    gt_prev = gt[frame_idx, joint_idx]
    gt_mid = gt[frame_idx + 1, joint_idx]
    gt_next = gt[frame_idx + 2, joint_idx]
    gt_acc = (gt_next - 2.0 * gt_mid + gt_prev) * (fps * fps / diagonal)
    target_raw = np.full(len(flat_indices), np.nan, dtype=np.float32)
    if hybrid_ok.any():
        target_raw[hybrid_ok] = np.linalg.norm(
            hybrid_acceleration[hybrid_ok] - gt_acc[hybrid_ok], axis=-1
        ).astype(np.float32)
    return X, target_raw, hybrid_ok


def _fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    good = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[good], y[good]
    if len(y) < 3 or X.shape[1] == 0:
        raise ValueError("Insufficient finite rows for triplet-risk Ridge")
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xz = (X - mean) / std
    design = np.column_stack([np.ones(len(Xz)), Xz])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(lhs) @ rhs
    return {"mean": mean, "std": std, "coef": coef, "n_samples": int(len(y))}


def _predict_ridge(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    design = np.column_stack([
        np.ones(len(X)),
        (X - np.asarray(model["mean"])) / np.asarray(model["std"]),
    ])
    return np.clip(design @ np.asarray(model["coef"]), 0.0, 1.0).astype(np.float32)


def _make_sequence_training_data(
    item: dict[str, Any],
    info: dict[str, Any],
    names: list[str],
    sample_indices: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = len(names)
    n_patterns = k ** 3
    X_all: np.ndarray | None = None
    y_all = np.full((len(sample_indices), n_patterns), np.nan, dtype=np.float32)
    support_all = np.zeros((len(sample_indices), n_patterns), dtype=bool)
    for pattern_id in range(n_patterns):
        combo = (pattern_id // (k * k), (pattern_id // k) % k, pattern_id % k)
        X, y, support = _triplet_features_target(
            item, info, names, combo, sample_indices, fps
        )
        if X_all is None:
            X_all = np.empty((len(sample_indices), n_patterns, X.shape[-1]), dtype=np.float32)
        X_all[:, pattern_id, :] = X
        y_all[:, pattern_id] = y
        support_all[:, pattern_id] = support
    assert X_all is not None
    return X_all, y_all, support_all


def _sequence_score(
    model_pred: np.ndarray,
    pattern_pred: np.ndarray,
    global_pred: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
) -> dict[str, float | int]:
    good = np.isfinite(target)
    y = target[good]
    pm = model_pred[good]
    pp = pattern_pred[good]
    pg = global_pred[good]
    if not len(y):
        raise ValueError("Sequence has no scoreable triplet targets")
    true_best = np.argmin(target, axis=1)
    model_best = np.argmin(model_pred, axis=1)
    pattern_best = np.argmin(pattern_pred, axis=1)
    rows = np.arange(len(target))
    return {
        "n_gt_valid_centers": int(len(target)),
        "mae_model": float(np.mean(np.abs(pm - y))),
        "mae_pattern_baseline": float(np.mean(np.abs(pp - y))),
        "mae_global_baseline": float(np.mean(np.abs(pg - y))),
        "mae_improvement_vs_pattern": float(np.mean(np.abs(pp - y)) - np.mean(np.abs(pm - y))),
        "mae_improvement_vs_global": float(np.mean(np.abs(pg - y)) - np.mean(np.abs(pm - y))),
        "spearman_model": float(pd.Series(pm).corr(pd.Series(y), method="spearman")) if len(y) >= 3 else float("nan"),
        "spearman_pattern_baseline": float(pd.Series(pp).corr(pd.Series(y), method="spearman")) if len(y) >= 3 else float("nan"),
        "top1_triplet_accuracy_model": float(np.mean(model_best == true_best)),
        "top1_triplet_accuracy_pattern_baseline": float(np.mean(pattern_best == true_best)),
        "mean_regret_model": float(np.mean(target[rows, model_best] - np.min(target, axis=1))),
        "mean_regret_pattern_baseline": float(np.mean(target[rows, pattern_best] - np.min(target, axis=1))),
        "per_center_oracle_mean_risk": float(np.mean(np.min(target, axis=1))),
        "temporal_support_coverage": float(np.mean(support)),
    }


def _bootstrap_ci(values: np.ndarray, seed: int, samples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out", type=Path, default=Path("outputs/selective_repair_crossfit"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--out", type=Path, default=Path("outputs/counterfactual_triplet_risk"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--max-train-centers-per-sequence", type=int, default=64)
    parser.add_argument("--test-batch-centers", type=int, default=1024)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    if args.fps <= 0 or args.ridge < 0:
        parser.error("FPS must be positive and ridge non-negative")
    if args.max_train_centers_per_sequence < 8 or args.test_batch_centers < 1 or args.bootstrap_samples < 100:
        parser.error("train centers >=8, test batch >=1, bootstrap samples >=100")

    started = time.time()
    source_out = args.source_out.resolve()
    dataset_root = args.dataset_root.resolve()
    metadata_path = source_out / "run_metadata.json"
    cache_dir = source_out / "feature_cache"
    if not metadata_path.is_file() or not cache_dir.is_dir():
        raise FileNotFoundError("Source output needs run_metadata.json and feature_cache/")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = [str(s).zfill(4) for s in metadata["selected_sequences"]]
    conditions = list(metadata["conditions"])
    names = list(metadata["candidate_order"])
    signature = str(metadata["feature_cache"]["signature"])
    if not names or names[0] != "raw" or len(selected) < 3:
        raise ValueError("Metadata requires raw-first candidates and at least 3 sequences")

    print("Validating sequence metadata and ground truth...", flush=True)
    infos: dict[str, dict[str, Any]] = {}
    for seq in selected:
        clean_path = cache_dir / f"{seq}_{conditions[0]}.npz"
        item = _load_cache_item(clean_path, names, signature)
        infos[seq] = _load_sequence_info(dataset_root, seq, item["gt_xy"])
        if infos[seq]["gt_xy"].shape != item["gt_xy"].shape or not np.allclose(
            infos[seq]["gt_xy"], item["gt_xy"], equal_nan=True
        ):
            raise ValueError(f"Ground-truth mismatch for {seq}")
        del item

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    n_patterns = len(names) ** 3

    for cond_idx, condition in enumerate(conditions):
        print(f"\n[{condition}] loading candidate arrays", flush=True)
        condition_cache: dict[str, dict[str, Any]] = {}
        train_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        # Sample GT-valid temporal centers once per sequence, independently of labels.
        for seq_idx, seq in enumerate(selected):
            path = cache_dir / f"{seq}_{condition}.npz"
            item = _load_cache_item(path, names, signature)
            if not np.allclose(item["gt_xy"], infos[seq]["gt_xy"], equal_nan=True):
                raise ValueError(f"Cache GT mismatch for {seq}/{condition}")
            valid = np.asarray(infos[seq]["valid_gt"], dtype=bool)
            gt_mask = valid[:-2] & valid[1:-1] & valid[2:]
            valid_indices = np.flatnonzero(gt_mask.reshape(-1))
            if not len(valid_indices):
                raise RuntimeError(f"No GT-valid temporal centers for {seq}/{condition}")
            rng = np.random.default_rng(args.seed + cond_idx * 100003 + seq_idx * 1009)
            if len(valid_indices) > args.max_train_centers_per_sequence:
                train_indices = np.sort(rng.choice(
                    valid_indices, size=args.max_train_centers_per_sequence, replace=False
                ))
            else:
                train_indices = valid_indices
            X_seq, y_seq, s_seq = _make_sequence_training_data(
                item, infos[seq], names, train_indices, args.fps
            )
            train_data[seq] = (X_seq, y_seq, s_seq, train_indices)
            # Keep only the current sequence cache for test-time use; reload per outer fold below.
            del item
            print(
                f"  sampled {len(train_indices)} centers for {seq}; patterns={n_patterns}",
                flush=False,
            )

        for outer_idx, heldout in enumerate(selected):
            training = [seq for seq in selected if seq != heldout]
            raw_finite_train = [
                train_data[seq][1][np.isfinite(train_data[seq][1])]
                for seq in training
            ]
            finite_train = [x for x in raw_finite_train if x.size]
            if not finite_train:
                raise RuntimeError(f"No finite training triplet targets for {condition}/{heldout}")
            target_scale = float(np.quantile(np.concatenate(finite_train), 0.95))
            if not np.isfinite(target_scale) or target_scale <= 1e-9:
                target_scale = 1.0

            X_blocks: list[np.ndarray] = []
            y_blocks: list[np.ndarray] = []
            for seq in training:
                X_seq, y_raw, support_seq, _ = train_data[seq]
                y_norm = np.ones(y_raw.shape, dtype=np.float32)
                finite_support = support_seq & np.isfinite(y_raw)
                y_norm[finite_support] = np.clip(y_raw[finite_support] / target_scale, 0.0, 1.0)
                X_blocks.append(X_seq.reshape(-1, X_seq.shape[-1]))
                y_blocks.append(y_norm.reshape(-1))
            X_train = np.concatenate(X_blocks, axis=0)
            y_train = np.concatenate(y_blocks, axis=0)
            model = _fit_ridge(X_train, y_train, args.ridge)

            # Exact per-triplet pattern means are a strong categorical baseline.
            y_train_matrix = np.concatenate(
                [
                    np.where(
                        train_data[seq][2] & np.isfinite(train_data[seq][1]),
                        np.clip(train_data[seq][1] / target_scale, 0.0, 1.0),
                        1.0,
                    ).astype(np.float32)
                    for seq in training
                ],
                axis=0,
            )
            pattern_mean = y_train_matrix.mean(axis=0)
            global_mean = float(y_train.mean())

            test_item = _load_cache_item(
                cache_dir / f"{heldout}_{condition}.npz", names, signature
            )
            valid = np.asarray(infos[heldout]["valid_gt"], dtype=bool)
            test_gt_mask = valid[:-2] & valid[1:-1] & valid[2:]
            test_indices = np.flatnonzero(test_gt_mask.reshape(-1))
            pred_all = np.empty((len(test_indices), n_patterns), dtype=np.float32)
            target_all = np.ones((len(test_indices), n_patterns), dtype=np.float32)
            support_all = np.zeros((len(test_indices), n_patterns), dtype=bool)

            for batch_start in range(0, len(test_indices), args.test_batch_centers):
                batch_ids = test_indices[batch_start:batch_start + args.test_batch_centers]
                batch_X = np.empty((len(batch_ids), n_patterns, X_train.shape[-1]), dtype=np.float32)
                batch_y = np.full((len(batch_ids), n_patterns), np.nan, dtype=np.float32)
                batch_support = np.zeros((len(batch_ids), n_patterns), dtype=bool)
                for pattern_id in range(n_patterns):
                    combo = (
                        pattern_id // (len(names) ** 2),
                        (pattern_id // len(names)) % len(names),
                        pattern_id % len(names),
                    )
                    x, y_raw, support = _triplet_features_target(
                        test_item, infos[heldout], names, combo, batch_ids, args.fps
                    )
                    batch_X[:, pattern_id, :] = x
                    batch_y[:, pattern_id] = y_raw
                    batch_support[:, pattern_id] = support
                pred = _predict_ridge(model, batch_X.reshape(-1, batch_X.shape[-1]))
                pred_all[batch_start:batch_start + len(batch_ids)] = pred.reshape(len(batch_ids), n_patterns)
                norm = np.ones(batch_y.shape, dtype=np.float32)
                ok = batch_support & np.isfinite(batch_y)
                norm[ok] = np.clip(batch_y[ok] / target_scale, 0.0, 1.0)
                target_all[batch_start:batch_start + len(batch_ids)] = norm
                support_all[batch_start:batch_start + len(batch_ids)] = batch_support

            pattern_all = np.broadcast_to(pattern_mean[None, :], pred_all.shape).copy()
            global_all = np.full(pred_all.shape, global_mean, dtype=np.float32)
            # All rows correspond to GT-valid centers by construction.
            flat_y = target_all.reshape(-1)
            flat_model = pred_all.reshape(-1)
            flat_pattern = pattern_all.reshape(-1)
            flat_global = global_all.reshape(-1)
            model_best = np.argmin(pred_all, axis=1)
            pattern_best = np.argmin(pattern_all, axis=1)
            oracle_best = np.argmin(target_all, axis=1)
            point_rows = np.arange(len(test_indices))
            model_regret = target_all[point_rows, model_best] - np.min(target_all, axis=1)
            pattern_regret = target_all[point_rows, pattern_best] - np.min(target_all, axis=1)

            row = {
                "condition": condition,
                "sequence": heldout,
                "action": test_item["action"],
                "n_gt_valid_centers": int(len(test_indices)),
                "n_candidate_triplet_patterns": int(n_patterns),
                "temporal_support_coverage_all_triplets": float(support_all.mean()),
                "target_scale_train_q95": target_scale,
                "n_train_centers_mean_per_sequence": float(np.mean([len(train_data[s][3]) for s in training])),
                "n_train_rows": int(len(y_train)),
                "mae_model": float(np.mean(np.abs(flat_model - flat_y))),
                "mae_pattern_baseline": float(np.mean(np.abs(flat_pattern - flat_y))),
                "mae_global_baseline": float(np.mean(np.abs(flat_global - flat_y))),
                "mae_improvement_vs_pattern": float(np.mean(np.abs(flat_pattern - flat_y)) - np.mean(np.abs(flat_model - flat_y))),
                "mae_improvement_vs_global": float(np.mean(np.abs(flat_global - flat_y)) - np.mean(np.abs(flat_model - flat_y))),
                "mae_model_supported_only": float(np.mean(np.abs(flat_model[support_all.reshape(-1)] - flat_y[support_all.reshape(-1)]))) if support_all.any() else float("nan"),
                "mae_pattern_supported_only": float(np.mean(np.abs(flat_pattern[support_all.reshape(-1)] - flat_y[support_all.reshape(-1)]))) if support_all.any() else float("nan"),
                "spearman_model": float(pd.Series(flat_model).corr(pd.Series(flat_y), method="spearman")),
                "spearman_pattern_baseline": float(pd.Series(flat_pattern).corr(pd.Series(flat_y), method="spearman")),
                "top1_triplet_accuracy_model": float(np.mean(model_best == oracle_best)),
                "top1_triplet_accuracy_pattern_baseline": float(np.mean(pattern_best == oracle_best)),
                "mean_regret_model": float(np.mean(model_regret)),
                "mean_regret_pattern_baseline": float(np.mean(pattern_regret)),
                "oracle_mean_risk": float(np.mean(np.min(target_all, axis=1))),
            }
            all_rows.append(row)
            del test_item, pred_all, target_all, support_all, pattern_all, global_all

            if (outer_idx + 1) % 5 == 0 or outer_idx == len(selected) - 1:
                print(
                    f"  outer {outer_idx + 1}/{len(selected)} ({heldout}): "
                    f"MAE={row['mae_model']:.4f}; pattern={row['mae_pattern_baseline']:.4f}; "
                    f"top1={row['top1_triplet_accuracy_model']:.3f}",
                    flush=True,
                )

        del train_data

    per_sequence = pd.DataFrame(all_rows)
    per_sequence_path = out / "counterfactual_triplet_risk_by_sequence.csv"
    per_sequence.to_csv(per_sequence_path, index=False)
    summary_rows: list[dict[str, Any]] = []
    for cond_idx, (condition, group) in enumerate(per_sequence.groupby("condition")):
        imp_pattern = group["mae_improvement_vs_pattern"].to_numpy(float)
        imp_global = group["mae_improvement_vs_global"].to_numpy(float)
        ci_p = _bootstrap_ci(imp_pattern, args.seed + cond_idx, args.bootstrap_samples)
        ci_g = _bootstrap_ci(imp_global, args.seed + 1000 + cond_idx, args.bootstrap_samples)
        top1_delta = (
            group["top1_triplet_accuracy_model"].to_numpy(float)
            - group["top1_triplet_accuracy_pattern_baseline"].to_numpy(float)
        )
        ci_top1 = _bootstrap_ci(top1_delta, args.seed + 2000 + cond_idx, args.bootstrap_samples)
        summary_rows.append({
            "condition": condition,
            "n_sequences": int(group["sequence"].nunique()),
            "mae_model_sequence_mean": float(group["mae_model"].mean()),
            "mae_pattern_baseline_sequence_mean": float(group["mae_pattern_baseline"].mean()),
            "mae_global_baseline_sequence_mean": float(group["mae_global_baseline"].mean()),
            "mae_improvement_vs_pattern_sequence_mean": float(imp_pattern.mean()),
            "mae_improvement_vs_pattern_ci95_low": ci_p[0],
            "mae_improvement_vs_pattern_ci95_high": ci_p[1],
            "mae_improvement_vs_global_sequence_mean": float(imp_global.mean()),
            "mae_improvement_vs_global_ci95_low": ci_g[0],
            "mae_improvement_vs_global_ci95_high": ci_g[1],
            "spearman_model_sequence_mean": float(group["spearman_model"].mean()),
            "spearman_pattern_baseline_sequence_mean": float(group["spearman_pattern_baseline"].mean()),
            "top1_triplet_accuracy_model_sequence_mean": float(group["top1_triplet_accuracy_model"].mean()),
            "top1_triplet_accuracy_pattern_baseline_sequence_mean": float(group["top1_triplet_accuracy_pattern_baseline"].mean()),
            "top1_delta_vs_pattern_sequence_mean": float(top1_delta.mean()),
            "top1_delta_vs_pattern_ci95_low": ci_top1[0],
            "top1_delta_vs_pattern_ci95_high": ci_top1[1],
            "mean_regret_model_sequence_mean": float(group["mean_regret_model"].mean()),
            "mean_regret_pattern_sequence_mean": float(group["mean_regret_pattern_baseline"].mean()),
            "per_center_oracle_mean_risk_sequence_mean": float(group["per_center_oracle_mean_risk"].mean()),
            "temporal_support_coverage_sequence_mean": float(group["temporal_support_coverage_all_triplets"].mean()),
            "n_train_rows_sequence_mean": float(group["n_train_rows"].mean()),
            "n_patterns": n_patterns,
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = out / "counterfactual_triplet_risk_summary.csv"
    summary.to_csv(summary_path, index=False)

    protocol = {
        "experiment": "counterfactual_all_candidate_triplet_temporal_risk_predictability",
        "dataset": "Penn Action",
        "official_split": "selected official training sequences only; sequence loader verifies train membership",
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_order": names,
        "candidate_triplets": "all K^3 combinations (candidate a at t-1, b at t, c at t+1)",
        "outer_protocol": "leave-one-sequence-out; held-out sequence excluded from fitting and target scale estimation",
        "target": "image-diagonal-normalized acceleration error against Penn Action annotations for each counterfactual hybrid candidate triplet; GT-valid triples with incomplete candidate support receive normalized target 1",
        "features": "center-frame cached inference features; three candidate identity one-hots; switch indicators; normalized candidate coordinates and offsets from raw at t-1,t,t+1; hybrid/raw velocity and acceleration descriptors; finite-support indicators",
        "model": "scalar Ridge regression; all candidate triplets are included in training at sampled GT-valid centers",
        "baselines": ["training-only mean per exact candidate-identity triplet", "training-only global mean"],
        "training_sampling": f"up to {args.max_train_centers_per_sequence} GT-valid centers per training sequence, sampled without replacement; every sampled center expands to all K^3 candidate triplets",
        "test_evaluation": "all GT-valid centers and all K^3 triplets; inference feature generation in batches",
        "metrics": ["sequence-macro MAE", "sequence-macro Spearman", "Top-1 triplet selection accuracy", "local per-center decision regret vs independent per-center oracle (not a feasible full-sequence route)", "temporal support coverage", "paired sequence bootstrap CIs"],
        "fps_assumed": args.fps,
        "ridge": args.ridge,
        "max_train_centers_per_sequence": args.max_train_centers_per_sequence,
        "test_batch_centers": args.test_batch_centers,
        "bootstrap": {"samples": args.bootstrap_samples, "unit": "sequence", "interval": "two-sided percentile 95% CI"},
        "official_test_split_read": False,
        "media_pipe_rerun": False,
        "runtime_seconds": round(time.time() - started, 2),
        "interpretation_guardrail": "This tests counterfactual triplet-risk predictability across all candidate identity combinations. It does not implement DP, establish a routing improvement, or report official test-set performance.",
    }
    metadata_path_out = out / "counterfactual_triplet_risk_metadata.json"
    metadata_path_out.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("\n=== Counterfactual triplet-risk predictability (sequence macro) ===")
    print(summary.round(5).to_string(index=False))
    print(f"\nSaved: {per_sequence_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {metadata_path_out}")
    print("Cache-only development pilot; official test split was not read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
