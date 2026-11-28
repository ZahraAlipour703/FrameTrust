#!/usr/bin/env python3
"""LOSO evaluation of a pairwise learning-to-rank model for hybrid triplets.

The previous scalar Ridge predictor reduced triplet-risk MAE but did not improve
which of the 216 candidate-identity triplets was selected. This experiment
changes the learning objective: for two supported triplets at the same temporal
center, learn which one has lower acceleration error.

A linear pairwise least-squares ranker is trained from within-center comparisons.
Its score is evaluated against the exact-triplet-mean pattern baseline using
supported Top-1 accuracy, decision regret, and sampled pairwise ranking accuracy.
Training centers are sampled equally per sequence; test metrics use every
GT-valid center in the held-out sequence. Outer LOSO is used throughout.

This is a ranking diagnostic, not a routing/DP implementation. It uses cached
candidate outputs only; MediaPipe is not rerun and the official test split is
never read.
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

from evaluate_counterfactual_triplet_risk import (  # noqa: E402
    _make_sequence_training_data,
)
from evaluate_sequence_routing import _load_cache_item  # noqa: E402
from evaluate_temporal_risk_predictability import _load_sequence_info  # noqa: E402


def _normalize_targets(
    y_raw: np.ndarray, support: np.ndarray, scale: float,
) -> np.ndarray:
    target = np.ones(y_raw.shape, dtype=np.float32)
    valid = support & np.isfinite(y_raw)
    target[valid] = np.clip(y_raw[valid] / max(float(scale), 1e-9), 0.0, 1.0)
    return target


def _fit_pairwise_ranker(
    train_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    training_sequences: list[str],
    target_scale: float,
    ridge: float,
    pairs_per_center: int,
    best_pairs_per_center: int,
    min_risk_gap: float,
    seed: int,
) -> tuple[dict[str, np.ndarray | float | int], np.ndarray, int]:
    """Fit a linear risk ranker from D=x_loser-x_winner, target margin=1.

    For every sampled center, pairs are drawn from supported candidate triplets.
    Half of the requested pairs emphasize the oracle-best triplet versus other
    candidates; the remainder are random within-center comparisons. Pair rows are
    oriented so the second member has greater observed temporal risk.
    """
    if pairs_per_center < 2 or best_pairs_per_center < 1:
        raise ValueError("pairs_per_center must be >=2 and best_pairs_per_center >=1")
    if best_pairs_per_center > pairs_per_center:
        raise ValueError("best_pairs_per_center cannot exceed pairs_per_center")
    rng = np.random.default_rng(seed)
    difference_blocks: list[np.ndarray] = []
    pattern_targets: list[np.ndarray] = []
    feature_count: int | None = None
    n_pairs = 0

    for seq in training_sequences:
        X_seq, y_raw, support_seq, _ = train_data[seq]
        y_norm = _normalize_targets(y_raw, support_seq, target_scale)
        pattern_targets.append(y_norm)
        if feature_count is None:
            feature_count = int(X_seq.shape[-1])
        elif X_seq.shape[-1] != feature_count:
            raise ValueError("Feature dimensions differ across training sequences")

        random_pairs_per_center = pairs_per_center - best_pairs_per_center
        for center in range(len(y_norm)):
            valid_ids = np.flatnonzero(support_seq[center] & np.isfinite(y_raw[center]))
            if len(valid_ids) < 2:
                continue
            risks = y_norm[center]
            order = valid_ids[np.argsort(risks[valid_ids])]
            best_id = int(order[0])
            others = order[1:]
            if not len(others):
                continue

            winners: list[int] = []
            losers: list[int] = []

            # Best-versus-other comparisons carry direct information about the
            # choice made by an argmin controller.
            best_gap = risks[others] - risks[best_id]
            eligible_others = others[best_gap >= min_risk_gap]
            n_best = min(best_pairs_per_center, len(eligible_others))
            if n_best:
                chosen_others = rng.choice(eligible_others, size=n_best, replace=(len(eligible_others) < n_best))
                winners.extend([best_id] * n_best)
                losers.extend(int(x) for x in chosen_others)

            # Add comparisons from across the risk distribution, avoiding
            # near-ties where annotation noise can make the preference unstable.
            accepted = 0
            attempts = 0
            max_attempts = max(64, random_pairs_per_center * 20)
            while accepted < random_pairs_per_center and attempts < max_attempts:
                attempts += 1
                pair = rng.choice(valid_ids, size=2, replace=False)
                left, right = int(pair[0]), int(pair[1])
                if abs(float(risks[left] - risks[right])) < min_risk_gap:
                    continue
                if risks[left] < risks[right]:
                    winners.append(left)
                    losers.append(right)
                else:
                    winners.append(right)
                    losers.append(left)
                accepted += 1
            # Do not fabricate preferences for tied/near-tied candidates. Some
            # centers can contribute fewer comparisons than the requested quota.
            if not winners:
                continue
            winner_idx = np.asarray(winners, dtype=np.int64)
            loser_idx = np.asarray(losers, dtype=np.int64)
            D = X_seq[center, loser_idx, :] - X_seq[center, winner_idx, :]
            finite = np.isfinite(D).all(axis=1)
            if finite.any():
                difference_blocks.append(D[finite].astype(np.float32, copy=False))
                n_pairs += int(finite.sum())

    if not difference_blocks or feature_count is None:
        raise RuntimeError("No valid pairwise training comparisons were generated")
    D_all = np.concatenate(difference_blocks, axis=0).astype(np.float64, copy=False)
    if len(D_all) < 100:
        raise RuntimeError(f"Too few pairwise comparisons: {len(D_all)}")

    # RMS scaling preserves the orientation of the pairwise differences (unlike
    # centering, which would change the meaning of a within-center difference).
    feature_scale = np.sqrt(np.mean(D_all * D_all, axis=0))
    feature_scale[~np.isfinite(feature_scale) | (feature_scale < 1e-8)] = 1.0
    Z = D_all / feature_scale
    gram = Z.T @ Z
    rhs = Z.sum(axis=0)
    lhs = gram + float(ridge) * np.eye(gram.shape[0], dtype=np.float64)
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(lhs) @ rhs

    pattern_matrix = np.concatenate(pattern_targets, axis=0)
    pattern_prior = pattern_matrix.mean(axis=0).astype(np.float32)
    model: dict[str, np.ndarray | float | int] = {
        "feature_scale": feature_scale.astype(np.float32),
        "coef": coef.astype(np.float32),
        "ridge": float(ridge),
        "n_pairs": int(n_pairs),
        "n_features": int(feature_count),
    }
    del D_all, Z, gram, lhs, rhs, difference_blocks
    return model, pattern_prior, n_pairs


def _sample_pair_indices(
    true_risk: np.ndarray,
    valid_ids: np.ndarray,
    rng: np.random.Generator,
    n_pairs: int,
    min_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample supported within-center pairs with non-ambiguous true ordering."""
    if len(valid_ids) < 2 or n_pairs <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    left_ids: list[int] = []
    right_ids: list[int] = []
    attempts = 0
    max_attempts = max(64, n_pairs * 20)
    while len(left_ids) < n_pairs and attempts < max_attempts:
        remaining = n_pairs - len(left_ids)
        draw_n = min(max(remaining * 2, 16), 512)
        a = rng.choice(valid_ids, size=draw_n, replace=True)
        b = rng.choice(valid_ids, size=draw_n, replace=True)
        keep = (a != b) & (np.abs(true_risk[a] - true_risk[b]) >= min_gap)
        for ia, ib in zip(a[keep], b[keep]):
            left_ids.append(int(ia))
            right_ids.append(int(ib))
            if len(left_ids) >= n_pairs:
                break
        attempts += draw_n
    return np.asarray(left_ids, dtype=np.int64), np.asarray(right_ids, dtype=np.int64)


def _pairwise_accuracy_for_pairs(
    true_risk: np.ndarray,
    predicted_risk: np.ndarray,
    left_ids: np.ndarray,
    right_ids: np.ndarray,
) -> float:
    if len(left_ids) == 0:
        return float("nan")
    true_delta = true_risk[left_ids] - true_risk[right_ids]
    pred_delta = predicted_risk[left_ids] - predicted_risk[right_ids]
    correct = pred_delta * true_delta > 0
    tied = np.isclose(pred_delta, 0.0, atol=1e-10)
    return float(np.mean(correct.astype(float) + 0.5 * tied.astype(float)))


def _score_batch(
    rank_scores: np.ndarray,
    pattern_prior: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
    rng: np.random.Generator,
    test_pairs_per_center: int,
    min_risk_gap: float,
) -> dict[str, float | int]:
    n_centers, n_patterns = target.shape
    centers = np.arange(n_centers)
    valid_center = support.sum(axis=1) >= 2
    if not valid_center.any():
        return {
            "n_centers": int(n_centers), "n_supported_centers": 0,
            "top1_ranker": float("nan"), "top1_pattern": float("nan"),
            "regret_ranker": float("nan"), "regret_pattern": float("nan"),
            "pairwise_accuracy_ranker": float("nan"),
            "pairwise_accuracy_pattern": float("nan"),
            "n_pairwise_centers": 0,
            "support_coverage": float(np.mean(support)),
        }

    # Pairwise fit makes the ranker output a risk score: a loser has a larger
    # score than its winner. Lower risk is preferred, as for the pattern prior.
    safe_rank = np.where(support, rank_scores, np.inf)
    safe_pattern = np.where(support, pattern_prior[None, :], np.inf)
    safe_target = np.where(support, target, np.inf)
    choice_rank = np.argmin(safe_rank, axis=1)
    choice_pattern = np.argmin(safe_pattern, axis=1)
    oracle = np.argmin(safe_target, axis=1)
    row_ids = centers[valid_center]

    rank_regret = target[row_ids, choice_rank[row_ids]] - target[row_ids, oracle[row_ids]]
    pattern_regret = target[row_ids, choice_pattern[row_ids]] - target[row_ids, oracle[row_ids]]
    rank_top1 = choice_rank[row_ids] == oracle[row_ids]
    pattern_top1 = choice_pattern[row_ids] == oracle[row_ids]

    pair_rank_values: list[float] = []
    pair_pattern_values: list[float] = []
    for center in row_ids:
        ids = np.flatnonzero(support[center])
        left_ids, right_ids = _sample_pair_indices(
            target[center], ids, rng, test_pairs_per_center, min_risk_gap
        )
        rank_risk = rank_scores[center]
        pair_rank_values.append(_pairwise_accuracy_for_pairs(
            target[center], rank_risk, left_ids, right_ids
        ))
        pair_pattern_values.append(_pairwise_accuracy_for_pairs(
            target[center], pattern_prior, left_ids, right_ids
        ))

    return {
        "n_centers": int(n_centers),
        "n_supported_centers": int(len(row_ids)),
        "top1_ranker": float(np.mean(rank_top1)),
        "top1_pattern": float(np.mean(pattern_top1)),
        "regret_ranker": float(np.mean(rank_regret)),
        "regret_pattern": float(np.mean(pattern_regret)),
        "pairwise_accuracy_ranker": float(np.nanmean(pair_rank_values)),
        "pairwise_accuracy_pattern": float(np.nanmean(pair_pattern_values)),
        "n_pairwise_centers": int(np.isfinite(pair_rank_values).sum()),
        "support_coverage": float(np.mean(support)),
    }


def _evaluate_full_sequence(
    item: dict[str, Any],
    info: dict[str, Any],
    names: list[str],
    center_indices: np.ndarray,
    model: dict[str, np.ndarray | float | int],
    pattern_prior: np.ndarray,
    fps: float,
    batch_centers: int,
    test_pairs_per_center: int,
    min_risk_gap: float,
    seed: int,
) -> dict[str, float | int]:
    k = len(names)
    n_patterns = k ** 3
    feature_scale = np.asarray(model["feature_scale"], dtype=np.float32)
    coef = np.asarray(model["coef"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    sums = {
        "rank_top1": 0.0, "pattern_top1": 0.0,
        "rank_regret": 0.0, "pattern_regret": 0.0,
        "rank_pair": 0.0, "pattern_pair": 0.0,
        "n_centers": 0.0, "n_supported_centers": 0.0,
        "n_rank_pair_centers": 0.0, "n_pattern_pair_centers": 0.0,
        "support_coverage_sum": 0.0, "support_coverage_centers": 0.0,
    }
    for start in range(0, len(center_indices), batch_centers):
        ids = center_indices[start:start + batch_centers]
        X_batch, y_raw, support = _make_sequence_training_data(
            item, info, names, ids, fps
        )
        # Use the training-fold pattern prior and model scale; normalization is
        # only needed for comparable regret and ranking-gap evaluation.
        target = np.ones(y_raw.shape, dtype=np.float32)
        valid = support & np.isfinite(y_raw)
        # Use target q95 from the training fold, passed in the model metadata.
        target_scale = float(model["target_scale"])
        target[valid] = np.clip(y_raw[valid] / max(target_scale, 1e-9), 0.0, 1.0)
        rank_scores = (X_batch / feature_scale[None, None, :]) @ coef
        batch = _score_batch(
            rank_scores, pattern_prior, target, support, rng,
            test_pairs_per_center, min_risk_gap,
        )
        n = int(batch["n_centers"])
        ns = int(batch["n_supported_centers"])
        sums["n_centers"] += n
        sums["n_supported_centers"] += ns
        sums["support_coverage_sum"] += float(batch["support_coverage"]) * n
        sums["support_coverage_centers"] += n
        if ns:
            sums["rank_top1"] += float(batch["top1_ranker"]) * ns
            sums["pattern_top1"] += float(batch["top1_pattern"]) * ns
            sums["rank_regret"] += float(batch["regret_ranker"]) * ns
            sums["pattern_regret"] += float(batch["regret_pattern"]) * ns
        # Pairwise accuracy is a mean over eligible centers within each batch.
        # Weight by supported-center count, but ignore batches with no valid pairs.
        pair_centers = int(batch["n_pairwise_centers"])
        if pair_centers and np.isfinite(batch["pairwise_accuracy_ranker"]):
            sums["rank_pair"] += float(batch["pairwise_accuracy_ranker"]) * pair_centers
            sums["n_rank_pair_centers"] += pair_centers
        if pair_centers and np.isfinite(batch["pairwise_accuracy_pattern"]):
            sums["pattern_pair"] += float(batch["pairwise_accuracy_pattern"]) * pair_centers
            sums["n_pattern_pair_centers"] += pair_centers
        del X_batch, y_raw, support, target, rank_scores

    ns = max(sums["n_supported_centers"], 1.0)
    return {
        "n_gt_valid_centers": int(sums["n_centers"]),
        "n_supported_centers": int(sums["n_supported_centers"]),
        "top1_ranker": sums["rank_top1"] / ns,
        "top1_pattern": sums["pattern_top1"] / ns,
        "regret_ranker": sums["rank_regret"] / ns,
        "regret_pattern": sums["pattern_regret"] / ns,
        "pairwise_accuracy_ranker": sums["rank_pair"] / max(sums["n_rank_pair_centers"], 1.0),
        "pairwise_accuracy_pattern": sums["pattern_pair"] / max(sums["n_pattern_pair_centers"], 1.0),
        "support_coverage": sums["support_coverage_sum"] / max(sums["support_coverage_centers"], 1.0),
    }


def _bootstrap_ci(values: np.ndarray, seed: int, samples: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(x), size=(samples, len(x)))
    means = x[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out", type=Path, default=Path("outputs/selective_repair_crossfit"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--out", type=Path, default=Path("outputs/pairwise_triplet_ranker"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-train-centers-per-sequence", type=int, default=64)
    parser.add_argument("--test-batch-centers", type=int, default=128)
    parser.add_argument("--pairs-per-center", type=int, default=48)
    parser.add_argument("--best-pairs-per-center", type=int, default=24)
    parser.add_argument("--min-risk-gap", type=float, default=0.005)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--test-pairs-per-center", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    if args.fps <= 0 or args.max_train_centers_per_sequence < 8:
        parser.error("FPS must be positive and training centers >=8")
    if args.test_batch_centers < 1 or args.pairs_per_center < 2:
        parser.error("Batch centers >=1 and pairs per center >=2 required")
    if not 1 <= args.best_pairs_per_center <= args.pairs_per_center:
        parser.error("best pairs per center must be between 1 and pairs per center")
    if args.min_risk_gap < 0 or args.ridge < 0 or args.test_pairs_per_center < 1:
        parser.error("Risk gap, ridge and test pair count must be non-negative/positive")
    if args.bootstrap_samples < 100:
        parser.error("At least 100 bootstrap samples are required")

    started = time.time()
    source_out = args.source_out.resolve()
    dataset_root = args.dataset_root.resolve()
    metadata_path = source_out / "run_metadata.json"
    cache_dir = source_out / "feature_cache"
    if not metadata_path.is_file() or not cache_dir.is_dir():
        raise FileNotFoundError("Source output must contain run_metadata.json and feature_cache/")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = [str(x).zfill(4) for x in meta["selected_sequences"]]
    conditions = list(meta["conditions"])
    names = list(meta["candidate_order"])
    signature = str(meta["feature_cache"]["signature"])
    if not names or names[0] != "raw" or len(selected) < 3:
        raise ValueError("Source metadata must define raw-first candidates and >=3 sequences")

    print("Validating selected training sequences and cache GT...", flush=True)
    infos: dict[str, dict[str, Any]] = {}
    for seq in selected:
        clean_item = _load_cache_item(cache_dir / f"{seq}_{conditions[0]}.npz", names, signature)
        infos[seq] = _load_sequence_info(dataset_root, seq, clean_item["gt_xy"])
        if not np.allclose(infos[seq]["gt_xy"], clean_item["gt_xy"], equal_nan=True):
            raise ValueError(f"GT/cache mismatch for {seq}")
        del clean_item

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    per_sequence_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    n_patterns = len(names) ** 3

    for condition_idx, condition in enumerate(conditions):
        print(f"\n[{condition}] preparing sequence-balanced training centers", flush=True)
        train_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        action: dict[str, str] = {}
        for seq_idx, seq in enumerate(selected):
            item = _load_cache_item(cache_dir / f"{seq}_{condition}.npz", names, signature)
            if item["gt_xy"].shape != infos[seq]["gt_xy"].shape or not np.allclose(
                item["gt_xy"], infos[seq]["gt_xy"], equal_nan=True
            ):
                raise ValueError(f"GT/cache mismatch for {seq}/{condition}")
            gt = np.asarray(infos[seq]["valid_gt"], dtype=bool)
            valid_centers = np.flatnonzero((gt[:-2] & gt[1:-1] & gt[2:]).reshape(-1))
            rng = np.random.default_rng(args.seed + condition_idx * 100003 + seq_idx * 1009)
            take = min(len(valid_centers), args.max_train_centers_per_sequence)
            sampled = np.sort(rng.choice(valid_centers, size=take, replace=False)) if take < len(valid_centers) else valid_centers
            X, y, support = _make_sequence_training_data(item, infos[seq], names, sampled, args.fps)
            train_data[seq] = (X, y, support, sampled)
            action[seq] = str(item["action"])
            del item

        for outer_idx, heldout in enumerate(selected):
            training = [seq for seq in selected if seq != heldout]
            finite_errors = [
                train_data[seq][1][train_data[seq][2] & np.isfinite(train_data[seq][1])]
                for seq in training
            ]
            finite_errors = [x for x in finite_errors if x.size]
            if not finite_errors:
                raise RuntimeError(f"No supported training errors for {condition}/{heldout}")
            target_scale = float(np.quantile(np.concatenate(finite_errors), 0.95))
            if not np.isfinite(target_scale) or target_scale <= 1e-9:
                target_scale = 1.0

            model, pattern_prior, n_pairs = _fit_pairwise_ranker(
                train_data, training, target_scale, args.ridge,
                args.pairs_per_center, args.best_pairs_per_center,
                args.min_risk_gap,
                args.seed + 1009 * outer_idx + 9176 * condition_idx,
            )
            model["target_scale"] = target_scale

            test_item = _load_cache_item(cache_dir / f"{heldout}_{condition}.npz", names, signature)
            valid = np.asarray(infos[heldout]["valid_gt"], dtype=bool)
            centers = np.flatnonzero((valid[:-2] & valid[1:-1] & valid[2:]).reshape(-1))
            metrics = _evaluate_full_sequence(
                test_item, infos[heldout], names, centers, model, pattern_prior,
                args.fps, args.test_batch_centers, args.test_pairs_per_center,
                args.min_risk_gap,
                args.seed + 1000003 * outer_idx + 9176 * condition_idx,
            )
            per_sequence_rows.append({
                "condition": condition,
                "sequence": heldout,
                "action": action[heldout],
                "n_training_sequences": len(training),
                "n_training_centers_per_sequence_mean": float(np.mean([len(train_data[s][3]) for s in training])),
                "n_pairwise_training_comparisons": n_pairs,
                "n_patterns": n_patterns,
                "target_scale_train_q95": target_scale,
                **metrics,
            })
            if (outer_idx + 1) % 5 == 0 or outer_idx == len(selected) - 1:
                print(
                    f"  outer {outer_idx + 1}/{len(selected)} ({heldout}): "
                    f"Top1 ranker={metrics['top1_ranker']:.3f}, pattern={metrics['top1_pattern']:.3f}; "
                    f"regret ranker={metrics['regret_ranker']:.4f}, pattern={metrics['regret_pattern']:.4f}",
                    flush=True,
                )
            del test_item, model, pattern_prior
        del train_data

    per_sequence = pd.DataFrame(per_sequence_rows)
    per_sequence.to_csv(out / "pairwise_triplet_ranker_by_sequence.csv", index=False)

    paired_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metrics = [
        ("top1_ranker", "top1_pattern", "top1_delta", "higher_better"),
        ("regret_ranker", "regret_pattern", "regret_delta", "lower_better"),
        ("pairwise_accuracy_ranker", "pairwise_accuracy_pattern", "pairwise_accuracy_delta", "higher_better"),
    ]
    for condition_idx, condition in enumerate(conditions):
        group = per_sequence[per_sequence["condition"] == condition].set_index("sequence")
        row: dict[str, Any] = {
            "condition": condition,
            "n_sequences": int(group.index.nunique()),
            "ranker_training_pairs_mean": float(group["n_pairwise_training_comparisons"].mean()),
            "support_coverage_sequence_mean": float(group["support_coverage"].mean()),
        }
        for rank_col, base_col, delta_col, direction in metrics:
            rank_mean = float(group[rank_col].mean())
            base_mean = float(group[base_col].mean())
            delta = group[rank_col].to_numpy(float) - group[base_col].to_numpy(float)
            lo, hi = _bootstrap_ci(
                delta, args.seed + condition_idx * 101 + metrics.index((rank_col, base_col, delta_col, direction)),
                args.bootstrap_samples,
            )
            row[f"{rank_col}_sequence_mean"] = rank_mean
            row[f"{base_col}_sequence_mean"] = base_mean
            row[f"{delta_col}_sequence_mean"] = float(np.nanmean(delta))
            row[f"{delta_col}_ci95_low"] = lo
            row[f"{delta_col}_ci95_high"] = hi
            paired_rows.append({
                "condition": condition,
                "metric": delta_col,
                "comparison": "pairwise_ranker_minus_pattern_prior",
                "n_paired_sequences": int(np.isfinite(delta).sum()),
                "delta_mean": float(np.nanmean(delta)),
                "delta_ci95_low": lo,
                "delta_ci95_high": hi,
                "direction": direction,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_unit": "sequence",
            })
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    summary.to_csv(out / "pairwise_triplet_ranker_summary.csv", index=False)
    paired.to_csv(out / "pairwise_triplet_ranker_paired_comparison.csv", index=False)

    protocol = {
        "experiment": "pairwise_least_squares_ranking_for_counterfactual_hybrid_triplets",
        "dataset": "Penn Action",
        "official_split": "selected official training sequences only; sequence loader verifies train membership",
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_order": names,
        "candidate_triplet_patterns": n_patterns,
        "outer_protocol": "leave-one-sequence-out; held-out sequence excluded from ranker and exact-pattern prior fitting",
        "training_sampling": f"up to {args.max_train_centers_per_sequence} GT-valid centers per training sequence, sampled without replacement; all candidate triplet patterns expanded per center",
        "pair_sampling": {
            "pairs_per_center": args.pairs_per_center,
            "best_vs_other_pairs_per_center": args.best_pairs_per_center,
            "random_pairs_per_center": args.pairs_per_center - args.best_pairs_per_center,
            "minimum_normalized_risk_gap": args.min_risk_gap,
            "sequence_balancing": "same maximum sampled centers per sequence; equal pair quota per center",
        },
        "ranker": "linear pairwise least-squares risk model trained on (X_loser - X_winner) with target margin 1; lower predicted score is preferred; RMS feature scaling; L2 regularization",
        "baseline": "training-only mean normalized temporal error for each exact triplet identity pattern",
        "decision_metrics": ["supported Top-1 accuracy", "supported mean decision regret", "sampled within-center pairwise ranking accuracy"],
        "pairwise_accuracy_test_sampling": args.test_pairs_per_center,
        "outer_test_centers": "all GT-visible valid centered three-frame windows for each held-out sequence",
        "target_scaling": "95th percentile of supported triplet acceleration errors from outer-training sampled centers only",
        "fps_assumed": args.fps,
        "ridge": args.ridge,
        "official_test_split_read": False,
        "media_pipe_rerun": False,
        "interpretation_guardrail": "This is a development-only ranking pilot; it does not implement DP or establish final/test-set performance.",
        "runtime_seconds": round(time.time() - started, 2),
    }
    (out / "pairwise_triplet_ranker_metadata.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print("\n=== Pairwise triplet-ranker sequence-macro summary ===")
    print(summary.round(5).to_string(index=False))
    print("\n=== Paired ranker minus pattern-prior comparison ===")
    print(paired.round(5).to_string(index=False))
    print(f"\nSaved outputs under {out}")
    print("Cache-only LOSO development experiment; official test split was not read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
