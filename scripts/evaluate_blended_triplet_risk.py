#!/usr/bin/env python3
"""Nested LOSO evaluation of a pattern-prior + contextual triplet-risk blend.

The counterfactual Ridge model substantially improves triplet-risk MAE but its
argmin decisions have higher regret than the exact-pattern mean baseline. This
script tests a decision-oriented shrinkage estimator:

    risk_blend = pattern_mean + alpha * (ridge_prediction - pattern_mean)

Alpha is selected only on action-balanced inner validation sequences by
minimizing supported-triplet decision regret. Outer held-out sequences are
never used to fit the Ridge model, pattern prior, risk scale, or alpha.

Uses the existing feature cache; no MediaPipe inference is rerun and the
official Penn Action test split is not read.
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
    _fit_ridge,
    _make_sequence_training_data,
    _predict_ridge,
)
from evaluate_sequence_routing import _load_cache_item  # noqa: E402
from evaluate_temporal_risk_predictability import _load_sequence_info  # noqa: E402


def _action_balanced_folds(
    sequences: list[str], actions: dict[str, str], n_folds: int, seed: int,
) -> list[list[str]]:
    if len(sequences) < 3:
        raise ValueError("At least three outer-training sequences are required")
    count = min(n_folds, len(sequences))
    folds: list[list[str]] = [[] for _ in range(count)]
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {}
    for seq in sequences:
        groups.setdefault(actions[seq], []).append(seq)
    for action_idx, action in enumerate(sorted(groups)):
        members = list(groups[action])
        rng.shuffle(members)
        start = action_idx % count
        for i, seq in enumerate(members):
            folds[(start + i) % count].append(seq)
    return [fold for fold in folds if fold]


def _normalise_target(
    y_raw: np.ndarray, support: np.ndarray, scale: float,
) -> np.ndarray:
    target = np.ones(y_raw.shape, dtype=np.float32)
    valid = support & np.isfinite(y_raw)
    target[valid] = np.clip(y_raw[valid] / max(scale, 1e-9), 0.0, 1.0)
    return target


def _fit_model_and_pattern_prior(
    training_sequences: list[str],
    train_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ridge: float,
) -> tuple[dict[str, Any], np.ndarray, float]:
    finite_targets: list[np.ndarray] = []
    for seq in training_sequences:
        _, y_raw, support, _ = train_data[seq]
        good = support & np.isfinite(y_raw)
        if good.any():
            finite_targets.append(y_raw[good])
    if not finite_targets:
        raise RuntimeError("No finite supported training triplet targets")
    scale = float(np.quantile(np.concatenate(finite_targets), 0.95))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0

    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    target_matrices: list[np.ndarray] = []
    for seq in training_sequences:
        X_seq, y_raw, support, _ = train_data[seq]
        y_norm = _normalise_target(y_raw, support, scale)
        x_blocks.append(X_seq.reshape(-1, X_seq.shape[-1]))
        y_blocks.append(y_norm.reshape(-1))
        target_matrices.append(y_norm)
    X_train = np.concatenate(x_blocks, axis=0)
    y_train = np.concatenate(y_blocks, axis=0)
    model = _fit_ridge(X_train, y_train, ridge)
    pattern_prior = np.concatenate(target_matrices, axis=0).mean(axis=0).astype(np.float32)
    return model, pattern_prior, scale


def _score_predictions(
    model_prediction: np.ndarray,
    pattern_prior: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
    alphas: list[float],
) -> dict[float, dict[str, float]]:
    n_centers, n_patterns = target.shape
    if support.shape != target.shape or model_prediction.shape != target.shape:
        raise ValueError("Prediction, target, and support shapes do not match")
    row = np.arange(n_centers)
    supported_centers = support.any(axis=1)
    safe_target = np.where(support, target, np.inf)
    oracle_supported = np.argmin(safe_target, axis=1)
    oracle_all = np.argmin(target, axis=1)
    oracle_supported_risk = np.min(safe_target, axis=1)
    oracle_all_risk = np.min(target, axis=1)
    results: dict[float, dict[str, float]] = {}

    for alpha in alphas:
        pred = np.clip(
            pattern_prior[None, :] + alpha * (model_prediction - pattern_prior[None, :]),
            0.0, 1.0,
        )
        mae_all = float(np.mean(np.abs(pred - target)))
        if support.any():
            mae_supported = float(np.mean(np.abs(pred[support] - target[support])))
        else:
            mae_supported = float("nan")

        choice_all = np.argmin(pred, axis=1)
        pred_supported = np.where(support, pred, np.inf)
        choice_supported = np.argmin(pred_supported, axis=1)
        valid_rows = supported_centers
        if valid_rows.any():
            ids = row[valid_rows]
            chosen_supported_risk = target[ids, choice_supported[valid_rows]]
            regret_supported = chosen_supported_risk - oracle_supported_risk[valid_rows]
            top1_supported = (choice_supported[valid_rows] == oracle_supported[valid_rows]).mean()
            unsupported_rate = float(np.mean(~support[row, choice_all]))
            regret_all = target[row, choice_all] - oracle_all_risk
            top1_all = (choice_all == oracle_all).mean()
            mean_regret_supported = float(np.mean(regret_supported))
        else:
            top1_supported = float("nan")
            mean_regret_supported = float("nan")
            unsupported_rate = float("nan")
            regret_all = target[row, choice_all] - oracle_all_risk
            top1_all = (choice_all == oracle_all).mean()

        results[alpha] = {
            "mae_all": mae_all,
            "mae_supported": mae_supported,
            "top1_all": float(top1_all),
            "regret_all": float(np.mean(regret_all)),
            "top1_supported": float(top1_supported),
            "regret_supported": mean_regret_supported,
            "unsupported_selection_rate": unsupported_rate,
            "supported_center_fraction": float(np.mean(supported_centers)),
        }
    return results


def _evaluate_sampled_sequence(
    X_seq: np.ndarray,
    y_raw: np.ndarray,
    support: np.ndarray,
    model: dict[str, Any],
    pattern_prior: np.ndarray,
    scale: float,
    alphas: list[float],
) -> dict[float, dict[str, float]]:
    target = _normalise_target(y_raw, support, scale)
    prediction = _predict_ridge(model, X_seq.reshape(-1, X_seq.shape[-1])).reshape(target.shape)
    return _score_predictions(prediction, pattern_prior, target, support, alphas)


def _evaluate_full_sequence(
    item: dict[str, Any],
    info: dict[str, Any],
    names: list[str],
    sample_indices: np.ndarray,
    model: dict[str, Any],
    pattern_prior: np.ndarray,
    scale: float,
    alphas: list[float],
    fps: float,
    batch_centers: int,
) -> dict[float, dict[str, float]]:
    n_patterns = len(names) ** 3
    # Sums are accumulated per sequence over centers, so long sequences do not
    # dominate the final sequence-macro summary.
    sums: dict[float, dict[str, float]] = {
        a: {
            "abs_all": 0.0, "count_all": 0.0,
            "abs_supported": 0.0, "count_supported": 0.0,
            "top1_all": 0.0, "regret_all": 0.0,
            "top1_supported": 0.0, "regret_supported": 0.0,
            "unsupported_selected": 0.0, "n_centers": 0.0,
            "n_supported_centers": 0.0,
        } for a in alphas
    }

    for start in range(0, len(sample_indices), batch_centers):
        ids = sample_indices[start:start + batch_centers]
        X_batch, y_raw, support = _make_sequence_training_data(
            item, info, names, ids, fps
        )
        target = _normalise_target(y_raw, support, scale)
        pred_model = _predict_ridge(model, X_batch.reshape(-1, X_batch.shape[-1])).reshape(
            len(ids), n_patterns
        )
        score = _score_predictions(pred_model, pattern_prior, target, support, alphas)
        for alpha in alphas:
            s = sums[alpha]
            # Recover sums from batch means with their exact denominators.
            s["abs_all"] += score[alpha]["mae_all"] * target.size
            s["count_all"] += target.size
            supported_count = int(support.sum())
            s["abs_supported"] += score[alpha]["mae_supported"] * supported_count if supported_count else 0.0
            s["count_supported"] += supported_count
            s["top1_all"] += score[alpha]["top1_all"] * len(ids)
            s["regret_all"] += score[alpha]["regret_all"] * len(ids)
            valid_centers = int(support.any(axis=1).sum())
            s["top1_supported"] += score[alpha]["top1_supported"] * valid_centers if valid_centers else 0.0
            s["regret_supported"] += score[alpha]["regret_supported"] * valid_centers if valid_centers else 0.0
            s["unsupported_selected"] += score[alpha]["unsupported_selection_rate"] * len(ids) if np.isfinite(score[alpha]["unsupported_selection_rate"]) else 0.0
            s["n_centers"] += len(ids)
            s["n_supported_centers"] += valid_centers
        del X_batch, y_raw, support, target, pred_model

    output: dict[float, dict[str, float]] = {}
    for alpha, s in sums.items():
        output[alpha] = {
            "mae_all": s["abs_all"] / max(s["count_all"], 1.0),
            "mae_supported": s["abs_supported"] / max(s["count_supported"], 1.0),
            "top1_all": s["top1_all"] / max(s["n_centers"], 1.0),
            "regret_all": s["regret_all"] / max(s["n_centers"], 1.0),
            "top1_supported": s["top1_supported"] / max(s["n_supported_centers"], 1.0),
            "regret_supported": s["regret_supported"] / max(s["n_supported_centers"], 1.0),
            "unsupported_selection_rate": s["unsupported_selected"] / max(s["n_centers"], 1.0),
            "supported_center_fraction": s["n_supported_centers"] / max(s["n_centers"], 1.0),
        }
    return output


def _bootstrap_ci(values: np.ndarray, seed: int, samples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out", type=Path, default=Path("outputs/selective_repair_crossfit"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--out", type=Path, default=Path("outputs/blended_triplet_risk"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--max-train-centers-per-sequence", type=int, default=64)
    parser.add_argument("--test-batch-centers", type=int, default=256)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--alphas", default="0,0.1,0.2,0.35,0.5,0.75,1.0")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    if args.fps <= 0 or args.ridge < 0:
        parser.error("FPS must be positive and ridge non-negative")
    if args.max_train_centers_per_sequence < 8 or args.test_batch_centers < 1:
        parser.error("Training centers must be >=8 and batch centers >=1")
    if args.inner_folds < 2 or args.bootstrap_samples < 100:
        parser.error("At least 2 inner folds and 100 bootstrap samples are required")
    alphas = sorted(set(float(x.strip()) for x in args.alphas.split(",") if x.strip()))
    if not alphas or any(not np.isfinite(a) or a < 0 or a > 1 for a in alphas) or 0.0 not in alphas or 1.0 not in alphas:
        parser.error("Alpha grid must include 0 and 1 and contain only values in [0,1]")

    started = time.time()
    source_out = args.source_out.resolve()
    dataset_root = args.dataset_root.resolve()
    metadata_path = source_out / "run_metadata.json"
    cache_dir = source_out / "feature_cache"
    if not metadata_path.is_file() or not cache_dir.is_dir():
        raise FileNotFoundError("Source output needs run_metadata.json and feature_cache/")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = [str(s).zfill(4) for s in meta["selected_sequences"]]
    conditions = list(meta["conditions"])
    names = list(meta["candidate_order"])
    signature = str(meta["feature_cache"]["signature"])
    if not names or names[0] != "raw" or len(selected) < 4:
        raise ValueError("Expected raw-first candidate order and at least four sequences")

    print("Validating sequence metadata and cached ground truth...", flush=True)
    infos: dict[str, dict[str, Any]] = {}
    actions: dict[tuple[str, str], str] = {}
    for seq in selected:
        clean_item = _load_cache_item(cache_dir / f"{seq}_{conditions[0]}.npz", names, signature)
        infos[seq] = _load_sequence_info(dataset_root, seq, clean_item["gt_xy"])
        for condition in conditions:
            item = clean_item if condition == conditions[0] else _load_cache_item(
                cache_dir / f"{seq}_{condition}.npz", names, signature
            )
            if item["gt_xy"].shape != infos[seq]["gt_xy"].shape or not np.allclose(
                item["gt_xy"], infos[seq]["gt_xy"], equal_nan=True
            ):
                raise ValueError(f"GT/cache mismatch for {seq}/{condition}")
            actions[(seq, condition)] = str(item["action"])
        del clean_item

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    outer_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for cond_idx, condition in enumerate(conditions):
        print(f"\n[{condition}] preparing sampled training centers", flush=True)
        train_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        action_map = {seq: actions[(seq, condition)] for seq in selected}
        for seq_idx, seq in enumerate(selected):
            item = _load_cache_item(cache_dir / f"{seq}_{condition}.npz", names, signature)
            valid = np.asarray(infos[seq]["valid_gt"], dtype=bool)
            gt_mask = valid[:-2] & valid[1:-1] & valid[2:]
            valid_indices = np.flatnonzero(gt_mask.reshape(-1))
            if not len(valid_indices):
                raise RuntimeError(f"No GT-valid centers for {seq}/{condition}")
            rng = np.random.default_rng(args.seed + cond_idx * 100003 + seq_idx * 1009)
            sample_n = min(len(valid_indices), args.max_train_centers_per_sequence)
            sample_indices = np.sort(rng.choice(valid_indices, size=sample_n, replace=False)) if sample_n < len(valid_indices) else valid_indices
            train_data[seq] = _make_sequence_training_data(
                item, infos[seq], names, sample_indices, args.fps
            ) + (sample_indices,)
            del item

        for outer_idx, heldout in enumerate(selected):
            outer_training = [s for s in selected if s != heldout]
            folds = _action_balanced_folds(
                outer_training, action_map, args.inner_folds,
                args.seed + 1009 * outer_idx + 9176 * cond_idx,
            )
            inner_scores: dict[float, list[dict[str, float]]] = {a: [] for a in alphas}
            for fold_idx, validation_sequences in enumerate(folds):
                validation_set = set(validation_sequences)
                inner_training = [s for s in outer_training if s not in validation_set]
                inner_model, inner_prior, inner_scale = _fit_model_and_pattern_prior(
                    inner_training, train_data, args.ridge
                )
                for val_seq in validation_sequences:
                    X_val, y_val, support_val, _ = train_data[val_seq]
                    scores = _evaluate_sampled_sequence(
                        X_val, y_val, support_val, inner_model, inner_prior,
                        inner_scale, alphas,
                    )
                    for alpha in alphas:
                        inner_scores[alpha].append(scores[alpha])
                del inner_model, inner_prior

            alpha_means: list[dict[str, float]] = []
            for alpha in alphas:
                frame = pd.DataFrame(inner_scores[alpha])
                means = frame.mean(numeric_only=True)
                row = {
                    "outer_heldout_sequence": heldout,
                    "condition": condition,
                    "alpha": alpha,
                    "n_inner_sequences": int(len(frame)),
                    **{f"{k}_mean": float(v) for k, v in means.items()},
                }
                sweep_rows.append(row)
                alpha_means.append(row)
            # Decision-aware selection: minimise supported regret; tie-break by
            # higher supported Top-1, then lower supported MAE, then smaller alpha.
            valid_alpha_rows = [r for r in alpha_means if np.isfinite(r.get("regret_supported_mean", np.nan))]
            if not valid_alpha_rows:
                raise RuntimeError(f"No scoreable inner alpha results for {condition}/{heldout}")
            chosen = sorted(
                valid_alpha_rows,
                key=lambda r: (
                    r["regret_supported_mean"],
                    -r["top1_supported_mean"],
                    r["mae_supported_mean"],
                    r["alpha"],
                ),
            )[0]
            chosen_alpha = float(chosen["alpha"])
            selection_rows.append({
                "outer_heldout_sequence": heldout,
                "condition": condition,
                "selected_alpha": chosen_alpha,
                "inner_selected_regret_supported": chosen["regret_supported_mean"],
                "inner_selected_top1_supported": chosen["top1_supported_mean"],
                "inner_selected_mae_supported": chosen["mae_supported_mean"],
                "selection_rule": "minimum sequence-macro supported-triplet decision regret; ties by higher supported Top-1, lower supported MAE, then smaller alpha",
                "inner_folds": len(folds),
                "n_inner_validation_sequences": chosen["n_inner_sequences"],
            })

            outer_model, outer_prior, outer_scale = _fit_model_and_pattern_prior(
                outer_training, train_data, args.ridge
            )
            item = _load_cache_item(cache_dir / f"{heldout}_{condition}.npz", names, signature)
            valid = np.asarray(infos[heldout]["valid_gt"], dtype=bool)
            gt_mask = valid[:-2] & valid[1:-1] & valid[2:]
            test_indices = np.flatnonzero(gt_mask.reshape(-1))
            outer_scores = _evaluate_full_sequence(
                item, infos[heldout], names, test_indices, outer_model,
                outer_prior, outer_scale, alphas, args.fps, args.test_batch_centers,
            )
            for alpha in alphas:
                outer_rows.append({
                    "sequence": heldout,
                    "action": item["action"],
                    "condition": condition,
                    "alpha": alpha,
                    "is_selected_alpha": bool(np.isclose(alpha, chosen_alpha)),
                    "selected_alpha_for_outer_fold": chosen_alpha,
                    "target_scale_train_q95": outer_scale,
                    "n_gt_valid_centers": len(test_indices),
                    **outer_scores[alpha],
                })
            del item, outer_model, outer_prior
            if (outer_idx + 1) % 5 == 0 or outer_idx == len(selected) - 1:
                print(
                    f"  outer {outer_idx + 1}/{len(selected)} ({heldout}): "
                    f"alpha={chosen_alpha:.2f}; "
                    f"selected regret={outer_scores[chosen_alpha]['regret_supported']:.4f}; "
                    f"pattern regret={outer_scores[0.0]['regret_supported']:.4f}",
                    flush=True,
                )
        del train_data

    per_sequence = pd.DataFrame(outer_rows)
    selection = pd.DataFrame(selection_rows)
    sweep = pd.DataFrame(sweep_rows)
    per_sequence.to_csv(out / "blended_triplet_risk_by_sequence.csv", index=False)
    selection.to_csv(out / "blended_triplet_risk_alpha_selection.csv", index=False)
    sweep.to_csv(out / "blended_triplet_risk_inner_sweep.csv", index=False)

    # Assemble one outer-tested selected-alpha row per sequence and pair it with
    # alpha=0 (pattern-prior-only) for sequence-level bootstrap comparisons.
    selected_rows = per_sequence[per_sequence["is_selected_alpha"]].copy()
    pattern_rows = per_sequence[np.isclose(per_sequence["alpha"], 0.0)].copy()
    selected_rows = selected_rows.set_index(["condition", "sequence"])
    pattern_rows = pattern_rows.set_index(["condition", "sequence"])
    metric_names = [
        "mae_all", "mae_supported", "top1_all", "regret_all",
        "top1_supported", "regret_supported", "unsupported_selection_rate",
    ]
    summary_rows: list[dict[str, Any]] = []
    for condition in conditions:
        sel = selected_rows.loc[condition]
        pat = pattern_rows.loc[condition]
        row: dict[str, Any] = {
            "condition": condition,
            "n_sequences": int(sel.index.nunique()),
            "selected_alpha_mean": float(sel["selected_alpha_for_outer_fold"].mean()),
            "selected_alpha_zero_fraction": float(np.mean(np.isclose(sel["selected_alpha_for_outer_fold"], 0.0))),
            "selected_alpha_one_fraction": float(np.mean(np.isclose(sel["selected_alpha_for_outer_fold"], 1.0))),
        }
        for metric in metric_names:
            row[f"selected_{metric}_sequence_mean"] = float(sel[metric].mean())
            row[f"pattern_{metric}_sequence_mean"] = float(pat[metric].mean())
        summary_rows.append(row)

        for metric in metric_names:
            delta = (sel[metric].to_numpy(float) - pat[metric].to_numpy(float))
            lo, hi = _bootstrap_ci(delta, args.seed + cond_idx * 100 + metric_names.index(metric), args.bootstrap_samples)
            paired_rows.append({
                "condition": condition,
                "metric": metric,
                "comparison": "nested_selected_blend_minus_pattern_prior",
                "n_paired_sequences": int(np.isfinite(delta).sum()),
                "delta_mean": float(np.nanmean(delta)),
                "delta_ci95_low": lo,
                "delta_ci95_high": hi,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_unit": "sequence",
            })
    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    summary.to_csv(out / "blended_triplet_risk_summary.csv", index=False)
    paired.to_csv(out / "blended_triplet_risk_paired_comparison.csv", index=False)

    protocol = {
        "experiment": "nested_loso_pattern_prior_contextual_triplet_risk_blend",
        "dataset": "Penn Action",
        "official_split": "selected official training sequences only; sequence loader verifies train membership",
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_order": names,
        "all_triplets": f"all {len(names) ** 3} candidate identity triplets at (t-1,t,t+1)",
        "outer_protocol": "leave-one-sequence-out; held-out sequence excluded from model, pattern prior, scale, and alpha selection",
        "inner_protocol": "action-balanced sequence folds; alpha selected on held-out inner sequences by supported-triplet decision regret",
        "blend": "clip(pattern_mean + alpha * (Ridge_prediction - pattern_mean), 0, 1)",
        "alpha_grid": alphas,
        "training_sampling": f"up to {args.max_train_centers_per_sequence} GT-valid frame-joint centers per sequence, sampled without replacement; all candidate triplets expanded for each center",
        "inner_validation_sampling": "uses the same sampled centers held out from Ridge fitting; selected outer alpha is then evaluated on every GT-valid center of the outer-held-out sequence",
        "target": "image-diagonal-normalized hybrid-triplet acceleration error; unsupported triplets assigned maximum normalized risk 1, while supported-only decision metrics enforce finite support",
        "decision_metrics": ["supported Top-1 accuracy", "supported mean decision regret", "unsupported-selection rate before hard support constraint", "MAE on all triplets", "MAE on supported triplets"],
        "selection_rule": "minimum inner sequence-macro supported-triplet decision regret; ties by higher supported Top-1, lower supported MAE, then smaller alpha",
        "bootstrap": {"samples": args.bootstrap_samples, "unit": "sequence", "interval": "two-sided percentile 95% CI"},
        "fps_assumed": args.fps,
        "ridge": args.ridge,
        "official_test_split_read": False,
        "media_pipe_rerun": False,
        "interpretation_guardrail": "This is a decision-oriented predictability development experiment. It does not implement sequence routing or establish final/test-set performance.",
        "runtime_seconds": round(time.time() - started, 2),
    }
    (out / "blended_triplet_risk_metadata.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("\n=== Nested blended counterfactual triplet-risk summary ===")
    print(summary.round(5).to_string(index=False))
    print("\n=== Paired selected-blend minus pattern-prior comparison ===")
    print(paired.round(5).to_string(index=False))
    print(f"\nSaved outputs under {out}")
    print("Cache-only nested development experiment; official test split was not read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
