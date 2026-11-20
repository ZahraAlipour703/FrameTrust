#!/usr/bin/env python3
"""LOSO evaluation of candidate temporal-risk prediction relative to raw.

This diagnostic reuses the existing FrameTrust temporal-risk pilot and feature
cache. It tests the decision that a temporal-risk veto would actually make:
for each repair candidate, can the model predict whether its temporal error is
higher or lower than raw on the same held-out frame/joint triples?

No MediaPipe inference is rerun. Only selected official Penn Action training
sequences are read. The official test split is never opened.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from frametrust.selective import fit_error_predictor, predict_candidate_errors  # noqa: E402
from evaluate_temporal_risk_predictability import (  # noqa: E402
    _build_temporal_targets_and_features,
    _load_cache_item,
    _load_sequence_info,
    _normalize_targets,
)


def _safe_metrics(true_delta: np.ndarray, predicted_delta: np.ndarray) -> dict[str, float]:
    """Score the binary decision: candidate temporal risk <= raw temporal risk."""
    true_delta = np.asarray(true_delta, dtype=float).ravel()
    predicted_delta = np.asarray(predicted_delta, dtype=float).ravel()
    good = np.isfinite(true_delta) & np.isfinite(predicted_delta)
    true_delta = true_delta[good]
    predicted_delta = predicted_delta[good]
    if not len(true_delta):
        return {
            "n_points": 0,
            "delta_mae": float("nan"),
            "sign_accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "harmful_precision": float("nan"),
            "harmful_recall": float("nan"),
            "harmful_acceptance_rate": float("nan"),
            "safe_acceptance_rate": float("nan"),
        }

    # Positive delta means the repair has greater acceleration error than raw.
    true_harmful = true_delta > 0.0
    predicted_harmful = predicted_delta > 0.0
    true_safe = ~true_harmful  # ties count as non-harmful
    predicted_safe = ~predicted_harmful

    if true_harmful.any() and true_safe.any():
        sensitivity = float(np.mean(predicted_harmful[true_harmful]))
        specificity = float(np.mean(predicted_safe[true_safe]))
        balanced = 0.5 * (sensitivity + specificity)
    else:
        # Balanced accuracy is undefined when a sequence has only one class.
        balanced = float("nan")

    predicted_harmful_count = int(predicted_harmful.sum())
    true_harmful_count = int(true_harmful.sum())
    precision = (
        float(np.sum(predicted_harmful & true_harmful) / predicted_harmful_count)
        if predicted_harmful_count else float("nan")
    )
    recall = (
        float(np.sum(predicted_harmful & true_harmful) / true_harmful_count)
        if true_harmful_count else float("nan")
    )
    return {
        "n_points": int(len(true_delta)),
        "delta_mae": float(np.mean(np.abs(predicted_delta - true_delta))),
        "sign_accuracy": float(np.mean(predicted_safe == true_safe)),
        "balanced_accuracy": balanced,
        "harmful_precision": precision,
        "harmful_recall": recall,
        # Among truly harmful repairs, how often would the zero-margin veto accept them?
        "harmful_acceptance_rate": (
            float(np.mean(predicted_safe[true_harmful]))
            if true_harmful.any() else float("nan")
        ),
        # Fraction of truly safe repair points retained by the zero-margin rule.
        "safe_acceptance_rate": (
            float(np.mean(predicted_safe[true_safe]))
            if true_safe.any() else float("nan")
        ),
    }


def _baseline_metrics(true_delta: np.ndarray, baseline_delta: float) -> dict[str, float]:
    true_delta = np.asarray(true_delta, dtype=float).ravel()
    true_delta = true_delta[np.isfinite(true_delta)]
    if not len(true_delta):
        return {
            "n_points": 0,
            "delta_mae": float("nan"),
            "sign_accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "harmful_precision": float("nan"),
            "harmful_recall": float("nan"),
            "harmful_acceptance_rate": float("nan"),
            "safe_acceptance_rate": float("nan"),
        }
    prediction = np.full_like(true_delta, float(baseline_delta))
    return _safe_metrics(true_delta, prediction)


def _bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-out", type=Path,
        default=Path("outputs/selective_repair_crossfit"),
    )
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("data/external/Penn_Action"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/temporal_risk_relative_to_raw"),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    if args.fps <= 0 or args.ridge < 0 or args.bootstrap_samples < 100:
        parser.error("fps must be positive, ridge non-negative, and bootstrap-samples >=100")

    source_out = args.source_out.resolve()
    dataset_root = args.dataset_root.resolve()
    metadata_path = source_out / "run_metadata.json"
    cache_dir = source_out / "feature_cache"
    if not metadata_path.is_file() or not cache_dir.is_dir():
        raise FileNotFoundError(
            "Source output must contain run_metadata.json and feature_cache/"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = [str(value).zfill(4) for value in metadata["selected_sequences"]]
    conditions = list(metadata["conditions"])
    candidate_names = list(metadata["candidate_order"])
    signature = str(metadata["feature_cache"]["signature"])
    if not candidate_names or candidate_names[0] != "raw":
        raise ValueError("candidate_order must start with raw")
    if len(selected) < 3:
        raise ValueError("At least three sequences are required for LOSO")

    print("Validating selected training sequences and cached ground truth...", flush=True)
    sequence_info: dict[str, dict[str, Any]] = {}
    for sequence in selected:
        clean_path = cache_dir / f"{sequence}_{conditions[0]}.npz"
        if not clean_path.is_file():
            raise FileNotFoundError(f"Missing cache: {clean_path}")
        with np.load(clean_path, allow_pickle=False) as saved:
            if str(saved["cache_signature"].item()) != signature:
                raise ValueError(f"Cache signature mismatch: {clean_path}")
            cached_gt = saved["gt_xy"].astype(float, copy=True)
        sequence_info[sequence] = _load_sequence_info(
            dataset_root, sequence, cached_gt
        )

    per_sequence_rows: list[dict[str, Any]] = []
    for condition_idx, condition in enumerate(conditions):
        print(f"\n[{condition}] loading cache", flush=True)
        items: dict[str, dict[str, Any]] = {}
        augmented_features: dict[str, np.ndarray] = {}
        temporal_errors: dict[str, np.ndarray] = {}
        gt_masks: dict[str, np.ndarray] = {}

        for sequence in selected:
            cache_path = cache_dir / f"{sequence}_{condition}.npz"
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing cache: {cache_path}")
            item = _load_cache_item(cache_path, candidate_names, signature)
            if not np.allclose(
                item["gt_xy"], sequence_info[sequence]["gt_xy"], equal_nan=True
            ):
                raise ValueError(f"Ground-truth/cache mismatch: {sequence}/{condition}")
            augmented, errors, gt_triples = _build_temporal_targets_and_features(
                item, sequence_info[sequence], candidate_names, args.fps
            )
            items[sequence] = item
            augmented_features[sequence] = augmented
            temporal_errors[sequence] = errors
            gt_masks[sequence] = gt_triples

        for outer_idx, heldout in enumerate(selected):
            training = [sequence for sequence in selected if sequence != heldout]
            finite_training_errors: list[np.ndarray] = []
            for sequence in training:
                mask = gt_masks[sequence]
                values = temporal_errors[sequence][1:-1][mask]
                values = values[np.isfinite(values)]
                if values.size:
                    finite_training_errors.append(values)
            if not finite_training_errors:
                raise RuntimeError(f"No finite temporal targets for {condition}/{heldout}")
            train_scale_values = np.concatenate(finite_training_errors)
            scale = float(np.quantile(train_scale_values, 0.95))
            if not np.isfinite(scale) or scale <= 1e-9:
                scale = 1.0

            train_x: list[np.ndarray] = []
            train_y: list[np.ndarray] = []
            for sequence in training:
                mask = gt_masks[sequence]
                x = augmented_features[sequence][1:-1][mask]
                y = _normalize_targets(
                    temporal_errors[sequence], mask, scale
                )[1:-1][mask]
                if x.size:
                    train_x.append(x)
                    train_y.append(y)
            n_train = sum(len(values) for values in train_x)
            if not train_x or n_train < 100:
                raise RuntimeError(f"Insufficient temporal rows for {condition}/{heldout}")

            model = fit_error_predictor(
                np.concatenate(train_x, axis=0),
                np.concatenate(train_y, axis=0),
                ridge=args.ridge,
            )
            predicted = predict_candidate_errors(
                model, augmented_features[heldout]
            )
            target = _normalize_targets(
                temporal_errors[heldout], gt_masks[heldout], scale
            )
            baseline_candidate_risk = np.concatenate(train_y, axis=0).mean(axis=0)
            valid = gt_masks[heldout]
            true_center = target[1:-1][valid]
            pred_center = predicted[1:-1][valid]
            if not len(true_center):
                raise RuntimeError(f"No temporal evaluation triples for {condition}/{heldout}")

            for candidate_idx, candidate_name in enumerate(candidate_names[1:], start=1):
                true_delta = true_center[:, candidate_idx] - true_center[:, 0]
                predicted_delta = pred_center[:, candidate_idx] - pred_center[:, 0]
                baseline_delta = float(
                    baseline_candidate_risk[candidate_idx] - baseline_candidate_risk[0]
                )
                model_scores = _safe_metrics(true_delta, predicted_delta)
                baseline_scores = _baseline_metrics(true_delta, baseline_delta)
                per_sequence_rows.append({
                    "condition": condition,
                    "sequence": heldout,
                    "candidate": candidate_name,
                    "n_training_sequences": len(training),
                    "temporal_target_scale_train_q95": scale,
                    **{f"model_{key}": value for key, value in model_scores.items()},
                    **{f"baseline_{key}": value for key, value in baseline_scores.items()},
                    "delta_mae_improvement": baseline_scores["delta_mae"] - model_scores["delta_mae"],
                    "sign_accuracy_delta": model_scores["sign_accuracy"] - baseline_scores["sign_accuracy"],
                    "balanced_accuracy_delta": model_scores["balanced_accuracy"] - baseline_scores["balanced_accuracy"],
                    "harmful_recall_delta": model_scores["harmful_recall"] - baseline_scores["harmful_recall"],
                    "harmful_acceptance_delta": model_scores["harmful_acceptance_rate"] - baseline_scores["harmful_acceptance_rate"],
                })
            print(
                f"  outer={heldout}: scored {int(valid.sum())} GT-valid temporal triples",
                flush=True,
            )

        del items, augmented_features, temporal_errors, gt_masks

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    per_sequence = pd.DataFrame(per_sequence_rows)
    per_sequence_path = out / "temporal_risk_relative_to_raw_by_sequence.csv"
    per_sequence.to_csv(per_sequence_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    metric_deltas = [
        "delta_mae_improvement",
        "sign_accuracy_delta",
        "balanced_accuracy_delta",
        "harmful_recall_delta",
        "harmful_acceptance_delta",
    ]
    for group_idx, ((condition, candidate), group) in enumerate(
        per_sequence.groupby(["condition", "candidate"], sort=False)
    ):
        row: dict[str, Any] = {
            "condition": condition,
            "candidate": candidate,
            "n_sequences": int(group["sequence"].nunique()),
        }
        for prefix in ("model", "baseline"):
            for metric in (
                "delta_mae", "sign_accuracy", "balanced_accuracy",
                "harmful_precision", "harmful_recall",
                "harmful_acceptance_rate", "safe_acceptance_rate",
            ):
                row[f"{prefix}_{metric}_sequence_mean"] = float(
                    group[f"{prefix}_{metric}"].mean()
                )
        for metric_idx, metric in enumerate(metric_deltas):
            values = group[metric].to_numpy(dtype=float)
            lo, hi = _bootstrap_ci(
                values,
                seed=args.seed + 101 * group_idx + metric_idx,
                n_bootstrap=args.bootstrap_samples,
            )
            row[f"{metric}_sequence_mean"] = float(np.nanmean(values))
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = out / "temporal_risk_relative_to_raw_summary.csv"
    summary.to_csv(summary_path, index=False)

    run_metadata = {
        "experiment": "candidate_temporal_risk_relative_to_raw",
        "dataset": "Penn Action",
        "official_split": "selected official training sequences only; load_sequence enforces train membership",
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_order": candidate_names,
        "source_selective_repair_output": str(source_out),
        "source_feature_cache_signature": signature,
        "outer_protocol": "leave-one-sequence-out; held-out sequence excluded from temporal predictor fitting and training-only q95 target scaling",
        "model": "same multi-output Ridge temporal-risk predictor as temporal-risk predictability pilot",
        "baseline": "training-only constant mean normalized temporal risk per candidate",
        "decision_target": "for each repair candidate, predict whether its normalized acceleration error is greater than raw on the same GT-visible centered three-frame windows",
        "positive_delta": "candidate temporal risk minus raw temporal risk > 0 means the repair is temporally worse than raw",
        "metrics": [
            "candidate-vs-raw delta MAE",
            "sign accuracy",
            "balanced accuracy",
            "harmful-repair precision and recall at zero margin",
            "harmful-repair acceptance rate at zero margin",
            "safe-repair acceptance rate at zero margin",
        ],
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "unit": "sequence",
            "interval": "two-sided percentile 95% CI",
        },
        "official_test_split_read": False,
        "interpretation_guardrail": "This is a pairwise relative-risk diagnostic, not a repair-selection result. It does not establish a calibrated safety guarantee or improve the composed hybrid trajectory.",
    }
    metadata_path = out / "temporal_risk_relative_to_raw_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print("\n=== Temporal risk relative-to-raw summary (sequence macro) ===")
    display_cols = [
        "condition", "candidate", "n_sequences",
        "model_delta_mae_sequence_mean", "baseline_delta_mae_sequence_mean",
        "model_sign_accuracy_sequence_mean", "baseline_sign_accuracy_sequence_mean",
        "model_balanced_accuracy_sequence_mean", "baseline_balanced_accuracy_sequence_mean",
        "model_harmful_recall_sequence_mean", "baseline_harmful_recall_sequence_mean",
        "model_harmful_acceptance_rate_sequence_mean",
        "baseline_harmful_acceptance_rate_sequence_mean",
        "balanced_accuracy_delta_sequence_mean",
        "balanced_accuracy_delta_ci95_low", "balanced_accuracy_delta_ci95_high",
    ]
    print(summary[display_cols].round(4).to_string(index=False))
    print(f"\nSaved: {per_sequence_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {metadata_path}")
    print("Cache-only development diagnostic; official test split was not read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
