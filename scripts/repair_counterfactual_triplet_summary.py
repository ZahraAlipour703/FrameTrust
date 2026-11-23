#!/usr/bin/env python3
"""Rebuild the counterfactual triplet-risk summary from saved per-sequence CSV.

This repairs the summary-stage KeyError caused by a column-name typo in the
original experiment script. It does not load the dataset, caches, or rerun any
models. Bootstrap resampling is performed over sequences, matching the original
LOSO summary protocol.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray, seed: int, samples: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(samples, len(x)))
    means = x[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/counterfactual_triplet_risk/counterfactual_triplet_risk_by_sequence.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/counterfactual_triplet_risk/counterfactual_triplet_risk_summary.csv"),
    )
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be >= 100")
    if not args.input.is_file():
        raise FileNotFoundError(
            f"Saved per-sequence results not found: {args.input}\n"
            "The expensive experiment may not have reached the CSV-save step. "
            "Do not rerun until the file path is checked."
        )

    df = pd.read_csv(args.input, dtype={"sequence": str})
    required = {
        "condition", "sequence", "mae_model", "mae_pattern_baseline",
        "mae_global_baseline", "mae_improvement_vs_pattern",
        "mae_improvement_vs_global", "spearman_model",
        "spearman_pattern_baseline", "top1_triplet_accuracy_model",
        "top1_triplet_accuracy_pattern_baseline", "mean_regret_model",
        "mean_regret_pattern_baseline", "oracle_mean_risk",
        "temporal_support_coverage_all_triplets", "n_train_rows",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Per-sequence CSV is missing required columns: {missing}")

    if df.duplicated(["condition", "sequence"]).any():
        raise ValueError("Duplicate condition/sequence rows found; refusing to summarize")

    conditions = sorted(df["condition"].dropna().unique().tolist())
    summary_rows: list[dict[str, object]] = []
    for cond_idx, condition in enumerate(conditions):
        group = df[df["condition"] == condition].copy()
        imp_pattern = group["mae_improvement_vs_pattern"].to_numpy(dtype=float)
        imp_global = group["mae_improvement_vs_global"].to_numpy(dtype=float)
        top1_delta = (
            group["top1_triplet_accuracy_model"].to_numpy(dtype=float)
            - group["top1_triplet_accuracy_pattern_baseline"].to_numpy(dtype=float)
        )
        ci_pattern = bootstrap_ci(imp_pattern, args.seed + cond_idx, args.bootstrap_samples)
        ci_global = bootstrap_ci(imp_global, args.seed + 1000 + cond_idx, args.bootstrap_samples)
        ci_top1 = bootstrap_ci(top1_delta, args.seed + 2000 + cond_idx, args.bootstrap_samples)

        # The original script wrote this per-sequence field as `oracle_mean_risk`.
        # Its summary tried to read `per_center_oracle_mean_risk`, causing KeyError.
        summary_rows.append({
            "condition": condition,
            "n_sequences": int(group["sequence"].nunique()),
            "mae_model_sequence_mean": float(group["mae_model"].mean()),
            "mae_pattern_baseline_sequence_mean": float(group["mae_pattern_baseline"].mean()),
            "mae_global_baseline_sequence_mean": float(group["mae_global_baseline"].mean()),
            "mae_improvement_vs_pattern_sequence_mean": float(np.nanmean(imp_pattern)),
            "mae_improvement_vs_pattern_ci95_low": ci_pattern[0],
            "mae_improvement_vs_pattern_ci95_high": ci_pattern[1],
            "mae_improvement_vs_global_sequence_mean": float(np.nanmean(imp_global)),
            "mae_improvement_vs_global_ci95_low": ci_global[0],
            "mae_improvement_vs_global_ci95_high": ci_global[1],
            "spearman_model_sequence_mean": float(group["spearman_model"].mean()),
            "spearman_pattern_baseline_sequence_mean": float(group["spearman_pattern_baseline"].mean()),
            "top1_triplet_accuracy_model_sequence_mean": float(group["top1_triplet_accuracy_model"].mean()),
            "top1_triplet_accuracy_pattern_baseline_sequence_mean": float(group["top1_triplet_accuracy_pattern_baseline"].mean()),
            "top1_delta_vs_pattern_sequence_mean": float(np.nanmean(top1_delta)),
            "top1_delta_vs_pattern_ci95_low": ci_top1[0],
            "top1_delta_vs_pattern_ci95_high": ci_top1[1],
            "mean_regret_model_sequence_mean": float(group["mean_regret_model"].mean()),
            "mean_regret_pattern_sequence_mean": float(group["mean_regret_pattern_baseline"].mean()),
            "per_center_oracle_mean_risk_sequence_mean": float(group["oracle_mean_risk"].mean()),
            "temporal_support_coverage_sequence_mean": float(group["temporal_support_coverage_all_triplets"].mean()),
            "n_train_rows_sequence_mean": float(group["n_train_rows"].mean()),
            "n_patterns": int(group["n_candidate_triplet_patterns"].max()) if "n_candidate_triplet_patterns" in group else 216,
        })

    summary = pd.DataFrame(summary_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    print("=== Rebuilt counterfactual triplet-risk summary ===")
    print(summary.round(5).to_string(index=False))
    print(f"\nRead existing per-sequence results: {args.input}")
    print(f"Saved repaired summary: {args.output}")
    print("No dataset loading, cache processing, or model fitting was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
