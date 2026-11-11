#!/usr/bin/env python3
"""Analyze cross-fitted selective-repair outputs with paired sequence bootstrap CIs.

Run after evaluate_selective_repair.py. This reads only development outputs; it does
not load or evaluate the official Penn Action test split.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/selective_repair_crossfit"))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")

    out = args.out.resolve()
    pred_path = out / "out_of_fold_candidate_predictions.csv"
    choice_path = out / "per_point_choices.csv"
    threshold_path = out / "threshold_sweep_per_sequence.csv"
    for path in (pred_path, choice_path, threshold_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run evaluate_selective_repair.py first.")

    pred = pd.read_csv(pred_path)
    choices = pd.read_csv(choice_path)
    thresholds = pd.read_csv(threshold_path)
    pred["scoreable_gt"] = as_bool(pred["scoreable_gt"])
    choices["scoreable_gt"] = as_bool(choices["scoreable_gt"])

    # Candidate metrics are computed per sequence first, then sequences receive equal
    # weight in condition-level means and in the paired bootstrap.
    scored = pred[pred["scoreable_gt"] & np.isfinite(pred["actual_error_norm"])].copy()
    scored["pck"] = scored["actual_error_norm"] <= 0.20
    candidate_seq = scored.groupby(
        ["sequence", "action", "condition", "candidate"], as_index=False
    ).agg(
        n_points=("actual_error_norm", "size"),
        pck=("pck", "mean"),
        mean_error=("actual_error_norm", "mean"),
    )

    sel = choices[choices["scoreable_gt"] & np.isfinite(choices["selected_actual_error"])].copy()
    sel["pck"] = sel["selected_actual_error"] <= 0.20
    selective_seq = sel.groupby(
        ["sequence", "condition"], as_index=False
    ).agg(
        n_points=("selected_actual_error", "size"),
        pck=("pck", "mean"),
        mean_error=("selected_actual_error", "mean"),
    )
    selective_seq["candidate"] = "selective"
    # per_point_choices.csv intentionally has no action column; recover the
    # sequence-level action from candidate predictions, which do include it.
    sequence_actions = pred[["sequence", "action"]].drop_duplicates()
    if sequence_actions["sequence"].duplicated().any():
        raise ValueError("A sequence maps to multiple action labels in candidate predictions.")
    selective_seq = selective_seq.merge(
        sequence_actions, on="sequence", how="left", validate="many_to_one"
    )
    if selective_seq["action"].isna().any():
        raise ValueError("Could not recover action labels for one or more sequences.")
    all_seq = pd.concat([
        candidate_seq[["sequence", "action", "condition", "candidate", "n_points", "pck", "mean_error"]],
        selective_seq[["sequence", "action", "condition", "candidate", "n_points", "pck", "mean_error"]],
    ], ignore_index=True)
    all_seq.to_csv(out / "sequence_candidate_metrics.csv", index=False)

    summary = all_seq.groupby(["condition", "candidate"], as_index=False).agg(
        n_sequences=("sequence", "nunique"),
        pck_mean=("pck", "mean"),
        mean_error_norm=("mean_error", "mean"),
        median_sequence_pck=("pck", "median"),
    )
    summary.to_csv(out / "candidate_sequence_summary.csv", index=False)

    rng = np.random.default_rng(args.seed)
    paired_rows = []
    for condition in sorted(all_seq["condition"].unique()):
        cond = all_seq[all_seq["condition"] == condition]
        wide_pck = cond.pivot(index="sequence", columns="candidate", values="pck")
        wide_err = cond.pivot(index="sequence", columns="candidate", values="mean_error")
        if "selective" not in wide_pck.columns:
            continue
        for baseline in [c for c in wide_pck.columns if c != "selective"]:
            paired = pd.concat(
                [wide_pck[["selective", baseline]], wide_err[["selective", baseline]].add_prefix("err_")],
                axis=1,
            ).dropna()
            if paired.empty:
                continue
            dpck = (paired["selective"] - paired[baseline]).to_numpy(dtype=float)
            derr = (paired["err_selective"] - paired[f"err_{baseline}"]).to_numpy(dtype=float)
            pck_ci = bootstrap_ci(dpck, rng, args.bootstrap_samples)
            err_ci = bootstrap_ci(derr, rng, args.bootstrap_samples)
            paired_rows.append({
                "condition": condition,
                "comparison": f"selective_minus_{baseline}",
                "baseline": baseline,
                "n_paired_sequences": len(paired),
                "delta_pck_mean": float(dpck.mean()),
                "delta_pck_ci95_low": pck_ci[0],
                "delta_pck_ci95_high": pck_ci[1],
                "delta_mean_error_norm": float(derr.mean()),
                "delta_mean_error_ci95_low": err_ci[0],
                "delta_mean_error_ci95_high": err_ci[1],
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_unit": "sequence",
            })
    pd.DataFrame(paired_rows).to_csv(out / "paired_baseline_comparison.csv", index=False)

    # Preserve the original sweep, and create a concise subset with the same
    # sequence-equal weighting used for the paired comparisons.
    threshold_summary = thresholds.groupby(
        ["condition", "min_predicted_gain"], as_index=False
    ).agg(
        n_sequences=("sequence", "nunique"),
        pck_mean=("pck_torso_0.20", "mean"),
        coverage_mean=("coverage", "mean"),
        repair_fraction_mean=("repair_fraction", "mean"),
        false_correction_rate_mean=("false_correction_rate", "mean"),
        mean_error_norm=("mean_error_norm", "mean"),
    )
    threshold_summary.to_csv(out / "threshold_sweep_summary_recomputed.csv", index=False)

    print("Development-only sequence-equal summaries written:")
    for name in (
        "sequence_candidate_metrics.csv",
        "candidate_sequence_summary.csv",
        "paired_baseline_comparison.csv",
        "threshold_sweep_summary_recomputed.csv",
    ):
        print(f"  {out / name}")
    print("Interpretation: positive delta_pck favors selective repair; negative delta_mean_error_norm favors selective repair.")
    print("Bootstrap intervals resample sequences, not individual correlated frame/joint observations.")
    print("Do not use this descriptive threshold sweep to claim an unbiased optimum threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
