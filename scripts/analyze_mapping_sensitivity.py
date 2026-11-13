#!/usr/bin/env python3
"""Sequence-level sensitivity analysis for Penn Action Mapping A vs B.

Run from the FrameTrust repository root:
    python scripts/analyze_mapping_sensitivity.py

Input:
    outputs/penn_action_mapping_holdout_v2_recheck/mapping_metrics_by_sequence.csv

Outputs (written beside the input):
    mapping_sensitivity_analysis.csv
    mapping_sensitivity_analysis.json

The analysis keeps the full 15-sequence result as primary, then reports
coverage-filtered sensitivity analysis at >= 0.20 and a stricter exploratory >= 0.40 subset.
Sequence bootstrap intervals are descriptive/exploratory and are not corrected
for evaluating multiple thresholds or metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("outputs/penn_action_mapping_holdout_v2_recheck/mapping_metrics_by_sequence.csv")
THRESHOLDS = (0.00, 0.20, 0.40)
BOOTSTRAP_SAMPLES = 50_000
BOOTSTRAP_SEED = 20261113
MAPPING_A = "mapping_a_current_empirical"
MAPPING_B = "mapping_b_previous_convention"

REQUIRED = {
    "sequence", "mapping", "prediction_coverage", "gt_points",
    "mean_error_bbox_norm_detected", "mean_capped_error_bbox_norm_all_gt",
    "pck_bbox_0.20", "common_support_points",
    "common_support_pck_bbox_0.20",
}


def macro_mean(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.average(v[keep], weights=w[keep])) if keep.any() else float("nan")


def bootstrap_mean_ci(deltas: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(deltas) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(deltas), size=(BOOTSTRAP_SAMPLES, len(deltas)))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> int:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT}. Run this script from the FrameTrust repository root."
        )

    df = pd.read_csv(INPUT, dtype={"sequence": str})
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    df["sequence"] = df["sequence"].str.zfill(4)
    df = df[df["mapping"].isin([MAPPING_A, MAPPING_B])].copy()
    if df.duplicated(["sequence", "mapping"]).any():
        raise ValueError("Duplicate sequence/mapping rows found in the input CSV.")

    wide = df.pivot(index="sequence", columns="mapping")
    if MAPPING_A not in wide.columns.get_level_values(1) or MAPPING_B not in wide.columns.get_level_values(1):
        raise ValueError("Both Mapping A and Mapping B must be present for every sequence.")

    # Retain only sequence IDs with paired A/B rows.
    paired_ids = sorted(set(df.loc[df.mapping == MAPPING_A, "sequence"]) &
                        set(df.loc[df.mapping == MAPPING_B, "sequence"]))
    a = df[df.mapping == MAPPING_A].set_index("sequence").loc[paired_ids].sort_index()
    b = df[df.mapping == MAPPING_B].set_index("sequence").loc[paired_ids].sort_index()

    coverage_a = pd.to_numeric(a["prediction_coverage"], errors="coerce")
    coverage_b = pd.to_numeric(b["prediction_coverage"], errors="coerce")
    if not np.allclose(coverage_a, coverage_b, equal_nan=True, atol=1e-10):
        raise ValueError("A/B prediction coverage differs; check whether inference outputs are paired.")
    if not np.allclose(pd.to_numeric(a["gt_points"]), pd.to_numeric(b["gt_points"]), atol=0):
        raise ValueError("A/B GT point counts differ; paired comparison is not valid.")
    if not np.allclose(pd.to_numeric(a["common_support_points"]),
                       pd.to_numeric(b["common_support_points"]), atol=0):
        raise ValueError("A/B common-support counts differ; paired comparison is not valid.")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out_rows = []
    details = []

    for threshold in THRESHOLDS:
        keep_ids = [sid for sid in paired_ids if float(coverage_a.loc[sid]) >= threshold]
        excluded_ids = [sid for sid in paired_ids if float(coverage_a.loc[sid]) < threshold]
        if not keep_ids:
            continue
        aa, bb = a.loc[keep_ids], b.loc[keep_ids]

        pck_a = pd.to_numeric(aa["pck_bbox_0.20"], errors="coerce")
        pck_b = pd.to_numeric(bb["pck_bbox_0.20"], errors="coerce")
        delta = (pck_a - pck_b).to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_mean_ci(delta, rng)

        row = {
            "minimum_sequence_coverage": threshold,
            "n_sequences": len(keep_ids),
            "included_sequences": ";".join(keep_ids),
            "excluded_sequences": ";".join(excluded_ids),
            "mean_prediction_coverage": macro_mean(aa["prediction_coverage"]),
            "mapping_a_pck20_sequence_macro": macro_mean(aa["pck_bbox_0.20"]),
            "mapping_b_pck20_sequence_macro": macro_mean(bb["pck_bbox_0.20"]),
            "pck20_delta_a_minus_b_sequence_macro": float(np.mean(delta)),
            "pck20_delta_bootstrap_95ci_low": ci_low,
            "pck20_delta_bootstrap_95ci_high": ci_high,
            "mapping_a_pck20_gt_point_weighted": weighted_mean(aa["pck_bbox_0.20"], aa["gt_points"]),
            "mapping_b_pck20_gt_point_weighted": weighted_mean(bb["pck_bbox_0.20"], bb["gt_points"]),
            "n_gt_points": int(pd.to_numeric(aa["gt_points"]).sum()),
            "mapping_a_detected_error_sequence_macro": macro_mean(aa["mean_error_bbox_norm_detected"]),
            "mapping_b_detected_error_sequence_macro": macro_mean(bb["mean_error_bbox_norm_detected"]),
            "mapping_a_capped_error_all_gt_sequence_macro": macro_mean(aa["mean_capped_error_bbox_norm_all_gt"]),
            "mapping_b_capped_error_all_gt_sequence_macro": macro_mean(bb["mean_capped_error_bbox_norm_all_gt"]),
            "mapping_a_common_support_pck20_sequence_macro": macro_mean(aa["common_support_pck_bbox_0.20"]),
            "mapping_b_common_support_pck20_sequence_macro": macro_mean(bb["common_support_pck_bbox_0.20"]),
            "mapping_a_common_support_pck20_point_weighted": weighted_mean(aa["common_support_pck_bbox_0.20"], aa["common_support_points"]),
            "mapping_b_common_support_pck20_point_weighted": weighted_mean(bb["common_support_pck_bbox_0.20"], bb["common_support_points"]),
            "n_common_support_points": int(pd.to_numeric(aa["common_support_points"]).sum()),
            "n_sequences_mapping_a_pck_better": int(np.sum(delta > 1e-12)),
            "n_sequences_pck_tied": int(np.sum(np.abs(delta) <= 1e-12)),
            "n_sequences_mapping_b_pck_better": int(np.sum(delta < -1e-12)),
        }
        out_rows.append(row)
        details.append({
            "minimum_sequence_coverage": threshold,
            "included_sequences": keep_ids,
            "excluded_sequences": excluded_ids,
            "sequence_level_pck20_delta_a_minus_b": {
                sid: float(pck_a.loc[sid] - pck_b.loc[sid]) for sid in keep_ids
            },
        })

    summary = pd.DataFrame(out_rows)
    out_csv = INPUT.parent / "mapping_sensitivity_analysis.csv"
    out_json = INPUT.parent / "mapping_sensitivity_analysis.json"
    summary.to_csv(out_csv, index=False)

    payload = {
        "input": str(INPUT).replace("\\", "/"),
        "mapping_a": MAPPING_A,
        "mapping_b": MAPPING_B,
        "primary_threshold": 0.00,
        "sensitivity_thresholds": {"0.20": "coverage-filtered exploratory analysis", "0.40": "stricter exploratory sensitivity analysis"},
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "unit": "sequence",
            "interval": "two-sided percentile 95% CI for paired sequence-macro PCK@0.20 difference",
            "seed": BOOTSTRAP_SEED,
            "multiplicity": "no correction across thresholds or metrics; exploratory intervals",
        },
        "interpretation_note": "Coverage filtering is a sensitivity analysis only; the full cohort remains the primary result. Mapping A and B share the same detector outputs, so coverage should be identical.",
        "threshold_details": details,
        "summary": out_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nMapping sensitivity analysis complete")
    print(summary[[
        "minimum_sequence_coverage", "n_sequences", "excluded_sequences",
        "mean_prediction_coverage", "mapping_a_pck20_sequence_macro",
        "mapping_b_pck20_sequence_macro", "pck20_delta_a_minus_b_sequence_macro",
        "pck20_delta_bootstrap_95ci_low", "pck20_delta_bootstrap_95ci_high",
        "mapping_a_common_support_pck20_point_weighted",
        "mapping_b_common_support_pck20_point_weighted",
    ]].to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

