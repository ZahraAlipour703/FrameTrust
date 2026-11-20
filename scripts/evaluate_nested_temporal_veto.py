#!/usr/bin/env python3
"""Nested threshold selection for FrameTrust's spatial + temporal-risk veto.

For each outer held-out sequence, fit spatial and temporal predictors only on
outer-training sequences. Select the temporal-veto threshold from action-balanced
cross-fitted predictions inside the outer-training set, then evaluate the locked
threshold on the outer held-out sequence.

The threshold is selected by minimizing composed-trajectory acceleration error
subject to conservative spatial/coverage/temporal-support guardrails relative to
the spatial-only selector. A threshold of 2.0 is an explicit no-veto fallback
(the predicted normalized risk difference is bounded to [-1, 1]).

This is a development-only nested LOSO experiment. It reuses the existing cache;
MediaPipe inference is not rerun and the official test split is not read.
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
    _sequence_spatial_metrics,
)
from evaluate_spatial_temporal_veto import (  # noqa: E402
    _evaluate_pose_and_choices,
    _fit_temporal_model,
    _make_veto_choices,
    _temporal_common_support,
    _bootstrap_ci,
)
from evaluate_temporal_fidelity import (  # noqa: E402
    TEMPORAL_METRICS,
    temporal_metric_values,
)
from evaluate_temporal_risk_predictability import (  # noqa: E402
    _build_temporal_targets_and_features,
    _load_sequence_info,
)

SPATIAL_METRICS = (
    "pck_torso_0.20",
    "mean_error_norm",
    "coverage",
    "false_correction_rate",
    "repair_fraction",
)
DEFAULT_THRESHOLDS = (-0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20, 2.0)


def _parse_thresholds(value: str) -> list[float]:
    values = sorted(set(float(x.strip()) for x in value.split(",") if x.strip()))
    if not values or not all(np.isfinite(x) for x in values):
        raise ValueError("Threshold grid must contain finite comma-separated numbers")
    if not any(x >= 2.0 for x in values):
        values.append(2.0)  # no-veto fallback because predicted deltas are in [-1, 1]
    return sorted(set(values))


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    return {
        column: float(frame[column].mean())
        for column in (
            *SPATIAL_METRICS,
            *TEMPORAL_METRICS,
            "temporal_support_coverage",
            "temporal_veto_rate",
        )
        if column in frame and frame[column].notna().any()
    }


def _choose_threshold(
    inner_rows: list[dict[str, Any]],
    pck_tolerance: float,
    minimum_coverage: float,
    maximum_false_correction_increase: float,
    maximum_temporal_support_drop: float,
) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    frame = pd.DataFrame(inner_rows)
    baseline = frame[frame["threshold"].isna()]
    if baseline.empty:
        raise RuntimeError("Inner-fold spatial-only baseline is missing")
    base = baseline.mean(numeric_only=True)

    sweep_rows: list[dict[str, Any]] = []
    for threshold, group in frame[frame["threshold"].notna()].groupby("threshold"):
        means = group.mean(numeric_only=True)
        row = {"threshold": float(threshold), "n_inner_sequences": int(group["sequence"].nunique())}
        for metric in (
            *SPATIAL_METRICS,
            *TEMPORAL_METRICS,
            "temporal_support_coverage",
            "temporal_veto_rate",
        ):
            if metric in means:
                row[f"{metric}_mean"] = float(means[metric])
        row["pck_floor"] = float(base["pck_torso_0.20"] - pck_tolerance)
        row["coverage_floor"] = float(max(minimum_coverage, base["coverage"] - 0.01))
        row["false_correction_ceiling"] = float(
            base["false_correction_rate"] + maximum_false_correction_increase
        )
        row["temporal_support_floor"] = float(
            base["temporal_support_coverage"] - maximum_temporal_support_drop
        )
        base_fc = float(base.get("false_correction_rate", np.nan))
        candidate_fc = float(row.get("false_correction_rate_mean", np.nan))
        false_correction_ok = (
            not np.isfinite(base_fc)
            or (
                np.isfinite(candidate_fc)
                and candidate_fc <= row["false_correction_ceiling"]
            )
        )
        row["feasible"] = bool(
            np.isfinite(row.get("pck_torso_0.20_mean", np.nan))
            and row["pck_torso_0.20_mean"] >= row["pck_floor"]
            and row.get("coverage_mean", -np.inf) >= row["coverage_floor"]
            and false_correction_ok
            and row.get("temporal_support_coverage_mean", -np.inf) >= row["temporal_support_floor"]
            and np.isfinite(row.get("acceleration_error_norm_s2_mean", np.nan))
        )
        sweep_rows.append(row)

    sweep = pd.DataFrame(sweep_rows)
    feasible = sweep[sweep["feasible"]].copy()
    if feasible.empty:
        # The explicit no-veto threshold should be feasible relative to itself.
        raise RuntimeError("No feasible threshold, including no-veto fallback")
    feasible = feasible.sort_values(
        ["acceleration_error_norm_s2_mean", "velocity_error_norm_s_mean", "temporal_jitter_norm_s2_mean", "threshold"],
        ascending=[True, True, True, False],
    )
    chosen = float(feasible.iloc[0]["threshold"])
    decision = {
        "selected_threshold": chosen,
        "inner_baseline_pck": float(base["pck_torso_0.20"]),
        "inner_baseline_coverage": float(base["coverage"]),
        "inner_baseline_false_correction_rate": float(base["false_correction_rate"]),
        "inner_baseline_temporal_support": float(base["temporal_support_coverage"]),
        "inner_selected_pck": float(feasible.iloc[0]["pck_torso_0.20_mean"]),
        "inner_selected_acceleration_error": float(feasible.iloc[0]["acceleration_error_norm_s2_mean"]),
        "inner_selected_velocity_error": float(feasible.iloc[0]["velocity_error_norm_s_mean"]),
        "inner_selected_veto_rate": float(feasible.iloc[0]["temporal_veto_rate_mean"]),
        "n_feasible_thresholds": int(len(feasible)),
        "selection_rule": "minimize composed acceleration error; velocity error, jitter and larger threshold break ties; require PCK within tolerance of spatial-only, coverage >= max(minimum, spatial-only minus 0.01), false-correction <= spatial-only plus configured allowance, and temporal support >= spatial-only minus configured drop",
        "pck_tolerance": pck_tolerance,
        "minimum_coverage": minimum_coverage,
        "maximum_false_correction_increase": maximum_false_correction_increase,
        "maximum_temporal_support_drop": maximum_temporal_support_drop,
    }
    return chosen, sweep, decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out", type=Path, default=Path("outputs/selective_repair_crossfit"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--out", type=Path, default=Path("outputs/nested_temporal_veto"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--spatial-ridge", type=float, default=None)
    parser.add_argument("--temporal-ridge", type=float, default=10.0)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)))
    parser.add_argument("--pck-tolerance", type=float, default=0.005)
    parser.add_argument("--minimum-coverage", type=float, default=0.90)
    parser.add_argument("--maximum-false-correction-increase", type=float, default=0.01)
    parser.add_argument("--maximum-temporal-support-drop", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()

    if args.fps <= 0 or args.spatial_ridge is not None and args.spatial_ridge < 0:
        parser.error("FPS and ridge parameters must be positive/non-negative")
    if args.temporal_ridge < 0 or args.inner_folds < 2 or args.bootstrap_samples < 100:
        parser.error("temporal ridge must be non-negative, inner folds >=2, bootstrap >=100")
    if not (0 < args.minimum_coverage <= 1) or args.pck_tolerance < 0:
        parser.error("minimum coverage must be in (0,1] and PCK tolerance non-negative")
    if args.maximum_false_correction_increase < 0 or args.maximum_temporal_support_drop < 0:
        parser.error("guardrail allowances must be non-negative")

    thresholds = _parse_thresholds(args.thresholds)
    started = time.time()
    source_out = args.source_out.resolve()
    dataset_root = args.dataset_root.resolve()
    metadata_path = source_out / "run_metadata.json"
    margin_path = source_out / "nested_threshold_selection.csv"
    cache_dir = source_out / "feature_cache"
    for path in (metadata_path, margin_path, cache_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required source artifact missing: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = [str(x).zfill(4) for x in metadata["selected_sequences"]]
    conditions = list(metadata["conditions"])
    names = list(metadata["candidate_order"])
    signature = str(metadata["feature_cache"]["signature"])
    spatial_ridge = float(args.spatial_ridge if args.spatial_ridge is not None else metadata.get("ridge", 10.0))
    margins = pd.read_csv(margin_path, dtype={"outer_heldout_sequence": str})
    margins["outer_heldout_sequence"] = margins["outer_heldout_sequence"].str.zfill(4)
    margin_map = {
        (str(row.condition), str(row.outer_heldout_sequence)): float(row.selected_margin)
        for row in margins.itertuples(index=False)
    }
    if not names or names[0] != "raw" or len(selected) < 3:
        raise ValueError("Source metadata requires raw-first candidates and >=3 sequences")

    print("Loading compatible caches and validating GT...", flush=True)
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    sequence_info: dict[str, dict[str, Any]] = {}
    for seq in selected:
        for condition in conditions:
            path = cache_dir / f"{seq}_{condition}.npz"
            cache[(seq, condition)] = _load_cache_item(path, names, signature)
        sequence_info[seq] = _load_sequence_info(
            dataset_root, seq, cache[(seq, conditions[0])]["gt_xy"]
        )
        for condition in conditions:
            cached_gt = cache[(seq, condition)]["gt_xy"]
            if cached_gt.shape != sequence_info[seq]["gt_xy"].shape or not np.allclose(
                cached_gt, sequence_info[seq]["gt_xy"], equal_nan=True
            ):
                raise ValueError(f"Ground-truth/cache mismatch for {seq}/{condition}")

    outer_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    sweep_rows_all: list[dict[str, Any]] = []

    for condition_idx, condition in enumerate(conditions):
        print(f"\n[{condition}] preparing temporal targets", flush=True)
        augmented: dict[str, np.ndarray] = {}
        temporal_errors: dict[str, np.ndarray] = {}
        gt_masks: dict[str, np.ndarray] = {}
        for seq in selected:
            x, y, mask = _build_temporal_targets_and_features(
                cache[(seq, condition)], sequence_info[seq], names, args.fps
            )
            augmented[seq], temporal_errors[seq], gt_masks[seq] = x, y, mask

        for outer_idx, heldout in enumerate(selected):
            outer_training = [seq for seq in selected if seq != heldout]
            try:
                spatial_margin = margin_map[(condition, heldout)]
            except KeyError as exc:
                raise ValueError(f"Missing spatial margin for {condition}/{heldout}") from exc

            folds = _action_balanced_folds(
                cache, condition, outer_training, args.inner_folds,
                args.seed + 1009 * outer_idx + 9176 * condition_idx,
            )
            inner_rows: list[dict[str, Any]] = []
            for fold_idx, validation_sequences in enumerate(folds):
                validation_set = set(validation_sequences)
                inner_training = [seq for seq in outer_training if seq not in validation_set]
                spatial_model = _fit_selector_for_sequences(
                    cache, condition, inner_training, spatial_ridge,
                    f"nested veto inner spatial fold={fold_idx}; outer={heldout}; condition={condition}",
                )
                temporal_model, _, _ = _fit_temporal_model(
                    inner_training, augmented, temporal_errors, gt_masks, args.temporal_ridge
                )
                for val_seq in validation_sequences:
                    item = cache[(val_seq, condition)]
                    pred_spatial = predict_candidate_errors(spatial_model, item["features"])
                    spatial_choices = select_candidates(pred_spatial, spatial_margin)
                    base_metrics, _ = _evaluate_pose_and_choices(
                        item, spatial_choices, names, sequence_info[val_seq], args.fps
                    )
                    inner_rows.append({
                        "outer_heldout_sequence": heldout,
                        "condition": condition,
                        "sequence": val_seq,
                        "threshold": np.nan,
                        **base_metrics,
                        "temporal_veto_rate": 0.0,
                    })

                    pred_temporal = predict_candidate_errors(temporal_model, augmented[val_seq])
                    for threshold in thresholds:
                        veto_choices, veto_mask = _make_veto_choices(
                            spatial_choices, pred_temporal, threshold=threshold
                        )
                        metrics, _ = _evaluate_pose_and_choices(
                            item, veto_choices, names, sequence_info[val_seq], args.fps
                        )
                        selected_count = int((spatial_choices > 0).sum())
                        veto_count = int(veto_mask.sum())
                        inner_rows.append({
                            "outer_heldout_sequence": heldout,
                            "condition": condition,
                            "sequence": val_seq,
                            "threshold": float(threshold),
                            **metrics,
                            "temporal_veto_rate": veto_count / selected_count if selected_count else 0.0,
                        })

            chosen_threshold, sweep, decision = _choose_threshold(
                inner_rows,
                args.pck_tolerance,
                args.minimum_coverage,
                args.maximum_false_correction_increase,
                args.maximum_temporal_support_drop,
            )
            for row in sweep.to_dict(orient="records"):
                sweep_rows_all.append({
                    "outer_heldout_sequence": heldout,
                    "condition": condition,
                    **row,
                })
            selection_rows.append({
                "outer_heldout_sequence": heldout,
                "condition": condition,
                "n_inner_sequences": len({r["sequence"] for r in inner_rows}),
                **decision,
            })

            # Final outer evaluation: both models are fitted on all outer-training sequences.
            outer_spatial_model = _fit_selector_for_sequences(
                cache, condition, outer_training, spatial_ridge,
                f"nested veto outer spatial; heldout={heldout}; condition={condition}",
            )
            outer_temporal_model, temporal_scale, _ = _fit_temporal_model(
                outer_training, augmented, temporal_errors, gt_masks, args.temporal_ridge
            )
            item = cache[(heldout, condition)]
            pred_spatial = predict_candidate_errors(outer_spatial_model, item["features"])
            spatial_choices = select_candidates(pred_spatial, spatial_margin)
            pred_temporal = predict_candidate_errors(outer_temporal_model, augmented[heldout])
            veto_choices, veto_mask = _make_veto_choices(
                spatial_choices, pred_temporal, threshold=chosen_threshold
            )
            base_metrics, base_pose = _evaluate_pose_and_choices(
                item, spatial_choices, names, sequence_info[heldout], args.fps
            )
            veto_metrics, veto_pose = _evaluate_pose_and_choices(
                item, veto_choices, names, sequence_info[heldout], args.fps
            )

            for method, metrics in (("spatial_only", base_metrics), ("nested_spatial_temporal_veto", veto_metrics)):
                outer_rows.append({
                    "sequence": heldout,
                    "action": item["action"],
                    "condition": condition,
                    "method": method,
                    "selected_spatial_margin": spatial_margin,
                    "selected_temporal_veto_threshold": chosen_threshold,
                    "temporal_target_scale_train_q95": temporal_scale,
                    "n_spatially_selected_repairs": int((spatial_choices > 0).sum()),
                    "n_vetoed_repairs": int(veto_mask.sum()) if method != "spatial_only" else 0,
                    "veto_rate": float(veto_mask.sum() / max(int((spatial_choices > 0).sum()), 1)) if method != "spatial_only" else 0.0,
                    **metrics,
                })

            common = _temporal_common_support(
                sequence_info[heldout]["valid_gt"], base_pose, veto_pose
            )
            base_temporal, _, _ = temporal_metric_values(
                base_pose, sequence_info[heldout]["gt_xy"], sequence_info[heldout]["valid_gt"],
                sequence_info[heldout]["diagonal"], args.fps, support=common,
            )
            veto_temporal, _, _ = temporal_metric_values(
                veto_pose, sequence_info[heldout]["gt_xy"], sequence_info[heldout]["valid_gt"],
                sequence_info[heldout]["diagonal"], args.fps, support=common,
            )
            for metric in SPATIAL_METRICS:
                paired_rows.append({
                    "sequence": heldout, "action": item["action"], "condition": condition,
                    "comparison": "nested_spatial_temporal_veto_minus_spatial_only",
                    "metric": metric, "delta": float(veto_metrics[metric] - base_metrics[metric]),
                })
            for metric in TEMPORAL_METRICS:
                paired_rows.append({
                    "sequence": heldout, "action": item["action"], "condition": condition,
                    "comparison": "nested_spatial_temporal_veto_minus_spatial_only",
                    "metric": metric, "delta": float(veto_temporal[metric] - base_temporal[metric]),
                    "n_common_temporal_triples": int(common.sum()),
                })

            if (outer_idx + 1) % 5 == 0 or outer_idx == len(selected) - 1:
                print(
                    f"  outer {outer_idx + 1}/{len(selected)} ({heldout}): "
                    f"tau={chosen_threshold:+.3f}; vetoed {int(veto_mask.sum())}/"
                    f"{int((spatial_choices > 0).sum())} repairs; "
                    f"accel Δ={veto_metrics['acceleration_error_norm_s2'] - base_metrics['acceleration_error_norm_s2']:+.5f}",
                    flush=True,
                )
        del augmented, temporal_errors, gt_masks

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    outer_df = pd.DataFrame(outer_rows)
    paired_df = pd.DataFrame(paired_rows)
    selection_df = pd.DataFrame(selection_rows)
    sweep_df = pd.DataFrame(sweep_rows_all)
    outer_path = out / "nested_temporal_veto_by_sequence.csv"
    summary_path = out / "nested_temporal_veto_summary.csv"
    paired_path = out / "nested_temporal_veto_paired_by_sequence.csv"
    paired_summary_path = out / "nested_temporal_veto_paired_summary.csv"
    selection_path = out / "nested_temporal_veto_threshold_selection.csv"
    sweep_path = out / "nested_temporal_veto_inner_sweep.csv"
    outer_df.to_csv(outer_path, index=False)
    selection_df.to_csv(selection_path, index=False)
    sweep_df.to_csv(sweep_path, index=False)
    paired_df.to_csv(paired_path, index=False)

    summary = outer_df.groupby(["condition", "method"], as_index=False).agg(
        n_sequences=("sequence", "nunique"),
        pck_mean=("pck_torso_0.20", "mean"),
        mean_error_norm_mean=("mean_error_norm", "mean"),
        coverage_mean=("coverage", "mean"),
        false_correction_rate_mean=("false_correction_rate", "mean"),
        repair_fraction_mean=("repair_fraction", "mean"),
        jitter_mean=("temporal_jitter_norm_s2", "mean"),
        velocity_error_mean=("velocity_error_norm_s", "mean"),
        acceleration_error_mean=("acceleration_error_norm_s2", "mean"),
        temporal_support_coverage_mean=("temporal_support_coverage", "mean"),
        veto_rate_mean=("veto_rate", "mean"),
    )
    summary.to_csv(summary_path, index=False)

    paired_summary_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    for (condition, comparison, metric), group in paired_df.groupby(["condition", "comparison", "metric"]):
        values = group["delta"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        low, high = _bootstrap_ci(values, rng, args.bootstrap_samples)
        paired_summary_rows.append({
            "condition": condition, "comparison": comparison, "metric": metric,
            "n_paired_sequences": len(values),
            "delta_mean": float(values.mean()) if len(values) else float("nan"),
            "delta_ci95_low": low, "delta_ci95_high": high,
            "bootstrap_samples": args.bootstrap_samples, "bootstrap_unit": "sequence",
        })
    paired_summary = pd.DataFrame(paired_summary_rows)
    paired_summary.to_csv(paired_summary_path, index=False)

    protocol = {
        "experiment": "nested_spatial_temporal_veto_threshold_selection",
        "dataset": "Penn Action",
        "official_split": "selected official training sequences only; load_sequence enforces train membership",
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_order": names,
        "outer_protocol": "leave-one-sequence-out; outer held-out sequence excluded from model fitting, temporal scaling, and threshold selection",
        "inner_protocol": "action-balanced sequence folds within each outer-training set; spatial and temporal models fitted without each inner validation fold; threshold selected from cross-fitted composed trajectories",
        "spatial_selector": "existing nested cross-fitted spatial Ridge selector with the previously selected outer-fold spatial margin",
        "temporal_model": "multi-output Ridge on candidate-specific centered acceleration errors; outer/inner train-only q95 target scaling",
        "threshold_grid": thresholds,
        "threshold_rule": "veto selected repair if predicted normalized candidate temporal risk minus predicted raw risk exceeds tau; negative tau requires predicted temporal improvement before accepting a repair",
        "selection_rule": "minimize actual composed-trajectory acceleration error on inner cross-fitted validation sequences subject to PCK, coverage, false-correction, and temporal-support guardrails relative to spatial-only",
        "guardrails": {
            "pck_tolerance": args.pck_tolerance,
            "minimum_coverage": args.minimum_coverage,
            "maximum_false_correction_increase": args.maximum_false_correction_increase,
            "maximum_temporal_support_drop": args.maximum_temporal_support_drop,
        },
        "evaluation": "outer held-out sequence; spatial and temporal metrics recomputed on the final framewise composed pose; temporal comparisons paired on common finite GT-visible triples",
        "bootstrap": {"samples": args.bootstrap_samples, "unit": "sequence", "interval": "two-sided percentile 95% CI"},
        "official_test_split_read": False,
        "source_feature_cache_signature": signature,
        "source_selective_repair_output": str(source_out),
        "runtime_seconds": round(time.time() - started, 2),
        "limitations": [
            "development-only results on selected official training sequences, not test performance",
            "the veto uses candidate-level centered temporal risk; it does not explicitly model all mixed candidate identity triplets",
            "temporal metrics measure agreement with Penn Action annotations, not biomechanical ground truth",
            "offline/non-causal due centered three-frame features",
        ],
    }
    metadata_out = out / "nested_temporal_veto_metadata.json"
    metadata_out.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("\n=== Nested spatial + temporal-veto summary ===")
    print(summary.round(5).to_string(index=False))
    print("\n=== Paired comparison (nested veto minus spatial-only) ===")
    print(paired_summary.round(5).to_string(index=False))
    print(f"\nSaved: {outer_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {selection_path}")
    print(f"Saved: {sweep_path}")
    print(f"Saved: {paired_path}")
    print(f"Saved: {paired_summary_path}")
    print(f"Saved: {metadata_out}")
    print("Nested cache-only development experiment; official test split was not read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
