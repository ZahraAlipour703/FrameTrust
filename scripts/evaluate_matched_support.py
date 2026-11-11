#!/usr/bin/env python3
"""Pairwise matched-support comparison for sequence-routing development results.

Reconstructs the existing outer-LOSO predictions from the feature cache and the
locked penalty selections. It compares disagreement routing directly with each
fixed candidate and selective baseline on identical finite GT-supported spatial
points and identical GT-visible finite temporal triples.

This is a post-processing development audit: no MediaPipe inference is rerun,
no penalty is retuned, and the official Penn Action test split is never read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from frametrust.selective import (  # noqa: E402
    DEFAULT_CANDIDATES,
    apply_candidate_choices,
    predict_candidate_errors,
    select_candidates,
)
from frametrust.sequence_routing import route_candidate_sequence  # noqa: E402
from evaluate_selective_repair import _fit_selector_for_sequences  # noqa: E402
from evaluate_sequence_routing import _load_cache_item  # noqa: E402
from evaluate_temporal_fidelity import (  # noqa: E402
    TEMPORAL_METRICS,
    temporal_ground_truth_mask,
    temporal_metric_values,
)
from penn_action_development import load_sequence  # noqa: E402


SPATIAL_METRICS = (
    "pck_torso_0.20",
    "mean_error_norm",
    "false_correction_rate",
)
ALL_METRICS = SPATIAL_METRICS + TEMPORAL_METRICS
ROUTED_METHOD = "nested_disagreement_cost"


def spatial_common_support(
    target: np.ndarray,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
) -> np.ndarray:
    """Scoreable points where both compared methods return finite coordinates."""
    target = np.asarray(target, dtype=bool)
    a = np.asarray(pose_a, dtype=float)
    b = np.asarray(pose_b, dtype=float)
    if a.shape != b.shape or a.shape[:-1] != target.shape or a.shape[-1] != 2:
        raise ValueError("target and pose shapes do not match")
    return (
        target
        & np.isfinite(a).all(axis=-1)
        & np.isfinite(b).all(axis=-1)
    )


def temporal_common_support(
    valid_gt: np.ndarray,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
) -> np.ndarray:
    """GT-visible three-frame triples where both compared poses are finite."""
    valid = np.asarray(valid_gt, dtype=bool)
    a = np.asarray(pose_a, dtype=float)
    b = np.asarray(pose_b, dtype=float)
    if a.shape != b.shape or a.shape[:2] != valid.shape or a.shape[-1] != 2:
        raise ValueError("valid_gt and pose shapes do not match")
    if len(valid) < 3:
        return np.zeros((0, valid.shape[1]), dtype=bool)
    gt_triples = valid[:-2] & valid[1:-1] & valid[2:]
    finite_a = (
        np.isfinite(a[:-2]).all(axis=-1)
        & np.isfinite(a[1:-1]).all(axis=-1)
        & np.isfinite(a[2:]).all(axis=-1)
    )
    finite_b = (
        np.isfinite(b[:-2]).all(axis=-1)
        & np.isfinite(b[1:-1]).all(axis=-1)
        & np.isfinite(b[2:]).all(axis=-1)
    )
    return gt_triples & finite_a & finite_b


def _spatial_values_on_support(
    errors: np.ndarray,
    support: np.ndarray,
    raw_errors: np.ndarray,
    raw_finite: np.ndarray,
) -> dict[str, float]:
    errors = np.asarray(errors, dtype=float)
    support = np.asarray(support, dtype=bool)
    raw_errors = np.asarray(raw_errors, dtype=float)
    raw_finite = np.asarray(raw_finite, dtype=bool)
    if not support.any():
        return {name: float("nan") for name in SPATIAL_METRICS}
    raw_correct = support & raw_finite & (raw_errors <= 0.20)
    harmed = raw_correct & (errors > 0.20)
    false_correction = (
        float(harmed.sum() / raw_correct.sum())
        if raw_correct.any() else float("nan")
    )
    return {
        "pck_torso_0.20": float(np.mean(errors[support] <= 0.20)),
        "mean_error_norm": float(np.mean(errors[support])),
        "false_correction_rate": false_correction,
    }


def _full_spatial_values(
    target: np.ndarray,
    pose: np.ndarray,
    errors: np.ndarray,
    raw_pose: np.ndarray,
    raw_errors: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(target, dtype=bool)
    finite = np.isfinite(pose).all(axis=-1)
    raw_finite = np.isfinite(raw_pose).all(axis=-1)
    raw_correct = target & raw_finite & (raw_errors <= 0.20)
    harmed = raw_correct & (~finite | (errors > 0.20))
    n_target = int(target.sum())
    return {
        "prediction_coverage": float((finite & target).sum() / n_target)
        if n_target else float("nan"),
        "pck_torso_0.20": float(((errors <= 0.20) & target).sum() / n_target)
        if n_target else float("nan"),
        "mean_error_norm": float(np.mean(errors[target]))
        if n_target else float("nan"),
        "false_correction_rate": float(harmed.sum() / raw_correct.sum())
        if raw_correct.any() else float("nan"),
        "repair_fraction": float("nan"),
    }


def _bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-out",
        type=Path,
        required=True,
        help="Original selective-repair output directory containing run_metadata.json and feature_cache/.",
    )
    parser.add_argument(
        "--routing-results",
        type=Path,
        required=True,
        help="Directory containing the expanded sequence-routing metadata and selected penalties.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New output directory for matched-support results.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/external/Penn_Action"),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()

    if args.fps <= 0 or args.bootstrap_samples < 100:
        parser.error("--fps must be positive and --bootstrap-samples must be >=100")

    source_out = args.source_out.resolve()
    routing_dir = args.routing_results.resolve()
    out = args.out.resolve()
    source_metadata_path = source_out / "run_metadata.json"
    cache_dir = source_out / "feature_cache"
    margin_path = source_out / "nested_threshold_selection.csv"
    route_metadata_path = routing_dir / "sequence_routing_metadata.json"
    penalty_path = routing_dir / "sequence_routing_penalty_selection.csv"

    for path in (
        source_metadata_path,
        cache_dir,
        margin_path,
        route_metadata_path,
        penalty_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required source artifact does not exist: {path}")

    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    route_metadata = json.loads(route_metadata_path.read_text(encoding="utf-8"))
    selected = [str(x).zfill(4) for x in source_metadata["selected_sequences"]]
    conditions = list(source_metadata["conditions"])
    names = list(source_metadata.get("candidate_order", DEFAULT_CANDIDATES))
    if not names or names[0] != "raw":
        raise ValueError("Source metadata must define raw as candidate zero")
    if route_metadata.get("official_test_split_read") is not False:
        raise ValueError("Routing metadata does not explicitly confirm official test split was not read")
    route_selected = [str(x).zfill(4) for x in route_metadata.get("selected_sequences", [])]
    if selected != route_selected or conditions != list(route_metadata.get("conditions", [])):
        raise ValueError("Source and routing result metadata do not describe the same sequences/conditions")

    signature = str(source_metadata["feature_cache"]["signature"])
    ridge = float(source_metadata.get("ridge", 10.0))
    margin_df = pd.read_csv(margin_path, dtype={"outer_heldout_sequence": str})
    margin_df["outer_heldout_sequence"] = margin_df["outer_heldout_sequence"].str.zfill(4)
    margin_map = {
        (str(row.condition), str(row.outer_heldout_sequence)): float(row.selected_margin)
        for row in margin_df.itertuples(index=False)
    }

    selection_df = pd.read_csv(
        penalty_path,
        dtype={"outer_heldout_sequence": str},
    )
    selection_df["outer_heldout_sequence"] = selection_df["outer_heldout_sequence"].str.zfill(4)
    required_selection_columns = {
        "outer_heldout_sequence", "condition", "mode", "selected_switch_penalty"
    }
    if not required_selection_columns.issubset(selection_df.columns):
        raise ValueError("Penalty-selection CSV is missing required columns")
    if selection_df.duplicated(["outer_heldout_sequence", "condition", "mode"]).any():
        raise ValueError("Penalty-selection CSV contains duplicate outer-fold decisions")
    penalty_map = {
        (str(row.condition), str(row.outer_heldout_sequence), str(row.mode)): float(row.selected_switch_penalty)
        for row in selection_df.itertuples(index=False)
    }

    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for sequence in selected:
        for condition in conditions:
            cache_path = cache_dir / f"{sequence}_{condition}.npz"
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing feature cache item: {cache_path}")
            cache[(sequence, condition)] = _load_cache_item(cache_path, names, signature)

    dataset_root = args.dataset_root.resolve()
    sequence_info: dict[str, dict[str, Any]] = {}
    for sequence in selected:
        data = load_sequence(dataset_root, sequence)
        item = cache[(sequence, conditions[0])]
        if len(data["frames"]) != len(item["gt_xy"]):
            raise ValueError(f"Frame/cache length mismatch for sequence {sequence}")
        gt = np.asarray(data["gt_xy"], dtype=float)
        if gt.shape != item["gt_xy"].shape or not np.allclose(
            gt, item["gt_xy"], equal_nan=True
        ):
            raise ValueError(f"Ground-truth/cache mismatch for sequence {sequence}")
        frame = cv2.imread(str(data["frames"][0]))
        if frame is None:
            raise ValueError(f"Could not decode first frame for sequence {sequence}")
        height, width = frame.shape[:2]
        sequence_info[sequence] = {
            "gt_xy": gt,
            "valid_gt": temporal_ground_truth_mask(
                gt, data["gt_vis"], width=width, height=height
            ),
            "diagonal": float(np.hypot(width, height)),
        }

    # Direct contrasts of the disagreement router against every fixed method
    # and the two relevant adaptive baselines. All metrics are pairwise matched.
    comparison_methods = [*names, "nested_selective", "nested_constant_switch_cost"]
    comparison_methods = [name for name in comparison_methods if name != ROUTED_METHOD]
    method_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    route_penalty_records: list[dict[str, Any]] = []

    for condition_idx, condition in enumerate(conditions):
        for outer_idx, heldout in enumerate(selected):
            training_sequences = [seq for seq in selected if seq != heldout]
            try:
                margin = margin_map[(condition, heldout)]
                constant_penalty = penalty_map[(condition, heldout, "constant")]
                disagreement_penalty = penalty_map[(condition, heldout, "disagreement")]
            except KeyError as exc:
                raise ValueError(f"Missing nested margin/penalty selection for {condition}/{heldout}") from exc

            model = _fit_selector_for_sequences(
                cache,
                condition,
                training_sequences,
                ridge,
                f"matched-support outer evaluation; heldout={heldout}, condition={condition}",
            )
            item = cache[(heldout, condition)]
            predicted = predict_candidate_errors(model, item["features"])
            choices: dict[str, np.ndarray] = {
                "nested_selective": select_candidates(predicted, margin),
                "nested_constant_switch_cost": route_candidate_sequence(
                    predicted,
                    item["candidates"],
                    names,
                    min_predicted_gain=margin,
                    switch_penalty=constant_penalty,
                    mode="constant",
                    torso_scale=item["torso_scale"],
                ),
                ROUTED_METHOD: route_candidate_sequence(
                    predicted,
                    item["candidates"],
                    names,
                    min_predicted_gain=margin,
                    switch_penalty=disagreement_penalty,
                    mode="disagreement",
                    torso_scale=item["torso_scale"],
                ),
            }
            poses: dict[str, np.ndarray] = {
                name: np.asarray(item["candidates"][name], dtype=float)
                for name in names
            }
            errors: dict[str, np.ndarray] = {
                name: np.asarray(item["errors"], dtype=float)[..., idx]
                for idx, name in enumerate(names)
            }
            for method, selection in choices.items():
                poses[method] = apply_candidate_choices(item["candidates"], selection, names)
                errors[method] = np.take_along_axis(
                    np.asarray(item["errors"], dtype=float),
                    selection[..., None],
                    axis=-1,
                )[..., 0]

            target = np.asarray(item["scoreable"], dtype=bool)
            raw_pose = poses["raw"]
            raw_errors = errors["raw"]
            full_metrics: dict[str, dict[str, float]] = {}
            for method, pose in poses.items():
                spatial = _full_spatial_values(
                    target, pose, errors[method], raw_pose, raw_errors
                )
                temporal, n_supported, n_possible = temporal_metric_values(
                    pose,
                    sequence_info[heldout]["gt_xy"],
                    sequence_info[heldout]["valid_gt"],
                    sequence_info[heldout]["diagonal"],
                    args.fps,
                )
                if method in choices:
                    repair_fraction = float((choices[method][target] != 0).mean()) if target.any() else float("nan")
                else:
                    repair_fraction = 0.0 if method == "raw" else float("nan")
                full_metrics[method] = {
                    **spatial,
                    **temporal,
                    "repair_fraction": repair_fraction,
                    "temporal_support_coverage": n_supported / n_possible if n_possible else float("nan"),
                }
                method_rows.append({
                    "sequence": heldout,
                    "action": item["action"],
                    "condition": condition,
                    "method": method,
                    "n_scoreable_points": int(target.sum()),
                    "n_valid_gt_triples": n_possible,
                    "n_temporal_triples_supported": n_supported,
                    **full_metrics[method],
                })

            valid_gt = sequence_info[heldout]["valid_gt"]
            gt_triples_count = (
                int((valid_gt[:-2] & valid_gt[1:-1] & valid_gt[2:]).sum())
                if len(valid_gt) >= 3 else 0
            )
            route_pose = poses[ROUTED_METHOD]
            route_errors = errors[ROUTED_METHOD]
            route_finite = np.isfinite(route_pose).all(axis=-1)

            for baseline in comparison_methods:
                baseline_pose = poses[baseline]
                baseline_errors = errors[baseline]
                baseline_finite = np.isfinite(baseline_pose).all(axis=-1)

                spatial_support = spatial_common_support(target, route_pose, baseline_pose)
                n_common_spatial = int(spatial_support.sum())
                n_target_spatial = int(target.sum())
                route_spatial = _spatial_values_on_support(
                    route_errors, spatial_support, raw_errors, np.isfinite(raw_pose).all(axis=-1)
                )
                baseline_spatial = _spatial_values_on_support(
                    baseline_errors, spatial_support, raw_errors, np.isfinite(raw_pose).all(axis=-1)
                )

                temporal_support = temporal_common_support(valid_gt, route_pose, baseline_pose)
                n_common_temporal = int(temporal_support.sum())
                route_temporal, _, _ = temporal_metric_values(
                    route_pose,
                    sequence_info[heldout]["gt_xy"],
                    valid_gt,
                    sequence_info[heldout]["diagonal"],
                    args.fps,
                    support=temporal_support,
                )
                baseline_temporal, _, _ = temporal_metric_values(
                    baseline_pose,
                    sequence_info[heldout]["gt_xy"],
                    valid_gt,
                    sequence_info[heldout]["diagonal"],
                    args.fps,
                    support=temporal_support,
                )

                combined_route = {**route_spatial, **route_temporal}
                combined_baseline = {**baseline_spatial, **baseline_temporal}
                for metric in ALL_METRICS:
                    route_value = float(combined_route[metric])
                    baseline_value = float(combined_baseline[metric])
                    delta = (
                        route_value - baseline_value
                        if np.isfinite(route_value) and np.isfinite(baseline_value)
                        else float("nan")
                    )
                    pair_rows.append({
                        "sequence": heldout,
                        "action": item["action"],
                        "condition": condition,
                        "comparison": f"{ROUTED_METHOD}_minus_{baseline}",
                        "route_method": ROUTED_METHOD,
                        "baseline_method": baseline,
                        "metric": metric,
                        "route_value": route_value,
                        "baseline_value": baseline_value,
                        "delta_mean": delta,
                        "n_common_spatial_points": n_common_spatial,
                        "n_target_spatial_points": n_target_spatial,
                        "common_spatial_coverage": (
                            n_common_spatial / n_target_spatial if n_target_spatial else float("nan")
                        ),
                        "route_native_spatial_coverage": full_metrics[ROUTED_METHOD]["prediction_coverage"],
                        "baseline_native_spatial_coverage": full_metrics[baseline]["prediction_coverage"],
                        "n_common_temporal_triples": n_common_temporal,
                        "n_valid_gt_triples": gt_triples_count,
                        "common_temporal_coverage": (
                            n_common_temporal / gt_triples_count if gt_triples_count else float("nan")
                        ),
                        "route_native_temporal_support": full_metrics[ROUTED_METHOD]["temporal_support_coverage"],
                        "baseline_native_temporal_support": full_metrics[baseline]["temporal_support_coverage"],
                    })

            route_penalty_records.append({
                "sequence": heldout,
                "condition": condition,
                "selected_margin": margin,
                "selected_disagreement_penalty": disagreement_penalty,
                "selected_constant_penalty": constant_penalty,
            })

            print(
                f"[{condition}] outer={heldout}: matched comparisons complete",
                flush=True,
            )

    method_df = pd.DataFrame(method_rows)
    pair_df = pd.DataFrame(pair_rows)
    penalty_df = pd.DataFrame(route_penalty_records)
    out.mkdir(parents=True, exist_ok=True)
    method_df.to_csv(out / "matched_support_method_metrics.csv", index=False)
    pair_df.to_csv(out / "matched_support_paired_per_sequence.csv", index=False)
    penalty_df.to_csv(out / "matched_support_selected_penalties.csv", index=False)

    method_summary = method_df.groupby(["condition", "method"], as_index=False).agg(
        n_sequences=("sequence", "nunique"),
        prediction_coverage_mean=("prediction_coverage", "mean"),
        pck_torso_0_20_mean=("pck_torso_0.20", "mean"),
        mean_error_norm_mean=("mean_error_norm", "mean"),
        false_correction_rate_mean=("false_correction_rate", "mean"),
        repair_fraction_mean=("repair_fraction", "mean"),
        temporal_support_coverage_mean=("temporal_support_coverage", "mean"),
        temporal_jitter_norm_s2_mean=("temporal_jitter_norm_s2", "mean"),
        velocity_error_norm_s_mean=("velocity_error_norm_s", "mean"),
        acceleration_error_norm_s2_mean=("acceleration_error_norm_s2", "mean"),
    )
    method_summary.to_csv(out / "matched_support_method_summary.csv", index=False)

    rng = np.random.default_rng(args.seed)
    paired_summary_rows: list[dict[str, Any]] = []
    for (condition, comparison, baseline, metric), group in pair_df.groupby(
        ["condition", "comparison", "baseline_method", "metric"], dropna=False
    ):
        values = group["delta_mean"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        ci_low, ci_high = _bootstrap_ci(values, rng, args.bootstrap_samples)
        paired_summary_rows.append({
            "condition": condition,
            "comparison": comparison,
            "baseline_method": baseline,
            "metric": metric,
            "n_paired_sequences": len(values),
            "delta_mean": float(values.mean()) if len(values) else float("nan"),
            "delta_ci95_low": ci_low,
            "delta_ci95_high": ci_high,
            "mean_common_spatial_points": float(group["n_common_spatial_points"].mean()),
            "mean_target_spatial_points": float(group["n_target_spatial_points"].mean()),
            "mean_common_spatial_coverage": float(group["common_spatial_coverage"].mean()),
            "mean_common_temporal_triples": float(group["n_common_temporal_triples"].mean()),
            "mean_valid_gt_triples": float(group["n_valid_gt_triples"].mean()),
            "mean_common_temporal_coverage": float(group["common_temporal_coverage"].mean()),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "sequence",
            "support_definition": (
                "spatial: GT torso-scoreable points finite for both methods; "
                "temporal: GT-visible in-frame triples finite for both methods"
            ),
        })
    paired_summary = pd.DataFrame(paired_summary_rows)
    paired_summary.to_csv(out / "matched_support_paired_comparison.csv", index=False)

    protocol = {
        "dataset": "Penn Action",
        "official_split": "selected official training sequences only; load_sequence enforces official train membership",
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_methods": names,
        "routed_method": ROUTED_METHOD,
        "comparison_methods": comparison_methods,
        "source_selective_repair_output": str(source_out),
        "source_routing_results": str(routing_dir),
        "outer_protocol": "leave-one-sequence-out; candidate-error predictor fitted on the other selected training sequences",
        "margin_and_penalty": "uses the previously selected nested threshold margin and the previously selected condition/sequence/mode penalty; no retuning is performed in this audit",
        "spatial_metrics": "torso-normalized PCK@0.20, clipped torso-normalized mean candidate error, and false-correction rate; pairwise comparisons use identical GT torso-scoreable frame/joint points where both methods are finite",
        "temporal_metrics": "image-diagonal-normalized temporal jitter, velocity error, and acceleration error at the assumed FPS; pairwise comparisons use identical GT-visible in-frame three-frame triples where both methods are finite",
        "coverage_reporting": "native per-method prediction and temporal support coverage are reported separately; pairwise common-support coverage is the fraction of GT targets retained after requiring both methods to be finite",
        "primary_interpretation": "paired common-support differences describe accuracy/fidelity conditional on both methods returning a prediction; they do not replace native coverage, full-target PCK, or false-correction reporting",
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "unit": "sequence",
            "interval": "two-sided percentile 95% CI",
            "multiplicity": "exploratory intervals; no multiplicity correction across conditions, metrics, or baselines",
        },
        "fps_assumed": args.fps,
        "official_test_split_read": False,
        "limitations": [
            "Development-only analysis of the already selected Penn Action official-training subset.",
            "This is pairwise common-support matching, not a learned or threshold-matched risk-coverage curve.",
            "Spatial common support requires both methods finite on the same GT torso-scoreable points; missing-prediction recovery is separately represented by native coverage and full-target PCK.",
            "Temporal common support requires both methods finite on the same GT-visible three-frame windows; report its size and coverage because the support can be small.",
            "The second-difference magnitude is a roughness/jitter descriptor; acceleration error versus annotations is the more direct temporal-fidelity measure.",
            "The router is offline/non-causal and does not enforce whole-skeleton consistency.",
        ],
    }
    (out / "matched_support_metadata.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    print("\n=== Matched-support paired comparison ===")
    key_metrics = {"pck_torso_0.20", "mean_error_norm", "false_correction_rate",
                   "temporal_jitter_norm_s2", "velocity_error_norm_s",
                   "acceleration_error_norm_s2"}
    display = paired_summary[
        paired_summary["metric"].isin(key_metrics)
        & paired_summary["baseline_method"].isin(
            ["linear_interpolation", "confidence_aware", "savgol", "one_euro", "nested_selective"]
        )
    ]
    if not display.empty:
        print(display.round(6).to_string(index=False))
    print(f"\nSaved matched-support results under {out}")
    print("Development-only. Official Penn Action test split was not accessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
