#!/usr/bin/env python3
"""Evaluate a zero-margin temporal-risk veto on the cross-fitted spatial selector.

For every held-out Penn Action development sequence, the existing spatial selector
chooses a repair candidate per frame/joint. A second LOSO Ridge model predicts
candidate-specific temporal risk. If the selected repair's predicted acceleration
error exceeds raw's predicted error, the choice is vetoed back to raw (tau=0).

Crucially, evaluation is performed on the *composed hybrid trajectory* after
candidate identities have been selected frame by frame. Temporal metrics are not
inferred from candidate-specific risks; they are recomputed on the resulting pose.

This is a fixed-threshold development test, not a tuned method or test-set result.
It reuses the existing feature cache and does not rerun MediaPipe.
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
    fit_error_predictor,
    predict_candidate_errors,
    select_candidates,
)
from evaluate_selective_repair import _fit_selector_for_sequences  # noqa: E402
from evaluate_sequence_routing import (  # noqa: E402
    _load_cache_item,
    _sequence_spatial_metrics,
)
from evaluate_temporal_fidelity import (  # noqa: E402
    TEMPORAL_METRICS,
    temporal_metric_values,
)
from evaluate_temporal_risk_predictability import (  # noqa: E402
    _build_temporal_targets_and_features,
    _load_sequence_info,
    _normalize_targets,
)


SPATIAL_METRICS = (
    "pck_torso_0.20",
    "mean_error_norm",
    "coverage",
    "false_correction_rate",
    "repair_fraction",
)
ALL_METRICS = SPATIAL_METRICS + TEMPORAL_METRICS


def _fit_temporal_model(
    training_sequences: list[str],
    augmented_features: dict[str, np.ndarray],
    temporal_errors: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
    ridge: float,
) -> tuple[dict[str, Any], float, np.ndarray]:
    """Fit temporal-risk Ridge using only the supplied training sequences."""
    finite_targets: list[np.ndarray] = []
    for seq in training_sequences:
        mask = gt_masks[seq]
        values = temporal_errors[seq][1:-1][mask]
        values = values[np.isfinite(values)]
        if values.size:
            finite_targets.append(values)
    if not finite_targets:
        raise RuntimeError("No finite temporal training targets")
    scale = float(np.quantile(np.concatenate(finite_targets), 0.95))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0

    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    for seq in training_sequences:
        mask = gt_masks[seq]
        x = augmented_features[seq][1:-1][mask]
        y = _normalize_targets(temporal_errors[seq], mask, scale)[1:-1][mask]
        if x.size:
            train_x.append(x)
            train_y.append(y)
    n_rows = sum(len(x) for x in train_x)
    if not train_x or n_rows < 100:
        raise RuntimeError(f"Insufficient temporal training rows: {n_rows}")

    x_train = np.concatenate(train_x, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    model = fit_error_predictor(x_train, y_train, ridge=ridge)
    candidate_mean = y_train.mean(axis=0)
    return model, scale, candidate_mean


def _make_veto_choices(
    spatial_choices: np.ndarray,
    predicted_temporal_risk: np.ndarray,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Veto spatial repairs whose predicted temporal risk exceeds raw's."""
    choices = np.asarray(spatial_choices, dtype=np.int64)
    predicted = np.asarray(predicted_temporal_risk, dtype=float)
    if predicted.ndim != 3 or predicted.shape[:2] != choices.shape:
        raise ValueError("Predicted temporal risk and spatial choices do not align")
    if predicted.shape[-1] < 2:
        raise ValueError("At least raw and one repair candidate are required")

    delta = predicted - predicted[..., :1]
    selected_delta = np.take_along_axis(delta, choices[..., None], axis=-1)[..., 0]
    veto = (choices > 0) & (selected_delta > float(threshold))

    # The target is a centered three-frame quantity. Do not veto sequence endpoints,
    # for which no centered temporal-risk target exists.
    if len(veto):
        veto[0] = False
        veto[-1] = False

    result = choices.copy()
    result[veto] = 0
    return result, veto


def _evaluate_pose_and_choices(
    item: dict[str, Any],
    choices: np.ndarray,
    candidate_names: list[str],
    sequence_info: dict[str, Any],
    fps: float,
) -> tuple[dict[str, float | int], np.ndarray]:
    pose = apply_candidate_choices(item["candidates"], choices, candidate_names)
    spatial = _sequence_spatial_metrics(item, choices, candidate_names)
    temporal, supported, possible = temporal_metric_values(
        pose,
        sequence_info["gt_xy"],
        sequence_info["valid_gt"],
        sequence_info["diagonal"],
        fps,
    )
    metrics: dict[str, float | int] = {
        **spatial,
        **temporal,
        "temporal_support_coverage": supported / possible if possible else float("nan"),
        "n_temporal_triples_supported": supported,
        "n_valid_gt_triples": possible,
    }
    return metrics, pose


def _temporal_common_support(
    valid_gt: np.ndarray,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
) -> np.ndarray:
    valid = np.asarray(valid_gt, dtype=bool)
    a = np.asarray(pose_a, dtype=float)
    b = np.asarray(pose_b, dtype=float)
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
    draws = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[draws].mean(axis=1)
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
        default=Path("outputs/spatial_temporal_veto"),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--temporal-veto-threshold", type=float, default=0.0)
    args = parser.parse_args()

    if args.fps <= 0 or args.ridge < 0 or args.bootstrap_samples < 100:
        parser.error("fps must be positive, ridge non-negative, bootstrap-samples >=100")

    started = time.time()
    source_out = args.source_out.resolve()
    dataset_root = args.dataset_root.resolve()
    source_metadata_path = source_out / "run_metadata.json"
    margin_path = source_out / "nested_threshold_selection.csv"
    cache_dir = source_out / "feature_cache"
    for path in (source_metadata_path, margin_path, cache_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required source artifact missing: {path}")

    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    selected = [str(value).zfill(4) for value in metadata["selected_sequences"]]
    conditions = list(metadata["conditions"])
    candidate_names = list(metadata["candidate_order"])
    signature = str(metadata["feature_cache"]["signature"])
    spatial_ridge = float(metadata.get("ridge", 10.0))
    if not candidate_names or candidate_names[0] != "raw":
        raise ValueError("Source candidate_order must start with raw")
    if len(selected) < 3:
        raise ValueError("At least three sequences are required for LOSO")

    margins = pd.read_csv(margin_path, dtype={"outer_heldout_sequence": str})
    margins["outer_heldout_sequence"] = margins["outer_heldout_sequence"].str.zfill(4)
    margin_map = {
        (str(row.condition), str(row.outer_heldout_sequence)): float(row.selected_margin)
        for row in margins.itertuples(index=False)
    }

    print("Loading compatible cached predictions...", flush=True)
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    sequence_info: dict[str, dict[str, Any]] = {}
    for seq in selected:
        for condition in conditions:
            path = cache_dir / f"{seq}_{condition}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing cache: {path}")
            cache[(seq, condition)] = _load_cache_item(path, candidate_names, signature)
        clean_item = cache[(seq, conditions[0])]
        sequence_info[seq] = _load_sequence_info(
            dataset_root, seq, clean_item["gt_xy"]
        )
        for condition in conditions:
            cached_gt = cache[(seq, condition)]["gt_xy"]
            if cached_gt.shape != sequence_info[seq]["gt_xy"].shape or not np.allclose(
                cached_gt, sequence_info[seq]["gt_xy"], equal_nan=True
            ):
                raise ValueError(f"Ground-truth/cache mismatch for {seq}/{condition}")

    all_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for condition_idx, condition in enumerate(conditions):
        print(f"\n[{condition}] constructing temporal features", flush=True)
        augmented_features: dict[str, np.ndarray] = {}
        temporal_errors: dict[str, np.ndarray] = {}
        gt_masks: dict[str, np.ndarray] = {}

        for seq in selected:
            item = cache[(seq, condition)]
            augmented, errors, gt_triples = _build_temporal_targets_and_features(
                item, sequence_info[seq], candidate_names, args.fps
            )
            augmented_features[seq] = augmented
            temporal_errors[seq] = errors
            gt_masks[seq] = gt_triples

        for outer_idx, heldout in enumerate(selected):
            training_sequences = [seq for seq in selected if seq != heldout]
            item = cache[(heldout, condition)]
            try:
                margin = margin_map[(condition, heldout)]
            except KeyError as exc:
                raise ValueError(f"Missing nested spatial margin for {condition}/{heldout}") from exc

            # Independent spatial model: the held-out sequence is excluded.
            spatial_model = _fit_selector_for_sequences(
                cache,
                condition,
                training_sequences,
                spatial_ridge,
                f"temporal veto outer spatial model; heldout={heldout}; condition={condition}",
            )
            predicted_spatial = predict_candidate_errors(spatial_model, item["features"])
            spatial_choices = select_candidates(predicted_spatial, margin)

            # Independent temporal model and target scaling: both use outer-train only.
            temporal_model, temporal_scale, _train_candidate_mean = _fit_temporal_model(
                training_sequences,
                augmented_features,
                temporal_errors,
                gt_masks,
                args.ridge,
            )
            predicted_temporal = predict_candidate_errors(
                temporal_model, augmented_features[heldout]
            )
            veto_choices, veto_mask = _make_veto_choices(
                spatial_choices,
                predicted_temporal,
                threshold=args.temporal_veto_threshold,
            )

            baseline_metrics, baseline_pose = _evaluate_pose_and_choices(
                item, spatial_choices, candidate_names, sequence_info[heldout], args.fps
            )
            veto_metrics, veto_pose = _evaluate_pose_and_choices(
                item, veto_choices, candidate_names, sequence_info[heldout], args.fps
            )

            # Fixed methods are included as context, not as tuned competitors.
            method_outputs: dict[str, tuple[dict[str, float | int], np.ndarray]] = {
                "spatial_only": (baseline_metrics, baseline_pose),
                "spatial_temporal_veto": (veto_metrics, veto_pose),
            }
            for candidate_idx, name in enumerate(candidate_names):
                fixed_choices = np.full(item["scoreable"].shape, candidate_idx, dtype=np.int64)
                fixed_metrics, fixed_pose = _evaluate_pose_and_choices(
                    item, fixed_choices, candidate_names, sequence_info[heldout], args.fps
                )
                method_outputs[f"fixed_{name}"] = (fixed_metrics, fixed_pose)

            total_scoreable = int(np.asarray(item["scoreable"], dtype=bool).sum())
            selected_repairs = (spatial_choices > 0) & np.asarray(item["scoreable"], dtype=bool)
            vetoed_repairs = veto_mask & np.asarray(item["scoreable"], dtype=bool)
            repair_count = int(selected_repairs.sum())
            veto_count = int(vetoed_repairs.sum())
            veto_rate = veto_count / repair_count if repair_count else 0.0

            for method, (metrics, _) in method_outputs.items():
                all_rows.append({
                    "sequence": heldout,
                    "action": item["action"],
                    "condition": condition,
                    "method": method,
                    "selected_spatial_margin": margin,
                    "temporal_veto_threshold": args.temporal_veto_threshold
                    if method == "spatial_temporal_veto" else float("nan"),
                    "temporal_target_scale_train_q95": temporal_scale
                    if method in {"spatial_only", "spatial_temporal_veto"} else float("nan"),
                    "n_scoreable_points": total_scoreable,
                    "n_spatially_selected_repairs": repair_count
                    if method in {"spatial_only", "spatial_temporal_veto"} else float("nan"),
                    "n_temporal_vetoed_repairs": veto_count
                    if method == "spatial_temporal_veto" else float("nan"),
                    "temporal_veto_rate_of_selected_repairs": veto_rate
                    if method == "spatial_temporal_veto" else float("nan"),
                    **metrics,
                })

            # Paired spatial metrics use the same scoreable GT points by construction.
            for comparison_method in (
                "spatial_temporal_veto",
                "fixed_savgol" if "savgol" in candidate_names else "spatial_only",
                "fixed_confidence_aware" if "confidence_aware" in candidate_names else "spatial_only",
            ):
                if comparison_method == "spatial_only":
                    continue
                other_metrics, other_pose = method_outputs[comparison_method]
                for metric in SPATIAL_METRICS:
                    a = float(other_metrics[metric])
                    b = float(baseline_metrics[metric])
                    paired_rows.append({
                        "sequence": heldout,
                        "action": item["action"],
                        "condition": condition,
                        "comparison": f"{comparison_method}_minus_spatial_only",
                        "metric": metric,
                        "delta": a - b,
                    })

                common = _temporal_common_support(
                    sequence_info[heldout]["valid_gt"], baseline_pose, other_pose
                )
                other_temporal, _, _ = temporal_metric_values(
                    other_pose,
                    sequence_info[heldout]["gt_xy"],
                    sequence_info[heldout]["valid_gt"],
                    sequence_info[heldout]["diagonal"],
                    args.fps,
                    support=common,
                )
                base_temporal, _, _ = temporal_metric_values(
                    baseline_pose,
                    sequence_info[heldout]["gt_xy"],
                    sequence_info[heldout]["valid_gt"],
                    sequence_info[heldout]["diagonal"],
                    args.fps,
                    support=common,
                )
                for metric in TEMPORAL_METRICS:
                    paired_rows.append({
                        "sequence": heldout,
                        "action": item["action"],
                        "condition": condition,
                        "comparison": f"{comparison_method}_minus_spatial_only",
                        "metric": metric,
                        "delta": other_temporal[metric] - base_temporal[metric],
                        "n_common_temporal_triples": int(common.sum()),
                    })

            if (outer_idx + 1) % 5 == 0 or outer_idx == len(selected) - 1:
                print(
                    f"  outer {outer_idx + 1}/{len(selected)} ({heldout}): "
                    f"vetoed {veto_count}/{repair_count} selected repairs; "
                    f"accel Δ={veto_metrics['acceleration_error_norm_s2'] - baseline_metrics['acceleration_error_norm_s2']:+.5f}",
                    flush=True,
                )

        del augmented_features, temporal_errors, gt_masks

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    per_sequence = pd.DataFrame(all_rows)
    paired = pd.DataFrame(paired_rows)
    per_sequence_path = out / "spatial_temporal_veto_by_sequence.csv"
    paired_path = out / "spatial_temporal_veto_paired_by_sequence.csv"
    per_sequence.to_csv(per_sequence_path, index=False)
    paired.to_csv(paired_path, index=False)

    summary = per_sequence.groupby(["condition", "method"], as_index=False).agg(
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
        veto_rate_mean=("temporal_veto_rate_of_selected_repairs", "mean"),
    )
    summary_path = out / "spatial_temporal_veto_summary.csv"
    summary.to_csv(summary_path, index=False)

    paired_summary_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    for (condition, comparison, metric), group in paired.groupby(
        ["condition", "comparison", "metric"]
    ):
        values = group["delta"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        low, high = _bootstrap_ci(values, rng, args.bootstrap_samples)
        paired_summary_rows.append({
            "condition": condition,
            "comparison": comparison,
            "metric": metric,
            "n_paired_sequences": len(values),
            "delta_mean": float(values.mean()) if len(values) else float("nan"),
            "delta_ci95_low": low,
            "delta_ci95_high": high,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "sequence",
        })
    paired_summary = pd.DataFrame(paired_summary_rows)
    paired_summary_path = out / "spatial_temporal_veto_paired_summary.csv"
    paired_summary.to_csv(paired_summary_path, index=False)

    protocol = {
        "experiment": "spatial_selector_with_temporal_risk_veto",
        "dataset": "Penn Action",
        "official_split": "selected official training sequences only; load_sequence enforces train membership",
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_order": candidate_names,
        "outer_protocol": "leave-one-sequence-out; each held-out sequence is excluded from both spatial and temporal model fitting and temporal target scaling",
        "spatial_selector": "existing cross-fitted spatial Ridge selector with its previously selected nested spatial margin for the outer fold",
        "temporal_model": "multi-output Ridge trained on centered candidate-specific acceleration error with outer-training-only q95 target scaling",
        "veto_rule": "if the spatially selected repair has predicted temporal risk greater than the predicted raw temporal risk by threshold tau, revert that frame/joint to raw; tau is fixed at the supplied value and not tuned on outer held-out data",
        "default_veto_threshold": args.temporal_veto_threshold,
        "endpoint_rule": "do not veto first/last sequence frames because the centered three-frame target is undefined there",
        "evaluation": "recompute spatial and temporal metrics on the composed framewise hybrid pose; temporal pairwise comparisons use common finite GT-visible triples",
        "bootstrap": {"samples": args.bootstrap_samples, "unit": "sequence", "interval": "two-sided percentile 95% CI"},
        "official_test_split_read": False,
        "limitations": [
            "This is a fixed-zero-threshold development experiment, not a tuned final method.",
            "Reverting a repair to raw can reduce prediction coverage when raw is missing.",
            "Candidate-level temporal risk is only a veto signal; actual hybrid trajectory metrics are recomputed after composition.",
            "Temporal metrics measure agreement with Penn Action annotations, not biomechanical ground truth.",
            "The feature protocol is offline/non-causal because it uses centered three-frame candidate descriptors.",
        ],
        "source_feature_cache_signature": signature,
        "source_selective_repair_output": str(source_out),
        "runtime_seconds": round(time.time() - started, 2),
    }
    metadata_path = out / "spatial_temporal_veto_metadata.json"
    metadata_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("\n=== Spatial + temporal-veto sequence-macro summary ===")
    print(summary.round(5).to_string(index=False))
    print("\n=== Paired comparison summary (method minus spatial-only) ===")
    print(paired_summary.round(5).to_string(index=False))
    print(f"\nSaved: {per_sequence_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {paired_path}")
    print(f"Saved: {paired_summary_path}")
    print(f"Saved: {metadata_path}")
    print("Cache-only LOSO development experiment; official test split was not read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
