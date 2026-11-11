#!/usr/bin/env python3
"""Nested cross-fitted evaluation of sequence-level candidate routing.

Consumes the existing FrameTrust feature cache and nested_threshold_selection.csv.
No MediaPipe inference is rerun, and only the selected official Penn Action
training sequences are loaded. The official test split is never opened.
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
    fit_error_predictor,
    predict_candidate_errors,
    select_candidates,
)
from frametrust.sequence_routing import route_candidate_sequence  # noqa: E402
from evaluate_selective_repair import _fit_selector_for_sequences  # noqa: E402
from evaluate_temporal_fidelity import (  # noqa: E402
    TEMPORAL_METRICS,
    temporal_ground_truth_mask,
    temporal_metric_values,
)
from penn_action_development import load_sequence  # noqa: E402


def _load_cache_item(path: Path, names: list[str], signature: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        observed = str(saved["cache_signature"].item())
        if observed != signature:
            raise ValueError(
                f"Cache signature mismatch in {path}; use the exact output folder "
                "from the source selective-repair run."
            )
        return {
            "action": str(saved["action"].item()),
            "features": saved["features"].copy(),
            "errors": saved["errors"].copy(),
            "scoreable": saved["scoreable"].copy(),
            "valid_gt": saved["valid_gt"].copy(),
            "torso_scale": saved["torso_scale"].copy(),
            "gt_xy": saved["gt_xy"].copy(),
            "raw": saved["raw"].copy(),
            "candidates": {
                name: saved[f"candidate__{name}"].copy() for name in names
            },
        }


def _action_balanced_folds(
    cache: dict[tuple[str, str], dict[str, Any]],
    condition: str,
    sequences: list[str],
    n_folds: int,
    seed: int,
) -> list[list[str]]:
    if len(sequences) < 3:
        raise ValueError("At least three outer-training sequences are required for inner CV")
    fold_count = min(int(n_folds), len(sequences))
    if fold_count < 2:
        raise ValueError("At least two inner folds are required")
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    rng = np.random.default_rng(seed)
    action_groups: dict[str, list[str]] = {}
    for seq in sequences:
        action_groups.setdefault(str(cache[(seq, condition)]["action"]), []).append(seq)
    for action_idx, action in enumerate(sorted(action_groups)):
        members = list(action_groups[action])
        rng.shuffle(members)
        start = action_idx % fold_count
        for index, seq in enumerate(members):
            folds[(start + index) % fold_count].append(seq)
    return [fold for fold in folds if fold]


def _sequence_spatial_metrics(
    item: dict[str, Any], choices: np.ndarray, names: list[str],
) -> dict[str, float]:
    target = np.asarray(item["scoreable"], dtype=bool)
    n_target = int(target.sum())
    pose = apply_candidate_choices(item["candidates"], choices, names)
    selected_errors = np.take_along_axis(
        np.asarray(item["errors"], dtype=float), choices[..., None], axis=-1
    )[..., 0]
    finite = np.isfinite(pose).all(axis=-1)
    raw_finite = np.isfinite(item["raw"]).all(axis=-1)
    raw_error = np.asarray(item["errors"], dtype=float)[..., 0]
    raw_correct = target & raw_finite & (raw_error <= 0.20)
    harmed = raw_correct & (~finite | (selected_errors > 0.20))
    return {
        "pck_torso_0.20": float(((selected_errors <= 0.20) & target).sum() / n_target)
        if n_target else float("nan"),
        "mean_error_norm": float(np.mean(selected_errors[target]))
        if n_target else float("nan"),
        "coverage": float((finite & target).sum() / n_target)
        if n_target else float("nan"),
        "false_correction_rate": float(harmed.sum() / raw_correct.sum())
        if raw_correct.any() else float("nan"),
        "repair_fraction": float((choices[target] != 0).mean())
        if n_target else float("nan"),
    }


def _evaluate_pose(
    pose: np.ndarray,
    gt: np.ndarray,
    valid_gt: np.ndarray,
    diagonal: float,
    fps: float,
) -> tuple[dict[str, float], int, int]:
    return temporal_metric_values(pose, gt, valid_gt, diagonal, fps)


def _evaluate_choices(
    item: dict[str, Any],
    choices: np.ndarray,
    names: list[str],
    sequence_info: dict[str, Any],
    fps: float,
) -> dict[str, float]:
    pose = apply_candidate_choices(item["candidates"], choices, names)
    spatial = _sequence_spatial_metrics(item, choices, names)
    temporal, supported, possible = _evaluate_pose(
        pose,
        sequence_info["gt_xy"],
        sequence_info["valid_gt"],
        sequence_info["diagonal"],
        fps,
    )
    return {
        **spatial,
        **temporal,
        "temporal_support_coverage": supported / possible if possible else float("nan"),
        "n_temporal_triples_supported": supported,
        "n_valid_gt_triples": possible,
    }


def _select_penalty(
    sweep: pd.DataFrame,
    pck_tolerance: float,
    min_coverage: float,
    max_false_correction: float,
    min_temporal_support: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    selected: dict[str, float] = {}
    decisions: list[dict[str, Any]] = []
    for mode in ("constant", "disagreement"):
        group = sweep[sweep["mode"] == mode].copy()
        feasible = group[
            group["pck_torso_0.20_mean"].notna()
            & (group["coverage_mean"] >= min_coverage)
            & (group["false_correction_rate_mean"] <= max_false_correction)
            & (group["temporal_support_coverage_mean"] >= min_temporal_support)
            & group["acceleration_error_norm_s2_mean"].notna()
        ].copy()
        fallback = "none"
        if feasible.empty:
            fallback = "false_correction_guardrail_relaxed"
            feasible = group[
                group["pck_torso_0.20_mean"].notna()
                & (group["coverage_mean"] >= min_coverage)
                & (group["temporal_support_coverage_mean"] >= min_temporal_support)
                & group["acceleration_error_norm_s2_mean"].notna()
            ].copy()
        if feasible.empty:
            fallback = "temporal_support_guardrail_relaxed"
            feasible = group[
                group["pck_torso_0.20_mean"].notna()
                & (group["coverage_mean"] >= min_coverage)
                & group["acceleration_error_norm_s2_mean"].notna()
            ].copy()
        if feasible.empty:
            fallback = "coverage_guardrail_relaxed"
            feasible = group[
                group["pck_torso_0.20_mean"].notna()
                & group["acceleration_error_norm_s2_mean"].notna()
            ].copy()
        if feasible.empty:
            raise RuntimeError(f"No scoreable inner-fold settings for {mode} routing")

        best_pck = float(feasible["pck_torso_0.20_mean"].max())
        near_best = feasible[
            feasible["pck_torso_0.20_mean"] >= best_pck - pck_tolerance
        ].copy()
        near_best = near_best.sort_values(
            ["acceleration_error_norm_s2_mean", "velocity_error_norm_s_mean", "switch_penalty"],
            ascending=[True, True, True],
        )
        chosen = near_best.iloc[0]
        penalty = float(chosen["switch_penalty"])
        selected[mode] = penalty
        decisions.append({
            "mode": mode,
            "selected_switch_penalty": penalty,
            "best_feasible_inner_pck": best_pck,
            "selected_inner_pck": float(chosen["pck_torso_0.20_mean"]),
            "selected_pck_gap": best_pck - float(chosen["pck_torso_0.20_mean"]),
            "selected_inner_coverage": float(chosen["coverage_mean"]),
            "selected_inner_false_correction_rate": float(chosen["false_correction_rate_mean"]),
            "selected_inner_temporal_support": float(chosen["temporal_support_coverage_mean"]),
            "selected_inner_velocity_error": float(chosen["velocity_error_norm_s_mean"]),
            "selected_inner_acceleration_error": float(chosen["acceleration_error_norm_s2_mean"]),
            "pck_tolerance": pck_tolerance,
            "minimum_coverage": min_coverage,
            "maximum_false_correction_rate": max_false_correction,
            "minimum_temporal_support": min_temporal_support,
            "guardrail_fallback": fallback,
            "selection_rule": "minimize acceleration error among settings within PCK tolerance and guardrails; velocity error and smaller penalty break ties",
        })
    return selected, pd.DataFrame(decisions)


def _paired_bootstrap(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--inner-folds", type=int, default=None)
    parser.add_argument("--switch-penalties", default="0,0.0025,0.005,0.01,0.02,0.04")
    parser.add_argument("--pck-tolerance", type=float, default=None)
    parser.add_argument("--min-coverage", type=float, default=None)
    parser.add_argument("--max-false-correction", type=float, default=None)
    parser.add_argument("--min-temporal-support", type=float, default=0.90)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    if args.fps <= 0 or args.bootstrap_samples < 100:
        parser.error("--fps must be positive and --bootstrap-samples must be >=100")
    if not 0 < args.min_temporal_support <= 1:
        parser.error("--min-temporal-support must be in (0, 1]")
    penalties = sorted(set(float(x.strip()) for x in args.switch_penalties.split(",") if x.strip()))
    if not penalties or any(x < 0 for x in penalties):
        parser.error("--switch-penalties must be a non-empty comma-separated list of non-negative values")

    out = args.out.resolve()
    metadata_path = out / "run_metadata.json"
    margin_path = out / "nested_threshold_selection.csv"
    cache_dir = out / "feature_cache"
    for path in (metadata_path, margin_path, cache_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required source artifact does not exist: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = [str(x).zfill(4) for x in metadata["selected_sequences"]]
    conditions = list(metadata["conditions"])
    names = list(metadata.get("candidate_order", DEFAULT_CANDIDATES))
    if not names or names[0] != "raw":
        raise ValueError("Source metadata must define raw as candidate zero")
    signature = str(metadata["feature_cache"]["signature"])
    ridge = float(metadata.get("ridge", 10.0))
    nested_meta = metadata.get("nested_threshold_selection", {})
    n_inner_folds = int(args.inner_folds or nested_meta.get("inner_folds", 3))
    pck_tolerance = float(args.pck_tolerance if args.pck_tolerance is not None else nested_meta.get("pck_tolerance", 0.005))
    min_coverage = float(args.min_coverage if args.min_coverage is not None else nested_meta.get("minimum_inner_coverage", 0.90))
    max_false_correction = float(args.max_false_correction if args.max_false_correction is not None else nested_meta.get("maximum_inner_false_correction_rate", 0.01))
    margins = pd.read_csv(margin_path, dtype={"outer_heldout_sequence": str})
    margins["outer_heldout_sequence"] = margins["outer_heldout_sequence"].str.zfill(4)
    margin_map = {
        (str(r.condition), str(r.outer_heldout_sequence)): float(r.selected_margin)
        for r in margins.itertuples(index=False)
    }

    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for seq in selected:
        for condition in conditions:
            path = cache_dir / f"{seq}_{condition}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing cache file: {path}")
            cache[(seq, condition)] = _load_cache_item(path, names, signature)

    # Metadata is loaded once. load_sequence enforces the official training flag.
    sequence_info: dict[str, dict[str, Any]] = {}
    dataset_root = args.dataset_root.resolve()
    for seq in selected:
        data = load_sequence(dataset_root, seq)
        item_shape = cache[(seq, conditions[0])]["gt_xy"].shape
        if len(data["frames"]) != item_shape[0]:
            raise ValueError(f"Frame/cache length mismatch for sequence {seq}")
        gt = np.asarray(data["gt_xy"], dtype=float)
        if gt.shape != item_shape or not np.allclose(
            gt, cache[(seq, conditions[0])]["gt_xy"], equal_nan=True
        ):
            raise ValueError(f"Ground-truth/cache mismatch for sequence {seq}")
        frame = cv2.imread(str(data["frames"][0]))
        if frame is None:
            raise ValueError(f"Could not read first frame of sequence {seq}")
        height, width = frame.shape[:2]
        sequence_info[seq] = {
            "gt_xy": gt,
            "valid_gt": temporal_ground_truth_mask(gt, data["gt_vis"], width, height),
            "diagonal": float(np.hypot(width, height)),
        }

    inner_sweep_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    candidate_methods = names
    route_methods = ["nested_selective", "nested_constant_switch_cost", "nested_disagreement_cost"]

    for condition_idx, condition in enumerate(conditions):
        for outer_idx, heldout in enumerate(selected):
            outer_training = [seq for seq in selected if seq != heldout]
            try:
                margin = margin_map[(condition, heldout)]
            except KeyError as exc:
                raise ValueError(f"Missing nested margin for {condition}/{heldout}") from exc
            folds = _action_balanced_folds(
                cache, condition, outer_training, n_inner_folds,
                args.seed + 1009 * outer_idx + 9176 * condition_idx,
            )
            fold_predictions: list[tuple[str, dict[str, np.ndarray], np.ndarray]] = []
            for fold_idx, validation in enumerate(folds):
                validation_set = set(validation)
                training = [seq for seq in outer_training if seq not in validation_set]
                model = _fit_selector_for_sequences(
                    cache, condition, training, ridge,
                    f"sequence-routing inner fold {fold_idx + 1}, outer={heldout}, condition={condition}",
                )
                for val_seq in validation:
                    val_item = cache[(val_seq, condition)]
                    predicted = predict_candidate_errors(model, val_item["features"])
                    fold_predictions.append((val_seq, val_item, predicted))

            for mode in ("constant", "disagreement"):
                for penalty in penalties:
                    rows: list[dict[str, Any]] = []
                    for val_seq, val_item, predicted in fold_predictions:
                        choices = route_candidate_sequence(
                            predicted, val_item["candidates"], names,
                            min_predicted_gain=margin,
                            switch_penalty=penalty,
                            mode=mode,
                            torso_scale=val_item["torso_scale"],
                        )
                        metrics = _evaluate_choices(
                            val_item, choices, names, sequence_info[val_seq], args.fps
                        )
                        rows.append({"sequence": val_seq, **metrics})
                    frame = pd.DataFrame(rows)
                    aggregate = frame.mean(numeric_only=True)
                    inner_sweep_rows.append({
                        "outer_heldout_sequence": heldout,
                        "condition": condition,
                        "mode": mode,
                        "switch_penalty": penalty,
                        "n_inner_sequences": frame["sequence"].nunique(),
                        "pck_torso_0.20_mean": float(aggregate.get("pck_torso_0.20", np.nan)),
                        "mean_error_norm_mean": float(aggregate.get("mean_error_norm", np.nan)),
                        "coverage_mean": float(aggregate.get("coverage", np.nan)),
                        "false_correction_rate_mean": float(aggregate.get("false_correction_rate", np.nan)),
                        "repair_fraction_mean": float(aggregate.get("repair_fraction", np.nan)),
                        "temporal_support_coverage_mean": float(aggregate.get("temporal_support_coverage", np.nan)),
                        "velocity_error_norm_s_mean": float(aggregate.get("velocity_error_norm_s", np.nan)),
                        "acceleration_error_norm_s2_mean": float(aggregate.get("acceleration_error_norm_s2", np.nan)),
                    })

            outer_sweep = pd.DataFrame([
                row for row in inner_sweep_rows
                if row["outer_heldout_sequence"] == heldout and row["condition"] == condition
            ])
            chosen_penalties, decisions = _select_penalty(
                outer_sweep,
                pck_tolerance,
                min_coverage,
                max_false_correction,
                args.min_temporal_support,
            )
            for decision in decisions.to_dict(orient="records"):
                selection_rows.append({
                    "outer_heldout_sequence": heldout,
                    "condition": condition,
                    **decision,
                })

            # Outer model: all other selected training sequences; heldout labels are
            # not used for predictor fitting or switch-penalty selection.
            model = _fit_selector_for_sequences(
                cache, condition, outer_training, ridge,
                f"sequence-routing outer evaluation, heldout={heldout}, condition={condition}",
            )
            item = cache[(heldout, condition)]
            predicted = predict_candidate_errors(model, item["features"])
            choices_by_method: dict[str, np.ndarray] = {
                "nested_selective": select_candidates(predicted, margin)
            }
            choices_by_method["nested_constant_switch_cost"] = route_candidate_sequence(
                predicted, item["candidates"], names,
                min_predicted_gain=margin,
                switch_penalty=chosen_penalties["constant"],
                mode="constant",
                torso_scale=item["torso_scale"],
            )
            choices_by_method["nested_disagreement_cost"] = route_candidate_sequence(
                predicted, item["candidates"], names,
                min_predicted_gain=margin,
                switch_penalty=chosen_penalties["disagreement"],
                mode="disagreement",
                torso_scale=item["torso_scale"],
            )

            poses: dict[str, np.ndarray] = {name: item["candidates"][name] for name in candidate_methods}
            for method, choices in choices_by_method.items():
                poses[method] = apply_candidate_choices(item["candidates"], choices, names)
            method_choices: dict[str, np.ndarray | None] = {name: None for name in candidate_methods}
            method_choices.update(choices_by_method)
            for method, pose in poses.items():
                if method in method_choices and method_choices[method] is not None:
                    spatial = _sequence_spatial_metrics(item, method_choices[method], names)
                else:
                    k = names.index(method)
                    onehot = np.full(item["scoreable"].shape, k, dtype=int)
                    spatial = _sequence_spatial_metrics(item, onehot, names)
                    spatial["repair_fraction"] = 0.0 if method == "raw" else float("nan")
                temporal, n_supported, n_possible = _evaluate_pose(
                    pose, sequence_info[heldout]["gt_xy"],
                    sequence_info[heldout]["valid_gt"],
                    sequence_info[heldout]["diagonal"], args.fps,
                )
                final_rows.append({
                    "sequence": heldout,
                    "action": item["action"],
                    "condition": condition,
                    "method": method,
                    "selected_margin": margin if method in route_methods else float("nan"),
                    "switch_penalty": (
                        chosen_penalties["constant"] if method == "nested_constant_switch_cost"
                        else chosen_penalties["disagreement"] if method == "nested_disagreement_cost"
                        else 0.0 if method == "nested_selective" else float("nan")
                    ),
                    **spatial,
                    **temporal,
                    "temporal_support_coverage": n_supported / n_possible if n_possible else float("nan"),
                    "n_temporal_triples_supported": n_supported,
                    "n_valid_gt_triples": n_possible,
                })

            # Sequence-paired comparisons; temporal metrics use identical finite
            # GT-visible triples for the routed method and nested-selective baseline.
            for method, pose in poses.items():
                if method == "nested_selective":
                    continue
                base_pose = poses["nested_selective"]
                valid_gt = sequence_info[heldout]["valid_gt"]
                gt_triples = valid_gt[:-2] & valid_gt[1:-1] & valid_gt[2:]
                common = gt_triples.copy()
                for candidate_pose in (base_pose, pose):
                    if len(candidate_pose) < 3:
                        common &= False
                    else:
                        common &= (
                            np.isfinite(candidate_pose[:-2]).all(axis=-1)
                            & np.isfinite(candidate_pose[1:-1]).all(axis=-1)
                            & np.isfinite(candidate_pose[2:]).all(axis=-1)
                        )
                method_temporal, _, _ = temporal_metric_values(
                    pose, sequence_info[heldout]["gt_xy"], valid_gt,
                    sequence_info[heldout]["diagonal"], args.fps, support=common,
                )
                base_temporal, _, _ = temporal_metric_values(
                    base_pose, sequence_info[heldout]["gt_xy"], valid_gt,
                    sequence_info[heldout]["diagonal"], args.fps, support=common,
                )
                method_spatial = next(r for r in final_rows[::-1] if r["sequence"] == heldout and r["condition"] == condition and r["method"] == method)
                base_spatial = next(r for r in final_rows[::-1] if r["sequence"] == heldout and r["condition"] == condition and r["method"] == "nested_selective")
                metric_deltas = {
                    "pck_torso_0.20": method_spatial["pck_torso_0.20"] - base_spatial["pck_torso_0.20"],
                    "mean_error_norm": method_spatial["mean_error_norm"] - base_spatial["mean_error_norm"],
                    "coverage": method_spatial["coverage"] - base_spatial["coverage"],
                    "false_correction_rate": method_spatial["false_correction_rate"] - base_spatial["false_correction_rate"],
                }
                for metric in (*TEMPORAL_METRICS, *metric_deltas.keys()):
                    if metric in metric_deltas:
                        delta = metric_deltas[metric]
                    else:
                        delta = method_temporal[metric] - base_temporal[metric]
                    paired_rows.append({
                        "sequence": heldout,
                        "action": item["action"],
                        "condition": condition,
                        "comparison": f"{method}_minus_nested_selective",
                        "method": method,
                        "metric": metric,
                        "delta_mean": delta,
                        "n_common_temporal_triples": int(common.sum()),
                    })

            print(
                f"[{condition}] outer={heldout}: "
                f"constant λ={chosen_penalties['constant']:g}, "
                f"disagreement λ={chosen_penalties['disagreement']:g}",
                flush=True,
            )

    inner_df = pd.DataFrame(inner_sweep_rows)
    selection_df = pd.DataFrame(selection_rows)
    final_df = pd.DataFrame(final_rows)
    paired_df = pd.DataFrame(paired_rows)
    inner_df.to_csv(out / "sequence_routing_inner_sweep.csv", index=False)
    selection_df.to_csv(out / "sequence_routing_penalty_selection.csv", index=False)
    final_df.to_csv(out / "sequence_routing_metrics.csv", index=False)
    paired_df.to_csv(out / "sequence_routing_paired_per_sequence.csv", index=False)

    summary = final_df.groupby(["condition", "method"], as_index=False).agg(
        n_sequences=("sequence", "nunique"),
        pck_mean=("pck_torso_0.20", "mean"),
        mean_error_norm_mean=("mean_error_norm", "mean"),
        coverage_mean=("coverage", "mean"),
        false_correction_rate_mean=("false_correction_rate", "mean"),
        repair_fraction_mean=("repair_fraction", "mean"),
        temporal_support_coverage_mean=("temporal_support_coverage", "mean"),
        velocity_error_norm_s_mean=("velocity_error_norm_s", "mean"),
        acceleration_error_norm_s2_mean=("acceleration_error_norm_s2", "mean"),
    )
    summary.to_csv(out / "sequence_routing_condition_summary.csv", index=False)

    paired_summary_rows = []
    rng = np.random.default_rng(args.seed)
    for (condition, comparison, method, metric), group in paired_df.groupby(
        ["condition", "comparison", "method", "metric"]
    ):
        values = group["delta_mean"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        lo, hi = _paired_bootstrap(values, rng, args.bootstrap_samples)
        paired_summary_rows.append({
            "condition": condition,
            "comparison": comparison,
            "method": method,
            "metric": metric,
            "n_paired_sequences": len(values),
            "delta_mean": float(values.mean()) if len(values) else float("nan"),
            "delta_ci95_low": lo,
            "delta_ci95_high": hi,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "sequence",
        })
    pd.DataFrame(paired_summary_rows).to_csv(
        out / "sequence_routing_paired_comparison.csv", index=False
    )

    protocol = {
        "dataset": "Penn Action",
        "official_split": "selected official training sequences only; load_sequence enforces train split",
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_order": names,
        "outer_protocol": "leave-one-sequence-out; outer held-out labels are excluded from predictor fitting and penalty tuning",
        "inner_protocol": "action-balanced sequence folds within each outer-training set; ridge models fit without each inner validation fold",
        "unary_cost": "inference-time ridge-predicted torso-normalized candidate error; min_predicted_gain added to non-raw candidates",
        "transition_modes": {
            "constant": "fixed switch_penalty for every candidate identity change",
            "disagreement": "switch_penalty multiplied by cross-candidate coordinate disagreement normalized by torso scale; if the pair is unavailable at both boundary endpoints, no disagreement penalty is applied",
        },
        "penalty_selection": "among inner settings meeting spatial coverage, false-correction, and temporal support guardrails, require PCK within tolerance of the best feasible setting and minimize acceleration error; velocity error and smaller penalty break ties",
        "switch_penalties": penalties,
        "pck_tolerance": pck_tolerance,
        "minimum_coverage": min_coverage,
        "maximum_false_correction_rate": max_false_correction,
        "minimum_temporal_support": args.min_temporal_support,
        "temporal_metrics": "same image-diagonal-normalized velocity and acceleration errors as evaluate_temporal_fidelity.py; sequence-level paired comparisons use common finite GT-visible triples",
        "ground_truth_use": "only for fitting candidate-error predictors on training sequences, inner-fold model/penalty selection, and evaluation; never used by the routing function or outer held-out decisions",
        "official_test_split_read": False,
        "limitations": [
            "This is a development-only prototype, not evidence of novelty or final superiority.",
            "The transition penalty is a simple regularizer; the disagreement mode is a heuristic based on candidate outputs.",
            "The inner CV is used to select the penalty and is not an independent estimate; only outer held-out sequence results estimate generalization.",
            "The objective is joint-wise and does not enforce skeletal/body-level consistency.",
            "This router is offline/non-causal because the existing features use neighboring frames and the DP optimizes a complete sequence.",
            "Temporal errors measure agreement with Penn Action annotations, not biomechanical ground truth.",
        ],
    }
    (out / "sequence_routing_metadata.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print("\n=== Sequence-routing condition summary ===")
    print(summary.round(5).to_string(index=False))
    print(f"\nSaved sequence-routing outputs under {out}")
    print("Development-only. Do not read or report official test split results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
