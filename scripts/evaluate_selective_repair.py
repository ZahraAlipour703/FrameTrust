#!/usr/bin/env python3
"""Cross-fitted, sequence-held-out evaluation of selective pose repair.

All sequences must be from the official Penn Action training split. Each held-out
sequence is evaluated by a selector fitted on other selected training sequences.
The official test split is never read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frametrust.selective import (  # noqa: E402
    DEFAULT_CANDIDATES,
    apply_candidate_choices,
    build_selective_features,
    candidate_error_targets,
    fit_error_predictor,
    predict_candidate_errors,
    select_candidates,
)
from penn_action_development import (  # noqa: E402
    CONDITIONS,
    MEDIAPIPE_JOINTS,
    PENN_JOINTS,
    confidence_aware_repair,
    extract_predictions,
    ground_truth_masks_and_scales,
    interpolate_missing,
    interpolate_missing_only,
    load_sequence,
    one_euro_filter,
    savgol_filter_pose,
    select_development_sequences,
)


def _save_feature_cache(
    path: Path,
    item: dict[str, Any],
    names: list[str],
    signature: str,
) -> None:
    """Persist inference-time features, targets, and candidate arrays for later re-analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "action": np.asarray(str(item["action"])),
        "cache_signature": np.asarray(signature),
        "features": item["features"],
        "errors": item["errors"],
        "scoreable": item["scoreable"],
        "valid_gt": item["valid_gt"],
        "torso_scale": item["torso_scale"],
        "gt_xy": item["gt_xy"],
        "raw": item["raw"],
    }
    for name in names:
        payload[f"candidate__{name}"] = item["candidates"][name]
    np.savez_compressed(path, **payload)


def _load_feature_cache(
    path: Path,
    names: list[str],
    expected_signature: str,
) -> dict[str, Any]:
    """Load one cached sequence-condition item and reject incompatible settings."""
    with np.load(path, allow_pickle=False) as saved:
        observed_signature = str(saved["cache_signature"].item())
        if observed_signature != expected_signature:
            raise ValueError(
                f"Feature cache settings do not match this run: {path}. "
                "Remove the feature_cache folder or rerun without --reuse-feature-cache."
            )
        candidates = {
            name: saved[f"candidate__{name}"].copy() for name in names
        }
        return {
            "action": str(saved["action"].item()),
            "features": saved["features"].copy(),
            "errors": saved["errors"].copy(),
            "scoreable": saved["scoreable"].copy(),
            "valid_gt": saved["valid_gt"].copy(),
            "torso_scale": saved["torso_scale"].copy(),
            "gt_xy": saved["gt_xy"].copy(),
            "raw": saved["raw"].copy(),
            "candidates": candidates,
        }


def _fit_selector_for_sequences(
    cache: dict[tuple[str, str], dict[str, Any]],
    condition: str,
    training_sequences: list[str],
    ridge: float,
    heldout_label: str,
) -> dict[str, np.ndarray | float | int]:
    """Fit on the supplied sequence IDs only; labels never cross the split."""
    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    for seq in training_sequences:
        item = cache[(seq, condition)]
        mask = item["scoreable"]
        if mask.any():
            train_x.append(item["features"][mask])
            train_y.append(item["errors"][mask])
    n_rows = sum(len(x) for x in train_x)
    if not train_x or n_rows < 100:
        raise RuntimeError(
            f"Insufficient scoreable training points ({n_rows}) for {heldout_label}"
        )
    return fit_error_predictor(
        np.concatenate(train_x, axis=0),
        np.concatenate(train_y, axis=0),
        ridge=ridge,
    )


def _score_margin(
    item: dict[str, Any],
    predicted: np.ndarray,
    margin: float,
    names: list[str],
) -> dict[str, float | int]:
    """Score one sequence at a given margin using sequence-level metrics."""
    target = item["scoreable"]
    n_target = int(target.sum())
    choices = select_candidates(predicted, margin)
    pose = apply_candidate_choices(item["candidates"], choices, names)
    selected_error = np.take_along_axis(
        item["errors"], choices[..., None], axis=-1
    )[..., 0]
    finite = np.isfinite(pose).all(axis=-1)
    raw_finite = np.isfinite(item["raw"]).all(axis=-1)
    raw_error = item["errors"][..., 0]
    raw_correct = target & raw_finite & (raw_error <= 0.20)
    harmed = raw_correct & (~finite | (selected_error > 0.20))
    return {
        "n_scored_points": n_target,
        "pck": float(((selected_error <= 0.20) & target).sum() / n_target)
        if n_target else float("nan"),
        "coverage": float((finite & target).sum() / n_target)
        if n_target else float("nan"),
        "false_correction_rate": float(harmed.sum() / raw_correct.sum())
        if raw_correct.any() else float("nan"),
        "repair_fraction": float((choices[target] != 0).mean())
        if n_target else float("nan"),
        "mean_error_norm": float(np.mean(selected_error[target]))
        if n_target else float("nan"),
    }


def nested_threshold_selection(
    cache: dict[tuple[str, str], dict[str, Any]],
    selected: list[str],
    conditions: list[str],
    names: list[str],
    thresholds: list[float],
    ridge: float,
    seed: int,
    n_inner_folds: int,
    pck_tolerance: float,
    min_coverage: float,
    max_false_correction: float,
) -> tuple[dict[tuple[str, str], float], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a conservative margin inside each outer training fold.

    Every outer sequence is excluded from both the inner-fold training data and
    the inner threshold-selection metrics. Inner splits are action-balanced by
    distributing sequences from each action across folds.
    """
    selected_margins: dict[tuple[str, str], float] = {}
    selection_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    margin_values = sorted(set(float(x) for x in thresholds))

    for condition_idx, condition in enumerate(conditions):
        for outer_idx, outer_sequence in enumerate(selected):
            outer_training = [seq for seq in selected if seq != outer_sequence]
            fold_count = min(n_inner_folds, len(outer_training))
            folds: list[list[str]] = [[] for _ in range(fold_count)]
            rng = np.random.default_rng(seed + 1009 * outer_idx + 9176 * condition_idx)

            action_groups: dict[str, list[str]] = {}
            for seq in outer_training:
                action = str(cache[(seq, condition)]["action"])
                action_groups.setdefault(action, []).append(seq)
            for action_idx, action in enumerate(sorted(action_groups)):
                seqs = list(action_groups[action])
                rng.shuffle(seqs)
                start_fold = action_idx % fold_count
                for i, seq in enumerate(seqs):
                    folds[(start_fold + i) % fold_count].append(seq)

            inner_rows: list[dict[str, Any]] = []
            for fold_idx, validation_sequences in enumerate(folds):
                if not validation_sequences:
                    continue
                validation_set = set(validation_sequences)
                training_sequences = [
                    seq for seq in outer_training if seq not in validation_set
                ]
                model = _fit_selector_for_sequences(
                    cache, condition, training_sequences, ridge,
                    f"nested inner fold {fold_idx + 1}/{fold_count}, outer={outer_sequence}, condition={condition}",
                )
                for validation_sequence in validation_sequences:
                    item = cache[(validation_sequence, condition)]
                    predicted = predict_candidate_errors(model, item["features"])
                    for margin in margin_values:
                        metrics = _score_margin(item, predicted, margin, names)
                        row = {
                            "outer_heldout_sequence": outer_sequence,
                            "condition": condition,
                            "inner_validation_sequence": validation_sequence,
                            "inner_validation_action": item["action"],
                            "inner_fold": fold_idx,
                            "min_predicted_gain": margin,
                            **metrics,
                        }
                        inner_rows.append(row)
                        sweep_rows.append(row)

            inner = pd.DataFrame(inner_rows)
            if inner.empty:
                raise RuntimeError(
                    f"No inner validation scores for outer={outer_sequence}, condition={condition}"
                )
            summary = inner.groupby("min_predicted_gain", as_index=False).agg(
                n_inner_sequences=("inner_validation_sequence", "nunique"),
                n_scoreable_inner_sequences=("pck", "count"),
                n_inner_sequences_with_false_correction_metric=("false_correction_rate", "count"),
                pck_mean=("pck", "mean"),
                coverage_mean=("coverage", "mean"),
                false_correction_rate_mean=("false_correction_rate", "mean"),
                repair_fraction_mean=("repair_fraction", "mean"),
                mean_error_norm=("mean_error_norm", "mean"),
            )

            feasible = summary[
                summary["pck_mean"].notna()
                & (summary["coverage_mean"] >= min_coverage)
                & (summary["false_correction_rate_mean"] <= max_false_correction)
            ].copy()
            fallback = "none"
            if feasible.empty:
                fallback = "false_correction_guardrail_relaxed"
                feasible = summary[
                    summary["pck_mean"].notna()
                    & (summary["coverage_mean"] >= min_coverage)
                ].copy()
            if feasible.empty:
                fallback = "coverage_guardrail_relaxed"
                feasible = summary[summary["pck_mean"].notna()].copy()
            if feasible.empty:
                raise RuntimeError(
                    f"No PCK-scoreable thresholds for outer={outer_sequence}, condition={condition}"
                )

            best_inner_pck = float(feasible["pck_mean"].max())
            near_best = feasible[
                feasible["pck_mean"] >= best_inner_pck - pck_tolerance
            ].sort_values("min_predicted_gain")
            chosen = near_best.iloc[-1]
            margin = float(chosen["min_predicted_gain"])
            selected_margins[(condition, outer_sequence)] = margin
            selection_rows.append({
                "outer_heldout_sequence": outer_sequence,
                "condition": condition,
                "selected_margin": margin,
                "best_feasible_inner_pck": best_inner_pck,
                "selected_inner_pck": float(chosen["pck_mean"]),
                "selected_inner_pck_gap_from_best": best_inner_pck - float(chosen["pck_mean"]),
                "selected_inner_coverage": float(chosen["coverage_mean"]),
                "selected_inner_false_correction_rate": float(chosen["false_correction_rate_mean"]),
                "selected_inner_repair_fraction": float(chosen["repair_fraction_mean"]),
                "selected_inner_mean_error_norm": float(chosen["mean_error_norm"]),
                "n_inner_validation_sequences": int(chosen["n_inner_sequences"]),
                "n_scoreable_inner_sequences": int(chosen["n_scoreable_inner_sequences"]),
                "n_inner_sequences_with_false_correction_metric": int(chosen["n_inner_sequences_with_false_correction_metric"]),
                "pck_tolerance": pck_tolerance,
                "minimum_coverage": min_coverage,
                "maximum_false_correction_rate": max_false_correction,
                "guardrail_fallback": fallback,
                "inner_folds": fold_count,
            })
            print(
                f"  nested threshold: {condition}, outer={outer_sequence}, "
                f"margin={margin:g}, inner PCK={float(chosen['pck_mean']):.3f}, "
                f"fallback={fallback}",
                flush=True,
            )

    return selected_margins, selection_rows, sweep_rows


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = 5000,
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
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--manifest", type=Path, default=Path("data/external/penn_action_qc_manifest.csv"))
    parser.add_argument("--sequences", default="", help="Optional comma-separated QC-approved official-train sequence IDs.")
    parser.add_argument("--max-per-action", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--gap-every", type=int, default=5)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--one-euro-min-cutoff", type=float, default=1.0)
    parser.add_argument("--one-euro-beta", type=float, default=0.007)
    parser.add_argument("--one-euro-derivative-cutoff", type=float, default=1.0)
    parser.add_argument("--savgol-window", type=int, default=5)
    parser.add_argument("--savgol-polyorder", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--min-predicted-gain", type=float, default=0.01,
                        help="Fixed reference margin used for the descriptive OOF threshold sweep.")
    parser.add_argument("--inner-folds", type=int, default=3,
                        help="Number of action-balanced inner folds used to choose a margin inside each outer fold.")
    parser.add_argument("--nested-pck-tolerance", type=float, default=0.005,
                        help="Choose the largest margin within this PCK tolerance of the best feasible inner-fold PCK (0.005 = 0.5 percentage points).")
    parser.add_argument("--nested-min-coverage", type=float, default=0.90,
                        help="Minimum mean inner-fold prediction coverage for threshold selection.")
    parser.add_argument("--nested-max-false-correction", type=float, default=0.01,
                        help="Maximum mean inner-fold false-correction rate for threshold selection.")
    parser.add_argument("--reuse-feature-cache", action="store_true",
                        help="Reuse compatible per-sequence feature caches in the output folder; fails on a configuration mismatch.")
    parser.add_argument("--out", type=Path, default=Path("outputs/selective_repair_crossfit"))
    args = parser.parse_args()

    if args.max_per_action < 1 or args.gap_every < 2 or args.fps <= 0:
        parser.error("max-per-action must be >=1, gap-every >=2 and fps positive")
    if args.savgol_window < 3 or args.savgol_window % 2 == 0:
        parser.error("savgol-window must be odd and >=3")
    if not 1 <= args.savgol_polyorder < args.savgol_window:
        parser.error("savgol-polyorder must be >=1 and less than savgol-window")
    if args.ridge < 0 or args.min_predicted_gain < 0:
        parser.error("ridge and min-predicted-gain must be non-negative")
    if args.inner_folds < 2:
        parser.error("--inner-folds must be at least 2")
    if args.nested_pck_tolerance < 0:
        parser.error("--nested-pck-tolerance must be non-negative")
    if not 0 < args.nested_min_coverage <= 1:
        parser.error("--nested-min-coverage must be in (0, 1]")
    if not 0 <= args.nested_max_false_correction <= 1:
        parser.error("--nested-max-false-correction must be in [0, 1]")

    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    unknown = set(conditions) - set(CONDITIONS)
    if not conditions or unknown:
        parser.error(f"unknown/empty conditions: {sorted(unknown)}")

    root = args.dataset_root.resolve()
    manifest_path = args.manifest.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path, dtype={"sequence": str})
    if args.sequences.strip():
        selected = [x.strip().zfill(4) for x in args.sequences.split(",") if x.strip()]
        eligible = manifest.copy()
        eligible["sequence"] = eligible["sequence"].astype(str).str.zfill(4)
        eligible["include_bool"] = eligible["include_primary"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
        index = eligible.set_index("sequence")
        for seq in selected:
            if seq not in index.index or str(index.loc[seq, "split"]).lower() != "train" or not bool(index.loc[seq, "include_bool"]):
                raise ValueError(f"{seq} is not QC-approved in official train split")
    else:
        selected = select_development_sequences(manifest, args.max_per_action, args.seed)
    if len(set(selected)) < 3:
        raise ValueError("Cross-fitted evaluation requires at least 3 distinct sequences.")
    if len(set(selected)) != len(selected):
        raise ValueError("Sequence IDs must be unique.")

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is required in the active environment") from exc

    start = time.time()
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    names = list(DEFAULT_CANDIDATES)
    cache_dir = out / "feature_cache"
    cache_configuration = {
        "format_version": 1,
        "dataset_root": str(root),
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_order": names,
        "penn_joint_indices_zero_based": PENN_JOINTS.tolist(),
        "mediapipe_joint_indices_zero_based": MEDIAPIPE_JOINTS.tolist(),
        "confidence_threshold": args.confidence_threshold,
        "gap_every": args.gap_every,
        "fps": args.fps,
        "one_euro_min_cutoff": args.one_euro_min_cutoff,
        "one_euro_beta": args.one_euro_beta,
        "one_euro_derivative_cutoff": args.one_euro_derivative_cutoff,
        "savgol_window": args.savgol_window,
        "savgol_polyorder": args.savgol_polyorder,
    }
    cache_signature = hashlib.sha256(
        json.dumps(cache_configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()
    for si, seq in enumerate(selected, 1):
        cache_paths = {
            condition: cache_dir / f"{seq}_{condition}.npz" for condition in conditions
        }
        if args.reuse_feature_cache and all(p.is_file() for p in cache_paths.values()):
            for condition, cache_path in cache_paths.items():
                cache[(seq, condition)] = _load_feature_cache(
                    cache_path, names, cache_signature
                )
            print(
                f"[{si}/{len(selected)}] {seq}: loaded compatible feature cache "
                f"for {len(conditions)} conditions",
                flush=True,
            )
            continue

        data = load_sequence(root, seq)
        frames = data["frames"]
        gt_xy, gt_vis, bbox = data["gt_xy"], data["gt_vis"], data["bbox_xyxy"]
        print(f"[{si}/{len(selected)}] {seq} ({data['action']}), {len(frames)} frames", flush=True)
        for condition in conditions:
            cache_path = cache_paths[condition]
            if args.reuse_feature_cache and cache_path.is_file():
                cache[(seq, condition)] = _load_feature_cache(
                    cache_path, names, cache_signature
                )
                print(f"  {condition}: loaded cached features", flush=True)
                continue
            raw, conf, sizes = extract_predictions(
                frames, condition, mp.solutions.pose, args.gap_every, seq
            )
            candidates = {
                "raw": raw,
                "linear_interpolation": interpolate_missing(raw, conf, args.confidence_threshold),
                "linear_missing_only": interpolate_missing_only(raw, conf, args.confidence_threshold),
                "confidence_aware": confidence_aware_repair(raw, conf, args.confidence_threshold),
                "one_euro": one_euro_filter(
                    raw, conf, sizes, args.confidence_threshold, args.fps,
                    args.one_euro_min_cutoff, args.one_euro_beta,
                    args.one_euro_derivative_cutoff,
                ),
                "savgol": savgol_filter_pose(
                    raw, conf, sizes, args.confidence_threshold,
                    args.savgol_window, args.savgol_polyorder,
                ),
            }
            masks = ground_truth_masks_and_scales(gt_xy, gt_vis, bbox, sizes)
            features = build_selective_features(
                raw, conf, sizes, candidates, args.confidence_threshold
            )
            errors, scoreable = candidate_error_targets(
                candidates, gt_xy, masks["valid_gt"], masks["torso_scale"], names
            )
            cache_item = {
                "action": data["action"], "raw": raw, "candidates": candidates,
                "features": features, "errors": errors, "scoreable": scoreable,
                "valid_gt": masks["valid_gt"], "torso_scale": masks["torso_scale"],
                "gt_xy": gt_xy,
            }
            cache[(seq, condition)] = cache_item
            _save_feature_cache(cache_path, cache_item, names, cache_signature)

    result_rows: list[dict[str, Any]] = []
    choices_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    # The original OOF sweep is descriptive only. The nested selector below chooses
    # a margin exclusively inside each outer training fold; the outer held-out
    # sequence's labels are not available to threshold selection.
    thresholds = sorted(set([0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, float(args.min_predicted_gain)]))
    nested_margins, nested_selection_rows, nested_sweep_rows = nested_threshold_selection(
        cache, selected, conditions, names, thresholds, args.ridge, args.seed,
        args.inner_folds, args.nested_pck_tolerance, args.nested_min_coverage,
        args.nested_max_false_correction,
    )
    nested_sequence_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for heldout in selected:
            train_x: list[np.ndarray] = []
            train_y: list[np.ndarray] = []
            for seq in selected:
                if seq == heldout:
                    continue
                item = cache[(seq, condition)]
                mask = item["scoreable"]
                if mask.any():
                    train_x.append(item["features"][mask])
                    train_y.append(item["errors"][mask])
            if not train_x or sum(len(x) for x in train_x) < 100:
                raise RuntimeError(f"Insufficient training points to fit selector for {heldout}/{condition}")
            x_train = np.concatenate(train_x, axis=0)
            y_train = np.concatenate(train_y, axis=0)
            model = fit_error_predictor(x_train, y_train, ridge=args.ridge)

            item = cache[(heldout, condition)]
            predicted = predict_candidate_errors(model, item["features"])
            choices = select_candidates(predicted, args.min_predicted_gain)
            selected_pose = apply_candidate_choices(item["candidates"], choices, names)
            actual_choice_errors = np.take_along_axis(item["errors"], choices[..., None], axis=-1)[..., 0]
            oracle_choice = np.argmin(item["errors"], axis=-1)
            best_predicted_repair = 1 + np.argmin(predicted[..., 1:], axis=-1)
            best_predicted_gain = predicted[..., 0] - np.min(predicted[..., 1:], axis=-1)
            best_actual_repair = 1 + np.argmin(item["errors"][..., 1:], axis=-1)
            best_actual_gain = item["errors"][..., 0] - np.min(item["errors"][..., 1:], axis=-1)
            actual_errors, scoreable = candidate_error_targets(
                {"selective": selected_pose}, item["gt_xy"], item["valid_gt"],
                item["torso_scale"], ["selective"]
            )
            raw_error = item["errors"][..., 0]
            sel_error = actual_errors[..., 0]
            target = scoreable
            n_target = int(target.sum())
            raw_finite = np.isfinite(item["raw"]).all(axis=-1)
            sel_finite = np.isfinite(selected_pose).all(axis=-1)
            raw_coverage = float((target & raw_finite).sum() / n_target) if n_target else float("nan")
            coverage = float((target & sel_finite).sum() / n_target) if n_target else float("nan")
            raw_pck = float(((raw_error <= 0.20) & target).sum() / n_target) if n_target else float("nan")
            pck = float(((sel_error <= 0.20) & target).sum() / n_target) if n_target else float("nan")
            harmed = target & raw_finite & (raw_error <= 0.20) & (~sel_finite | (sel_error > 0.20))
            raw_correct_n = int((target & raw_finite & (raw_error <= 0.20)).sum())
            false_correction = float(harmed.sum() / raw_correct_n) if raw_correct_n else float("nan")
            frequencies = {f"selected_{name}_fraction": float((choices == i).mean()) for i, name in enumerate(names)}
            result_rows.append({
                "sequence": heldout, "action": item["action"], "condition": condition,
                "n_training_sequences": len(selected) - 1, "n_scored_points": n_target,
                "raw_coverage": raw_coverage, "selective_coverage": coverage,
                "raw_pck_torso_0.20": raw_pck, "selective_pck_torso_0.20": pck,
                "delta_pck_vs_raw": pck - raw_pck,
                "raw_mean_error_norm": float(np.mean(raw_error[target])) if n_target else float("nan"),
                "selective_mean_error_norm": float(np.mean(sel_error[target])) if n_target else float("nan"),
                "false_correction_rate": false_correction,
                "repair_fraction": float((choices != 0).mean()),
                "ridge": args.ridge, "min_predicted_gain": args.min_predicted_gain,
                **frequencies,
            })

            # Evaluate the margin selected only from nested inner-fold predictions.
            # This outer sequence was excluded from both selector fitting and margin tuning.
            nested_margin = nested_margins[(condition, heldout)]
            nested_choices = select_candidates(predicted, nested_margin)
            nested_selected_pose = apply_candidate_choices(item["candidates"], nested_choices, names)
            nested_selected_errors = np.take_along_axis(
                item["errors"], nested_choices[..., None], axis=-1
            )[..., 0]
            raw_correct_for_nested = (
                target & raw_finite & (raw_error <= 0.20)
            )
            raw_correct_n_for_nested = int(raw_correct_for_nested.sum())
            eval_methods: list[tuple[str, np.ndarray, np.ndarray, float]] = [
                (
                    candidate_name,
                    item["candidates"][candidate_name],
                    item["errors"][..., k],
                    float("nan"),
                )
                for k, candidate_name in enumerate(names)
            ]
            eval_methods.append((
                "nested_selective",
                nested_selected_pose,
                nested_selected_errors,
                float(nested_margin),
            ))
            for method_name, method_pose, method_errors, selected_margin in eval_methods:
                method_finite = np.isfinite(method_pose).all(axis=-1)
                method_harmed = raw_correct_for_nested & (
                    ~method_finite | (method_errors > 0.20)
                )
                method_false_correction = (
                    float(method_harmed.sum() / raw_correct_n_for_nested)
                    if raw_correct_n_for_nested else float("nan")
                )
                nested_sequence_rows.append({
                    "sequence": heldout,
                    "action": item["action"],
                    "condition": condition,
                    "method": method_name,
                    "n_scored_points": n_target,
                    "pck_torso_0.20": (
                        float(((method_errors <= 0.20) & target).sum() / n_target)
                        if n_target else float("nan")
                    ),
                    "mean_error_norm_clipped": (
                        float(np.mean(method_errors[target])) if n_target else float("nan")
                    ),
                    "coverage": (
                        float((method_finite & target).sum() / n_target)
                        if n_target else float("nan")
                    ),
                    "false_correction_rate": method_false_correction,
                    "repair_fraction": (
                        float((nested_choices[target] != 0).mean())
                        if method_name == "nested_selective" and n_target else float("nan")
                    ),
                    "selected_margin": selected_margin,
                })

            for t in range(choices.shape[0]):
                for j in range(choices.shape[1]):
                    choices_rows.append({
                        "sequence": heldout, "condition": condition, "frame": t,
                        "joint": j, "choice": names[int(choices[t, j])],
                        "raw_finite": bool(raw_finite[t, j]),
                        "selected_finite": bool(sel_finite[t, j]),
                        "scoreable_gt": bool(target[t, j]),
                        "selected_actual_error": float(actual_choice_errors[t, j]),
                        "oracle_best_candidate": names[int(oracle_choice[t, j])],
                        "selected_matches_oracle": bool(choices[t, j] == oracle_choice[t, j]),
                        "best_predicted_repair": names[int(best_predicted_repair[t, j])],
                        "best_predicted_gain_vs_raw": float(best_predicted_gain[t, j]),
                        "best_actual_repair": names[int(best_actual_repair[t, j])],
                        "best_actual_gain_vs_raw": float(best_actual_gain[t, j]),
                    })
                    for k, candidate_name in enumerate(names):
                        prediction_rows.append({
                            "sequence": heldout, "action": item["action"],
                            "condition": condition, "frame": t, "joint": j,
                            "candidate": candidate_name,
                            "predicted_error_norm": float(predicted[t, j, k]),
                            "actual_error_norm": float(item["errors"][t, j, k]),
                            "scoreable_gt": bool(target[t, j]),
                            "selected_by_default_margin": bool(choices[t, j] == k),
                        })
            # Out-of-fold risk/coverage diagnostics over fixed margins. These rows are
            # descriptive; choosing the best margin here would require a separate
            # nested validation layer to avoid optimistic threshold selection.
            for margin in thresholds:
                margin_choices = select_candidates(predicted, margin)
                margin_pose = apply_candidate_choices(item["candidates"], margin_choices, names)
                margin_errors = np.take_along_axis(item["errors"], margin_choices[..., None], axis=-1)[..., 0]
                margin_finite = np.isfinite(margin_pose).all(axis=-1)
                margin_coverage = float((target & margin_finite).sum() / n_target) if n_target else float("nan")
                margin_pck = float(((margin_errors <= 0.20) & target).sum() / n_target) if n_target else float("nan")
                margin_harmed = target & raw_finite & (raw_error <= 0.20) & (~margin_finite | (margin_errors > 0.20))
                margin_raw_correct = int((target & raw_finite & (raw_error <= 0.20)).sum())
                threshold_rows.append({
                    "sequence": heldout, "action": item["action"], "condition": condition,
                    "min_predicted_gain": margin, "n_scored_points": n_target,
                    "pck_torso_0.20": margin_pck, "coverage": margin_coverage,
                    "repair_fraction": float((margin_choices != 0).mean()),
                    "false_correction_rate": float(margin_harmed.sum() / margin_raw_correct) if margin_raw_correct else float("nan"),
                    "mean_error_norm": float(np.mean(margin_errors[target])) if n_target else float("nan"),
                })
            print(
                f"  {condition}: {heldout} PCK raw={raw_pck:.3f}, "
                f"selective={pck:.3f}, repair={float((choices != 0).mean()):.1%}",
                flush=True,
            )

    detail = pd.DataFrame(result_rows)
    detail.to_csv(out / "sequence_condition_metrics.csv", index=False)
    pd.DataFrame(choices_rows).to_csv(out / "per_point_choices.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out / "out_of_fold_candidate_predictions.csv", index=False)
    threshold_detail = pd.DataFrame(threshold_rows)
    threshold_detail.to_csv(out / "threshold_sweep_per_sequence.csv", index=False)
    threshold_summary = threshold_detail.groupby(["condition", "min_predicted_gain"], as_index=False).agg(
        n_sequences=("sequence", "nunique"),
        pck_mean=("pck_torso_0.20", "mean"),
        coverage_mean=("coverage", "mean"),
        repair_fraction_mean=("repair_fraction", "mean"),
        false_correction_rate_mean=("false_correction_rate", "mean"),
        mean_error_norm=("mean_error_norm", "mean"),
    )
    threshold_summary.to_csv(out / "threshold_sweep_summary.csv", index=False)
    summary = detail.groupby("condition", as_index=False).agg(
        n_sequences=("sequence", "nunique"),
        raw_pck_torso_0_20_mean=("raw_pck_torso_0.20", "mean"),
        selective_pck_torso_0_20_mean=("selective_pck_torso_0.20", "mean"),
        delta_pck_vs_raw_mean=("delta_pck_vs_raw", "mean"),
        raw_coverage_mean=("raw_coverage", "mean"),
        selective_coverage_mean=("selective_coverage", "mean"),
        false_correction_rate_mean=("false_correction_rate", "mean"),
        repair_fraction_mean=("repair_fraction", "mean"),
        selective_mean_error_norm_mean=("selective_mean_error_norm", "mean"),
    )
    summary.to_csv(out / "condition_summary.csv", index=False)

    nested_selection_detail = pd.DataFrame(nested_selection_rows)
    nested_selection_detail.to_csv(out / "nested_threshold_selection.csv", index=False)
    pd.DataFrame(nested_sweep_rows).to_csv(out / "nested_inner_threshold_sweep.csv", index=False)
    nested_detail = pd.DataFrame(nested_sequence_rows)
    nested_detail.to_csv(out / "nested_selective_sequence_metrics.csv", index=False)
    nested_summary = nested_detail.groupby(["condition", "method"], as_index=False).agg(
        n_sequences=("sequence", "nunique"),
        n_scoreable_sequences=("pck_torso_0.20", "count"),
        pck_mean=("pck_torso_0.20", "mean"),
        mean_error_norm_clipped=("mean_error_norm_clipped", "mean"),
        coverage_mean=("coverage", "mean"),
        false_correction_rate_mean=("false_correction_rate", "mean"),
        repair_fraction_mean=("repair_fraction", "mean"),
    )
    nested_summary.to_csv(out / "nested_selective_condition_summary.csv", index=False)

    # Paired sequence-bootstrap intervals on the outer-held-out predictions.
    # These are still exploratory estimates from one development subset.
    nested_paired_rows: list[dict[str, Any]] = []
    nested_rng = np.random.default_rng(args.seed + 887)
    paired_metrics = [
        "pck_torso_0.20", "mean_error_norm_clipped", "coverage", "false_correction_rate"
    ]
    for condition in conditions:
        cond = nested_detail[nested_detail["condition"] == condition]
        for baseline in [m for m in names]:
            if baseline == "nested_selective":
                continue
            for metric in paired_metrics:
                wide = cond.pivot(index="sequence", columns="method", values=metric)
                if "nested_selective" not in wide.columns or baseline not in wide.columns:
                    continue
                paired = wide[["nested_selective", baseline]].dropna()
                if paired.empty:
                    continue
                delta = (
                    paired["nested_selective"] - paired[baseline]
                ).to_numpy(dtype=float)
                ci_low, ci_high = _bootstrap_mean_ci(delta, nested_rng, n_bootstrap=5000)
                nested_paired_rows.append({
                    "condition": condition,
                    "comparison": f"nested_selective_minus_{baseline}",
                    "baseline": baseline,
                    "metric": metric,
                    "n_paired_sequences": len(paired),
                    "delta_mean": float(delta.mean()),
                    "delta_ci95_low": ci_low,
                    "delta_ci95_high": ci_high,
                    "bootstrap_samples": 5000,
                    "bootstrap_unit": "sequence",
                })
    pd.DataFrame(nested_paired_rows).to_csv(
        out / "nested_selective_paired_comparison.csv", index=False
    )

    metadata = {
        "protocol": "leave-one-sequence-out cross-fitting, independently per condition",
        "dataset": "Penn Action",
        "official_split": "training only; each sequence is validated as train by load_sequence",
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_order": names,
        "feature_cache": {
            "directory": str(cache_dir),
            "format_version": 1,
            "signature": cache_signature,
            "reuse_enabled_this_run": args.reuse_feature_cache,
            "cached_per_sequence_condition": True,
        },
        "features_use_ground_truth": False,
        "ground_truth_used_only_for_training_labels_and_evaluation": True,
        "ridge": args.ridge,
        "min_predicted_gain_torso_normalized": args.min_predicted_gain,
        "nested_threshold_selection": {
            "protocol": "outer leave-one-sequence-out evaluation; within each outer training set, action-balanced inner folds choose the threshold; outer-held-out sequence labels are excluded from fitting and threshold selection",
            "inner_folds": args.inner_folds,
            "pck_tolerance": args.nested_pck_tolerance,
            "minimum_inner_coverage": args.nested_min_coverage,
            "maximum_inner_false_correction_rate": args.nested_max_false_correction,
            "selection_rule": "Among inner thresholds meeting coverage and false-correction guardrails, choose the largest margin whose mean sequence PCK is within pck_tolerance of the best feasible threshold; relax false-correction guardrail first, then coverage, only if no threshold is feasible",
        },
        "target_loss": "candidate localization error normalized by left-shoulder/right-hip torso scale, clipped to [0,1]; missing prediction loss=1",
        "limitations": [
            "Development-only cross-fit on the selected official-training subset; not official test performance.",
            "The swapped left/right mapping is supported by a separate training-sequence audit but remains provisional for view-specific correspondence; see docs/selective-repair-development-results.md.",
            "Selector is a linear error predictor, not a calibrated probability of repair success.",
            "Features use both neighboring frames and are therefore offline/non-causal.",
            "Torso-scale-scored points only; bbox-normalized metrics are not included in this selector script.",
            "The descriptive threshold_sweep_summary.csv reuses ordinary out-of-fold predictions and must not be used to claim an unbiased optimum threshold; use nested_threshold_selection.csv and nested_selective_paired_comparison.csv for the nested analysis.",
            "The nested estimate uses one action-balanced inner partition per outer held-out sequence; robustness to inner-fold assignment should be checked in a separate sensitivity analysis.",
            "Candidate prediction rows expose calibration/ranking errors and oracle agreement for audit.",
            "The per-sequence output is exploratory and should not be treated as a confirmatory statistical test.",
        ],
        "runtime_seconds": round(time.time() - start, 2),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\n=== Cross-fitted summary: sequence means ===")
    print(summary.round(4).to_string(index=False))
    print("\n=== Nested threshold-selected method: sequence means ===")
    print(nested_summary.round(4).to_string(index=False))
    print(f"\nSaved nested thresholds: {out / 'nested_threshold_selection.csv'}")
    print(f"Saved nested paired comparisons: {out / 'nested_selective_paired_comparison.csv'}")
    print(f"\nSaved outputs to {out}")
    print("Development-only. Do not describe as official test-set performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
