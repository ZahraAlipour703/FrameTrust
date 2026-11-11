#!/usr/bin/env python3
"""Evaluate temporal fidelity on cached, nested Penn Action development predictions.

This post-processing script does not rerun MediaPipe. It reconstructs the
outer-fold nested-selective predictions from the run's compatible feature cache,
then compares temporal trajectory metrics against the cached candidate baselines.
It uses the official Penn Action training split only; it is not a test evaluation.
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
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frametrust.selective import (  # noqa: E402
    apply_candidate_choices,
    fit_error_predictor,
    predict_candidate_errors,
    select_candidates,
)
from penn_action_development import load_sequence  # noqa: E402


TEMPORAL_METRICS = (
    "temporal_jitter_norm_s2",
    "velocity_error_norm_s",
    "acceleration_error_norm_s2",
)


def temporal_metric_values(
    pose: np.ndarray,
    gt_xy: np.ndarray,
    valid_gt: np.ndarray,
    frame_diagonal: float,
    fps: float,
    support: np.ndarray | None = None,
) -> tuple[dict[str, float], int, int]:
    """Return second-difference magnitude/error on GT-visible three-frame windows.

    Coordinates are normalized by a fixed image diagonal for the sequence.
    A triple is eligible only if GT is valid at t-1,t,t+1 and the prediction is
    finite at all three frames. If support is supplied, it further restricts
    triples to a shared support for a paired method comparison.
    """
    pose = np.asarray(pose, dtype=float)
    gt_xy = np.asarray(gt_xy, dtype=float)
    valid_gt = np.asarray(valid_gt, dtype=bool)
    if pose.ndim != 3 or pose.shape[-1] != 2:
        raise ValueError("pose must have shape [frames, joints, 2]")
    if gt_xy.shape != pose.shape or valid_gt.shape != pose.shape[:2]:
        raise ValueError("pose, ground truth, and validity mask shapes do not match")
    if frame_diagonal <= 0 or fps <= 0:
        raise ValueError("frame_diagonal and fps must be positive")
    if len(pose) < 3:
        return {name: float("nan") for name in TEMPORAL_METRICS}, 0, 0

    gt_triples = valid_gt[:-2] & valid_gt[1:-1] & valid_gt[2:]
    pred_finite = (
        np.isfinite(pose[:-2]).all(axis=-1)
        & np.isfinite(pose[1:-1]).all(axis=-1)
        & np.isfinite(pose[2:]).all(axis=-1)
    )
    eligible = gt_triples & pred_finite
    if support is not None:
        if support.shape != eligible.shape:
            raise ValueError("support mask must have shape [frames-2, joints]")
        eligible &= support

    gt_normalized = gt_xy / frame_diagonal
    pred_normalized = pose / frame_diagonal
    pred_velocity = (pred_normalized[2:] - pred_normalized[:-2]) * (fps / 2.0)
    gt_velocity = (gt_normalized[2:] - gt_normalized[:-2]) * (fps / 2.0)
    pred_acceleration = (
        pred_normalized[2:] - 2.0 * pred_normalized[1:-1] + pred_normalized[:-2]
    ) * (fps ** 2)
    gt_acceleration = (
        gt_normalized[2:] - 2.0 * gt_normalized[1:-1] + gt_normalized[:-2]
    ) * (fps ** 2)

    jitter = np.linalg.norm(pred_acceleration, axis=-1)
    velocity_error = np.linalg.norm(pred_velocity - gt_velocity, axis=-1)
    acceleration_error = np.linalg.norm(pred_acceleration - gt_acceleration, axis=-1)
    values = {
        "temporal_jitter_norm_s2": jitter,
        "velocity_error_norm_s": velocity_error,
        "acceleration_error_norm_s2": acceleration_error,
    }
    result = {
        name: float(np.mean(arr[eligible])) if eligible.any() else float("nan")
        for name, arr in values.items()
    }
    return result, int(eligible.sum()), int(gt_triples.sum())


def temporal_ground_truth_mask(
    gt_xy: np.ndarray,
    gt_visible: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Return all visible, in-frame annotated joints for temporal evaluation.

    Temporal metrics use image-diagonal normalization and therefore must not
    reuse the localization evaluator's torso-scale scoreability mask. In
    particular, a frame remains eligible when a torso anchor is not annotated.
    """
    gt_xy = np.asarray(gt_xy, dtype=float)
    gt_visible = np.asarray(gt_visible, dtype=bool)
    if gt_xy.ndim != 3 or gt_xy.shape[-1] != 2:
        raise ValueError("gt_xy must have shape [frames, joints, 2]")
    if gt_visible.shape != gt_xy.shape[:2]:
        raise ValueError("gt_visible must have shape [frames, joints]")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    finite = np.isfinite(gt_xy).all(axis=-1)
    in_frame = (
        (gt_xy[..., 0] >= 0) & (gt_xy[..., 0] < width)
        & (gt_xy[..., 1] >= 0) & (gt_xy[..., 1] < height)
    )
    return gt_visible & finite & in_frame


def _load_cache_item(
    path: Path, candidate_names: list[str], expected_signature: str,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        observed_signature = str(saved["cache_signature"].item())
        if observed_signature != expected_signature:
            raise ValueError(
                f"Cache signature mismatch in {path}. Use the output directory "
                "from the exact run whose run_metadata.json you supplied."
            )
        candidates = {
            name: saved[f"candidate__{name}"].copy()
            for name in candidate_names
        }
        return {
            "action": str(saved["action"].item()),
            "features": saved["features"].copy(),
            "errors": saved["errors"].copy(),
            "scoreable": saved["scoreable"].copy(),
            "valid_gt": saved["valid_gt"].copy(),
            "gt_xy": saved["gt_xy"].copy(),
            "candidates": candidates,
        }


def _bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, n_bootstrap: int = 5000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Existing selective-repair output directory containing run_metadata.json and feature_cache/.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    if args.fps <= 0 or args.bootstrap_samples < 100:
        parser.error("--fps must be positive and --bootstrap-samples must be >=100")

    out = args.out.resolve()
    metadata_path = out / "run_metadata.json"
    cache_dir = out / "feature_cache"
    margin_path = out / "nested_threshold_selection.csv"
    for path in (metadata_path, margin_path, cache_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required cached run artifact not found: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = [str(x).zfill(4) for x in metadata["selected_sequences"]]
    conditions = list(metadata["conditions"])
    candidate_names = list(metadata["candidate_order"])
    signature = str(metadata["feature_cache"]["signature"])
    ridge = float(metadata.get("ridge", 10.0))
    margins_df = pd.read_csv(margin_path, dtype={"outer_heldout_sequence": str})
    margins_df["outer_heldout_sequence"] = margins_df["outer_heldout_sequence"].str.zfill(4)
    margin_map = {
        (str(row.condition), str(row.outer_heldout_sequence).zfill(4)): float(row.selected_margin)
        for row in margins_df.itertuples(index=False)
    }

    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for seq in selected:
        for condition in conditions:
            cache_path = cache_dir / f"{seq}_{condition}.npz"
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing per-sequence cache: {cache_path}")
            cache[(seq, condition)] = _load_cache_item(
                cache_path, candidate_names, signature
            )

    per_sequence_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    n_sequences = len(selected)
    method_names = candidate_names + ["nested_selective"]
    dataset_root = args.dataset_root.resolve()

    for condition_idx, condition in enumerate(conditions):
        for heldout_idx, heldout in enumerate(selected):
            item = cache[(heldout, condition)]
            # load_sequence validates the official train flag. Only the first image
            # is decoded to obtain the fixed pixel normalization scale.
            sequence_data = load_sequence(dataset_root, heldout)
            if len(sequence_data["frames"]) != len(item["gt_xy"]):
                raise ValueError(f"Frame/cache length mismatch for sequence {heldout}")
            first_frame = cv2.imread(str(sequence_data["frames"][0]))
            if first_frame is None:
                raise ValueError(f"Could not decode first frame for sequence {heldout}")
            height, width = first_frame.shape[:2]
            diagonal = float(np.hypot(width, height))

            train_x: list[np.ndarray] = []
            train_y: list[np.ndarray] = []
            for train_seq in selected:
                if train_seq == heldout:
                    continue
                train_item = cache[(train_seq, condition)]
                mask = train_item["scoreable"]
                if mask.any():
                    train_x.append(train_item["features"][mask])
                    train_y.append(train_item["errors"][mask])
            if not train_x or sum(len(values) for values in train_x) < 100:
                raise RuntimeError(f"Insufficient training points for {heldout}/{condition}")
            model = fit_error_predictor(
                np.concatenate(train_x, axis=0),
                np.concatenate(train_y, axis=0),
                ridge=ridge,
            )
            predicted = predict_candidate_errors(model, item["features"])
            try:
                margin = margin_map[(condition, heldout)]
            except KeyError as exc:
                raise ValueError(f"No nested threshold recorded for {condition}/{heldout}") from exc
            choices = select_candidates(predicted, margin)
            nested_pose = apply_candidate_choices(item["candidates"], choices, candidate_names)
            poses = dict(item["candidates"])
            poses["nested_selective"] = nested_pose

            # Temporal evaluation does not require torso anchors. Rebuild
            # validity from all Penn Action-visible, in-frame landmarks instead
            # of reusing item["valid_gt"], which is restricted to torso-scale
            # scoreability for the localization/PCK evaluation.
            gt = np.asarray(sequence_data["gt_xy"], dtype=float)
            if gt.shape != item["gt_xy"].shape or not np.allclose(
                gt, item["gt_xy"], equal_nan=True
            ):
                raise ValueError(f"Ground-truth/cache mismatch for sequence {heldout}")
            valid_gt = temporal_ground_truth_mask(
                gt,
                sequence_data["gt_vis"],
                width=width,
                height=height,
            )
            gt_triples = (
                valid_gt[:-2] & valid_gt[1:-1] & valid_gt[2:]
                if len(valid_gt) >= 3 else np.zeros((0, valid_gt.shape[1]), dtype=bool)
            )
            n_gt_triples = int(gt_triples.sum())
            method_support: dict[str, np.ndarray] = {}
            for method in method_names:
                pose = poses[method]
                if len(pose) >= 3:
                    finite_triples = (
                        np.isfinite(pose[:-2]).all(axis=-1)
                        & np.isfinite(pose[1:-1]).all(axis=-1)
                        & np.isfinite(pose[2:]).all(axis=-1)
                    )
                    method_support[method] = gt_triples & finite_triples
                else:
                    method_support[method] = gt_triples.copy() & False
                metrics, n_supported, n_possible = temporal_metric_values(
                    pose, gt, valid_gt, diagonal, args.fps,
                )
                per_sequence_rows.append({
                    "sequence": heldout,
                    "action": item["action"],
                    "condition": condition,
                    "method": method,
                    "frame_diagonal_px": diagonal,
                    "fps_assumed": args.fps,
                    "n_valid_gt_triples": n_possible,
                    "n_temporal_triples_supported": n_supported,
                    "temporal_support_coverage": (
                        n_supported / n_possible if n_possible else float("nan")
                    ),
                    **metrics,
                })

            # Paired comparisons use identical, GT-visible three-frame windows for
            # each method pair to prevent a method benefiting from dropped support.
            for baseline in candidate_names:
                pair_support = method_support["nested_selective"] & method_support[baseline]
                for metric_name in TEMPORAL_METRICS:
                    method_values = temporal_metric_values(
                        poses["nested_selective"], gt, valid_gt, diagonal, args.fps,
                        support=pair_support,
                    )[0]
                    baseline_values = temporal_metric_values(
                        poses[baseline], gt, valid_gt, diagonal, args.fps,
                        support=pair_support,
                    )[0]
                    a = method_values[metric_name]
                    b = baseline_values[metric_name]
                    paired_rows.append({
                        "sequence": heldout,
                        "action": item["action"],
                        "condition": condition,
                        "comparison": f"nested_selective_minus_{baseline}",
                        "baseline": baseline,
                        "metric": metric_name,
                        "n_common_temporal_triples": int(pair_support.sum()),
                        "delta_mean": a - b if np.isfinite(a) and np.isfinite(b) else float("nan"),
                    })

            print(
                f"[{condition_idx + 1}/{len(conditions)}] {condition}: "
                f"{heldout} temporal metrics complete",
                flush=True,
            )

    sequence_metrics = pd.DataFrame(per_sequence_rows)
    sequence_metrics.to_csv(out / "temporal_fidelity_per_sequence.csv", index=False)
    paired_detail = pd.DataFrame(paired_rows)
    paired_detail.to_csv(out / "temporal_fidelity_paired_per_sequence.csv", index=False)

    summary = sequence_metrics.groupby(["condition", "method"], as_index=False).agg(
        n_sequences=("sequence", "nunique"),
        temporal_support_coverage_mean=("temporal_support_coverage", "mean"),
        temporal_jitter_norm_s2_mean=("temporal_jitter_norm_s2", "mean"),
        velocity_error_norm_s_mean=("velocity_error_norm_s", "mean"),
        acceleration_error_norm_s2_mean=("acceleration_error_norm_s2", "mean"),
    )
    summary.to_csv(out / "temporal_fidelity_condition_summary.csv", index=False)

    paired_summary_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    detail_groups = paired_detail.groupby(["condition", "comparison", "baseline", "metric"], dropna=False)
    for (condition, comparison, baseline, metric_name), group in detail_groups:
        values = group["delta_mean"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        ci_low, ci_high = _bootstrap_mean_ci(values, rng, args.bootstrap_samples)
        supports = group.loc[np.isfinite(group["delta_mean"]), "n_common_temporal_triples"]
        paired_summary_rows.append({
            "condition": condition,
            "comparison": comparison,
            "baseline": baseline,
            "metric": metric_name,
            "n_paired_sequences": len(values),
            "mean_temporal_triples_per_sequence": (
                float(supports.mean()) if len(supports) else float("nan")
            ),
            "delta_mean": float(values.mean()) if len(values) else float("nan"),
            "delta_ci95_low": ci_low,
            "delta_ci95_high": ci_high,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "sequence",
            "paired_support": "same GT-visible and finite three-frame windows for both methods",
        })
    paired_summary = pd.DataFrame(paired_summary_rows)
    paired_summary.to_csv(out / "temporal_fidelity_paired_comparison.csv", index=False)

    protocol = {
        "dataset": "Penn Action",
        "official_split": "training only; load_sequence checks each selected sequence's official train flag",
        "selected_sequences": selected,
        "conditions": conditions,
        "methods": method_names,
        "temporal_jitter_definition": "mean magnitude of the second temporal difference of normalized 2D pose coordinates, multiplied by fps^2; this measures trajectory acceleration/roughness, not jitter in isolation because real motion also accelerates",
        "velocity_error_definition": "mean vector error between centered predicted and annotated GT velocity, normalized by fixed image diagonal and expressed per second",
        "acceleration_error_definition": "mean vector error between predicted and annotated GT second temporal difference, normalized by fixed image diagonal and expressed per second squared",
        "temporal_support": "All three GT frames must be visible and valid; method-specific metrics additionally require finite predictions at all three frames. Paired comparisons use the same GT-visible triples where both compared methods have finite predictions.",
        "normalization": "fixed image diagonal from the first decoded frame of each sequence; not mixed with torso-normalized PCK",
        "fps_assumed": args.fps,
        "selection": "nested thresholds from nested_threshold_selection.csv; selector retrained on all other selected development sequences and never on the held-out sequence",
        "official_test_split_read": False,
        "limitations": [
            "Development-only results from the selected official-training subset.",
            "Second-difference magnitude alone cannot distinguish unwanted jitter from genuine rapid motion; acceleration error relative to annotations is the more direct temporal-fidelity measure.",
            "Sparse annotation visibility and detector missingness can reduce temporal support; always report support coverage alongside temporal errors.",
            "The annotation trajectory itself contains noise, and the evaluation measures agreement with that trajectory rather than biomechanical truth.",
            "Paired confidence intervals resample sequences, not frames or joints, and are exploratory and unadjusted for multiple comparisons.",
            "The selector is offline/non-causal because it uses neighboring-frame features.",
        ],
    }
    (out / "temporal_fidelity_metadata.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    print("\n=== Temporal fidelity: sequence-macro means ===")
    print(summary.round(5).to_string(index=False))
    print("\n=== Paired nested-selective deltas (negative favors selective for all metrics) ===")
    print(paired_summary.round(5).to_string(index=False))
    print(f"\nSaved temporal outputs under {out}")
    print("Development-only. Do not report as official test-set performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
