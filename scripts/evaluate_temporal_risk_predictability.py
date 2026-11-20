#!/usr/bin/env python3
"""Cache-only, sequence-held-out pilot for predicting candidate temporal distortion.

This is a diagnostic precursor to a spatial-temporal pose-repair selector.
It does NOT rerun MediaPipe and does NOT modify the source feature cache.

For each condition and each outer held-out Penn Action training sequence:
  1. Build candidate-specific kinematic descriptors from cached pose tracks.
  2. Build candidate temporal-error targets from annotated three-frame
     acceleration error (GT is used only as a training/evaluation target).
  3. Fit a multi-output Ridge model on the other selected sequences.
  4. Evaluate temporal-error prediction and candidate ranking on the held-out
     sequence, against a training-only constant candidate-mean baseline.

The unit of summary/inference is the sequence, not the frame/joint point.
The script is a predictability pilot, not the final repair-method evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

from frametrust.selective import fit_error_predictor, predict_candidate_errors  # noqa: E402
from evaluate_temporal_fidelity import temporal_ground_truth_mask  # noqa: E402
from penn_action_development import load_sequence  # noqa: E402


def _load_cache_item(
    path: Path,
    candidate_names: list[str],
    expected_signature: str,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        observed = str(saved["cache_signature"].item())
        if observed != expected_signature:
            raise ValueError(
                f"Cache signature mismatch in {path}: expected {expected_signature}, got {observed}"
            )
        return {
            "action": str(saved["action"].item()),
            "features": saved["features"].astype(np.float32, copy=True),
            "gt_xy": saved["gt_xy"].astype(float, copy=True),
            "raw": saved["raw"].astype(float, copy=True),
            "candidates": {
                name: saved[f"candidate__{name}"].astype(float, copy=True)
                for name in candidate_names
            },
        }


def _load_sequence_info(
    dataset_root: Path,
    sequence: str,
    expected_gt: np.ndarray,
) -> dict[str, Any]:
    data = load_sequence(dataset_root, sequence)  # Enforces official train membership.
    gt = np.asarray(data["gt_xy"], dtype=float)
    if gt.shape != expected_gt.shape or not np.allclose(gt, expected_gt, equal_nan=True):
        raise ValueError(f"Ground-truth/cache mismatch for sequence {sequence}")
    if not data["frames"]:
        raise ValueError(f"No frames found for sequence {sequence}")
    frame = cv2.imread(str(data["frames"][0]))
    if frame is None:
        raise ValueError(f"Could not decode first frame for sequence {sequence}")
    height, width = frame.shape[:2]
    valid_gt = temporal_ground_truth_mask(
        gt,
        np.asarray(data["gt_vis"], dtype=bool),
        width=width,
        height=height,
    )
    return {
        "gt_xy": gt,
        "valid_gt": valid_gt,
        "diagonal": float(np.hypot(width, height)),
        "n_frames": len(data["frames"]),
    }


def _build_temporal_targets_and_features(
    item: dict[str, Any],
    sequence_info: dict[str, Any],
    candidate_names: list[str],
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return augmented features [T,J,F+4K], raw temporal errors [T,J,K], GT triple mask [T-2,J].

    Temporal target is image-diagonal-normalized acceleration error over a
    centered three-frame window. Candidate triples that are not fully finite
    receive NaN here; after training-fold scaling they are assigned the maximum
    normalized target (1.0), while native temporal support is still reported
    separately in downstream evaluation.
    """
    base = np.asarray(item["features"], dtype=np.float32)
    gt = np.asarray(sequence_info["gt_xy"], dtype=float)
    valid = np.asarray(sequence_info["valid_gt"], dtype=bool)
    diagonal = float(sequence_info["diagonal"])
    if base.ndim != 3 or gt.ndim != 3 or gt.shape[-1] != 2:
        raise ValueError("Unexpected cache feature or ground-truth shape")
    t_count, joint_count, _ = base.shape
    if gt.shape[:2] != (t_count, joint_count) or valid.shape != (t_count, joint_count):
        raise ValueError("Cache/sequence dimensions do not match")
    if t_count < 3:
        raise ValueError("All selected sequences must contain at least three frames")

    gt_triples = valid[:-2] & valid[1:-1] & valid[2:]
    gt_acc = (
        gt[2:] - 2.0 * gt[1:-1] + gt[:-2]
    ) * (fps ** 2 / diagonal)

    errors = np.full((t_count, joint_count, len(candidate_names)), np.nan, dtype=np.float32)
    extra_blocks: list[np.ndarray] = []
    raw = np.asarray(item["raw"], dtype=float)
    raw_finite3 = (
        np.isfinite(raw[:-2]).all(axis=-1)
        & np.isfinite(raw[1:-1]).all(axis=-1)
        & np.isfinite(raw[2:]).all(axis=-1)
    )
    raw_acc = (raw[2:] - 2.0 * raw[1:-1] + raw[:-2]) * (fps ** 2 / diagonal)

    for candidate_idx, name in enumerate(candidate_names):
        pose = np.asarray(item["candidates"][name], dtype=float)
        finite3 = (
            np.isfinite(pose[:-2]).all(axis=-1)
            & np.isfinite(pose[1:-1]).all(axis=-1)
            & np.isfinite(pose[2:]).all(axis=-1)
        )
        pred_acc = (pose[2:] - 2.0 * pose[1:-1] + pose[:-2]) * (fps ** 2 / diagonal)
        pred_vel = (pose[2:] - pose[:-2]) * (fps / (2.0 * diagonal))
        accel_error = np.linalg.norm(pred_acc - gt_acc, axis=-1)
        accel_magnitude = np.linalg.norm(pred_acc, axis=-1)
        velocity_magnitude = np.linalg.norm(pred_vel, axis=-1)
        accel_delta_raw = np.linalg.norm(pred_acc - raw_acc, axis=-1)

        # Only visible, in-frame GT triples are eligible temporal targets.
        usable = gt_triples & finite3
        center_error = np.full((t_count - 2, joint_count), np.nan, dtype=np.float32)
        center_error[usable] = accel_error[usable].astype(np.float32)
        errors[1:-1, :, candidate_idx] = center_error

        # Candidate-specific inference-time descriptors, centered at t.
        block = np.zeros((t_count, joint_count, 4), dtype=np.float32)
        block[1:-1, :, 0] = np.where(finite3, velocity_magnitude, 0.0)
        block[1:-1, :, 1] = np.where(finite3, accel_magnitude, 0.0)
        block[1:-1, :, 2] = np.where(finite3 & raw_finite3, accel_delta_raw, 0.0)
        block[1:-1, :, 3] = finite3.astype(np.float32)
        extra_blocks.append(block)

    if extra_blocks:
        extras = np.concatenate(extra_blocks, axis=-1)
        augmented = np.concatenate([base, extras], axis=-1)
    else:
        augmented = base.copy()

    augmented = np.nan_to_num(augmented, nan=0.0, posinf=10.0, neginf=0.0)
    augmented = np.clip(augmented, -10.0, 10.0).astype(np.float32)
    return augmented, errors, gt_triples


def _normalize_targets(
    errors: np.ndarray,
    gt_triples: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Normalize candidate temporal error to [0,1]; missing candidate triples map to 1."""
    t_count, joint_count, n_candidates = errors.shape
    result = np.ones((t_count, joint_count, n_candidates), dtype=np.float32)
    center = errors[1:-1]
    normalized = np.clip(center / max(float(scale), 1e-9), 0.0, 1.0)
    # GT-invalid triples are not used in fitting/evaluation; their stored value is irrelevant.
    result[1:-1] = np.where(np.isfinite(center), normalized, 1.0).astype(np.float32)
    return result


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return float("nan")
    return float(pd.Series(x[good]).corr(pd.Series(y[good]), method="spearman"))


def _score_sequence(
    predicted: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
    gt_triples: np.ndarray,
) -> dict[str, float | int]:
    y_true = target[1:-1][gt_triples]
    y_pred = predicted[1:-1][gt_triples]
    y_base = baseline[1:-1][gt_triples]
    if y_true.size == 0:
        return {
            "n_gt_temporal_points": 0,
            "mae_model": float("nan"),
            "mae_candidate_mean_baseline": float("nan"),
            "mae_improvement_over_baseline": float("nan"),
            "spearman_model": float("nan"),
            "top1_candidate_accuracy_model": float("nan"),
            "top1_candidate_accuracy_baseline": float("nan"),
        }
    mae_model = float(np.mean(np.abs(y_pred - y_true)))
    mae_base = float(np.mean(np.abs(y_base - y_true)))
    true_best = np.argmin(y_true, axis=-1)
    model_best = np.argmin(y_pred, axis=-1)
    base_best = np.argmin(y_base, axis=-1)
    return {
        "n_gt_temporal_points": int(gt_triples.sum()),
        "mae_model": mae_model,
        "mae_candidate_mean_baseline": mae_base,
        "mae_improvement_over_baseline": mae_base - mae_model,
        "spearman_model": _spearman(y_pred, y_true),
        "top1_candidate_accuracy_model": float(np.mean(model_best == true_best)),
        "top1_candidate_accuracy_baseline": float(np.mean(base_best == true_best)),
    }


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out", type=Path, default=Path("outputs/selective_repair_crossfit"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--out", type=Path, default=Path("outputs/temporal_risk_predictability"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()

    if args.fps <= 0 or args.ridge < 0 or args.bootstrap_samples < 100:
        parser.error("fps must be positive, ridge non-negative, and bootstrap-samples >=100")

    started = time.time()
    source_out = args.source_out.resolve()
    dataset_root = args.dataset_root.resolve()
    metadata_path = source_out / "run_metadata.json"
    cache_dir = source_out / "feature_cache"
    if not metadata_path.is_file() or not cache_dir.is_dir():
        raise FileNotFoundError("Source output must contain run_metadata.json and feature_cache/")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = [str(x).zfill(4) for x in metadata["selected_sequences"]]
    conditions = list(metadata["conditions"])
    candidate_names = list(metadata["candidate_order"])
    signature = str(metadata["feature_cache"]["signature"])
    if not candidate_names or candidate_names[0] != "raw":
        raise ValueError("candidate_order must start with raw")
    if len(selected) < 3:
        raise ValueError("At least three sequences are required for LOSO evaluation")

    # Validate dataset membership and retrieve the temporal GT support/scale once.
    sequence_info: dict[str, dict[str, Any]] = {}
    for seq in selected:
        # A clean-condition cache is sufficient to verify GT shape/content.
        clean_path = cache_dir / f"{seq}_{conditions[0]}.npz"
        if not clean_path.is_file():
            raise FileNotFoundError(f"Missing cache: {clean_path}")
        with np.load(clean_path, allow_pickle=False) as saved:
            if str(saved["cache_signature"].item()) != signature:
                raise ValueError(f"Cache signature mismatch: {clean_path}")
            cached_gt = saved["gt_xy"].astype(float, copy=True)
        sequence_info[seq] = _load_sequence_info(dataset_root, seq, cached_gt)
        if sequence_info[seq]["n_frames"] != len(cached_gt):
            raise ValueError(f"Frame/cache length mismatch for sequence {seq}")

    result_rows: list[dict[str, Any]] = []

    for condition_idx, condition in enumerate(conditions):
        print(f"\n[{condition}] loading compatible caches", flush=True)
        condition_items: dict[str, dict[str, Any]] = {}
        augmented_features: dict[str, np.ndarray] = {}
        raw_temporal_errors: dict[str, np.ndarray] = {}
        gt_triple_masks: dict[str, np.ndarray] = {}

        for seq in selected:
            path = cache_dir / f"{seq}_{condition}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing cache: {path}")
            item = _load_cache_item(path, candidate_names, signature)
            if not np.allclose(item["gt_xy"], sequence_info[seq]["gt_xy"], equal_nan=True):
                raise ValueError(f"Ground-truth/cache mismatch for {seq}/{condition}")
            augmented, temporal_errors, triple_mask = _build_temporal_targets_and_features(
                item, sequence_info[seq], candidate_names, args.fps
            )
            condition_items[seq] = item
            augmented_features[seq] = augmented
            raw_temporal_errors[seq] = temporal_errors
            gt_triple_masks[seq] = triple_mask

        for outer_idx, heldout in enumerate(selected):
            training_sequences = [seq for seq in selected if seq != heldout]

            finite_training_temporal = []
            for seq in training_sequences:
                mask = gt_triple_masks[seq]
                values = raw_temporal_errors[seq][1:-1][mask]
                if values.size:
                    finite_training_temporal.append(values[np.isfinite(values)])
            finite_training_temporal = [x for x in finite_training_temporal if x.size]
            if not finite_training_temporal:
                raise RuntimeError(f"No finite temporal training targets for {condition}/{heldout}")
            all_train_errors = np.concatenate(finite_training_temporal)
            scale = float(np.quantile(all_train_errors, 0.95))
            if not np.isfinite(scale) or scale <= 1e-9:
                scale = 1.0

            train_x: list[np.ndarray] = []
            train_y: list[np.ndarray] = []
            for seq in training_sequences:
                mask = gt_triple_masks[seq]
                features = augmented_features[seq][1:-1][mask]
                targets = _normalize_targets(raw_temporal_errors[seq], mask, scale)[1:-1][mask]
                if features.size:
                    train_x.append(features)
                    train_y.append(targets)
            if not train_x or sum(len(x) for x in train_x) < 100:
                raise RuntimeError(f"Insufficient temporal training rows for {condition}/{heldout}")

            model = fit_error_predictor(
                np.concatenate(train_x, axis=0),
                np.concatenate(train_y, axis=0),
                ridge=args.ridge,
            )
            test_features = augmented_features[heldout]
            predicted = predict_candidate_errors(model, test_features)
            target = _normalize_targets(
                raw_temporal_errors[heldout], gt_triple_masks[heldout], scale
            )
            # Training-only constant per-candidate mean predictor.
            candidate_mean = np.concatenate(train_y, axis=0).mean(axis=0)
            baseline = np.broadcast_to(candidate_mean, predicted.shape).astype(np.float32)
            scores = _score_sequence(
                predicted, target, baseline, gt_triple_masks[heldout]
            )
            result_rows.append({
                "condition": condition,
                "sequence": heldout,
                "action": condition_items[heldout]["action"],
                "temporal_target_scale_train_q95": scale,
                "n_training_sequences": len(training_sequences),
                **scores,
            })
            print(
                f"  outer={heldout}: MAE model={scores['mae_model']:.4f}, "
                f"baseline={scores['mae_candidate_mean_baseline']:.4f}, "
                f"top1={scores['top1_candidate_accuracy_model']:.3f}",
                flush=True,
            )

        # Release condition-specific arrays before moving to the next condition.
        del condition_items, augmented_features, raw_temporal_errors, gt_triple_masks

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    per_sequence_path = out / "temporal_risk_predictability_by_sequence.csv"
    results.to_csv(per_sequence_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    for condition, group in results.groupby("condition"):
        model_mae = group["mae_model"].to_numpy(dtype=float)
        base_mae = group["mae_candidate_mean_baseline"].to_numpy(dtype=float)
        improvement = group["mae_improvement_over_baseline"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(
            improvement, args.bootstrap_samples, args.seed + conditions.index(condition)
        )
        summary_rows.append({
            "condition": condition,
            "n_sequences": int(group["sequence"].nunique()),
            "mae_model_sequence_mean": float(np.nanmean(model_mae)),
            "mae_candidate_mean_baseline_sequence_mean": float(np.nanmean(base_mae)),
            "mae_improvement_sequence_mean": float(np.nanmean(improvement)),
            "mae_improvement_ci95_low": ci_low,
            "mae_improvement_ci95_high": ci_high,
            "spearman_model_sequence_mean": float(group["spearman_model"].mean()),
            "top1_candidate_accuracy_model_sequence_mean": float(group["top1_candidate_accuracy_model"].mean()),
            "top1_candidate_accuracy_baseline_sequence_mean": float(group["top1_candidate_accuracy_baseline"].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = out / "temporal_risk_predictability_summary.csv"
    summary.to_csv(summary_path, index=False)

    run_metadata = {
        "experiment": "candidate_temporal_risk_predictability_pilot",
        "dataset": "Penn Action",
        "official_split": "selected official training sequences only; load_sequence enforces train membership",
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_order": candidate_names,
        "source_selective_repair_output": str(source_out),
        "source_feature_cache_signature": signature,
        "outer_protocol": "leave-one-sequence-out; the held-out sequence is excluded from fitting and target-scale estimation",
        "model": "multi-output Ridge using cached inference features plus candidate-specific centered velocity, acceleration magnitude, acceleration deviation from raw, and three-frame finite-support indicator",
        "target": "candidate-specific image-diagonal-normalized acceleration error against Penn Action annotations on GT-visible, in-frame centered three-frame windows; candidate-incomplete triples receive maximum normalized target; GT-invalid triples are excluded",
        "target_scaling": "95th percentile of finite candidate temporal errors from outer-training sequences only; targets clipped to [0,1]",
        "baseline": "training-only constant mean temporal error per candidate",
        "metrics": ["sequence-level MAE", "sequence-level Spearman correlation", "per-point top-1 candidate ranking accuracy"],
        "fps_assumed": args.fps,
        "bootstrap": {"samples": args.bootstrap_samples, "unit": "sequence", "interval": "two-sided percentile 95% CI"},
        "official_test_split_read": False,
        "runtime_seconds": round(time.time() - started, 2),
        "interpretation_guardrail": "This pilot tests whether temporal distortion is predictable from inference-time inputs. It does not establish repair improvement, calibrated risk guarantees, or test-set performance.",
    }
    metadata_path = out / "temporal_risk_predictability_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print("\n=== Temporal-risk predictability summary (sequence macro) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {per_sequence_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {metadata_path}")
    print("This is a cache-only development pilot; do not report it as final test performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
