#!/usr/bin/env python3
"""Nested LOSO evaluation of pattern-prior second-order pose-repair routing.

The method combines:
  (1) inference-time spatial candidate-error predictions (unary costs),
  (2) a training-only sequence-balanced mean temporal risk for each candidate
      identity triplet (a,b,c), and
  (3) a constant penalty for candidate switches.

For each joint, the second-order dynamic program minimizes

  sum_t U[t, z_t]
  + lambda * sum_t prior[z_(t-1), z_t, z_(t+1)]
  + rho * sum_t 1[z_t != z_(t-1)].

The pattern prior is estimated from sampled GT-valid centers in training
sequences only. Each training sequence contributes equally to the prior.
Hyperparameters are selected inside action-balanced inner folds; outer held-out
sequences are not used for fitting or selection. The evaluation reuses the
existing MediaPipe feature cache and reads only selected official Penn Action
training sequences. It does not read the official test split.

This is a development experiment, not a claim of state-of-the-art performance.
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
from frametrust.sequence_routing import route_candidate_sequence  # noqa: E402
from evaluate_selective_repair import _fit_selector_for_sequences  # noqa: E402
from evaluate_sequence_routing import (  # noqa: E402
    _action_balanced_folds,
    _load_cache_item,
    _sequence_spatial_metrics,
)
from evaluate_temporal_fidelity import (  # noqa: E402
    TEMPORAL_METRICS,
    temporal_metric_values,
)
from evaluate_temporal_risk_predictability import _load_sequence_info  # noqa: E402

SPATIAL_MARGIN = 0.01
SPATIAL_METRIC_NAMES = (
    "pck_torso_0.20",
    "mean_error_norm",
    "coverage",
    "false_correction_rate",
    "repair_fraction",
)
ALL_METRIC_NAMES = SPATIAL_METRIC_NAMES + TEMPORAL_METRICS + (
    "temporal_support_coverage",
    "switch_rate",
)


def _sample_gt_valid_centers(
    valid_gt: np.ndarray,
    max_centers: int,
    seed: int,
) -> np.ndarray:
    """Sample flattened [T-2,J] centers without using prediction targets."""
    valid = np.asarray(valid_gt, dtype=bool)
    if len(valid) < 3:
        raise ValueError("Sequence must contain at least three frames")
    mask = valid[:-2] & valid[1:-1] & valid[2:]
    indices = np.flatnonzero(mask.reshape(-1))
    if not len(indices):
        raise RuntimeError("No GT-valid temporal centers in sequence")
    if len(indices) > max_centers:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_centers, replace=False))
    return indices.astype(np.int64)


def _sequence_pattern_targets(
    item: dict[str, Any],
    info: dict[str, Any],
    names: list[str],
    flat_indices: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Raw acceleration error [sampled center, K^3] for every hybrid triplet.

    A pattern (a,b,c) takes candidate a at t-1, b at t, and c at t+1.
    GT validity is guaranteed by the sampled-center mask. Unsupported candidate
    triplets remain NaN and are assigned maximum normalized risk when the
    training-only prior is formed.
    """
    raw = np.asarray(item["raw"], dtype=float)
    gt = np.asarray(info["gt_xy"], dtype=float)
    diagonal = float(info["diagonal"])
    _, n_joints, _ = raw.shape
    frame_idx = flat_indices // n_joints
    joint_idx = flat_indices % n_joints
    gt_prev = gt[frame_idx, joint_idx]
    gt_mid = gt[frame_idx + 1, joint_idx]
    gt_next = gt[frame_idx + 2, joint_idx]
    gt_acc = (gt_next - 2.0 * gt_mid + gt_prev) * (fps * fps / diagonal)

    k_count = len(names)
    out = np.full((len(flat_indices), k_count**3), np.nan, dtype=np.float32)
    pattern_id = 0
    for a in range(k_count):
        coord_a = np.asarray(item["candidates"][names[a]], dtype=float)
        prev = coord_a[frame_idx, joint_idx]
        prev_ok = np.isfinite(prev).all(axis=-1)
        for b in range(k_count):
            coord_b = np.asarray(item["candidates"][names[b]], dtype=float)
            mid = coord_b[frame_idx + 1, joint_idx]
            mid_ok = np.isfinite(mid).all(axis=-1)
            for c in range(k_count):
                coord_c = np.asarray(item["candidates"][names[c]], dtype=float)
                nxt = coord_c[frame_idx + 2, joint_idx]
                valid = prev_ok & mid_ok & np.isfinite(nxt).all(axis=-1)
                if valid.any():
                    acc = (nxt - 2.0 * mid + prev) * (fps * fps / diagonal)
                    errors = np.linalg.norm(acc - gt_acc, axis=-1)
                    out[valid, pattern_id] = errors[valid].astype(np.float32)
                pattern_id += 1
    return out


def _make_pattern_prior(
    training_sequences: list[str],
    pattern_targets: dict[str, np.ndarray],
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a sequence-balanced pattern prior and training-only q95 scale."""
    finite_parts = [
        pattern_targets[seq][np.isfinite(pattern_targets[seq])]
        for seq in training_sequences
    ]
    finite_parts = [values for values in finite_parts if values.size]
    if not finite_parts:
        raise RuntimeError("No finite temporal targets for pattern prior")
    scale = float(np.quantile(np.concatenate(finite_parts), 0.95))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0

    sequence_means: list[np.ndarray] = []
    sequence_support: list[np.ndarray] = []
    for seq in training_sequences:
        raw = pattern_targets[seq]
        finite = np.isfinite(raw)
        normalized = np.ones(raw.shape, dtype=np.float32)
        normalized[finite] = np.clip(raw[finite] / scale, 0.0, 1.0)
        # Equal sequence weighting prevents longer videos dominating the prior.
        sequence_means.append(normalized.mean(axis=0))
        sequence_support.append(finite.mean(axis=0))
    prior_flat = np.mean(np.stack(sequence_means, axis=0), axis=0)
    support_flat = np.mean(np.stack(sequence_support, axis=0), axis=0)
    k_count = int(round(prior_flat.size ** (1.0 / 3.0)))
    if k_count**3 != prior_flat.size:
        raise ValueError("Pattern prior length is not a perfect cube")
    return (
        prior_flat.reshape(k_count, k_count, k_count),
        scale,
        support_flat.reshape(k_count, k_count, k_count),
    )


def second_order_route(
    predicted_spatial_errors: np.ndarray,
    candidates: dict[str, np.ndarray],
    names: list[str],
    pattern_prior: np.ndarray,
    temporal_weight: float,
    switch_penalty: float,
    min_predicted_gain: float = SPATIAL_MARGIN,
) -> np.ndarray:
    """Exact second-order Viterbi/DP over candidate identities per joint.

    DP state is the previous pair (z[t-1], z[t]). The temporal pattern prior is
    charged for every centered three-frame combination. Invalid candidate
    triplets receive maximum normalized temporal cost (1.0); invalid unary
    candidate coordinates also receive spatial cost 1.0.
    """
    predicted = np.asarray(predicted_spatial_errors, dtype=float)
    if predicted.ndim != 3:
        raise ValueError("predicted_spatial_errors must be [T,J,K]")
    t_count, n_joints, k_count = predicted.shape
    if t_count < 1 or n_joints < 1:
        raise ValueError("Prediction array must contain frames and joints")
    if len(names) != k_count or names[0] != "raw":
        raise ValueError("Candidate names must match prediction axis and begin with raw")
    prior = np.asarray(pattern_prior, dtype=float)
    if prior.shape != (k_count, k_count, k_count):
        raise ValueError("pattern_prior must have shape [K,K,K]")
    if temporal_weight < 0 or switch_penalty < 0 or min_predicted_gain < 0:
        raise ValueError("Weights and spatial margin must be non-negative")
    if not np.isfinite(temporal_weight + switch_penalty + min_predicted_gain):
        raise ValueError("Weights and margin must be finite")

    available = np.stack(
        [
            np.isfinite(np.asarray(candidates[name], dtype=float)).all(axis=-1)
            for name in names
        ],
        axis=-1,
    )  # [T,J,K]
    if available.shape != (t_count, n_joints, k_count):
        raise ValueError("Candidate arrays do not match prediction dimensions")

    unary = predicted.copy()
    unary[..., 1:] += float(min_predicted_gain)
    unary = np.nan_to_num(unary, nan=1.0, posinf=1.0, neginf=0.0)
    unary = np.where(available, unary, 1.0)

    if t_count == 1:
        return np.argmin(unary[0], axis=-1)[None, :].astype(np.int64)

    switch = (~np.eye(k_count, dtype=bool)).astype(np.float64)
    dp = unary[0, :, :, None] + unary[1, :, None, :]
    dp += float(switch_penalty) * switch[None, :, :]

    if t_count == 2:
        best_flat = np.argmin(dp.reshape(n_joints, -1), axis=1)
        best_a = best_flat // k_count
        best_b = best_flat % k_count
        out = np.empty((2, n_joints), dtype=np.int64)
        out[0] = best_a
        out[1] = best_b
        return out

    # back[t,j,b,c] stores the best a=z[t-2] for state (b=z[t-1],c=z[t]).
    back = np.zeros((t_count, n_joints, k_count, k_count), dtype=np.uint8)
    for frame in range(2, t_count):
        valid_triplet = (
            available[frame - 2, :, :, None, None]
            & available[frame - 1, :, None, :, None]
            & available[frame, :, None, None, :]
        )
        temporal_cost = np.where(valid_triplet, prior[None, :, :, :], 1.0)
        totals = (
            dp[:, :, :, None]
            + float(temporal_weight) * temporal_cost
            + float(switch_penalty) * switch[None, None, :, :]
            + unary[frame, :, None, None, :]
        )  # [J,a,b,c]
        best_prevprev = np.argmin(totals, axis=1)  # [J,b,c]
        dp = np.take_along_axis(
            totals, best_prevprev[:, None, :, :], axis=1
        )[:, 0, :, :]
        back[frame] = best_prevprev.astype(np.uint8)

    last_flat = np.argmin(dp.reshape(n_joints, -1), axis=1)
    state_b = last_flat // k_count
    state_c = last_flat % k_count
    choices = np.zeros((t_count, n_joints), dtype=np.int64)
    choices[-2] = state_b
    choices[-1] = state_c
    joint_ids = np.arange(n_joints)
    for frame in range(t_count - 1, 1, -1):
        state_a = back[frame, joint_ids, state_b, state_c].astype(np.int64)
        choices[frame - 2] = state_a
        state_b, state_c = state_a, state_b
    return choices


def _switch_rate(choices: np.ndarray) -> float:
    choices = np.asarray(choices, dtype=int)
    if len(choices) < 2:
        return 0.0
    return float(np.mean(choices[1:] != choices[:-1]))


def _evaluate_choices(
    item: dict[str, Any],
    choices: np.ndarray,
    names: list[str],
    info: dict[str, Any],
    fps: float,
) -> tuple[dict[str, float], np.ndarray]:
    pose = apply_candidate_choices(item["candidates"], choices, names)
    spatial = _sequence_spatial_metrics(item, choices, names)
    temporal, supported, possible = temporal_metric_values(
        pose,
        info["gt_xy"],
        info["valid_gt"],
        info["diagonal"],
        fps,
    )
    metrics = {
        **spatial,
        **temporal,
        "temporal_support_coverage": supported / possible if possible else float("nan"),
        "switch_rate": _switch_rate(choices),
    }
    return metrics, pose


def _choose_parameters(
    rows: list[dict[str, Any]],
    spatial_rows: list[dict[str, Any]],
    allowed_lambdas: list[float],
    pck_tolerance: float,
    minimum_coverage: float,
    max_false_correction_increase: float,
    max_temporal_support_drop: float,
) -> tuple[float, float, pd.DataFrame, dict[str, Any]]:
    """Choose (lambda, rho) using inner validation sequences only.

    In addition to the sequence-macro metric means, record the number of
    distinct validation sequences contributing a finite value to every metric.
    The fixed-margin spatial baseline counts are recorded separately.

    This is an audit addition; it does not change the original feasibility
    constraints or parameter ranking.
    """
    base_frame = pd.DataFrame(spatial_rows)
    frame = pd.DataFrame(rows)
    if base_frame.empty or frame.empty:
        return (
            float("nan"),
            float("nan"),
            pd.DataFrame(),
            {
                "selection_status": "empty_inner_validation_data",
                "n_feasible_parameter_pairs": 0,
            },
        )

    base = base_frame.mean(numeric_only=True)
    sweep_rows: list[dict[str, Any]] = []

    filtered = frame[frame["temporal_weight"].isin(allowed_lambdas)]
    for (lam, rho), group in filtered.groupby(
        ["temporal_weight", "switch_penalty"]
    ):
        row: dict[str, Any] = {
            "temporal_weight": float(lam),
            "switch_penalty": float(rho),
            "n_validation_sequences": int(group["sequence"].nunique()),
        }

        # Candidate-configuration metric means and valid-sequence counts.
        for metric in ALL_METRIC_NAMES:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce")
                valid_mask = values.notna()
                row[f"{metric}_mean"] = (
                    float(values.mean()) if valid_mask.any() else float("nan")
                )
                row[f"n_valid_sequences_{metric}"] = int(
                    group.loc[valid_mask, "sequence"].nunique()
                )
                row[f"n_total_sequences_{metric}"] = int(
                    group["sequence"].nunique()
                )

            # The baseline is the same fixed-margin spatial method for every
            # candidate pair. Its per-metric denominator is therefore constant
            # across pairs, but is written explicitly for auditability.
            if metric in base_frame.columns:
                base_values = pd.to_numeric(base_frame[metric], errors="coerce")
                base_valid_mask = base_values.notna()
                row[f"base_n_valid_sequences_{metric}"] = int(
                    base_frame.loc[base_valid_mask, "sequence"].nunique()
                )
                row[f"base_n_total_sequences_{metric}"] = int(
                    base_frame["sequence"].nunique()
                )

        # Guardrails preserve the original selection protocol.
        row["pck_floor"] = float(
            base["pck_torso_0.20"] - pck_tolerance
        )
        row["coverage_floor"] = float(
            max(minimum_coverage, base["coverage"] - 0.01)
        )
        row["false_correction_ceiling"] = float(
            base["false_correction_rate"] + max_false_correction_increase
        )
        row["temporal_support_floor"] = float(
            base["temporal_support_coverage"] - max_temporal_support_drop
        )
        row["feasible"] = bool(
            np.isfinite(row.get("pck_torso_0.20_mean", np.nan))
            and row["pck_torso_0.20_mean"] >= row["pck_floor"]
            and row.get("coverage_mean", -np.inf) >= row["coverage_floor"]
            and row.get("false_correction_rate_mean", np.inf)
            <= row["false_correction_ceiling"]
            and row.get("temporal_support_coverage_mean", -np.inf)
            >= row["temporal_support_floor"]
            and np.isfinite(
                row.get("acceleration_error_norm_s2_mean", np.nan)
            )
        )
        # Exactly one row per (lambda, rho).
        sweep_rows.append(row)

    sweep = pd.DataFrame(sweep_rows)
    if sweep.empty:
        return (
            float("nan"),
            float("nan"),
            sweep,
            {
                "selection_status": "empty_parameter_sweep",
                "n_feasible_parameter_pairs": 0,
            },
        )

    feasible = sweep[sweep["feasible"]].copy()
    if feasible.empty:
        return (
            float("nan"),
            float("nan"),
            sweep,
            {
                "selection_status": "no_feasible_parameter_pair",
                "n_feasible_parameter_pairs": 0,
                "selection_rule": (
                    "no parameter pair satisfied the inner-validation guardrails"
                ),
            },
        )

    feasible = feasible.sort_values(
        [
            "acceleration_error_norm_s2_mean",
            "velocity_error_norm_s_mean",
            "temporal_jitter_norm_s2_mean",
            "temporal_weight",
            "switch_penalty",
        ],
        ascending=[True, True, True, True, True],
    )
    chosen = feasible.iloc[0]
    decision = {
        "selected_temporal_weight": float(chosen["temporal_weight"]),
        "selected_switch_penalty": float(chosen["switch_penalty"]),
        "inner_spatial_pck": float(base["pck_torso_0.20"]),
        "inner_spatial_coverage": float(base["coverage"]),
        "inner_spatial_false_correction": float(base["false_correction_rate"]),
        "inner_spatial_temporal_support": float(base["temporal_support_coverage"]),
        "inner_selected_pck": float(chosen["pck_torso_0.20_mean"]),
        "inner_selected_acceleration_error": float(
            chosen["acceleration_error_norm_s2_mean"]
        ),
        "inner_selected_velocity_error": float(
            chosen["velocity_error_norm_s_mean"]
        ),
        "inner_selected_switch_rate": float(chosen["switch_rate_mean"]),
        "n_feasible_parameter_pairs": int(len(feasible)),
        "selection_rule": (
            "minimize inner sequence-macro acceleration error subject to PCK "
            "within tolerance, coverage floor, false-correction ceiling, and "
            "temporal-support floor; velocity, jitter, lower temporal weight, "
            "and lower switch penalty break ties"
        ),
    }
    return (
        float(chosen["temporal_weight"]),
        float(chosen["switch_penalty"]),
        sweep,
        decision,
    )


def _paired_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _common_temporal_support(
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
        default=Path("outputs/pattern_prior_second_order_routing"),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--spatial-ridge", type=float, default=None)
    parser.add_argument("--max-pattern-centers-per-sequence", type=int, default=64)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--inner-validation-sequences-per-fold", type=int, default=1)
    parser.add_argument("--temporal-weights", default="0,0.1,0.5")
    parser.add_argument("--switch-penalties", default="0,0.005")
    parser.add_argument("--pck-tolerance", type=float, default=0.005)
    parser.add_argument("--minimum-coverage", type=float, default=0.90)
    parser.add_argument(
        "--maximum-false-correction-increase", type=float, default=0.01
    )
    parser.add_argument("--maximum-temporal-support-drop", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261128)
    args = parser.parse_args()

    if args.fps <= 0 or args.max_pattern_centers_per_sequence < 8:
        parser.error("FPS must be positive and pattern centers per sequence >=8")
    if args.inner_folds < 2 or args.inner_validation_sequences_per_fold < 1:
        parser.error("inner folds >=2 and validation sequences per fold >=1")
    if args.bootstrap_samples < 100 or args.pck_tolerance < 0:
        parser.error("bootstrap samples >=100 and PCK tolerance non-negative")
    if not (0 < args.minimum_coverage <= 1):
        parser.error("minimum coverage must be in (0,1]")
    if (
        args.maximum_false_correction_increase < 0
        or args.maximum_temporal_support_drop < 0
    ):
        parser.error("guardrail allowances must be non-negative")

    temporal_weights = sorted(
        set(float(x.strip()) for x in args.temporal_weights.split(",") if x.strip())
    )
    switch_penalties = sorted(
        set(float(x.strip()) for x in args.switch_penalties.split(",") if x.strip())
    )
    if not temporal_weights or not switch_penalties:
        parser.error("temporal weights and switch penalties must be non-empty")
    if any(
        x < 0 or not np.isfinite(x)
        for x in temporal_weights + switch_penalties
    ):
        parser.error("temporal weights and switch penalties must be finite and non-negative")
    if 0.0 not in temporal_weights:
        temporal_weights.insert(0, 0.0)

    started = time.time()
    source_out = args.source_out.resolve()
    dataset_root = args.dataset_root.resolve()
    meta_path = source_out / "run_metadata.json"
    cache_dir = source_out / "feature_cache"
    margin_path = source_out / "nested_threshold_selection.csv"
    for path in (meta_path, cache_dir, margin_path):
        if not path.exists():
            raise FileNotFoundError(f"Required source artifact not found: {path}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    selected = [str(s).zfill(4) for s in metadata["selected_sequences"]]
    conditions = list(metadata["conditions"])
    names = list(metadata["candidate_order"])
    signature = str(metadata["feature_cache"]["signature"])
    ridge = float(
        args.spatial_ridge
        if args.spatial_ridge is not None
        else metadata.get("ridge", 10.0)
    )
    if not names or names[0] != "raw" or len(selected) < 3:
        raise ValueError(
            "Metadata must contain raw-first candidates and at least three sequences"
        )

    old_margins = pd.read_csv(
        margin_path, dtype={"outer_heldout_sequence": str}
    )
    old_margins["outer_heldout_sequence"] = (
        old_margins["outer_heldout_sequence"].str.zfill(4)
    )
    margin_map = {
        (str(r.condition), str(r.outer_heldout_sequence)): float(r.selected_margin)
        for r in old_margins.itertuples(index=False)
    }

    print(
        "Validating selected training sequences and cached ground truth...",
        flush=True,
    )
    infos: dict[str, dict[str, Any]] = {}
    for seq in selected:
        clean_path = cache_dir / f"{seq}_{conditions[0]}.npz"
        clean_item = _load_cache_item(clean_path, names, signature)
        infos[seq] = _load_sequence_info(
            dataset_root, seq, clean_item["gt_xy"]
        )
        if not np.allclose(
            infos[seq]["gt_xy"], clean_item["gt_xy"], equal_nan=True
        ):
            raise ValueError(f"GT/cache mismatch for {seq}")
        del clean_item

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inner_sweep_all: list[dict[str, Any]] = []
    selection_all: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    k_count = len(names)

    for condition_idx, condition in enumerate(conditions):
        print(
            f"\n[{condition}] loading cache and computing sequence-level pattern targets",
            flush=True,
        )
        cache: dict[tuple[str, str], dict[str, Any]] = {}
        pattern_targets: dict[str, np.ndarray] = {}
        for seq_idx, seq in enumerate(selected):
            item = _load_cache_item(
                cache_dir / f"{seq}_{condition}.npz", names, signature
            )
            if not np.allclose(
                item["gt_xy"], infos[seq]["gt_xy"], equal_nan=True
            ):
                raise ValueError(f"GT/cache mismatch for {seq}/{condition}")
            cache[(seq, condition)] = item
            valid_centers = _sample_gt_valid_centers(
                infos[seq]["valid_gt"],
                args.max_pattern_centers_per_sequence,
                args.seed + condition_idx * 100003 + seq_idx * 1009,
            )
            pattern_targets[seq] = _sequence_pattern_targets(
                item, infos[seq], names, valid_centers, args.fps
            )
            print(
                f"  {seq}: {len(valid_centers)} sampled centers; "
                f"{k_count**3} triplet patterns",
                flush=False,
            )

        for outer_idx, heldout in enumerate(selected):
            outer_training = [seq for seq in selected if seq != heldout]
            folds = _action_balanced_folds(
                cache,
                condition,
                outer_training,
                args.inner_folds,
                args.seed + 1009 * outer_idx + 9176 * condition_idx,
            )
            inner_config_rows: list[dict[str, Any]] = []
            inner_spatial_rows: list[dict[str, Any]] = []

            for fold_idx, validation_sequences in enumerate(folds):
                val_set = set(validation_sequences)
                inner_training = [
                    seq for seq in outer_training if seq not in val_set
                ]
                if len(inner_training) < 3:
                    raise RuntimeError(
                        "Too few sequences remain for inner spatial/pattern training"
                    )
                inner_spatial_model = _fit_selector_for_sequences(
                    cache,
                    condition,
                    inner_training,
                    ridge,
                    f"pattern-prior inner spatial fold={fold_idx}; "
                    f"outer={heldout}; condition={condition}",
                )
                inner_prior, _, _ = _make_pattern_prior(
                    inner_training, pattern_targets
                )

                # Deterministic validation subset; when the requested size is
                # >= the fold size, every sequence in that fold is evaluated.
                n_val = min(
                    args.inner_validation_sequences_per_fold,
                    len(validation_sequences),
                )
                rng = np.random.default_rng(
                    args.seed
                    + 500009 * outer_idx
                    + 1009 * fold_idx
                    + condition_idx
                )
                if n_val < len(validation_sequences):
                    eval_sequences = sorted(
                        rng.choice(
                            validation_sequences,
                            size=n_val,
                            replace=False,
                        ).tolist()
                    )
                else:
                    eval_sequences = list(validation_sequences)

                for val_seq in eval_sequences:
                    val_item = cache[(val_seq, condition)]
                    pred = predict_candidate_errors(
                        inner_spatial_model, val_item["features"]
                    )
                    baseline_choices = select_candidates(pred, SPATIAL_MARGIN)
                    baseline_metrics, _ = _evaluate_choices(
                        val_item,
                        baseline_choices,
                        names,
                        infos[val_seq],
                        args.fps,
                    )
                    inner_spatial_rows.append(
                        {"sequence": val_seq, **baseline_metrics}
                    )

                    for lam in temporal_weights:
                        for rho in switch_penalties:
                            if lam == 0.0:
                                # Keep the first-order tuning path exactly aligned
                                # with the established constant-switch router.
                                choices = route_candidate_sequence(
                                    pred,
                                    val_item["candidates"],
                                    names,
                                    min_predicted_gain=SPATIAL_MARGIN,
                                    switch_penalty=rho,
                                    mode="constant",
                                    torso_scale=val_item["torso_scale"],
                                )
                            else:
                                choices = second_order_route(
                                    pred,
                                    val_item["candidates"],
                                    names,
                                    inner_prior,
                                    temporal_weight=lam,
                                    switch_penalty=rho,
                                    min_predicted_gain=SPATIAL_MARGIN,
                                )
                            metrics, _ = _evaluate_choices(
                                val_item,
                                choices,
                                names,
                                infos[val_seq],
                                args.fps,
                            )
                            inner_config_rows.append(
                                {
                                    "sequence": val_seq,
                                    "fold": fold_idx,
                                    "temporal_weight": lam,
                                    "switch_penalty": rho,
                                    **metrics,
                                }
                            )

            # Tune first-order (lambda=0) and pattern-prior second-order
            # (lambda>0) separately, against the same fixed-margin baseline.
            first_lambdas = [x for x in temporal_weights if x == 0.0]
            second_lambdas = [x for x in temporal_weights if x > 0.0]
            if not first_lambdas:
                raise ValueError(
                    "temporal_weights must include 0 for first-order baseline"
                )
            if not second_lambdas:
                raise ValueError(
                    "At least one positive temporal weight is required"
                )

            first_weight, first_rho, first_sweep, first_decision = _choose_parameters(
                inner_config_rows,
                inner_spatial_rows,
                [0.0],
                args.pck_tolerance,
                args.minimum_coverage,
                args.maximum_false_correction_increase,
                args.maximum_temporal_support_drop,
            )
            if not np.isfinite(first_weight) or not np.isfinite(first_rho):
                first_weight, first_rho = 0.0, 0.0
                first_decision["fallback"] = (
                    "spatial_only_no_feasible_first_order_setting"
                )

            second_weight, second_rho, second_sweep, second_decision = _choose_parameters(
                inner_config_rows,
                inner_spatial_rows,
                second_lambdas,
                args.pck_tolerance,
                args.minimum_coverage,
                args.maximum_false_correction_increase,
                args.maximum_temporal_support_drop,
            )
            if not np.isfinite(second_weight) or not np.isfinite(second_rho):
                # Do not force a pattern-prior setting when no positive-lambda
                # configuration survives the guardrails. Report a first-order
                # fallback explicitly rather than failing the experiment.
                second_weight, second_rho = 0.0, first_rho
                second_decision["fallback"] = (
                    "first_order_setting_no_feasible_positive_temporal_weight"
                )
                second_decision["fallback_temporal_weight"] = 0.0
                second_decision["fallback_switch_penalty"] = first_rho

            for kind, sweep in (
                ("first_order", first_sweep),
                ("second_order", second_sweep),
            ):
                for row in sweep.to_dict(orient="records"):
                    inner_sweep_all.append(
                        {
                            "outer_heldout_sequence": heldout,
                            "condition": condition,
                            "selection_kind": kind,
                            **row,
                        }
                    )

            selection_all.append(
                {
                    "outer_heldout_sequence": heldout,
                    "condition": condition,
                    "n_inner_validation_sequences": len(
                        {r["sequence"] for r in inner_spatial_rows}
                    ),
                    "selected_spatial_margin_fixed": SPATIAL_MARGIN,
                    "first_order_selected_lambda": first_weight,
                    "first_order_selected_switch_penalty": first_rho,
                    "second_order_selected_lambda": second_weight,
                    "second_order_selected_switch_penalty": second_rho,
                    **{f"first_{k}": v for k, v in first_decision.items()},
                    **{f"second_{k}": v for k, v in second_decision.items()},
                }
            )

            # Refit only on outer-training sequences and evaluate the untouched
            # outer sequence using the inner-selected parameter pairs.
            outer_spatial_model = _fit_selector_for_sequences(
                cache,
                condition,
                outer_training,
                ridge,
                f"pattern-prior outer spatial; heldout={heldout}; "
                f"condition={condition}",
            )
            outer_prior, outer_scale, outer_pattern_support = _make_pattern_prior(
                outer_training, pattern_targets
            )
            item = cache[(heldout, condition)]
            pred = predict_candidate_errors(outer_spatial_model, item["features"])

            nested_margin = margin_map.get((condition, heldout), SPATIAL_MARGIN)
            spatial_nested_choices = select_candidates(pred, nested_margin)
            spatial_fixed_choices = select_candidates(pred, SPATIAL_MARGIN)
            first_choices = route_candidate_sequence(
                pred,
                item["candidates"],
                names,
                min_predicted_gain=SPATIAL_MARGIN,
                switch_penalty=first_rho,
                mode="constant",
                torso_scale=item["torso_scale"],
            )
            second_choices = second_order_route(
                pred,
                item["candidates"],
                names,
                outer_prior,
                temporal_weight=second_weight,
                switch_penalty=second_rho,
                min_predicted_gain=SPATIAL_MARGIN,
            )

            method_choices: dict[str, np.ndarray] = {
                "spatial_only_fixed_margin_0.01": spatial_fixed_choices,
                "spatial_only_nested_margin": spatial_nested_choices,
                "nested_first_order_constant": first_choices,
                "nested_pattern_second_order": second_choices,
            }
            method_choices.update(
                {
                    f"fixed_{name}": np.full(
                        item["scoreable"].shape, idx, dtype=np.int64
                    )
                    for idx, name in enumerate(names)
                }
            )

            method_metrics: dict[str, dict[str, float]] = {}
            method_poses: dict[str, np.ndarray] = {}
            for method, choices in method_choices.items():
                metrics, pose = _evaluate_choices(
                    item, choices, names, infos[heldout], args.fps
                )
                method_metrics[method] = metrics
                method_poses[method] = pose
                outer_rows.append(
                    {
                        "sequence": heldout,
                        "action": item["action"],
                        "condition": condition,
                        "method": method,
                        "selected_first_order_lambda": first_weight,
                        "selected_first_order_switch_penalty": first_rho,
                        "selected_second_order_lambda": second_weight,
                        "selected_second_order_switch_penalty": second_rho,
                        "pattern_prior_train_q95_scale": outer_scale,
                        "pattern_prior_mean_support": float(
                            np.mean(outer_pattern_support)
                        ),
                        **metrics,
                    }
                )

            # Paired, sequence-level comparisons use common finite temporal support.
            comparison_pairs = (
                (
                    "nested_first_order_constant",
                    "spatial_only_fixed_margin_0.01",
                ),
                (
                    "nested_pattern_second_order",
                    "spatial_only_fixed_margin_0.01",
                ),
                (
                    "nested_pattern_second_order",
                    "nested_first_order_constant",
                ),
            )
            for method, base_method in comparison_pairs:
                base_pose = method_poses[base_method]
                method_pose = method_poses[method]
                common = _common_temporal_support(
                    infos[heldout]["valid_gt"], base_pose, method_pose
                )
                base_temp = temporal_metric_values(
                    base_pose,
                    infos[heldout]["gt_xy"],
                    infos[heldout]["valid_gt"],
                    infos[heldout]["diagonal"],
                    args.fps,
                    support=common,
                )[0]
                method_temp = temporal_metric_values(
                    method_pose,
                    infos[heldout]["gt_xy"],
                    infos[heldout]["valid_gt"],
                    infos[heldout]["diagonal"],
                    args.fps,
                    support=common,
                )[0]
                comparison_name = f"{method}_minus_{base_method}"
                for metric in SPATIAL_METRIC_NAMES:
                    paired_rows.append(
                        {
                            "sequence": heldout,
                            "action": item["action"],
                            "condition": condition,
                            "comparison": comparison_name,
                            "metric": metric,
                            "delta": (
                                method_metrics[method][metric]
                                - method_metrics[base_method][metric]
                            ),
                            "n_common_temporal_triples": int(common.sum()),
                        }
                    )
                for metric in TEMPORAL_METRICS:
                    paired_rows.append(
                        {
                            "sequence": heldout,
                            "action": item["action"],
                            "condition": condition,
                            "comparison": comparison_name,
                            "metric": metric,
                            "delta": method_temp[metric] - base_temp[metric],
                            "n_common_temporal_triples": int(common.sum()),
                        }
                    )

            print(
                f"[{condition}] outer={heldout}: first-order rho={first_rho:g}; "
                f"second-order lambda={second_weight:g}, rho={second_rho:g}; "
                f"PCK={method_metrics['nested_pattern_second_order']['pck_torso_0.20']:.4f}; "
                f"accel={method_metrics['nested_pattern_second_order']['acceleration_error_norm_s2']:.4f}",
                flush=True,
            )

        # Free sequence-condition cache before proceeding to the next condition.
        del cache, pattern_targets

    outer_df = pd.DataFrame(outer_rows)
    selection_df = pd.DataFrame(selection_all)
    sweep_df = pd.DataFrame(inner_sweep_all)
    paired_df = pd.DataFrame(paired_rows)
    outer_df.to_csv(
        out / "pattern_prior_second_order_by_sequence.csv", index=False
    )
    selection_df.to_csv(
        out / "pattern_prior_second_order_parameter_selection.csv", index=False
    )
    sweep_df.to_csv(
        out / "pattern_prior_second_order_inner_sweep.csv", index=False
    )
    paired_df.to_csv(
        out / "pattern_prior_second_order_paired_by_sequence.csv", index=False
    )

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
        switch_rate_mean=("switch_rate", "mean"),
    )
    summary.to_csv(
        out / "pattern_prior_second_order_summary.csv", index=False
    )

    paired_summary_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    for (condition, comparison, metric), group in paired_df.groupby(
        ["condition", "comparison", "metric"]
    ):
        values = group["delta"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        low, high = _paired_ci(values, rng, args.bootstrap_samples)
        paired_summary_rows.append(
            {
                "condition": condition,
                "comparison": comparison,
                "metric": metric,
                "n_paired_sequences": int(len(values)),
                "delta_mean": float(values.mean()) if len(values) else float("nan"),
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_unit": "sequence",
                "interpretation": (
                    "negative favors routed method for error metrics; "
                    "positive favors routed method for PCK/coverage"
                ),
            }
        )
    paired_summary = pd.DataFrame(paired_summary_rows)
    paired_summary.to_csv(
        out / "pattern_prior_second_order_paired_summary.csv", index=False
    )

    protocol = {
        "experiment": "pattern_prior_second_order_routing_nested_loso",
        "dataset": "Penn Action",
        "official_split": (
            "selected official training sequences only; "
            "load_sequence enforces train membership"
        ),
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_order": names,
        "outer_protocol": (
            "leave-one-sequence-out; outer held-out sequence excluded from "
            "spatial fitting, pattern-prior fitting, and parameter selection"
        ),
        "inner_protocol": (
            "action-balanced folds; spatial predictor and pattern prior fit "
            "only on inner-training sequences; validation sequences are "
            "deterministically subsampled per fold, or all evaluated when "
            "the requested per-fold count covers the fold"
        ),
        "pattern_prior": (
            "for each of K^3 identity triplets, sequence-balanced mean of "
            "image-diagonal-normalized acceleration error; each sequence "
            "contributes equally; unsupported triplets assigned maximum "
            "normalized risk 1; q95 scaling fitted on training sequences only"
        ),
        "spatial_unary": (
            "LOSO Ridge-predicted torso-normalized localization error; "
            "fixed min_predicted_gain=0.01, with the same margin used for "
            "spatial-only and routed comparators"
        ),
        "objective": (
            "sum unary spatial risks + lambda times per-triple training-only "
            "pattern prior + rho times candidate identity switches"
        ),
        "dp_state": "ordered pair of previous candidate identities; transition enumerates K^3 triplets",
        "parameter_grid": {
            "temporal_weights": temporal_weights,
            "switch_penalties": switch_penalties,
        },
        "inner_validation_sequences_per_fold": args.inner_validation_sequences_per_fold,
        "inner_metric_support_audit": (
            "inner sweep records per-metric valid and total sequence counts "
            "for each parameter pair, plus fixed-margin baseline counts; "
            "selection means retain the existing NaN-skipping sequence-macro behavior"
        ),
        "guardrails": {
            "pck_tolerance": args.pck_tolerance,
            "minimum_coverage": args.minimum_coverage,
            "maximum_false_correction_increase": args.maximum_false_correction_increase,
            "maximum_temporal_support_drop": args.maximum_temporal_support_drop,
        },
        "sampling": (
            f"up to {args.max_pattern_centers_per_sequence} GT-valid frame-joint "
            "centers per training sequence; deterministic and independent of model targets"
        ),
        "fps_assumed": args.fps,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "unit": "sequence",
            "interval": "two-sided percentile 95% CI",
        },
        "official_test_split_read": False,
        "limitations": [
            "Development-only results on the selected official-training subset; no official test performance is estimated.",
            "The pattern prior is a condition-specific global prior over candidate identity triplets and is not a context-conditioned temporal risk model.",
            "Inner hyperparameter selection uses deterministic validation subsets when requested; the outer LOSO evaluation remains sequence-held-out.",
            "The fixed spatial margin 0.01 is used for the paired Spatial-only and routing comparisons; the existing nested-margin Spatial-only result is reported separately as a reference.",
            "Temporal annotations are the evaluation target, not biomechanical ground truth; FPS is assumed to be 30 unless overridden.",
            "The method is offline/non-causal because it optimizes a complete sequence and uses centered three-frame costs.",
            "Inner parameter-selection metric means use available finite sequence values; per-metric support counts are written to the inner sweep for audit.",
        ],
    }
    (out / "pattern_prior_second_order_metadata.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    print("\n=== Pattern-prior second-order routing: sequence-macro summary ===")
    print(summary.round(5).to_string(index=False))
    print("\n=== Paired comparisons (routed method minus fixed-margin Spatial-only) ===")
    print(paired_summary.round(5).to_string(index=False))
    print(f"\nSaved outputs under {out}")
    print("Cache-only nested development experiment; official test split was not read.")
    print(f"Runtime seconds: {time.time() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
