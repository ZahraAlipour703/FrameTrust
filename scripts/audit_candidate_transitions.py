#!/usr/bin/env python3
"""Audit temporal switching in the nested sequence-cross-fitted selector.

Reconstructs the *nested* selector choices from the compatible feature cache
and nested_threshold_selection.csv. It does not use the fixed-margin
per_point_choices.csv, which is not the selector used by the corrected
nested temporal evaluation.

The audit is descriptive/diagnostic. Ground-truth labels are used only to
measure localization and temporal outcomes after choices are reconstructed;
they never affect the held-out sequence's selector choices.
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
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from frametrust.selective import (  # noqa: E402
    apply_candidate_choices,
    fit_error_predictor,
    predict_candidate_errors,
    select_candidates,
)
from penn_action_development import JOINT_NAMES, load_sequence  # noqa: E402
from frametrust.transition_audit import (  # noqa: E402
    candidate_run_lengths,
    conditional_velocity_metrics,
    summarize_choice_sequence,
    transition_masks,
    transition_matrix_rows,
)
from evaluate_temporal_fidelity import (  # noqa: E402
    _bootstrap_mean_ci,
    _load_cache_item,
    temporal_ground_truth_mask,
    temporal_metric_values,
)

TEMPORAL_METRICS = (
    "temporal_jitter_norm_s2",
    "velocity_error_norm_s",
    "acceleration_error_norm_s2",
)


def conditional_temporal_metrics(
    pose: np.ndarray,
    gt_xy: np.ndarray,
    valid_gt: np.ndarray,
    frame_diagonal: float,
    fps: float,
    group_support: np.ndarray,
) -> tuple[dict[str, float], int]:
    """Evaluate temporal metrics on a supplied subset of candidate-state triples."""
    values, n_supported, _ = temporal_metric_values(
        pose, gt_xy, valid_gt, frame_diagonal, fps, support=group_support,
    )
    return values, int(n_supported)


def _sequence_temporal_row(
    sequence: str,
    action: str,
    condition: str,
    group: str,
    values: dict[str, float],
    n_triples: int,
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "action": action,
        "condition": condition,
        "choice_state": group,
        "n_temporal_triples": n_triples,
        **values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Existing selective-repair output directory with run_metadata.json, nested_threshold_selection.csv and feature_cache/.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    if args.fps <= 0 or args.bootstrap_samples < 100:
        parser.error("--fps must be positive and --bootstrap-samples must be >=100")

    out = args.out.resolve()
    metadata_path = out / "run_metadata.json"
    margin_path = out / "nested_threshold_selection.csv"
    cache_dir = out / "feature_cache"
    for path in (metadata_path, margin_path, cache_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required cached run artifact not found: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = [str(x).zfill(4) for x in metadata["selected_sequences"]]
    conditions = list(metadata["conditions"])
    candidate_names = list(metadata["candidate_order"])
    signature = str(metadata["feature_cache"]["signature"])
    ridge = float(metadata.get("ridge", 10.0))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("run_metadata selected_sequences must be non-empty and unique")
    if not candidate_names or candidate_names[0] != "raw":
        raise ValueError("candidate_order must begin with 'raw'")

    margins_df = pd.read_csv(margin_path, dtype={"outer_heldout_sequence": str})
    required_margin_cols = {"condition", "outer_heldout_sequence", "selected_margin"}
    if missing := required_margin_cols - set(margins_df.columns):
        raise ValueError(f"nested_threshold_selection.csv is missing columns: {sorted(missing)}")
    margins_df["outer_heldout_sequence"] = margins_df["outer_heldout_sequence"].str.zfill(4)
    if margins_df.duplicated(["condition", "outer_heldout_sequence"]).any():
        raise ValueError("nested_threshold_selection.csv has duplicate condition/outer-sequence rows")
    margin_map = {
        (str(row.condition), str(row.outer_heldout_sequence)): float(row.selected_margin)
        for row in margins_df.itertuples(index=False)
    }

    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for sequence in selected:
        for condition in conditions:
            cache_path = cache_dir / f"{sequence}_{condition}.npz"
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing feature cache item: {cache_path}")
            cache[(sequence, condition)] = _load_cache_item(
                cache_path, candidate_names, signature,
            )

    sequence_rows: list[dict[str, Any]] = []
    dwell_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    dataset_root = args.dataset_root.resolve()

    for condition_idx, condition in enumerate(conditions, start=1):
        for sequence_idx, heldout in enumerate(selected, start=1):
            item = cache[(heldout, condition)]
            try:
                margin = margin_map[(condition, heldout)]
            except KeyError as exc:
                raise ValueError(f"Missing nested margin for {condition}/{heldout}") from exc

            # Refit exactly as the nested temporal evaluator: the held-out sequence
            # contributes neither labels nor features to its selector fit.
            train_x: list[np.ndarray] = []
            train_y: list[np.ndarray] = []
            for train_sequence in selected:
                if train_sequence == heldout:
                    continue
                train_item = cache[(train_sequence, condition)]
                mask = train_item["scoreable"]
                if mask.any():
                    train_x.append(train_item["features"][mask])
                    train_y.append(train_item["errors"][mask])
            if not train_x or sum(len(x) for x in train_x) < 100:
                raise RuntimeError(f"Insufficient outer-training points for {heldout}/{condition}")
            model = fit_error_predictor(
                np.concatenate(train_x, axis=0),
                np.concatenate(train_y, axis=0),
                ridge=ridge,
            )
            predicted_errors = predict_candidate_errors(model, item["features"])
            choices = select_candidates(predicted_errors, margin)
            selected_pose = apply_candidate_choices(item["candidates"], choices, candidate_names)

            n_frames, n_joints = choices.shape
            data = load_sequence(dataset_root, heldout)
            if len(data["frames"]) != n_frames:
                raise ValueError(f"Frame/cache length mismatch for {heldout}")
            if not np.allclose(data["gt_xy"], item["gt_xy"], equal_nan=True):
                raise ValueError(f"Ground-truth/cache mismatch for {heldout}")
            first_frame = cv2.imread(str(data["frames"][0]))
            if first_frame is None:
                raise ValueError(f"Could not decode first frame for {heldout}")
            height, width = first_frame.shape[:2]
            diagonal = float(np.hypot(width, height))
            valid_gt = temporal_ground_truth_mask(
                data["gt_xy"], data["gt_vis"], width=width, height=height,
            )

            summary = summarize_choice_sequence(choices, candidate_names, item["scoreable"])
            sequence_rows.append({
                "sequence": heldout,
                "action": item["action"],
                "condition": condition,
                "selected_margin": margin,
                **summary,
            })
            matrix_rows.extend(transition_matrix_rows(choices, candidate_names, heldout, condition))

            for event_t in range(1, n_frames):
                changed_joints = np.flatnonzero(choices[event_t] != choices[event_t - 1])
                for joint in changed_joints:
                    old_idx = int(choices[event_t - 1, joint])
                    new_idx = int(choices[event_t, joint])
                    if old_idx == 0 and new_idx > 0:
                        event_type = "repair_entry"
                    elif old_idx > 0 and new_idx == 0:
                        event_type = "repair_exit"
                    else:
                        event_type = "repair_method_switch"
                    event_rows.append({
                        "sequence": heldout,
                        "action": item["action"],
                        "condition": condition,
                        "transition_at_frame": event_t,
                        "joint_index": int(joint),
                        "joint_name": JOINT_NAMES[int(joint)] if int(joint) < len(JOINT_NAMES) else str(joint),
                        "from_candidate": candidate_names[old_idx],
                        "to_candidate": candidate_names[new_idx],
                        "transition_type": event_type,
                        "torso_scoreable_at_both_frames": bool(item["scoreable"][event_t - 1, joint] and item["scoreable"][event_t, joint]),
                    })

            runs = candidate_run_lengths(choices, candidate_names)
            for candidate in candidate_names:
                lengths = np.asarray(runs[candidate], dtype=float)
                dwell_rows.append({
                    "sequence": heldout,
                    "action": item["action"],
                    "condition": condition,
                    "candidate": candidate,
                    "n_runs": int(len(lengths)),
                    "n_selected_frame_joints": int(lengths.sum()) if len(lengths) else 0,
                    "mean_dwell_frames": float(lengths.mean()) if len(lengths) else float("nan"),
                    "median_dwell_frames": float(np.median(lengths)) if len(lengths) else float("nan"),
                    "p90_dwell_frames": float(np.quantile(lengths, 0.90)) if len(lengths) else float("nan"),
                    "max_dwell_frames": int(lengths.max()) if len(lengths) else 0,
                })

            adjacent, transition_triples, stable_triples = transition_masks(choices)
            gt_triples = (
                valid_gt[:-2] & valid_gt[1:-1] & valid_gt[2:]
                if n_frames >= 3 else np.zeros((0, n_joints), dtype=bool)
            )
            finite_triples = (
                np.isfinite(selected_pose[:-2]).all(axis=-1)
                & np.isfinite(selected_pose[1:-1]).all(axis=-1)
                & np.isfinite(selected_pose[2:]).all(axis=-1)
                if n_frames >= 3 else np.zeros((0, n_joints), dtype=bool)
            )
            transition_support = gt_triples & finite_triples & transition_triples
            stable_support = gt_triples & finite_triples & stable_triples
            for group, support in (("transition", transition_support), ("stable", stable_support)):
                metric_values, n_supported = conditional_temporal_metrics(
                    selected_pose, data["gt_xy"], valid_gt, diagonal, args.fps, support,
                )
                conditional_rows.append(_sequence_temporal_row(
                    heldout, item["action"], condition, group, metric_values, n_supported,
                ))

            # Directly test whether a candidate-state boundary coincides with a
            # discontinuity in one-step motion, using only GT-valid finite pairs.
            interval_gt = (valid_gt[:-1] & valid_gt[1:]
                           if n_frames >= 2 else np.zeros((0, n_joints), dtype=bool))
            interval_finite = (
                np.isfinite(selected_pose[:-1]).all(axis=-1)
                & np.isfinite(selected_pose[1:]).all(axis=-1)
                if n_frames >= 2 else np.zeros((0, n_joints), dtype=bool)
            )
            switch_intervals = interval_gt & interval_finite & adjacent
            stable_intervals = interval_gt & interval_finite & ~adjacent
            for group, support in (("switch_interval", switch_intervals), ("stable_interval", stable_intervals)):
                velocity_values, n_intervals = conditional_velocity_metrics(
                    selected_pose, data["gt_xy"], valid_gt, diagonal, args.fps, support,
                )
                interval_rows.append({
                    "sequence": heldout, "action": item["action"], "condition": condition,
                    "interval_state": group, "n_intervals": n_intervals, **velocity_values,
                })

            # Additional localization diagnostic: compare GT-scoreable frame/joints
            # adjacent to a state change with those not adjacent to any state change.
            frame_transition = np.zeros((n_frames, n_joints), dtype=bool)
            if n_frames >= 2:
                frame_transition[1:] |= adjacent
                frame_transition[:-1] |= adjacent
            selected_actual_errors = np.take_along_axis(
                item["errors"], choices[..., None], axis=-1,
            )[..., 0]
            local_mask = np.asarray(item["scoreable"], dtype=bool) & np.isfinite(selected_actual_errors)
            transition_local = local_mask & frame_transition
            stable_local = local_mask & ~frame_transition
            sequence_rows[-1].update({
                "n_transition_adjacent_torso_scoreable_frame_joints": int(transition_local.sum()),
                "n_stable_torso_scoreable_frame_joints": int(stable_local.sum()),
                "transition_frame_torso_norm_error_mean": (
                    float(selected_actual_errors[transition_local].mean()) if transition_local.any() else float("nan")
                ),
                "stable_frame_torso_norm_error_mean": (
                    float(selected_actual_errors[stable_local].mean()) if stable_local.any() else float("nan")
                ),
            })

            print(
                f"[{condition_idx}/{len(conditions)}] {condition} [{sequence_idx}/{len(selected)}] {heldout}: "
                f"switches={summary['n_candidate_transitions']}, "
                f"rate={summary['transitions_per_1000_joint_steps']:.2f}/1000, "
                f"transition triples={int(transition_support.sum())}, stable triples={int(stable_support.sum())}",
                flush=True,
            )

    seq_df = pd.DataFrame(sequence_rows)
    dwell_df = pd.DataFrame(dwell_rows)
    matrix_df = pd.DataFrame(matrix_rows)
    event_df = pd.DataFrame(event_rows)
    temporal_df = pd.DataFrame(conditional_rows)
    interval_df = pd.DataFrame(interval_rows)

    seq_df.to_csv(out / "candidate_transition_sequence_summary.csv", index=False)
    dwell_df.to_csv(out / "candidate_dwell_by_sequence.csv", index=False)
    matrix_df.to_csv(out / "candidate_transition_matrix_by_sequence.csv", index=False)
    event_df.to_csv(out / "candidate_transition_events.csv", index=False)
    temporal_df.to_csv(out / "candidate_transition_temporal_by_sequence.csv", index=False)
    interval_df.to_csv(out / "candidate_transition_velocity_by_sequence.csv", index=False)

    numeric_cols = [
        col for col in seq_df.columns
        if col not in {"sequence", "action", "condition", "selected_margin"}
        and pd.api.types.is_numeric_dtype(seq_df[col])
    ]
    cond_summary = seq_df.groupby("condition", as_index=False).agg(
        n_sequences=("sequence", "nunique"),
        **{f"{col}_sequence_macro": (col, "mean") for col in numeric_cols},
        **{f"{col}_sequence_sd": (col, "std") for col in numeric_cols},
    )
    cond_summary.to_csv(out / "candidate_transition_condition_summary.csv", index=False)

    if not matrix_df.empty:
        matrix_summary = matrix_df.groupby(
            ["condition", "from_candidate", "to_candidate"], as_index=False,
        ).agg(transition_count=("transition_count", "sum"))
    else:
        matrix_summary = pd.DataFrame(columns=["condition", "from_candidate", "to_candidate", "transition_count"])
    matrix_summary.to_csv(out / "candidate_transition_matrix_condition_summary.csv", index=False)

    temporal_summary = temporal_df.groupby(
        ["condition", "choice_state"], as_index=False,
    ).agg(
        n_sequences=("sequence", "nunique"),
        n_temporal_triples=("n_temporal_triples", "sum"),
        **{f"{metric}_sequence_macro": (metric, "mean") for metric in TEMPORAL_METRICS},
    )
    temporal_summary.to_csv(out / "candidate_transition_temporal_condition_summary.csv", index=False)

    paired_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed + 17)
    for condition in conditions:
        subset = temporal_df[temporal_df["condition"] == condition]
        for metric in TEMPORAL_METRICS:
            wide = subset.pivot(index="sequence", columns="choice_state", values=metric)
            if not {"transition", "stable"}.issubset(wide.columns):
                continue
            paired = wide[["transition", "stable"]].dropna()
            delta = (paired["transition"] - paired["stable"]).to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_mean_ci(delta, rng, args.bootstrap_samples)
            paired_rows.append({
                "condition": condition,
                "metric": metric,
                "n_paired_sequences": int(len(delta)),
                "transition_minus_stable_mean": float(delta.mean()) if len(delta) else float("nan"),
                "delta_ci95_low": ci_low,
                "delta_ci95_high": ci_high,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_unit": "sequence",
                "interpretation": "descriptive association, not causal evidence that switching creates temporal error",
            })
    pd.DataFrame(paired_rows).to_csv(out / "candidate_transition_temporal_paired.csv", index=False)

    interval_paired_rows: list[dict[str, Any]] = []
    interval_rng = np.random.default_rng(args.seed + 29)
    for condition in conditions:
        subset = interval_df[interval_df["condition"] == condition]
        for metric in (
            "one_step_velocity_error_norm_s",
            "predicted_displacement_norm_mean",
            "ground_truth_displacement_norm_mean",
        ):
            wide = subset.pivot(index="sequence", columns="interval_state", values=metric)
            if not {"switch_interval", "stable_interval"}.issubset(wide.columns):
                continue
            paired = wide[["switch_interval", "stable_interval"]].dropna()
            delta = (paired["switch_interval"] - paired["stable_interval"]).to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_mean_ci(delta, interval_rng, args.bootstrap_samples)
            interval_paired_rows.append({
                "condition": condition, "metric": metric,
                "n_paired_sequences": int(len(delta)),
                "switch_minus_stable_mean": float(delta.mean()) if len(delta) else float("nan"),
                "delta_ci95_low": ci_low, "delta_ci95_high": ci_high,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_unit": "sequence",
                "interpretation": "descriptive association, not causal evidence",
            })
    pd.DataFrame(interval_paired_rows).to_csv(
        out / "candidate_transition_velocity_paired.csv", index=False,
    )

    protocol = {
        "dataset": "Penn Action",
        "official_split": "training only; each selected sequence is loaded through the existing train-flag validator",
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_order": candidate_names,
        "selection_reconstruction": "For every outer held-out sequence and condition, refit the candidate-error predictor on all other selected sequences, then apply the margin from nested_threshold_selection.csv. Matches the nested temporal-evaluation reconstruction.",
        "heldout_ground_truth_usage": "Ground truth is used only for diagnostic outcome metrics after choices are reconstructed; it never contributes to held-out selector fitting or choice-making.",
        "transition_definition": "A transition occurs when the selected candidate identity for a joint changes between adjacent frames. Transition-containing temporal triples have at least one identity change among their two adjacent frame pairs; stable triples use one candidate identity across all three frames.",
        "temporal_metrics": "Same image-diagonal normalization, FPS, visible/in-frame GT mask, and second-difference/velocity/acceleration definitions as evaluate_temporal_fidelity.py; one-step velocity boundary diagnostics use adjacent-frame finite pairs.",
        "bootstrap_unit": "sequence",
        "causal_warning": "Transition-vs-stable comparisons are descriptive. Difficult frames, occlusion, missingness, or motion can cause both switching and temporal error; association does not establish that switching causes error.",
        "not_used": [
            "per_point_choices.csv (it contains the fixed-margin selector, not nested-threshold choices)",
            "official Penn Action test split",
        ],
        "limitations": [
            "Selected 30-sequence official-training development subset only.",
            "One pose estimator and the current finite candidate set are evaluated.",
            "Candidate identity transitions are not necessarily coordinate discontinuities; coordinate jumps must be interpreted alongside trajectory errors.",
            "Sequence-bootstrap intervals are exploratory and are not corrected for multiple comparisons.",
            "The selector uses two-sided temporal features and is offline/non-causal.",
        ],
    }
    (out / "candidate_transition_audit_metadata.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )

    print("\nCandidate-transition audit complete. Wrote:")
    for name in (
        "candidate_transition_condition_summary.csv",
        "candidate_transition_matrix_condition_summary.csv",
        "candidate_dwell_by_sequence.csv",
        "candidate_transition_temporal_condition_summary.csv",
        "candidate_transition_temporal_paired.csv",
        "candidate_transition_velocity_by_sequence.csv",
        "candidate_transition_velocity_paired.csv",
        "candidate_transition_audit_metadata.json",
    ):
        print(f"  {out / name}")
    print("Interpret transition-vs-stable results as associations, not causal effects.")
    print("Development-only. Official Penn Action test split was not read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
