#!/usr/bin/env python3
"""Counterfactual analysis of temporal errors at nested-selector switch boundaries.

Reconstructs exactly the nested outer-fold selector from feature_cache and
nested_threshold_selection.csv. At each observed candidate switch, it compares
the hybrid interval (old candidate at t-1, new candidate at t) with two
same-candidate continuation counterfactuals on that identical adjacent pair.
Ground truth is used only to score these alternatives after choices are fixed.
The best-continuation comparator is an oracle diagnostic, not a deployable model.
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
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frametrust.selective import (  # noqa: E402
    fit_error_predictor,
    predict_candidate_errors,
    select_candidates,
)
from frametrust.switch_counterfactual import (  # noqa: E402
    switch_counterfactual_events,
)
from penn_action_development import load_sequence  # noqa: E402
from evaluate_temporal_fidelity import (  # noqa: E402
    _load_cache_item,
    _bootstrap_mean_ci,
    temporal_ground_truth_mask,
)

COMPARE_METRICS = {
    "hybrid_minus_old_continuation": "hybrid_minus_old_continuation_mean",
    "hybrid_minus_new_continuation": "hybrid_minus_new_continuation_mean",
    "hybrid_minus_best_continuation": "hybrid_minus_best_continuation_mean",
}


def _summarize_by_sequence(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    valid = events[events["gt_pair_valid"] & events["hybrid_supported"]].copy()
    if valid.empty:
        return pd.DataFrame()
    grouped = valid.groupby(
        ["sequence", "action", "condition", "transition_type"], as_index=False,
    ).agg(
        n_switch_events=("transition_at_frame", "size"),
        n_old_continuation_supported=("old_continuation_supported", "sum"),
        n_new_continuation_supported=("new_continuation_supported", "sum"),
        n_both_continuations_supported=("both_continuations_supported", "sum"),
        n_hybrid_old_comparisons=("hybrid_minus_old_continuation", "count"),
        n_hybrid_new_comparisons=("hybrid_minus_new_continuation", "count"),
        n_hybrid_best_comparisons=("hybrid_minus_best_continuation", "count"),
        hybrid_velocity_error_norm_s_mean=("hybrid_velocity_error_norm_s", "mean"),
        old_continuation_velocity_error_norm_s_mean=("old_continuation_velocity_error_norm_s", "mean"),
        new_continuation_velocity_error_norm_s_mean=("new_continuation_velocity_error_norm_s", "mean"),
        best_continuation_velocity_error_norm_s_mean=("best_continuation_velocity_error_norm_s", "mean"),
        hybrid_minus_old_continuation_mean=("hybrid_minus_old_continuation", "mean"),
        hybrid_minus_new_continuation_mean=("hybrid_minus_new_continuation", "mean"),
        hybrid_minus_best_continuation_mean=("hybrid_minus_best_continuation", "mean"),
        hybrid_worse_than_both_fraction=("hybrid_worse_than_both_continuations", "mean"),
    )
    return grouped


def _paired_bootstrap(
    seq_summary: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_rows: list[dict[str, Any]] = []
    type_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for keys, group in seq_summary.groupby(["condition", "transition_type"], dropna=False):
        condition, transition_type = keys
        # Sequence-level means are the inferential units, avoiding event-level
        # pseudo-replication from many correlated switches in one sequence.
        for comparison, metric_column in COMPARE_METRICS.items():
            values = group[metric_column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            low, high = _bootstrap_mean_ci(values, rng, bootstrap_samples)
            delta = float(values.mean()) if len(values) else float("nan")
            row = {
                "condition": condition,
                "transition_type": transition_type,
                "comparison": comparison,
                "n_paired_sequences": int(len(values)),
                "delta_mean": delta,
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_unit": "sequence",
                "positive_means_hybrid_switch_has_higher_velocity_error": True,
            }
            type_rows.append(row)
            # Condition-level pool is computed separately below using all event
            # types per sequence; it is a broad summary, not a substitute for type.
    if not seq_summary.empty:
        weighted_rows: list[dict[str, Any]] = []
        for (sequence, condition), group in seq_summary.groupby(["sequence", "condition"]):
            row: dict[str, Any] = {
                "sequence": sequence,
                "condition": condition,
                "n_switch_events": int(group["n_switch_events"].sum()),
                "n_both_continuations_supported": int(group["n_both_continuations_supported"].sum()),
            }
            weight_specs = (
                ("hybrid_minus_old_continuation_mean", "n_hybrid_old_comparisons"),
                ("hybrid_minus_new_continuation_mean", "n_hybrid_new_comparisons"),
                ("hybrid_minus_best_continuation_mean", "n_hybrid_best_comparisons"),
            )
            for metric, weight_col in weight_specs:
                weights = group[weight_col].to_numpy(dtype=float)
                values = group[metric].to_numpy(dtype=float)
                finite = np.isfinite(values) & (weights > 0)
                row[metric] = (
                    float(np.average(values[finite], weights=weights[finite]))
                    if finite.any() else float("nan")
                )
            weighted_rows.append(row)
        all_types = pd.DataFrame(weighted_rows)
        for condition, group in all_types.groupby("condition"):
            for comparison, metric_column in COMPARE_METRICS.items():
                values = group[metric_column].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                low, high = _bootstrap_mean_ci(values, rng, bootstrap_samples)
                condition_rows.append({
                    "condition": condition,
                    "comparison": comparison,
                    "n_paired_sequences": int(len(values)),
                    "delta_mean": float(values.mean()) if len(values) else float("nan"),
                    "delta_ci95_low": low,
                    "delta_ci95_high": high,
                    "bootstrap_samples": bootstrap_samples,
                    "bootstrap_unit": "sequence",
                    "positive_means_hybrid_switch_has_higher_velocity_error": True,
                })
    return pd.DataFrame(condition_rows), pd.DataFrame(type_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="Existing selective-repair output with run_metadata.json, nested_threshold_selection.csv and feature_cache/.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    if args.fps <= 0 or args.bootstrap_samples < 100:
        parser.error("--fps must be positive and --bootstrap-samples must be >=100")

    out = args.out.resolve()
    metadata_path = out / "run_metadata.json"
    margin_path = out / "nested_threshold_selection.csv"
    cache_dir = out / "feature_cache"
    for required in (metadata_path, margin_path, cache_dir):
        if not required.exists():
            raise FileNotFoundError(f"Required cached artifact not found: {required}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = [str(value).zfill(4) for value in metadata["selected_sequences"]]
    conditions = list(metadata["conditions"])
    candidate_names = list(metadata["candidate_order"])
    signature = str(metadata["feature_cache"]["signature"])
    ridge = float(metadata.get("ridge", 10.0))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("run_metadata selected_sequences must be non-empty and unique")

    margin_df = pd.read_csv(margin_path, dtype={"outer_heldout_sequence": str})
    margin_df["outer_heldout_sequence"] = margin_df["outer_heldout_sequence"].str.zfill(4)
    margin_map = {
        (str(row.condition), str(row.outer_heldout_sequence)): float(row.selected_margin)
        for row in margin_df.itertuples(index=False)
    }

    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for sequence in selected:
        for condition in conditions:
            cache_path = cache_dir / f"{sequence}_{condition}.npz"
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing feature cache item: {cache_path}")
            cache[(sequence, condition)] = _load_cache_item(cache_path, candidate_names, signature)

    all_events: list[dict[str, Any]] = []
    dataset_root = args.dataset_root.resolve()
    for condition_idx, condition in enumerate(conditions, start=1):
        for sequence_idx, heldout in enumerate(selected, start=1):
            item = cache[(heldout, condition)]
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
            if not train_x or sum(len(part) for part in train_x) < 100:
                raise RuntimeError(f"Insufficient outer-training points for {heldout}/{condition}")
            model = fit_error_predictor(np.concatenate(train_x), np.concatenate(train_y), ridge=ridge)
            predicted = predict_candidate_errors(model, item["features"])
            try:
                margin = margin_map[(condition, heldout)]
            except KeyError as exc:
                raise ValueError(f"Missing nested margin for {condition}/{heldout}") from exc
            choices = select_candidates(predicted, margin)

            data = load_sequence(dataset_root, heldout)  # validates official training split
            if len(data["frames"]) != len(choices):
                raise ValueError(f"Frame/cache length mismatch for {heldout}")
            if not np.allclose(data["gt_xy"], item["gt_xy"], equal_nan=True):
                raise ValueError(f"Ground-truth/cache mismatch for {heldout}")
            first_frame = cv2.imread(str(data["frames"][0]))
            if first_frame is None:
                raise ValueError(f"Could not decode first frame for {heldout}")
            height, width = first_frame.shape[:2]
            valid_gt = temporal_ground_truth_mask(
                data["gt_xy"], data["gt_vis"], width=width, height=height,
            )
            diagonal = float(np.hypot(width, height))
            events = switch_counterfactual_events(
                item["candidates"], choices, data["gt_xy"], valid_gt,
                candidate_names, diagonal, args.fps,
                sequence=heldout, condition=condition, action=item["action"],
            )
            all_events.extend(events)
            print(
                f"[{condition_idx}/{len(conditions)}] {condition} "
                f"[{sequence_idx}/{len(selected)}] {heldout}: "
                f"{len(events)} switch events",
                flush=True,
            )

    events_df = pd.DataFrame(all_events)
    sequence_summary = _summarize_by_sequence(events_df) if not events_df.empty else pd.DataFrame()
    condition_paired, type_paired = _paired_bootstrap(
        sequence_summary, args.bootstrap_samples, args.seed,
    )
    if not events_df.empty:
        event_summary = events_df.groupby(["condition", "transition_type"], as_index=False).agg(
            n_switch_events=("transition_at_frame", "size"),
            n_gt_pair_valid_events=("gt_pair_valid", "sum"),
            n_hybrid_supported_events=("hybrid_supported", "sum"),
            n_both_continuations_supported_events=("both_continuations_supported", "sum"),
        )
    else:
        event_summary = pd.DataFrame()

    events_df.to_csv(out / "candidate_switch_counterfactual_events.csv", index=False)
    sequence_summary.to_csv(out / "candidate_switch_counterfactual_by_sequence.csv", index=False)
    condition_paired.to_csv(out / "candidate_switch_counterfactual_condition_paired.csv", index=False)
    type_paired.to_csv(out / "candidate_switch_counterfactual_type_paired.csv", index=False)
    event_summary.to_csv(out / "candidate_switch_counterfactual_event_support.csv", index=False)

    protocol = {
        "dataset": "Penn Action",
        "official_split": "training only; load_sequence validates every selected sequence train flag",
        "selected_sequences": selected,
        "conditions": conditions,
        "candidate_order": candidate_names,
        "selector_reconstruction": "outer leave-one-sequence-out refit on all other selected training sequences; use the selected margin from nested_threshold_selection.csv for the held-out sequence",
        "switch_interval_definition": "For an old->new candidate switch between t-1 and t, the hybrid output uses old(t-1) and new(t).",
        "counterfactuals": "On the exact same adjacent GT-valid interval, score old(t-1)->old(t) and new(t-1)->new(t) as same-candidate continuation alternatives.",
        "ground_truth_usage": "GT is used only after selector choices are reconstructed, to score the hybrid interval and counterfactual continuations; it does not affect choices.",
        "best_continuation_warning": "The lower-error of old/new continuation alternatives is selected using GT for diagnostic analysis only. It is an oracle comparator, not a deployable method or a fair learned baseline.",
        "primary_interpretation": "Positive hybrid-minus-continuation velocity-error means the observed switch boundary has more one-step motion error than keeping that candidate identity across the same pair.",
        "bootstrap_unit": "sequence; event metrics are first averaged within sequence to avoid event-level pseudo-replication",
        "official_test_split_read": False,
        "limitations": [
            "Development-only selected 30-sequence official-training subset.",
            "A counterfactual continuation can be undefined when that candidate is non-finite at either endpoint; support counts are reported.",
            "The oracle best-continuation comparator uses GT after the fact and must not be presented as an inference-time policy.",
            "This is a local one-step boundary diagnostic; it does not prove that switching is causal or that a transition-aware selector will improve end-to-end metrics.",
            "Sequence-bootstrap intervals are exploratory and unadjusted for multiple comparisons.",
            "The selector uses two-sided temporal features and is offline/non-causal.",
        ],
    }
    (out / "candidate_switch_counterfactual_metadata.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print("\nCounterfactual switch audit complete. Wrote:")
    for name in (
        "candidate_switch_counterfactual_event_support.csv",
        "candidate_switch_counterfactual_by_sequence.csv",
        "candidate_switch_counterfactual_condition_paired.csv",
        "candidate_switch_counterfactual_type_paired.csv",
        "candidate_switch_counterfactual_metadata.json",
    ):
        print(f"  {out / name}")
    print("Positive hybrid-minus-continuation deltas indicate higher error at the observed switch boundary.")
    print("Diagnostic only; official test split was not read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
