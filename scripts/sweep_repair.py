#!/usr/bin/env python
"""Sweep temporal smoothing strengths using already-extracted FrameTrust predictions.

This script does not rerun pose estimation. It uses landmarks_<condition>.csv files
from a completed run, scores manual ground truth, and reports temporal jitter only
for the selected anatomical profile.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from frametrust.ground_truth import evaluate_ground_truth, read_ground_truth_csv
from frametrust.metrics import temporal_jitter
from frametrust.pose import frame_to_arrays
from frametrust.repair import (
    confidence_aware_repair,
    confidence_aware_symmetric_repair,
    interpolate_missing,
)

PROFILES = {
    "upper_body": [0, 11, 12, 13, 14, 15, 16],
    "squat": [23, 24, 25, 26, 27, 28, 31, 32],
    "full_body": [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
}
# Include current defaults and progressively weaker smoothing settings.
ALPHA_PAIRS = [
    (0.20, 0.75),
    (0.40, 0.85),
    (0.60, 0.90),
    (0.80, 1.00),
]
CONDITIONS = ["clean", "frame_drop", "occlusion", "blur", "low_resolution"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep FrameTrust smoothing strengths without rerunning pose inference."
    )
    parser.add_argument(
        "--landmarks-dir", type=Path, default=Path("outputs/annotated_upper_body_v2"),
        help="Directory containing landmarks_<condition>.csv from a completed run",
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=Path("data/annotations_upper_body_v2.csv"),
        help="Manual annotation CSV",
    )
    parser.add_argument(
        "--profile", choices=sorted(PROFILES), default="upper_body",
        help="Anatomical region used for temporal-jitter calculation",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/repair_sweep_upper_body.csv"),
        help="CSV path for the sweep results",
    )
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Visibility threshold used for repair (default: 0.35)")
    parser.add_argument("--pck-fraction", type=float, default=0.1,
                        help="PCK threshold relative to annotated body-box diagonal")
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    if not 0 < args.pck_fraction <= 1:
        parser.error("--pck-fraction must be in (0, 1]")
    return args


def evaluate_common_support_metrics(
    annotations: pd.DataFrame,
    frame_indices: object,
    raw_estimate: object,
    estimate: object,
    pck_fraction: float = 0.1,
) -> dict[str, float | int]:
    """Compare a method and raw output on the same raw-detected labeled points.

    This avoids treating the raw method's missing predictions as though they were
    simply absent from its mean localization error while repaired methods are
    scored on additionally interpolated points. Overall PCK and coverage from
    evaluate_ground_truth remain the all-label metrics.
    """
    import numpy as np

    indices = np.asarray(frame_indices, dtype=int)
    raw = np.asarray(raw_estimate, dtype=float)
    pred = np.asarray(estimate, dtype=float)
    row_for_frame = {int(frame_id): row for row, frame_id in enumerate(indices)}
    visible = annotations[
        (annotations["visible"] == 1)
        & annotations["frame_index"].isin(row_for_frame)
    ]

    total = 0
    detected = 0
    correct = 0
    image_errors: list[float] = []
    body_errors: list[float] = []
    for frame_id, group in visible.groupby("frame_index", sort=True):
        points = group[["x", "y"]].to_numpy(dtype=float)
        if len(points) < 2:
            continue
        body_scale = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        if not np.isfinite(body_scale) or body_scale <= 1e-12:
            continue
        row = row_for_frame[int(frame_id)]
        for item in group.itertuples(index=False):
            joint = int(item.joint_index)
            raw_xy = raw[row, joint, :2]
            if not np.isfinite(raw_xy).all():
                continue
            total += 1
            pred_xy = pred[row, joint, :2]
            if not np.isfinite(pred_xy).all():
                continue
            detected += 1
            err = float(np.linalg.norm(pred_xy - np.array([item.x, item.y], dtype=float)))
            body_err = err / body_scale
            image_errors.append(err)
            body_errors.append(body_err)
            if body_err <= pck_fraction:
                correct += 1

    return {
        "gt_common_support_points": int(total),
        "gt_common_support_coverage": float(detected / total) if total else float("nan"),
        "gt_common_support_mean_error_normalized_image": float(np.mean(image_errors)) if image_errors else float("nan"),
        "gt_common_support_mean_error_body_diagonal": float(np.mean(body_errors)) if body_errors else float("nan"),
        "gt_common_support_pck": float(correct / total) if total else float("nan"),
    }


def main() -> int:
    args = parse_args()
    ground_truth = read_ground_truth_csv(args.ground_truth)
    target_joints = PROFILES[args.profile]
    rows: list[dict[str, float | int | str]] = []

    for condition in CONDITIONS:
        source = args.landmarks_dir / f"landmarks_{condition}.csv"
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing {source}. Run scripts/run_experiment.py first to extract poses."
            )
        frame = pd.read_csv(source)
        coords, confidence = frame_to_arrays(frame)
        frame_indices = frame["frame_index"].to_numpy(dtype=int)

        estimates = [
            (
                "raw", None, None, coords,
            ),
            (
                "linear_interpolation", None, None,
                interpolate_missing(coords, confidence, threshold=args.threshold),
            ),
        ]
        for alpha_low, alpha_high in ALPHA_PAIRS:
            name = f"confidence_aware_a{alpha_low:.2f}_h{alpha_high:.2f}"
            estimate = confidence_aware_repair(
                coords,
                confidence,
                threshold=args.threshold,
                alpha_low=alpha_low,
                alpha_high=alpha_high,
            )
            estimates.append((name, alpha_low, alpha_high, estimate))
            symmetric_name = f"symmetric_{name}"
            symmetric_estimate = confidence_aware_symmetric_repair(
                coords,
                confidence,
                threshold=args.threshold,
                alpha_low=alpha_low,
                alpha_high=alpha_high,
            )
            estimates.append((symmetric_name, alpha_low, alpha_high, symmetric_estimate))

        for method, alpha_low, alpha_high, estimate in estimates:
            scores = evaluate_ground_truth(
                ground_truth,
                frame_indices,
                estimate,
                pck_fraction=args.pck_fraction,
            )
            rows.append({
                "corruption": condition,
                "method": method,
                "alpha_low": alpha_low,
                "alpha_high": alpha_high,
                "frames": int(len(frame)),
                "profile": args.profile,
                "profile_temporal_jitter_normalized_per_frame2": temporal_jitter(
                    estimate[:, target_joints, :]
                ),
                **scores,
                **evaluate_common_support_metrics(
                    ground_truth,
                    frame_indices,
                    coords,
                    estimate,
                    pck_fraction=args.pck_fraction,
                ),
            })

    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    columns = [
        "corruption", "method",
        "profile_temporal_jitter_normalized_per_frame2",
        "gt_prediction_coverage", "gt_mean_error_body_diagonal", "gt_pck",
        "gt_common_support_points", "gt_common_support_coverage",
        "gt_common_support_mean_error_body_diagonal", "gt_common_support_pck",
    ]
    print(result[columns].to_string(index=False))
    print(f"Saved smoothing sweep to {args.out}")
    print("Interpret jointly: lower jitter is not sufficient if ground-truth error or PCK worsens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
