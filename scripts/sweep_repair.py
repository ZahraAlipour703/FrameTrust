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
from frametrust.repair import confidence_aware_repair, interpolate_missing

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
            })

    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    columns = [
        "corruption", "method",
        "profile_temporal_jitter_normalized_per_frame2",
        "gt_prediction_coverage", "gt_mean_error_body_diagonal", "gt_pck",
    ]
    print(result[columns].to_string(index=False))
    print(f"Saved smoothing sweep to {args.out}")
    print("Interpret jointly: lower jitter is not sufficient if ground-truth error or PCK worsens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
