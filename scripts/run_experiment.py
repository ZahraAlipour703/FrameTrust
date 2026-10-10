#!/usr/bin/env python
"""Run FrameTrust on a local video."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permit direct execution from a source checkout before editable installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from frametrust.corruptions import CORRUPTIONS
from frametrust.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark 2D human-pose robustness under video degradation."
    )
    parser.add_argument("--video", required=True, type=Path, help="Path to a local video file")
    parser.add_argument("--out", type=Path, default=Path("outputs/run"), help="Output directory")
    parser.add_argument(
        "--max-frames", type=int, default=300,
        help="Maximum frames per condition (default: 300; use 0 for all frames)",
    )
    parser.add_argument(
        "--drop-every", type=int, default=5,
        help="Repeat previous frame at every Nth frame for frame_drop condition",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.35,
        help="Visibility threshold for interpolation/repair",
    )
    parser.add_argument(
        "--corruptions", default=",".join(CORRUPTIONS),
        help=f"Comma-separated conditions, chosen from: {', '.join(CORRUPTIONS)}",
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=None,
        help="Optional CSV created by scripts/annotate_video.py for accuracy evaluation",
    )
    parser.add_argument(
        "--pck-fraction", type=float, default=0.1,
        help="PCK threshold as a fraction of each frame's visible annotated body-box diagonal (default: 0.1)",
    )
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    if args.drop_every < 2:
        parser.error("--drop-every must be >= 2")
    if not 0 < args.pck_fraction <= 1:
        parser.error("--pck-fraction must be in (0, 1]")
    args.max_frames = None if args.max_frames == 0 else args.max_frames
    args.corruptions = [x.strip() for x in args.corruptions.split(",") if x.strip()]
    if not args.corruptions:
        parser.error("--corruptions must name at least one condition")
    return args


if __name__ == "__main__":
    args = parse_args()
    result = run_experiment(
        video_path=args.video,
        out_dir=args.out,
        corruptions=args.corruptions,
        max_frames=args.max_frames,
        drop_every=args.drop_every,
        confidence_threshold=args.threshold,
        ground_truth_path=args.ground_truth,
        pck_fraction=args.pck_fraction,
    )
    print(result.to_string(index=False))
