#!/usr/bin/env python
"""Create sparse, manually labeled 2D pose ground truth from a local video."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# MediaPipe Pose landmark indices.
# Squat-focused profile: lower-body landmarks relevant to a squat clip.
SQUAT_JOINTS = [
    (23, "left_hip"),
    (24, "right_hip"),
    (25, "left_knee"),
    (26, "right_knee"),
    (27, "left_ankle"),
    (28, "right_ankle"),
    (31, "left_foot_index"),
    (32, "right_foot_index"),
]

# Upper-body profile for clips where lower-body landmarks are not reliably visible.
UPPER_BODY_JOINTS = [
    (0, "nose"),
    (11, "left_shoulder"),
    (12, "right_shoulder"),
    (13, "left_elbow"),
    (14, "right_elbow"),
    (15, "left_wrist"),
    (16, "right_wrist"),
]

# Keep the broad set available for videos framed to show the full body.
FULL_BODY_JOINTS = [
    (0, "nose"),
    (11, "left_shoulder"),
    (12, "right_shoulder"),
    (13, "left_elbow"),
    (14, "right_elbow"),
    (15, "left_wrist"),
    (16, "right_wrist"),
    (23, "left_hip"),
    (24, "right_hip"),
    (25, "left_knee"),
    (26, "right_knee"),
    (27, "left_ankle"),
    (28, "right_ankle"),
]
JOINT_PROFILES = {
    "squat": SQUAT_JOINTS,
    "upper_body": UPPER_BODY_JOINTS,
    "full_body": FULL_BODY_JOINTS,
}
OUTPUT_COLUMNS = [
    "frame_index", "timestamp_seconds", "joint_index", "joint_name",
    "x", "y", "visible",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually annotate sparse MediaPipe-indexed pose landmarks in a video."
    )
    parser.add_argument("--video", required=True, type=Path, help="Original video to annotate")
    parser.add_argument(
        "--out", type=Path, default=Path("data/annotations.csv"),
        help="Output annotation CSV (default: data/annotations.csv)",
    )
    parser.add_argument(
        "--profile", choices=sorted(JOINT_PROFILES), default="squat",
        help="Landmark set: squat=lower body (default); upper_body=nose/shoulders/elbows/wrists; full_body=13 landmarks",
    )
    parser.add_argument("--stride", type=int, default=15,
                        help="Annotate every Nth frame, starting with frame 0 (default: 15)")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Inspect only the first N video frames; use 0 for the full video (default: 300)")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    args.max_frames = None if args.max_frames == 0 else args.max_frames
    return args


def annotate_frame(
    frame: np.ndarray,
    frame_index: int,
    timestamp_seconds: float,
    window_name: str,
    joints: list[tuple[int, str]],
) -> tuple[list[dict[str, int | float | str]], bool]:
    """Return annotations for one frame and whether the user requested exit."""
    height, width = frame.shape[:2]
    cursor = 0
    picked: dict[int, dict[str, int | float | str]] = {}
    state: dict[str, tuple[int, int] | None] = {"click": None}

    def on_mouse(event: int, x: int, y: int, _flags: int, _params: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.setMouseCallback(window_name, on_mouse)

    def row_for(joint_index: int, joint_name: str,
                x: float | None, y: float | None, visible: int):
        return {
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "joint_index": joint_index,
            "joint_name": joint_name,
            "x": float(x) if x is not None else float("nan"),
            "y": float(y) if y is not None else float("nan"),
            "visible": visible,
        }

    should_quit = False
    while True:
        complete = cursor >= len(joints)
        if complete:
            joint_index, joint_name = -1, "FRAME COMPLETE"
            prompt = "Enter: accept frame | U: undo last point | Q: save and quit"
        else:
            joint_index, joint_name = joints[cursor]
            prompt = "Click joint | S: not visible | U: undo | Q: save and quit"

        display = frame.copy()
        for saved in picked.values():
            if int(saved["visible"]) == 1:
                px = int(float(saved["x"]) * max(width - 1, 1))
                py = int(float(saved["y"]) * max(height - 1, 1))
                cv2.circle(display, (px, py), 5, (0, 215, 255), -1)
                cv2.putText(
                    display, str(saved["joint_name"]), (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 215, 255), 1, cv2.LINE_AA,
                )
        cv2.rectangle(display, (0, 0), (width, min(86, height)), (20, 20, 20), -1)
        cv2.putText(
            display, f"Frame {frame_index} | Label: {joint_name}",
            (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            display, prompt, (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            display, f"Annotated {cursor}/{len(joints)} landmarks",
            (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 220, 190), 1, cv2.LINE_AA,
        )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF

        click = state["click"]
        if not complete and click is not None:
            state["click"] = None
            px, py = click
            picked[joint_index] = row_for(
                joint_index, joint_name,
                px / max(width - 1, 1), py / max(height - 1, 1), 1,
            )
            cursor += 1
        elif not complete and key == ord("s"):
            picked[joint_index] = row_for(joint_index, joint_name, None, None, 0)
            cursor += 1
        elif key == ord("u") and cursor > 0:
            cursor -= 1
            picked.pop(joints[cursor][0], None)
            state["click"] = None
        elif complete and key in (10, 13, ord("n")):
            break
        elif key in (ord("q"), 27):
            should_quit = True
            break

    current_rows = [picked[index] for index, _ in joints if index in picked]
    return current_rows, should_quit


def annotate_video(video_path: Path, out_path: Path, stride: int, max_frames: int | None, joints: list[tuple[int, str]]) -> int:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    window_name = "FrameTrust — manual ground-truth annotation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    records: list[dict[str, int | float | str]] = []
    frame_index = 0
    stop = False
    try:
        while not stop:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_index >= max_frames:
                break
            if frame_index % stride == 0:
                frame_rows, stop = annotate_frame(
                    frame, frame_index, frame_index / fps, window_name, joints
                )
                records.extend(frame_rows)
                if records:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(records, columns=OUTPUT_COLUMNS).to_csv(out_path, index=False)
            frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=OUTPUT_COLUMNS).to_csv(out_path, index=False)
    print(f"[FrameTrust] saved {len(records)} point annotations to {out_path}")
    print("Only manually labeled points are written; unlabeled joints are not treated as invisible.")
    return len(records)


if __name__ == "__main__":
    args = parse_args()
    annotate_video(args.video, args.out, args.stride, args.max_frames, JOINT_PROFILES[args.profile])
