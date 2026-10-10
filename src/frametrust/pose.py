"""Video loading and MediaPipe Pose landmark extraction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .corruptions import CORRUPTIONS, corrupt_frame


def extract_video_landmarks(
    video_path: str | Path,
    corruption: str = "clean",
    max_frames: int | None = None,
    drop_every: int = 5,
) -> pd.DataFrame:
    """Run MediaPipe Pose and return one row per frame with landmark columns.

    MediaPipe is imported lazily so the numeric repair/metric package can be
    tested without installing the pose-estimation runtime.
    """
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is required for video inference. Run `pip install -e .`.") from exc

    if corruption not in CORRUPTIONS:
        raise ValueError(f"corruption must be one of {CORRUPTIONS}")
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    pose_api: Any = mp.solutions.pose
    landmark_names = [item.name.lower() for item in pose_api.PoseLandmark]
    records: list[dict[str, float | int]] = []
    previous_clean: np.ndarray | None = None
    frame_index = 0
    try:
        with pose_api.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                ok, clean_frame = cap.read()
                if not ok:
                    break
                if max_frames is not None and frame_index >= max_frames:
                    break
                frame = corrupt_frame(clean_frame, corruption, frame_index, previous_clean, drop_every)
                previous_clean = clean_frame.copy()
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                row: dict[str, float | int] = {
                    "frame_index": frame_index,
                    "timestamp_seconds": frame_index / fps,
                    "width": int(clean_frame.shape[1]),
                    "height": int(clean_frame.shape[0]),
                }
                lm = result.pose_landmarks.landmark if result.pose_landmarks else None
                for idx, name in enumerate(landmark_names):
                    if lm is not None and idx < len(lm):
                        point = lm[idx]
                        row[f"x_{idx}"] = float(point.x)
                        row[f"y_{idx}"] = float(point.y)
                        row[f"z_{idx}"] = float(point.z)
                        row[f"visibility_{idx}"] = float(point.visibility)
                    else:
                        row[f"x_{idx}"] = float("nan")
                        row[f"y_{idx}"] = float("nan")
                        row[f"z_{idx}"] = float("nan")
                        row[f"visibility_{idx}"] = 0.0
                records.append(row)
                frame_index += 1
    finally:
        cap.release()
    if not records:
        raise RuntimeError(f"No video frames were read from {path}")
    frame = pd.DataFrame.from_records(records)
    frame.attrs["landmark_names"] = landmark_names
    frame.attrs["fps"] = fps
    return frame


def frame_to_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert extracted DataFrame to coords [T,J,3], confidence [T,J]."""
    x_cols = sorted((c for c in frame.columns if c.startswith("x_")), key=lambda c: int(c.split("_")[1]))
    if not x_cols:
        raise ValueError("No landmark columns found")
    n_frames, n_joints = len(frame), len(x_cols)
    coords = np.empty((n_frames, n_joints, 3), dtype=float)
    confidence = np.empty((n_frames, n_joints), dtype=float)
    for j in range(n_joints):
        for d, axis in enumerate(("x", "y", "z")):
            coords[:, j, d] = frame[f"{axis}_{j}"].to_numpy(dtype=float)
        confidence[:, j] = frame[f"visibility_{j}"].to_numpy(dtype=float)
    return coords, confidence


def arrays_to_frame(original: pd.DataFrame, coords: np.ndarray, method: str) -> pd.DataFrame:
    """Build a long-form repaired output table, retaining original confidence."""
    rows: list[dict[str, float | int | str]] = []
    for t in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            rows.append({
                "frame_index": int(original.iloc[t]["frame_index"]),
                "timestamp_seconds": float(original.iloc[t]["timestamp_seconds"]),
                "joint_index": j,
                "x": float(coords[t, j, 0]),
                "y": float(coords[t, j, 1]),
                "z": float(coords[t, j, 2]),
                "visibility": float(original.iloc[t][f"visibility_{j}"]),
                "method": method,
            })
    return pd.DataFrame(rows)
