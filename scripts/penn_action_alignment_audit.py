#!/usr/bin/env python3
"""Visual and numeric alignment audit for Penn Action vs MediaPipe Pose.

Runs MediaPipe over each selected sequence in chronological order, saves overlays
at evenly spaced frames, and writes per-joint errors plus sequence summaries.
This is a diagnostic utility, not a final benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

PENN_JOINTS = np.arange(1, 13, dtype=int)
MEDIAPIPE_JOINTS = np.array([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
JOINT_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
# Edges use selected-joint indices after excluding the Penn Action head point.
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5),
    (0, 6), (1, 7), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11),
]


def matlab_scalar(v: Any) -> Any:
    return np.asarray(v).squeeze().item()


def load_one(root: Path, sequence: str) -> dict[str, Any]:
    label_path = root / "labels" / f"{sequence}.mat"
    frame_dir = root / "frames" / sequence
    if not label_path.is_file() or not frame_dir.is_dir():
        raise FileNotFoundError(f"Missing label file or frame directory for sequence {sequence}")
    labels = loadmat(label_path)
    frames = sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    n = int(matlab_scalar(labels["nframes"]))
    if len(frames) != n:
        raise ValueError(f"{sequence}: found {len(frames)} frames; annotation says {n}")
    x = np.asarray(labels["x"], dtype=float)
    y = np.asarray(labels["y"], dtype=float)
    vis = np.asarray(labels["visibility"]).astype(bool)
    if x.shape != (n, 13) or y.shape != (n, 13) or vis.shape != (n, 13):
        raise ValueError(f"{sequence}: unexpected x/y/visibility shapes {x.shape}, {y.shape}, {vis.shape}")
    gt = np.stack([x[:, PENN_JOINTS], y[:, PENN_JOINTS]], axis=-1)
    gt_vis = vis[:, PENN_JOINTS]
    return {"frames": frames, "gt": gt, "vis": gt_vis, "nframes": n}


def infer_sequence(frames: list[Path], mp_pose: Any, seq: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(frames)
    pred = np.full((n, 12, 2), np.nan, dtype=float)
    sizes = np.zeros((n, 2), dtype=int)
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
        smooth_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for t, path in enumerate(frames):
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Cannot read image: {path}")
            h, w = image.shape[:2]
            sizes[t] = (w, h)
            result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                for j, k in enumerate(MEDIAPIPE_JOINTS):
                    p = lm[int(k)]
                    pred[t, j] = (float(p.x) * w, float(p.y) * h)
            if (t + 1) % 100 == 0 or t + 1 == n:
                print(f"  {seq}: inferred {t + 1}/{n} frames", flush=True)
    return pred, sizes


def draw_overlay(image: np.ndarray, gt: np.ndarray, visible: np.ndarray,
                 pred: np.ndarray, frame_idx: int, seq: str) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    gt_ok = (
        visible & np.isfinite(gt).all(axis=1)
        & (gt[:, 0] >= 0) & (gt[:, 0] < w)
        & (gt[:, 1] >= 0) & (gt[:, 1] < h)
    )
    pred_ok = np.isfinite(pred).all(axis=1)

    # Ground-truth skeleton in red; model skeleton in green.
    for a, b in SKELETON:
        if gt_ok[a] and gt_ok[b]:
            cv2.line(out, tuple(np.round(gt[a]).astype(int)), tuple(np.round(gt[b]).astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
        if pred_ok[a] and pred_ok[b]:
            cv2.line(out, tuple(np.round(pred[a]).astype(int)), tuple(np.round(pred[b]).astype(int)), (0, 210, 0), 2, cv2.LINE_AA)
    for j, name in enumerate(JOINT_NAMES):
        if gt_ok[j]:
            p = tuple(np.round(gt[j]).astype(int))
            cv2.circle(out, p, 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(out, f"G{j}", (p[0] + 3, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
        if pred_ok[j]:
            p = tuple(np.round(pred[j]).astype(int))
            cv2.circle(out, p, 4, (0, 210, 0), 2, cv2.LINE_AA)
            cv2.putText(out, f"P{j}", (p[0] + 3, p[1] + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 210, 0), 1, cv2.LINE_AA)
        if gt_ok[j] and pred_ok[j]:
            cv2.line(out, tuple(np.round(gt[j]).astype(int)), tuple(np.round(pred[j]).astype(int)), (255, 120, 0), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.putText(out, f"Penn Action GT (red) vs MediaPipe (green) | seq {seq}, frame {frame_idx}",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "Numbers 0-11: shoulder, elbow, wrist, hip, knee, ankle pairs | orange line = point offset",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    ap.add_argument("--sequences", default="1797,1219,1033", help="Comma-separated sequence IDs; must exist locally")
    ap.add_argument("--n-overlays", type=int, default=4, help="Number of evenly spaced frame overlays per sequence")
    ap.add_argument("--out", type=Path, default=Path("outputs/penn_action_alignment_audit"))
    args = ap.parse_args()
    if args.n_overlays < 1:
        ap.error("--n-overlays must be >= 1")

    root = args.dataset_root.resolve()
    out = args.out.resolve()
    overlay_dir = out / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is required in the active Python environment") from exc

    rows: list[dict[str, Any]] = []
    seq_summaries: list[dict[str, Any]] = []
    for raw_id in args.sequences.split(","):
        seq = raw_id.strip().zfill(4)
        if not seq:
            continue
        data = load_one(root, seq)
        print(f"\nSequence {seq}: {data['nframes']} frames", flush=True)
        pred, sizes = infer_sequence(data["frames"], mp.solutions.pose, seq)
        gt, vis = data["gt"], data["vis"]
        w, h = sizes[:, 0], sizes[:, 1]
        inside = (
            np.isfinite(gt).all(axis=-1)
            & (gt[:, :, 0] >= 0) & (gt[:, :, 0] < w[:, None])
            & (gt[:, :, 1] >= 0) & (gt[:, :, 1] < h[:, None])
        )
        valid = vis & inside
        scales = np.full(len(gt), np.nan, dtype=float)
        scale_valid = valid[:, 0] & valid[:, 7]
        scales[scale_valid] = np.linalg.norm(gt[scale_valid, 0] - gt[scale_valid, 7], axis=1)
        scale_valid &= np.isfinite(scales) & (scales > 1e-6)
        valid_score = valid & scale_valid[:, None]
        finite = np.isfinite(pred).all(axis=-1)
        diff = np.linalg.norm(pred - gt, axis=-1)
        err_norm = np.full(valid.shape, np.nan, dtype=float)
        scored = valid_score & finite
        err_norm[scored] = diff[scored] / np.broadcast_to(scales[:, None], valid.shape)[scored]
        n_gt = int(valid_score.sum())
        n_detected = int((valid_score & finite).sum())
        n_error = int(np.isfinite(err_norm).sum())
        summary = {
            "sequence": seq,
            "frames": int(len(gt)),
            "gt_points_scored": n_gt,
            "prediction_coverage": float(n_detected / n_gt) if n_gt else None,
            "mean_error_torso_normalized_detected": float(np.nanmean(err_norm)) if n_error else None,
            "median_error_torso_normalized_detected": float(np.nanmedian(err_norm)) if n_error else None,
            "pck_torso_0.20": float(((valid_score & finite & (err_norm <= .20)).sum() / n_gt)) if n_gt else None,
            "pck_torso_0.50": float(((valid_score & finite & (err_norm <= .50)).sum() / n_gt)) if n_gt else None,
            "mapped_joint_order": JOINT_NAMES,
        }
        seq_summaries.append(summary)
        for t in range(len(gt)):
            for j, name in enumerate(JOINT_NAMES):
                rows.append({
                    "sequence": seq, "frame": t, "joint_index": j, "joint": name,
                    "gt_visible": bool(vis[t, j]), "gt_in_bounds": bool(inside[t, j]),
                    "gt_x": float(gt[t, j, 0]) if np.isfinite(gt[t, j, 0]) else np.nan,
                    "gt_y": float(gt[t, j, 1]) if np.isfinite(gt[t, j, 1]) else np.nan,
                    "pred_x": float(pred[t, j, 0]) if np.isfinite(pred[t, j, 0]) else np.nan,
                    "pred_y": float(pred[t, j, 1]) if np.isfinite(pred[t, j, 1]) else np.nan,
                    "error_px": float(diff[t, j]) if np.isfinite(diff[t, j]) else np.nan,
                    "torso_scale_px": float(scales[t]) if np.isfinite(scales[t]) else np.nan,
                    "error_torso_normalized": float(err_norm[t, j]) if np.isfinite(err_norm[t, j]) else np.nan,
                })
        frame_ids = np.unique(np.linspace(0, len(gt) - 1, min(args.n_overlays, len(gt))).round().astype(int))
        for t in frame_ids:
            image = cv2.imread(str(data["frames"][int(t)]))
            overlay = draw_overlay(image, gt[t], vis[t] & inside[t], pred[t], int(t), seq)
            dest = overlay_dir / f"seq_{seq}_frame_{int(t):04d}.jpg"
            if not cv2.imwrite(str(dest), overlay):
                raise RuntimeError(f"Could not write overlay {dest}")
        def fmt(value):
            return f"{value:.3f}" if value is not None and np.isfinite(value) else "NA"

        print(
            f"  coverage={fmt(summary['prediction_coverage'])}; "
            f"mean normalized error={fmt(summary['mean_error_torso_normalized_detected'])}; "
            f"PCK@0.20={fmt(summary['pck_torso_0.20'])}; overlays={len(frame_ids)}",
            flush=True,
        )

    pd.DataFrame(rows).to_csv(out / "per_joint_alignment.csv", index=False)
    pd.DataFrame(seq_summaries).to_csv(out / "sequence_alignment_summary.csv", index=False)
    (out / "audit_metadata.json").write_text(json.dumps({
        "purpose": "diagnostic only; verify GT-coordinate alignment and joint mapping visually",
        "sequences": [s.strip().zfill(4) for s in args.sequences.split(",") if s.strip()],
        "mediapipe_model_complexity": 0,
        "mediapipe_smooth_landmarks": False,
        "landmark_mapping": {name: int(mp_idx) for name, mp_idx in zip(JOINT_NAMES, MEDIAPIPE_JOINTS)},
        "penn_action_source_indices_zero_based": PENN_JOINTS.tolist(),
        "normalization_scale": "distance between visible, in-bounds selected left shoulder and right hip (indices 0 and 7)",
    }, indent=2), encoding="utf-8")
    print(f"\nOverlays: {overlay_dir}")
    print(f"Joint errors: {out / 'per_joint_alignment.csv'}")
    print(f"Sequence summary: {out / 'sequence_alignment_summary.csv'}")
    print("Inspect overlays before interpreting the development PCK values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
