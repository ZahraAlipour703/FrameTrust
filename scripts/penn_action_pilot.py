#!/usr/bin/env python3
"""Small Penn Action -> MediaPipe -> FrameTrust pilot benchmark.

Uses only quality-control-approved sequences in the official training split.
Conditions: clean, blur, low-resolution, central occlusion, and missing pose
observations (skip inference on every Nth frame). This is a pipeline pilot, not
an official benchmark implementation or a publication result.
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
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frametrust.repair import confidence_aware_repair, interpolate_missing  # noqa: E402

# Penn Action order: head, L/R shoulder, L/R elbow, L/R wrist,
# L/R hip, L/R knee, L/R ankle. Exclude head initially because its point
# definition is not directly equivalent to MediaPipe's nose landmark.
PENN_JOINTS = np.arange(1, 13, dtype=int)
MEDIAPIPE_JOINTS = np.array([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
DEFAULT_SEQUENCES = ["0002", "0960", "1354"]
CONDITIONS = ["clean", "blur", "low_resolution", "occlusion", "missing_frames"]
METHODS = ["raw", "linear_interpolation", "confidence_aware"]


def corrupt_image(frame: np.ndarray, condition: str) -> np.ndarray:
    """Apply one deterministic spatial corruption to a copy of a frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    if condition == "clean" or condition == "missing_frames":
        return out
    if condition == "blur":
        k = max(5, int(round(min(h, w) * 0.025)))
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(out, (k, k), sigmaX=0)
    if condition == "low_resolution":
        sw, sh = max(2, w // 4), max(2, h // 4)
        small = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if condition == "occlusion":
        cv2.rectangle(
            out,
            (int(0.36 * w), int(0.25 * h)),
            (max(int(0.36 * w) + 1, int(0.64 * w)),
             max(int(0.25 * h) + 1, int(0.70 * h))),
            (0, 0, 0), thickness=-1,
        )
        return out
    raise ValueError(f"Unknown condition: {condition}")


def load_sequence(dataset_root: Path, sequence_id: str) -> tuple[list[Path], dict[str, Any]]:
    label_path = dataset_root / "labels" / f"{sequence_id}.mat"
    frame_dir = dataset_root / "frames" / sequence_id
    if not label_path.is_file() or not frame_dir.is_dir():
        raise FileNotFoundError(f"Missing label or frame directory for sequence {sequence_id}")
    labels = loadmat(label_path)
    frames = sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    nframes = int(np.asarray(labels["nframes"]).squeeze())
    train_flag = int(np.asarray(labels["train"]).squeeze())
    if train_flag != 1:
        raise ValueError(f"Sequence {sequence_id} is not in the official training split (train={train_flag})")
    if len(frames) != nframes:
        raise ValueError(f"Sequence {sequence_id}: {len(frames)} JPEGs but nframes={nframes}")
    x = np.asarray(labels["x"], dtype=float)
    y = np.asarray(labels["y"], dtype=float)
    visible = np.asarray(labels["visibility"]).astype(bool)
    if x.shape != (nframes, 13) or y.shape != (nframes, 13) or visible.shape != (nframes, 13):
        raise ValueError(f"Unexpected annotation shape in sequence {sequence_id}")
    gt_xy = np.stack([x[:, PENN_JOINTS], y[:, PENN_JOINTS]], axis=-1)
    gt_vis = visible[:, PENN_JOINTS]
    return frames, {"raw": labels, "gt_xy": gt_xy, "gt_vis": gt_vis}


def extract_predictions(
    frames: list[Path], condition: str, mp_pose_module: Any,
    gap_every: int, sequence_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(frames)
    coords = np.full((n, len(PENN_JOINTS), 2), np.nan, dtype=float)
    conf = np.zeros((n, len(PENN_JOINTS)), dtype=float)
    sizes = np.zeros((n, 2), dtype=int)  # width, height per frame
    previous_clean: np.ndarray | None = None

    with mp_pose_module.Pose(
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

            if t and condition == "missing_frames" and t % gap_every == 0:
                # Simulate absent pose observations; no estimate is generated.
                previous_clean = image.copy()
                continue

            input_image = corrupt_image(image, condition)
            previous_clean = image.copy()
            result = pose.process(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            if not result.pose_landmarks:
                continue

            lm = result.pose_landmarks.landmark
            for j, mp_index in enumerate(MEDIAPIPE_JOINTS):
                point = lm[int(mp_index)]
                # MediaPipe x/y are normalized to image width/height.
                coords[t, j, 0] = float(point.x) * w
                coords[t, j, 1] = float(point.y) * h
                conf[t, j] = float(point.visibility)

            if (t + 1) % 50 == 0 or t + 1 == n:
                print(f"    {sequence_id} / {condition}: {t+1}/{n} frames", flush=True)

    return coords, conf, sizes


def get_gt_validity(gt_xy: np.ndarray, gt_vis: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = gt_xy.shape[0]
    valid = np.zeros(gt_vis.shape, dtype=bool)
    scale = np.full(n, np.nan, dtype=float)
    for t in range(n):
        w, h = sizes[t]
        points = gt_xy[t]
        inside = (
            np.isfinite(points).all(axis=-1)
            & (points[:, 0] >= 0) & (points[:, 0] <= w)
            & (points[:, 1] >= 0) & (points[:, 1] <= h)
        )
        valid[t] = gt_vis[t] & inside
        visible_points = points[valid[t]]
        if len(visible_points) >= 2:
            diag = float(np.linalg.norm(visible_points.max(axis=0) - visible_points.min(axis=0)))
            if np.isfinite(diag) and diag > 1e-9:
                scale[t] = diag
    valid &= np.isfinite(scale[:, None])
    return valid, scale


def score_method(
    pred: np.ndarray, raw: np.ndarray, gt_xy: np.ndarray,
    valid_gt: np.ndarray, scale: np.ndarray, pck_fraction: float,
) -> dict[str, float | int]:
    gt_total = int(valid_gt.sum())
    finite = np.isfinite(pred).all(axis=-1)
    evaluated = valid_gt & finite
    detected = int(evaluated.sum())

    errors_px: list[float] = []
    errors_norm: list[float] = []
    pck_correct = 0
    for t, j in zip(*np.where(evaluated)):
        err = float(np.linalg.norm(pred[t, j] - gt_xy[t, j]))
        norm_err = err / scale[t]
        errors_px.append(err)
        errors_norm.append(norm_err)
        if norm_err <= pck_fraction:
            pck_correct += 1

    common = valid_gt & np.isfinite(raw).all(axis=-1) & finite
    common_indices = list(zip(*np.where(common)))
    common_errors = [
        float(np.linalg.norm(pred[t, j] - gt_xy[t, j]) / scale[t])
        for t, j in common_indices
    ]
    common_pck = sum(e <= pck_fraction for e in common_errors)

    # Motion-fidelity metric: error in second differences (acceleration-like
    # trajectory change), normalized by the visible GT body diagonal.
    accel_errors: list[float] = []
    jitter_values: list[float] = []
    triples = 0
    for t in range(1, len(pred) - 1):
        ok = (
            valid_gt[t - 1] & valid_gt[t] & valid_gt[t + 1]
            & np.isfinite(pred[t - 1]).all(axis=-1)
            & np.isfinite(pred[t]).all(axis=-1)
            & np.isfinite(pred[t + 1]).all(axis=-1)
        )
        for j in np.flatnonzero(ok):
            gt_second = gt_xy[t + 1, j] - 2 * gt_xy[t, j] + gt_xy[t - 1, j]
            pr_second = pred[t + 1, j] - 2 * pred[t, j] + pred[t - 1, j]
            accel_errors.append(float(np.linalg.norm(pr_second - gt_second) / scale[t]))
            jitter_values.append(float(np.linalg.norm(pr_second) / scale[t]))
            triples += 1

    return {
        "gt_points": gt_total,
        "prediction_coverage": detected / gt_total if gt_total else float("nan"),
        "mean_error_px_detected": float(np.mean(errors_px)) if errors_px else float("nan"),
        "mean_error_body_diag_detected": float(np.mean(errors_norm)) if errors_norm else float("nan"),
        "pck": pck_correct / gt_total if gt_total else float("nan"),
        "common_support_points": len(common_indices),
        "common_support_mean_error_body_diag": float(np.mean(common_errors)) if common_errors else float("nan"),
        "common_support_pck": common_pck / len(common_indices) if common_indices else float("nan"),
        "temporal_triples": triples,
        "gt_acceleration_error_body_diag": float(np.mean(accel_errors)) if accel_errors else float("nan"),
        "temporal_jitter_body_diag_per_frame2": float(np.sqrt(np.mean(np.square(jitter_values)))) if jitter_values else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--manifest", type=Path, default=Path("data/external/penn_action_qc_manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/penn_action_pilot"))
    parser.add_argument("--sequences", default=",".join(DEFAULT_SEQUENCES))
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--pck-fraction", type=float, default=0.10)
    parser.add_argument("--gap-every", type=int, default=5)
    args = parser.parse_args()

    if not 0 < args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be in (0, 1]")
    if not 0 < args.pck_fraction <= 1:
        parser.error("--pck-fraction must be in (0, 1]")
    if args.gap_every < 2:
        parser.error("--gap-every must be >= 2")

    root = args.dataset_root.resolve()
    manifest_path = args.manifest.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"QC manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype={"sequence": str})
    manifest["sequence"] = manifest["sequence"].str.zfill(4)
    selected = [s.strip().zfill(4) for s in args.sequences.split(",") if s.strip()]
    approved = manifest.set_index("sequence")
    for seq in selected:
        if seq not in approved.index:
            raise ValueError(f"Sequence {seq} is absent from the QC manifest")
        record = approved.loc[seq]
        if record["split"] != "train" or not bool(record["include_primary"]):
            raise ValueError(f"Sequence {seq} is not a QC-approved training sequence")

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is required in the active environment") from exc

    all_rows: list[dict[str, Any]] = []
    for seq in selected:
        frames, data = load_sequence(root, seq)
        gt_xy = data["gt_xy"]
        gt_vis = data["gt_vis"]
        print(f"\nSequence {seq}: {len(frames)} frames; running {len(CONDITIONS)} conditions.", flush=True)

        for condition in CONDITIONS:
            raw, conf, sizes = extract_predictions(
                frames, condition, mp.solutions.pose, args.gap_every, seq
            )
            valid_gt, scale = get_gt_validity(gt_xy, gt_vis, sizes)
            estimates = {
                "raw": raw,
                "linear_interpolation": interpolate_missing(
                    raw, conf, threshold=args.confidence_threshold
                ),
                "confidence_aware": confidence_aware_repair(
                    raw, conf, threshold=args.confidence_threshold
                ),
            }
            for method, pred in estimates.items():
                metrics = score_method(
                    pred, raw, gt_xy, valid_gt, scale, args.pck_fraction
                )
                all_rows.append({
                    "sequence": seq,
                    "condition": condition,
                    "method": method,
                    "frames": len(frames),
                    "gt_visible_mapped_points": int(valid_gt.sum()),
                    "confidence_threshold": args.confidence_threshold,
                    "pck_fraction_body_diagonal": args.pck_fraction,
                    "missing_every_nth_frame": args.gap_every if condition == "missing_frames" else 0,
                    **metrics,
                })
            print(f"  completed condition: {condition}", flush=True)

    result = pd.DataFrame(all_rows)
    csv_path = out_dir / "penn_action_pilot_metrics.csv"
    result.to_csv(csv_path, index=False)
    metadata = {
        "dataset": "Penn Action",
        "pilot_sequences": selected,
        "official_split": "training only",
        "conditions": CONDITIONS,
        "methods": METHODS,
        "penn_joint_indices_zero_based": PENN_JOINTS.tolist(),
        "mediapipe_joint_indices_zero_based": MEDIAPIPE_JOINTS.tolist(),
        "head_joint_excluded": True,
        "media_pipe_smooth_landmarks": False,
        "confidence_threshold": args.confidence_threshold,
        "pck_fraction_body_diagonal": args.pck_fraction,
        "missing_frame_policy": f"skip pose inference for frame indices t>0 where t % {args.gap_every} == 0; do not synthesize an image observation",
        "coordinate_policy": "MediaPipe normalized x/y converted to actual JPEG pixels; Penn Action labels compared in pixel coordinates without rescaling; ground-truth points outside inclusive image bounds excluded per point",
        "warning": "Small pipeline pilot only; not an official benchmark and not paper evidence. Check joint overlay/coordinate convention before interpreting accuracy.",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\n=== Pilot summary ===")
    print(result[[
        "sequence", "condition", "method", "gt_points", "prediction_coverage",
        "mean_error_body_diag_detected", "pck", "common_support_mean_error_body_diag",
        "common_support_pck", "gt_acceleration_error_body_diag",
        "temporal_jitter_body_diag_per_frame2", "temporal_triples"
    ]].to_string(index=False))
    print(f"\nSaved metrics: {csv_path}")
    print(f"Saved metadata: {out_dir / 'run_metadata.json'}")
    print("Interpret the pilot only as a pipeline check; its small sequence count is not enough for general claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
