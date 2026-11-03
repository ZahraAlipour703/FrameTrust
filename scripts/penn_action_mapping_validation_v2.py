#!/usr/bin/env python3
"""Independent Penn Action landmark-mapping audit, version 2.

This diagnostic compares the current empirical MediaPipe-to-Penn mapping with
its left/right-pair-swapped alternative. Unlike v1, it runs MediaPipe inside a
padded Penn Action ground-truth person bounding box, so the audit isolates
landmark correspondence from full-frame person-association failures.

IMPORTANT: the ground-truth bounding box is an oracle ROI. Results from this
script are for mapping validation only and must not be presented as detector
performance or as a deployable camera-only pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

PENN_JOINTS = np.arange(1, 13, dtype=int)  # remove Penn Action head joint
# Mapping A: current empirical mapping used by FrameTrust.
MAPPING_A_MEDIAPIPE = np.array([12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27], dtype=int)
# Penn order is L/R pairs; this permutation produces the previous convention.
SWAP_LR = np.array([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10], dtype=int)
JOINT_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5),
    (0, 6), (1, 7), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11),
]
PCK_THRESHOLDS = (0.05, 0.10, 0.20, 0.30)


def matlab_scalar(value: Any) -> Any:
    return np.asarray(value).squeeze().item()


def matlab_text(value: Any) -> str:
    arr = np.asarray(value).squeeze()
    if arr.ndim == 0:
        item = arr.item()
        return item.decode("utf-8", errors="replace").strip() if isinstance(item, bytes) else str(item).strip()
    if arr.dtype.kind in "US":
        return "".join(str(x) for x in arr.reshape(-1)).strip()
    return str(arr.tolist())


def load_sequence(dataset_root: Path, sequence: str) -> dict[str, Any]:
    label_path = dataset_root / "labels" / f"{sequence}.mat"
    frame_dir = dataset_root / "frames" / sequence
    if not label_path.is_file() or not frame_dir.is_dir():
        raise FileNotFoundError(f"Missing Penn Action labels or frames for sequence {sequence}")
    labels = loadmat(label_path)
    frames = sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    nframes = int(matlab_scalar(labels["nframes"]))
    if len(frames) != nframes:
        raise ValueError(f"{sequence}: found {len(frames)} frames; annotation says {nframes}")
    if int(matlab_scalar(labels["train"])) != 1:
        raise ValueError(f"{sequence}: not in the official Penn Action training split")

    x = np.asarray(labels["x"], dtype=float)
    y = np.asarray(labels["y"], dtype=float)
    visibility = np.asarray(labels["visibility"]).astype(bool)
    bbox = np.asarray(labels["bbox"], dtype=float)
    if x.shape != (nframes, 13) or y.shape != (nframes, 13) or visibility.shape != (nframes, 13):
        raise ValueError(f"{sequence}: unexpected x/y/visibility shapes")
    if bbox.shape != (nframes, 4):
        raise ValueError(f"{sequence}: unexpected bbox shape {bbox.shape}; expected {(nframes, 4)}")
    return {
        "frames": frames,
        "gt_xy": np.stack([x[:, PENN_JOINTS], y[:, PENN_JOINTS]], axis=-1),
        "gt_visible": visibility[:, PENN_JOINTS],
        "bbox_xyxy": bbox,
        "action": matlab_text(labels["action"]),
        "pose": matlab_text(labels["pose"]).lower(),
        "nframes": nframes,
    }


def make_padded_crop(
    image: np.ndarray, bbox_xyxy: np.ndarray, padding: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Return bbox crop and its (x1,y1,x2,y2) image coordinates."""
    h, w = image.shape[:2]
    box = np.asarray(bbox_xyxy, dtype=float).reshape(-1)
    if box.shape != (4,) or not np.isfinite(box).all():
        return None
    x1, y1, x2, y2 = box.tolist()
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return None
    pad_x, pad_y = padding * bw, padding * bh
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(w, int(math.ceil(x2 + pad_x)))
    bottom = min(h, int(math.ceil(y2 + pad_y)))
    if right - left < 16 or bottom - top < 16:
        return None
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    return crop, (left, top, right, bottom)


def infer_bbox_cropped_sequence(
    frames: list[Path],
    bboxes: np.ndarray,
    mp_pose: Any,
    sequence: str,
    padding: float,
    model_complexity: int,
    min_detection_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Infer canonical mapped points on per-frame GT person crops.

    Returns predictions in Mapping-A Penn joint order, per-frame image sizes,
    per-frame ROI boxes, and a Boolean indicating whether a valid crop existed.
    """
    n = len(frames)
    predictions = np.full((n, 12, 2), np.nan, dtype=float)
    sizes = np.zeros((n, 2), dtype=int)  # width, height
    roi_boxes = np.full((n, 4), np.nan, dtype=float)
    roi_valid = np.zeros(n, dtype=bool)
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        enable_segmentation=False,
        smooth_landmarks=False,
        min_detection_confidence=min_detection_confidence,
    ) as pose:
        for t, frame_path in enumerate(frames):
            image = cv2.imread(str(frame_path))
            if image is None:
                raise RuntimeError(f"Cannot read frame: {frame_path}")
            h, w = image.shape[:2]
            sizes[t] = (w, h)
            made = make_padded_crop(image, bboxes[t], padding)
            if made is None:
                continue
            crop, (left, top, right, bottom) = made
            roi_boxes[t] = (left, top, right, bottom)
            roi_valid[t] = True
            result = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                crop_w, crop_h = right - left, bottom - top
                for j, mp_index in enumerate(MAPPING_A_MEDIAPIPE):
                    point = landmarks[int(mp_index)]
                    predictions[t, j] = (
                        left + float(point.x) * crop_w,
                        top + float(point.y) * crop_h,
                    )
            if (t + 1) % 100 == 0 or t + 1 == n:
                print(f"  {sequence}: processed {t + 1}/{n} frames", flush=True)
    return predictions, sizes, roi_boxes, roi_valid


def frame_masks_and_scales(
    gt_xy: np.ndarray, gt_visible: np.ndarray, bbox_xyxy: np.ndarray, sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return visible/in-bounds GT mask, valid bbox mask, and bbox diagonal."""
    n = len(gt_xy)
    visible_in_bounds = np.zeros(gt_visible.shape, dtype=bool)
    bbox_diag = np.full(n, np.nan, dtype=float)
    bbox_valid = np.zeros(n, dtype=bool)
    for t in range(n):
        w, h = sizes[t]
        pts = gt_xy[t]
        inside = (
            np.isfinite(pts).all(axis=1)
            & (pts[:, 0] >= 0) & (pts[:, 0] < w)
            & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        )
        visible_in_bounds[t] = gt_visible[t] & inside
        box = np.asarray(bbox_xyxy[t], dtype=float)
        if box.shape == (4,) and np.isfinite(box).all():
            x1, y1, x2, y2 = box
            if x2 > x1 and y2 > y1:
                bbox_diag[t] = float(np.hypot(x2 - x1, y2 - y1))
                bbox_valid[t] = np.isfinite(bbox_diag[t]) and bbox_diag[t] > 1e-6
    score_mask = visible_in_bounds & bbox_valid[:, None]
    return score_mask, bbox_valid, bbox_diag


def evaluate_mapping(
    pred: np.ndarray, gt_xy: np.ndarray, score_mask: np.ndarray, bbox_diag: np.ndarray,
) -> dict[str, Any]:
    finite = np.isfinite(pred).all(axis=-1)
    scale = bbox_diag[:, None]
    dist = np.linalg.norm(pred - gt_xy, axis=-1)
    norm_error = np.full(score_mask.shape, np.nan, dtype=float)
    detected = score_mask & finite
    norm_error[detected] = dist[detected] / np.broadcast_to(scale, score_mask.shape)[detected]
    n_gt = int(score_mask.sum())
    n_detected = int(detected.sum())
    coverage = float(n_detected / n_gt) if n_gt else float("nan")
    mean_detected = float(np.mean(norm_error[detected])) if n_detected else float("nan")
    # The capped all-target loss assigns a transparent unit penalty to a missing prediction.
    all_losses = np.ones(n_gt, dtype=float)
    if n_detected:
        selected_errors = norm_error[score_mask]
        selected_finite = finite[score_mask]
        all_losses[selected_finite] = np.minimum(selected_errors[selected_finite], 1.0)
    metrics: dict[str, Any] = {
        "gt_points": n_gt,
        "detected_points": n_detected,
        "prediction_coverage": coverage,
        "mean_error_bbox_norm_detected": mean_detected,
        "mean_capped_error_bbox_norm_all_gt": float(np.mean(all_losses)) if n_gt else float("nan"),
    }
    for threshold in PCK_THRESHOLDS:
        metrics[f"pck_bbox_{threshold:.2f}"] = (
            float((detected & (norm_error <= threshold)).sum() / n_gt) if n_gt else float("nan")
        )
    return {**metrics, "finite": finite, "norm_error": norm_error}


def aggregate_sequence_macro(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        "prediction_coverage", "mean_error_bbox_norm_detected",
        "mean_capped_error_bbox_norm_all_gt", "pck_bbox_0.05", "pck_bbox_0.10",
        "pck_bbox_0.20", "pck_bbox_0.30",
    ]
    if rows.empty:
        return pd.DataFrame()
    return rows.groupby(group_cols, dropna=False, as_index=False).agg(
        n_sequences=("sequence", "nunique"),
        **{f"{col}_sequence_macro": (col, "mean") for col in metric_cols},
        **{f"{col}_sequence_sd": (col, "std") for col in metric_cols},
        n_gt_points=("gt_points", "sum"),
    )


def draw_mapping_panel(
    image: np.ndarray, gt: np.ndarray, visible: np.ndarray, pred: np.ndarray,
    frame_idx: int, seq: str, panel_title: str, pred_color: tuple[int, int, int],
    annotation_bbox: np.ndarray, inference_roi: np.ndarray,
) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    gt_ok = (
        visible & np.isfinite(gt).all(axis=1)
        & (gt[:, 0] >= 0) & (gt[:, 0] < w)
        & (gt[:, 1] >= 0) & (gt[:, 1] < h)
    )
    pred_ok = np.isfinite(pred).all(axis=1)
    for a, b in SKELETON:
        if gt_ok[a] and gt_ok[b]:
            cv2.line(out, tuple(np.round(gt[a]).astype(int)), tuple(np.round(gt[b]).astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
        if pred_ok[a] and pred_ok[b]:
            cv2.line(out, tuple(np.round(pred[a]).astype(int)), tuple(np.round(pred[b]).astype(int)), pred_color, 2, cv2.LINE_AA)
    for j in range(len(JOINT_NAMES)):
        if gt_ok[j]:
            p = tuple(np.round(gt[j]).astype(int))
            cv2.circle(out, p, 4, (0, 0, 255), -1, cv2.LINE_AA)
        if pred_ok[j]:
            p = tuple(np.round(pred[j]).astype(int))
            cv2.circle(out, p, 4, pred_color, 2, cv2.LINE_AA)
        if gt_ok[j] and pred_ok[j]:
            cv2.line(out, tuple(np.round(gt[j]).astype(int)), tuple(np.round(pred[j]).astype(int)), (0, 180, 255), 1, cv2.LINE_AA)
    # Original Penn Action bbox (magenta) and padded inference crop (yellow).
    if np.isfinite(annotation_bbox).all():
        ax1, ay1, ax2, ay2 = np.round(annotation_bbox).astype(int)
        cv2.rectangle(out, (ax1, ay1), (ax2, ay2), (255, 0, 255), 1, cv2.LINE_AA)
    if np.isfinite(inference_roi).all():
        rx1, ry1, rx2, ry2 = np.round(inference_roi).astype(int)
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (w, 42), (0, 0, 0), -1)
    cv2.putText(out, f"{seq} frame {frame_idx}: {panel_title}",
                (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "GT red | prediction colored | bbox magenta | padded ROI yellow",
                (7, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def save_overlays(
    sequence: str, frames: list[Path], gt_xy: np.ndarray, gt_visible: np.ndarray,
    bboxes: np.ndarray, roi_boxes: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    n_overlays: int, output_dir: Path,
) -> list[int]:
    ids = np.unique(np.linspace(0, len(frames) - 1, min(n_overlays, len(frames))).round().astype(int))
    for t in ids:
        image = cv2.imread(str(frames[int(t)]))
        if image is None:
            raise RuntimeError(f"Cannot read frame for overlay: {frames[int(t)]}")
        panel_a = draw_mapping_panel(
            image, gt_xy[t], gt_visible[t], pred_a[t], int(t), sequence,
            "Mapping A (current empirical)", (0, 210, 0), bboxes[t], roi_boxes[t],
        )
        panel_b = draw_mapping_panel(
            image, gt_xy[t], gt_visible[t], pred_b[t], int(t), sequence,
            "Mapping B (previous L/R convention)", (255, 160, 0), bboxes[t], roi_boxes[t],
        )
        combined = np.concatenate([panel_a, panel_b], axis=1)
        dest = output_dir / f"seq_{sequence}_frame_{int(t):04d}.jpg"
        if not cv2.imwrite(str(dest), combined):
            raise RuntimeError(f"Failed to write overlay: {dest}")
    return ids.tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--selection", type=Path, default=Path("outputs/penn_action_mapping_holdout/selected_mapping_validation_sequences.csv"))
    parser.add_argument("--sequences", default="", help="Optional comma-separated sequence IDs; overrides --selection")
    parser.add_argument("--out", type=Path, default=Path("outputs/penn_action_mapping_holdout_v2"))
    parser.add_argument("--bbox-padding", type=float, default=0.15, help="Fractional context added around each GT bbox")
    parser.add_argument("--n-overlays", type=int, default=5)
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.bbox_padding <= 1.0:
        parser.error("--bbox-padding must be in [0, 1]")
    if args.n_overlays < 1:
        parser.error("--n-overlays must be >= 1")
    if not 0.0 < args.min_detection_confidence <= 1.0:
        parser.error("--min-detection-confidence must be in (0, 1]")

    if args.sequences.strip():
        selected = [value.strip().zfill(4) for value in args.sequences.split(",") if value.strip()]
    else:
        selection_path = args.selection.resolve()
        if not selection_path.is_file():
            raise FileNotFoundError(
                f"Selection file not found: {selection_path}. Run the v1 mapping selection first "
                "or provide --sequences explicitly."
            )
        selection_df = pd.read_csv(selection_path, dtype={"sequence": str})
        if "sequence" not in selection_df.columns:
            raise ValueError("Selection file must contain a 'sequence' column")
        selected = sorted(set(selection_df["sequence"].astype(str).str.zfill(4)))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Sequence selection must be non-empty and contain unique IDs")

    dataset_root = args.dataset_root.resolve()
    out = args.out.resolve()
    overlay_dir = out / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is required in the active Python environment") from exc
    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            f"MediaPipe {getattr(mp, '__version__', 'unknown')} does not expose mp.solutions. "
            "Use the project's frametrust310 environment with MediaPipe <0.10.31."
        )

    sequence_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    for index, sequence in enumerate(selected, 1):
        data = load_sequence(dataset_root, sequence)
        print(
            f"\n[{index}/{len(selected)}] {sequence} ({data['action']}, {data['pose']}), "
            f"{data['nframes']} frames",
            flush=True,
        )
        pred_a, sizes, roi_boxes, roi_valid = infer_bbox_cropped_sequence(
            data["frames"], data["bbox_xyxy"], mp.solutions.pose, sequence,
            args.bbox_padding, args.model_complexity, args.min_detection_confidence,
        )
        pred_b = pred_a[:, SWAP_LR, :].copy()
        score_mask, bbox_valid, bbox_diag = frame_masks_and_scales(
            data["gt_xy"], data["gt_visible"], data["bbox_xyxy"], sizes,
        )
        eval_a = evaluate_mapping(pred_a, data["gt_xy"], score_mask, bbox_diag)
        eval_b = evaluate_mapping(pred_b, data["gt_xy"], score_mask, bbox_diag)
        n_gt = int(score_mask.sum())
        common = score_mask & eval_a["finite"] & eval_b["finite"]
        common_count = int(common.sum())
        common_error_a = eval_a["norm_error"][common]
        common_error_b = eval_b["norm_error"][common]
        for mapping_name, metrics in (("mapping_a_current_empirical", eval_a), ("mapping_b_previous_convention", eval_b)):
            sequence_rows.append({
                "sequence": sequence,
                "action": data["action"],
                "viewpoint": data["pose"],
                "mapping": mapping_name,
                "frames": data["nframes"],
                "valid_bbox_frames": int(bbox_valid.sum()),
                "valid_roi_frames": int(roi_valid.sum()),
                "gt_points": metrics["gt_points"],
                "detected_points": metrics["detected_points"],
                "prediction_coverage": metrics["prediction_coverage"],
                "mean_error_bbox_norm_detected": metrics["mean_error_bbox_norm_detected"],
                "mean_capped_error_bbox_norm_all_gt": metrics["mean_capped_error_bbox_norm_all_gt"],
                **{key: metrics[key] for key in [f"pck_bbox_{t:.2f}" for t in PCK_THRESHOLDS]},
                "common_support_points": common_count,
                "common_support_mean_error_bbox_norm": (
                    float(np.mean(common_error_a if mapping_name.startswith("mapping_a") else common_error_b))
                    if common_count else float("nan")
                ),
                "common_support_pck_bbox_0.20": (
                    float(np.mean((common_error_a if mapping_name.startswith("mapping_a") else common_error_b) <= 0.20))
                    if common_count else float("nan")
                ),
            })

        for t in range(data["nframes"]):
            for j, joint_name in enumerate(JOINT_NAMES):
                gt_point = data["gt_xy"][t, j]
                gt_ok = bool(score_mask[t, j])
                point_a, point_b = pred_a[t, j], pred_b[t, j]
                finite_a, finite_b = np.isfinite(point_a).all(), np.isfinite(point_b).all()
                error_a = eval_a["norm_error"][t, j]
                error_b = eval_b["norm_error"][t, j]
                joint_rows.append({
                    "sequence": sequence, "action": data["action"], "viewpoint": data["pose"],
                    "frame": t, "joint_index": j, "joint": joint_name,
                    "gt_visible_in_bounds_with_bbox": gt_ok,
                    "gt_x": float(gt_point[0]) if np.isfinite(gt_point[0]) else np.nan,
                    "gt_y": float(gt_point[1]) if np.isfinite(gt_point[1]) else np.nan,
                    "mapping_a_x": float(point_a[0]) if finite_a else np.nan,
                    "mapping_a_y": float(point_a[1]) if finite_a else np.nan,
                    "mapping_b_x": float(point_b[0]) if finite_b else np.nan,
                    "mapping_b_y": float(point_b[1]) if finite_b else np.nan,
                    "mapping_a_detected": bool(finite_a), "mapping_b_detected": bool(finite_b),
                    "mapping_a_error_bbox_norm": float(error_a) if np.isfinite(error_a) else np.nan,
                    "mapping_b_error_bbox_norm": float(error_b) if np.isfinite(error_b) else np.nan,
                    "bbox_diagonal_px": float(bbox_diag[t]) if np.isfinite(bbox_diag[t]) else np.nan,
                })
        common_rows.append({
            "sequence": sequence, "action": data["action"], "viewpoint": data["pose"],
            "common_support_points": common_count,
            "mapping_a_mean_error_common_support": float(np.mean(common_error_a)) if common_count else np.nan,
            "mapping_b_mean_error_common_support": float(np.mean(common_error_b)) if common_count else np.nan,
            "mapping_a_pck20_common_support": float(np.mean(common_error_a <= 0.20)) if common_count else np.nan,
            "mapping_b_pck20_common_support": float(np.mean(common_error_b <= 0.20)) if common_count else np.nan,
            "mapping_a_better_common_support": bool(np.mean(common_error_a) < np.mean(common_error_b)) if common_count else None,
        })
        overlay_frames = save_overlays(
            sequence, data["frames"], data["gt_xy"], data["gt_visible"],
            data["bbox_xyxy"], roi_boxes, pred_a, pred_b, args.n_overlays, overlay_dir,
        )
        print(
            f"  bbox-ROI frames={int(roi_valid.sum())}/{data['nframes']}; "
            f"scored GT points={n_gt}; common-support points={common_count}; "
            f"A PCK@0.20={eval_a['pck_bbox_0.20']:.3f}; "
            f"B PCK@0.20={eval_b['pck_bbox_0.20']:.3f}; overlays={len(overlay_frames)}",
            flush=True,
        )

    seq_df = pd.DataFrame(sequence_rows)
    joints_df = pd.DataFrame(joint_rows)
    common_df = pd.DataFrame(common_rows)
    seq_df.to_csv(out / "mapping_metrics_by_sequence.csv", index=False)
    joints_df.to_csv(out / "mapping_metrics_by_joint.csv", index=False)
    common_df.to_csv(out / "mapping_common_support_by_sequence.csv", index=False)
    aggregate_sequence_macro(seq_df, ["mapping"]).to_csv(out / "mapping_summary_overall.csv", index=False)
    aggregate_sequence_macro(seq_df, ["viewpoint", "mapping"]).to_csv(out / "mapping_summary_by_viewpoint.csv", index=False)
    aggregate_sequence_macro(seq_df, ["action", "mapping"]).to_csv(out / "mapping_summary_by_action.csv", index=False)

    metadata = {
        "purpose": "independent training-split landmark mapping audit; not final benchmark evaluation",
        "official_split": "Penn Action training only; each sequence MAT train flag checked",
        "sequences": selected,
        "mapping_a_mediapipe_indices_zero_based": MAPPING_A_MEDIAPIPE.tolist(),
        "mapping_b_previous_convention": MAPPING_A_MEDIAPIPE[SWAP_LR].tolist(),
        "penn_action_indices_zero_based_excluding_head": PENN_JOINTS.tolist(),
        "joint_names_in_penn_order": JOINT_NAMES,
        "inference_roi": "per-frame Penn Action ground-truth person bbox plus configured context padding",
        "oracle_bbox_crop": True,
        "oracle_bbox_warning": "This crop uses ground-truth person boxes to isolate mapping from person association. Results are not end-to-end detector performance and must not be described as deployable camera-only results.",
        "bbox_order_assumption": "[x1,y1,x2,y2], consistent with the existing FrameTrust evaluation code; verify with overlays before publication.",
        "normalization": "Euclidean diagonal of the same-frame Penn Action person bbox; only visible, in-bounds joints in frames with valid boxes are scored.",
        "missing_prediction_policy": "Missing predictions count as PCK failures; coverage and a capped all-GT normalized error (missing=1.0) are reported separately.",
        "comparison_policy": "Each mapping is scored on all eligible GT points independently; common-support metrics additionally compare only points detected by both mappings.",
        "mediapipe_static_image_mode": True,
        "mediapipe_model_complexity": args.model_complexity,
        "min_detection_confidence": args.min_detection_confidence,
        "bbox_padding_fraction": args.bbox_padding,
        "overlay_frames_per_sequence": args.n_overlays,
        "pck_thresholds_bbox_diagonal": list(PCK_THRESHOLDS),
        "limitations": [
            "Oracle ground-truth ROI is used solely for mapping diagnosis.",
            "Only the selected independent official-training sequences are included.",
            "A single 2D detector and one person-box protocol are evaluated.",
            "A sequence with no detected pose contributes zero coverage and PCK, not evidence of a mapping error.",
        ],
    }
    (out / "mapping_audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved mapping-v2 outputs to: {out}")
    print("Review mapping_summary_overall.csv and mapping_summary_by_viewpoint.csv before accepting a mapping.")
    print("This remains a mapping diagnostic with oracle person boxes, not the final pose-estimation benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
