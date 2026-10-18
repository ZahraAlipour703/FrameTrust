#!/usr/bin/env python3
"""Action-stratified Penn Action development evaluation for FrameTrust.

Runs only QC-approved sequences in Penn Action's official training split.
Compares raw MediaPipe predictions, confidence-thresholded interpolation,
a missing-only interpolation baseline, FrameTrust's confidence-aware repair,
One Euro filtering, and Savitzky-Golay smoothing.

This is a development experiment, not a final test-set evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frametrust.repair import confidence_aware_repair, interpolate_missing  # noqa: E402

# Penn Action annotation order: head, L/R shoulder, L/R elbow, L/R wrist,
# L/R hip, L/R knee, L/R ankle. Exclude head because its definition does not
# directly correspond to MediaPipe's nose landmark.
PENN_JOINTS = np.arange(1, 13, dtype=int)  # source indices 1..12, zero-based
# Empirical Penn Action alignment correction: reverse each MediaPipe
# left/right pair to match the annotated ground-truth correspondence.
# This is a landmark-mapping correction, not a model change.
MEDIAPIPE_JOINTS = np.array([12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27])
JOINT_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
CONDITIONS = ["clean", "blur", "low_resolution", "occlusion", "missing_pose_observations"]
METHODS = ["raw", "linear_interpolation", "linear_missing_only", "confidence_aware", "one_euro", "savgol"]


def as_scalar(value: Any) -> Any:
    return np.asarray(value).squeeze().item()


def decode_matlab_string(value: Any) -> str:
    v = np.asarray(value).squeeze()
    if v.ndim == 0:
        return str(v.item())
    if v.dtype.kind in "US":
        return "".join(str(x) for x in v.reshape(-1).tolist())
    return str(v.tolist())


def as_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def select_development_sequences(
    manifest: pd.DataFrame, max_per_action: int, seed: int,
) -> list[str]:
    required = {"sequence", "split", "include_primary", "action"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"QC manifest missing required columns: {sorted(missing)}")
    work = manifest.copy()
    work["sequence"] = work["sequence"].astype(str).str.zfill(4)
    work["include_primary_bool"] = as_bool_series(work["include_primary"])
    eligible = work[
        (work["split"].astype(str).str.lower() == "train")
        & work["include_primary_bool"]
    ].copy()
    if "visible_points" in eligible:
        eligible = eligible[pd.to_numeric(eligible["visible_points"], errors="coerce").fillna(0) > 0]
    if eligible.empty:
        raise ValueError("No QC-approved training sequences in the manifest")

    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    actions = sorted(eligible["action"].astype(str).unique())
    for action in actions:
        ids = sorted(eligible.loc[eligible["action"].astype(str) == action, "sequence"].tolist())
        take = min(max_per_action, len(ids))
        sampled = rng.choice(np.asarray(ids, dtype=object), size=take, replace=False)
        chosen.extend(sorted(str(x) for x in sampled))
    if not chosen:
        raise ValueError("Sequence selection returned no sequences")
    print(f"Action-stratified selection: {len(chosen)} sequences across {len(actions)} action categories.")
    for action in actions:
        selected_for_action = eligible[
            (eligible["action"].astype(str) == action)
            & eligible["sequence"].isin(chosen)
        ]["sequence"].tolist()
        print(f"  {action}: {', '.join(sorted(selected_for_action)) or '(none)'}")
    return chosen


def corrupt_image(frame: np.ndarray, condition: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    if condition in ("clean", "missing_pose_observations"):
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


def load_sequence(dataset_root: Path, sequence_id: str) -> dict[str, Any]:
    label_path = dataset_root / "labels" / f"{sequence_id}.mat"
    frame_dir = dataset_root / "frames" / sequence_id
    if not label_path.is_file() or not frame_dir.is_dir():
        raise FileNotFoundError(f"Missing label or frame directory for sequence {sequence_id}")
    labels = loadmat(label_path)
    frames = sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    nframes = int(as_scalar(labels["nframes"]))
    train_flag = int(as_scalar(labels["train"]))
    if train_flag != 1:
        raise ValueError(f"Sequence {sequence_id} is not in official training split (train={train_flag})")
    if len(frames) != nframes:
        raise ValueError(f"Sequence {sequence_id}: {len(frames)} JPEGs but nframes={nframes}")

    x = np.asarray(labels["x"], dtype=float)
    y = np.asarray(labels["y"], dtype=float)
    visible = np.asarray(labels["visibility"]).astype(bool)
    bbox = np.asarray(labels["bbox"], dtype=float)
    if x.shape != (nframes, 13) or y.shape != (nframes, 13) or visible.shape != (nframes, 13):
        raise ValueError(f"Unexpected annotation shape in sequence {sequence_id}")
    if bbox.shape != (nframes, 4):
        raise ValueError(f"Unexpected bbox shape {bbox.shape} in sequence {sequence_id}")

    return {
        "frames": frames,
        "gt_xy": np.stack([x[:, PENN_JOINTS], y[:, PENN_JOINTS]], axis=-1),
        "gt_vis": visible[:, PENN_JOINTS],
        "bbox_xyxy": bbox,
        "action": decode_matlab_string(labels["action"]),
    }


def extract_predictions(
    frames: list[Path], condition: str, mp_pose_module: Any,
    gap_every: int, sequence_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(frames)
    coords = np.full((n, len(PENN_JOINTS), 2), np.nan, dtype=float)
    conf = np.zeros((n, len(PENN_JOINTS)), dtype=float)
    sizes = np.zeros((n, 2), dtype=int)  # width, height

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

            # Temporal observation-loss experiment: keep the GT frame/timestamp,
            # but intentionally omit that frame's pose observation.
            if t > 0 and condition == "missing_pose_observations" and t % gap_every == 0:
                continue

            input_image = corrupt_image(image, condition)
            result = pose.process(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            if not result.pose_landmarks:
                continue

            lm = result.pose_landmarks.landmark
            for j, mp_index in enumerate(MEDIAPIPE_JOINTS):
                point = lm[int(mp_index)]
                coords[t, j, 0] = float(point.x) * w
                coords[t, j, 1] = float(point.y) * h
                conf[t, j] = float(point.visibility)

            if (t + 1) % 100 == 0 or t + 1 == n:
                print(f"    {sequence_id}/{condition}: {t+1}/{n} frames", flush=True)

    return coords, conf, sizes


def alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro_filter(
    coords: np.ndarray, conf: np.ndarray, sizes: np.ndarray,
    threshold: float, fps: float, min_cutoff: float, beta: float, d_cutoff: float,
) -> np.ndarray:
    """Causal One Euro filter on frame-normalized 2D joints after gap filling.

    One cutoff is shared by x/y for each joint using the filtered 2D speed.
    Default parameters are a fixed reference configuration, not tuned on test.
    """
    filled = interpolate_missing(coords, conf, threshold=threshold)
    out = np.full_like(filled, np.nan, dtype=float)
    if len(filled) == 0:
        return out
    w = sizes[:, 0].astype(float)
    h = sizes[:, 1].astype(float)
    norm = filled.copy()
    norm[:, :, 0] /= w[:, None]
    norm[:, :, 1] /= h[:, None]
    dt = 1.0 / fps

    for j in range(norm.shape[1]):
        valid = np.isfinite(norm[:, j, :]).all(axis=1)
        if not valid.any():
            continue
        ids = np.flatnonzero(valid)
        first = int(ids[0])
        prev_x = norm[first, j].copy()
        prev_dx = np.zeros(2, dtype=float)
        out_norm = np.full((len(norm), 2), np.nan, dtype=float)
        out_norm[first] = prev_x
        prev_t = first
        for t in ids[1:]:
            t = int(t)
            delta_t = max((t - prev_t) * dt, dt)
            dx = (norm[t, j] - prev_x) / delta_t
            a_d = alpha(d_cutoff, delta_t)
            dx_hat = a_d * dx + (1.0 - a_d) * prev_dx
            speed = float(np.linalg.norm(dx_hat))
            cutoff = min_cutoff + beta * speed
            a = alpha(cutoff, delta_t)
            x_hat = a * norm[t, j] + (1.0 - a) * prev_x
            out_norm[t] = x_hat
            prev_x, prev_dx, prev_t = x_hat, dx_hat, t
        out[:, j, 0] = out_norm[:, 0] * w
        out[:, j, 1] = out_norm[:, 1] * h

    return out


def savgol_filter_pose(
    coords: np.ndarray, conf: np.ndarray, sizes: np.ndarray,
    threshold: float, window: int, polyorder: int,
) -> np.ndarray:
    """Offline Savitzky-Golay smoother applied after linear gap filling."""
    filled = interpolate_missing(coords, conf, threshold=threshold)
    out = filled.copy()
    t_count = len(filled)
    if t_count < 3:
        return out

    window = min(window, t_count if t_count % 2 else t_count - 1)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return out

    w = sizes[:, 0].astype(float)
    h = sizes[:, 1].astype(float)
    norm = filled.copy()
    norm[:, :, 0] /= w[:, None]
    norm[:, :, 1] /= h[:, None]

    for j in range(norm.shape[1]):
        for d in range(2):
            series = norm[:, j, d]
            valid = np.isfinite(series)
            # Interpolation helper normally fills this trajectory whenever it has
            # at least one reliable observation. Keep all-NaN trajectories missing.
            if valid.sum() < window or not valid.any():
                continue
            if not valid.all():
                idx = np.arange(t_count)
                series = np.interp(idx, idx[valid], series[valid])
            norm[:, j, d] = savgol_filter(
                series, window_length=window, polyorder=polyorder, mode="interp"
            )
    out[:, :, 0] = norm[:, :, 0] * w[:, None]
    out[:, :, 1] = norm[:, :, 1] * h[:, None]
    return out


def interpolate_missing_only(
    coords: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.35,
) -> np.ndarray:
    """Fill non-finite coordinate components without changing finite values.

    Only high-confidence, fully finite observations are used as anchors. Existing
    finite coordinates are preserved exactly, including finite low-confidence
    observations. As with numpy.interp, gaps before/after available anchors are
    filled with the nearest anchor value; callers should interpret long gaps with
    care.
    """
    c = np.asarray(coords, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    if c.ndim != 3 or conf.shape != c.shape[:2]:
        raise ValueError(
            "Expected coords [frames, joints, dims] and "
            "confidence [frames, joints]"
        )

    out = c.copy()
    frames = np.arange(c.shape[0])
    for j in range(c.shape[1]):
        finite_point = np.isfinite(c[:, j, :]).all(axis=1)
        good = (
            finite_point
            & np.isfinite(conf[:, j])
            & (conf[:, j] >= threshold)
        )
        if not good.any():
            continue

        for d in range(c.shape[2]):
            missing = ~np.isfinite(c[:, j, d])
            if not missing.any():
                continue
            filled = np.interp(frames, frames[good], c[good, j, d])
            out[missing, j, d] = filled[missing]
    return out


def ground_truth_masks_and_scales(
    gt_xy: np.ndarray, gt_vis: np.ndarray, bbox_xyxy: np.ndarray, sizes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create valid GT masks and explicitly defined normalization scales.

    Primary scale is Penn Action torso diameter defined as distance from the
    left shoulder to the right hip (selected joint indices 0 and 7). A frame
    is scoreable only when both reference points are visible and in bounds.
    Bounding-box diagonal is retained as a secondary, separately labelled scale.
    """
    n = len(gt_xy)
    valid = np.zeros(gt_vis.shape, dtype=bool)
    torso_scale = np.full(n, np.nan, dtype=float)
    bbox_scale = np.full(n, np.nan, dtype=float)
    for t in range(n):
        w, h = sizes[t]
        pts = gt_xy[t]
        inside = (
            np.isfinite(pts).all(axis=-1)
            & (pts[:, 0] >= 0) & (pts[:, 0] < w)
            & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        )
        valid[t] = gt_vis[t] & inside
        # Penn Action literature commonly defines torso diameter using the
        # left-shoulder to right-hip diagonal; selected indices 0 and 7.
        if valid[t, 0] and valid[t, 7]:
            td = float(np.linalg.norm(pts[0] - pts[7]))
            if np.isfinite(td) and td > 1e-6:
                torso_scale[t] = td
        x1, y1, x2, y2 = bbox_xyxy[t]
        if np.isfinite([x1, y1, x2, y2]).all() and x2 > x1 and y2 > y1:
            bbox_scale[t] = float(np.hypot(x2 - x1, y2 - y1))
    scored = valid & np.isfinite(torso_scale[:, None])
    return {"valid_gt": scored, "valid_gt_any_scale": valid,
            "torso_scale": torso_scale, "bbox_diag": bbox_scale}


def error_arrays(pred: np.ndarray, gt_xy: np.ndarray, valid_gt: np.ndarray,
                 torso_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(pred).all(axis=-1)
    mask = valid_gt & finite
    px_error = np.full(valid_gt.shape, np.nan, dtype=float)
    norm_error = np.full(valid_gt.shape, np.nan, dtype=float)
    diffs = np.linalg.norm(pred - gt_xy, axis=-1)
    px_error[mask] = diffs[mask]
    scales = np.broadcast_to(torso_scale[:, None], valid_gt.shape)
    norm_error[mask] = diffs[mask] / scales[mask]
    return px_error, norm_error


def score_method(
    pred: np.ndarray, raw: np.ndarray, raw_conf: np.ndarray, gt_xy: np.ndarray,
    valid_gt: np.ndarray, torso_scale: np.ndarray, bbox_diag: np.ndarray,
    confidence_threshold: float,
) -> dict[str, float | int]:
    n_gt = int(valid_gt.sum())
    pred_finite = np.isfinite(pred).all(axis=-1)
    detected_mask = valid_gt & pred_finite
    err_px, err_torso = error_arrays(pred, gt_xy, valid_gt, torso_scale)
    detected_torso = err_torso[detected_mask]
    coverage = float(detected_mask.sum() / n_gt) if n_gt else float("nan")

    # PCK is reported at multiple declared thresholds normalized by the torso
    # diameter. Missing predictions count as incorrect and remain in denominator.
    pcks: dict[str, float] = {}
    for threshold in (0.05, 0.10, 0.20):
        correct = valid_gt & pred_finite & np.isfinite(err_torso) & (err_torso <= threshold)
        pcks[f"pck_torso_{threshold:.2f}"] = float(correct.sum() / n_gt) if n_gt else float("nan")

    # Secondary normalized error by the annotation bounding-box diagonal.
    bbox_valid = valid_gt & pred_finite & np.isfinite(bbox_diag[:, None])
    bbox_norm_err = np.full(valid_gt.shape, np.nan, dtype=float)
    distance_px = np.linalg.norm(pred - gt_xy, axis=-1)
    bbox_scales = np.broadcast_to(bbox_diag[:, None], valid_gt.shape)
    bbox_norm_err[bbox_valid] = distance_px[bbox_valid] / bbox_scales[bbox_valid]

    raw_finite = np.isfinite(raw).all(axis=-1)
    common = valid_gt & raw_finite & pred_finite
    common_norm_errors = err_torso[common]
    common_correct20 = int((common_norm_errors <= 0.20).sum())

    # Separate raw observations into mutually exclusive reliability strata.
    # Missing coordinates take precedence over low reported confidence.
    raw_missing_state = ~raw_finite
    raw_low_conf_finite_state = (
        raw_finite
        & (~np.isfinite(raw_conf) | (raw_conf < confidence_threshold))
    )
    raw_initially_reliable_state = (
        raw_finite
        & np.isfinite(raw_conf)
        & (raw_conf >= confidence_threshold)
    )

    raw_err_px, raw_err_torso = error_arrays(
        raw, gt_xy, valid_gt, torso_scale
    )

    # Overall false correction: a prediction correct at PCK@0.20 becomes
    # incorrect or missing after the evaluated method.
    raw_correct = (
        valid_gt & raw_finite & np.isfinite(raw_err_torso)
        & (raw_err_torso <= 0.20)
    )
    harmed_raw_correct = raw_correct & (
        ~pred_finite | ~np.isfinite(err_torso) | (err_torso > 0.20)
    )
    n_raw_correct = int(raw_correct.sum())
    false_correction_rate = (
        float(harmed_raw_correct.sum() / n_raw_correct)
        if n_raw_correct else float("nan")
    )

    def score_raw_stratum(prefix: str, raw_state: np.ndarray) -> dict[str, float | int]:
        # The target set is defined only by raw predictions and GT validity.
        # It remains fixed across all evaluated methods.
        target = valid_gt & raw_state
        n_target = int(target.sum())
        detected = target & pred_finite & np.isfinite(err_torso)
        correct = detected & (err_torso <= 0.20)

        source_correct = (
            target & raw_finite & np.isfinite(raw_err_torso)
            & (raw_err_torso <= 0.20)
        )
        harmed = source_correct & (
            ~pred_finite | ~np.isfinite(err_torso) | (err_torso > 0.20)
        )
        n_source_correct = int(source_correct.sum())
        errors = err_torso[detected]

        return {
            f"{prefix}_target_points": n_target,
            f"{prefix}_method_coverage": (
                float(detected.sum() / n_target) if n_target else float("nan")
            ),
            f"{prefix}_method_mean_error_torso_norm_detected": (
                float(np.mean(errors)) if errors.size else float("nan")
            ),
            f"{prefix}_method_pck_torso_0.20": (
                float(correct.sum() / n_target) if n_target else float("nan")
            ),
            f"{prefix}_source_correct_points": n_source_correct,
            f"{prefix}_false_correction_rate": (
                float(harmed.sum() / n_source_correct)
                if n_source_correct else float("nan")
            ),
        }

    stratum_scores: dict[str, float | int] = {}
    for prefix, state in [
        ("missing_raw", raw_missing_state),
        ("low_conf_finite_raw", raw_low_conf_finite_state),
        ("initially_reliable_raw", raw_initially_reliable_state),
    ]:
        stratum_scores.update(score_raw_stratum(prefix, state))

    # Preserve the combined repair-target metrics for continuity with prior
    # outputs; the disjoint strata above are the more informative analysis.
    unreliable_raw = raw_missing_state | raw_low_conf_finite_state
    recovery_target = valid_gt & unreliable_raw
    recovery_detected = recovery_target & pred_finite & np.isfinite(err_torso)
    recovery_errors = err_torso[recovery_detected]
    recovery_n = int(recovery_target.sum())

    # Trajectory-fidelity metrics are computed on three adjacent GT-labeled
    # frames. Also provide a matched-triple comparison that includes only triples
    # where the raw model had all three observations.
    gt_accel_errors: list[float] = []
    pred_jitter: list[float] = []
    gt_accel_common: list[float] = []
    pred_jitter_common: list[float] = []
    triples = common_triples = 0
    for t in range(1, len(pred) - 1):
        scale = torso_scale[t]
        if not np.isfinite(scale) or scale <= 1e-9:
            continue
        gt3 = valid_gt[t - 1] & valid_gt[t] & valid_gt[t + 1]
        pr3 = (
            np.isfinite(pred[t - 1]).all(axis=-1)
            & np.isfinite(pred[t]).all(axis=-1)
            & np.isfinite(pred[t + 1]).all(axis=-1)
        )
        ok = gt3 & pr3
        for j in np.flatnonzero(ok):
            gt_second = gt_xy[t + 1, j] - 2 * gt_xy[t, j] + gt_xy[t - 1, j]
            pr_second = pred[t + 1, j] - 2 * pred[t, j] + pred[t - 1, j]
            gt_accel_errors.append(float(np.linalg.norm(pr_second - gt_second) / scale))
            pred_jitter.append(float(np.linalg.norm(pr_second) / scale))
            triples += 1

        raw3 = (
            np.isfinite(raw[t - 1]).all(axis=-1)
            & np.isfinite(raw[t]).all(axis=-1)
            & np.isfinite(raw[t + 1]).all(axis=-1)
        )
        ok_common = gt3 & pr3 & raw3
        for j in np.flatnonzero(ok_common):
            gt_second = gt_xy[t + 1, j] - 2 * gt_xy[t, j] + gt_xy[t - 1, j]
            pr_second = pred[t + 1, j] - 2 * pred[t, j] + pred[t - 1, j]
            gt_accel_common.append(float(np.linalg.norm(pr_second - gt_second) / scale))
            pred_jitter_common.append(float(np.linalg.norm(pr_second) / scale))
            common_triples += 1

    return {
        "gt_points_scored": n_gt,
        "prediction_coverage": coverage,
        "mean_error_px_detected": float(np.nanmean(err_px[detected_mask])) if detected_mask.any() else float("nan"),
        "mean_error_torso_norm_detected": float(np.nanmean(detected_torso)) if len(detected_torso) else float("nan"),
        "mean_error_bbox_diag_norm_detected": float(np.nanmean(bbox_norm_err[bbox_valid])) if bbox_valid.any() else float("nan"),
        **pcks,
        "common_support_points": int(common.sum()),
        "common_support_mean_error_torso_norm": float(np.nanmean(common_norm_errors)) if common.sum() else float("nan"),
        "common_support_pck_torso_0.20": float(common_correct20 / common.sum()) if common.sum() else float("nan"),
        "repair_target_points": recovery_n,
        "repair_target_coverage": float(recovery_detected.sum() / recovery_n) if recovery_n else float("nan"),
        "repair_target_mean_error_torso_norm": float(np.nanmean(recovery_errors)) if len(recovery_errors) else float("nan"),
        "repair_target_pck_torso_0.20": float(np.sum(recovery_errors <= 0.20) / recovery_n) if recovery_n else float("nan"),
        "raw_correct_points": n_raw_correct,
        "false_correction_rate": false_correction_rate,
        **stratum_scores,
        "temporal_triples": triples,
        "temporal_jitter_torso_norm": float(np.sqrt(np.mean(np.square(pred_jitter)))) if pred_jitter else float("nan"),
        "gt_acceleration_error_torso_norm": float(np.mean(gt_accel_errors)) if gt_accel_errors else float("nan"),
        "common_raw_temporal_triples": common_triples,
        "temporal_jitter_torso_norm_common_raw": float(np.sqrt(np.mean(np.square(pred_jitter_common)))) if pred_jitter_common else float("nan"),
        "gt_acceleration_error_torso_norm_common_raw": float(np.mean(gt_accel_common)) if gt_accel_common else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--manifest", type=Path, default=Path("data/external/penn_action_qc_manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/penn_action_development"))
    parser.add_argument("--sequences", default="", help="Optional comma-separated training sequence IDs; otherwise sample action-stratified from QC-approved train split.")
    parser.add_argument("--max-per-action", type=int, default=2, help="Number of training sequences sampled per action (default: 2).")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--gap-every", type=int, default=5)
    parser.add_argument("--fps", type=float, default=30.0, help="Assumed frame rate for One Euro filter; Penn Action MAT files do not include frame timestamps.")
    parser.add_argument("--one-euro-min-cutoff", type=float, default=1.0)
    parser.add_argument("--one-euro-beta", type=float, default=0.007)
    parser.add_argument("--one-euro-derivative-cutoff", type=float, default=1.0)
    parser.add_argument("--savgol-window", type=int, default=5)
    parser.add_argument("--savgol-polyorder", type=int, default=2)
    args = parser.parse_args()

    if args.max_per_action < 1:
        parser.error("--max-per-action must be >= 1")
    if not 0 < args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be in (0, 1]")
    if args.gap_every < 2 or args.fps <= 0:
        parser.error("--gap-every must be >= 2 and --fps must be positive")
    if args.savgol_window < 3 or args.savgol_window % 2 == 0:
        parser.error("--savgol-window must be an odd integer >= 3")
    if args.savgol_polyorder < 1 or args.savgol_polyorder >= args.savgol_window:
        parser.error("--savgol-polyorder must be >= 1 and < --savgol-window")

    root = args.dataset_root.resolve()
    manifest_path = args.manifest.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"QC manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype={"sequence": str})

    if args.sequences.strip():
        selected = [s.strip().zfill(4) for s in args.sequences.split(",") if s.strip()]
        # Validate chosen IDs against the QC manifest and official train split.
        tmp = manifest.copy()
        tmp["sequence"] = tmp["sequence"].astype(str).str.zfill(4)
        tmp["include_primary_bool"] = as_bool_series(tmp["include_primary"])
        ix = tmp.set_index("sequence")
        for seq in selected:
            if seq not in ix.index:
                raise ValueError(f"Sequence {seq} missing from QC manifest")
            row = ix.loc[seq]
            if str(row["split"]).lower() != "train" or not bool(row["include_primary_bool"]):
                raise ValueError(f"Sequence {seq} is not QC-approved in official train split")
    else:
        selected = select_development_sequences(manifest, args.max_per_action, args.seed)

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is required in the active environment") from exc

    started = time.time()
    all_rows: list[dict[str, Any]] = []
    for si, seq in enumerate(selected, 1):
        data = load_sequence(root, seq)
        frames = data["frames"]
        gt_xy, gt_vis, bbox = data["gt_xy"], data["gt_vis"], data["bbox_xyxy"]
        print(f"\n[{si}/{len(selected)}] Sequence {seq} ({data['action']}): {len(frames)} frames", flush=True)

        for condition in CONDITIONS:
            raw, conf, sizes = extract_predictions(
                frames, condition, mp.solutions.pose, args.gap_every, seq
            )
            masks = ground_truth_masks_and_scales(gt_xy, gt_vis, bbox, sizes)
            valid_gt = masks["valid_gt"]
            torso_scale = masks["torso_scale"]
            bbox_diag = masks["bbox_diag"]

            estimates = {
                "raw": raw,
                "linear_interpolation": interpolate_missing(
                    raw, conf, threshold=args.confidence_threshold
                ),
                "linear_missing_only": interpolate_missing_only(
                    raw, conf, threshold=args.confidence_threshold
                ),
                "confidence_aware": confidence_aware_repair(
                    raw, conf, threshold=args.confidence_threshold
                ),
                "one_euro": one_euro_filter(
                    raw, conf, sizes, args.confidence_threshold, args.fps,
                    args.one_euro_min_cutoff, args.one_euro_beta,
                    args.one_euro_derivative_cutoff,
                ),
                "savgol": savgol_filter_pose(
                    raw, conf, sizes, args.confidence_threshold,
                    args.savgol_window, args.savgol_polyorder,
                ),
            }
            for method, pred in estimates.items():
                metrics = score_method(
                    pred, raw, conf, gt_xy, valid_gt, torso_scale, bbox_diag,
                    args.confidence_threshold,
                )
                all_rows.append({
                    "sequence": seq,
                    "action": data["action"],
                    "official_split": "train",
                    "condition": condition,
                    "method": method,
                    "frames": len(frames),
                    "n_gt_frames_with_torso_scale": int(np.isfinite(torso_scale).sum()),
                    "confidence_threshold": args.confidence_threshold,
                    "pck_primary_normalization": "distance(left_shoulder,right_hip)",
                    "pck_thresholds_torso_diameter": "0.05;0.10;0.20",
                    "missing_pose_every_nth_frame": args.gap_every if condition == "missing_pose_observations" else 0,
                    **metrics,
                })
            print(f"  completed {condition}", flush=True)

    result = pd.DataFrame(all_rows)
    sequence_path = out_dir / "sequence_condition_method_metrics.csv"
    result.to_csv(sequence_path, index=False)

    metric_cols = [
        "prediction_coverage", "mean_error_px_detected", "mean_error_torso_norm_detected",
        "mean_error_bbox_diag_norm_detected", "pck_torso_0.05", "pck_torso_0.10",
        "pck_torso_0.20", "common_support_mean_error_torso_norm",
        "common_support_pck_torso_0.20", "repair_target_coverage",
        "repair_target_mean_error_torso_norm", "repair_target_pck_torso_0.20",
        "false_correction_rate",
        "missing_raw_target_points", "missing_raw_method_coverage",
        "missing_raw_method_mean_error_torso_norm_detected",
        "missing_raw_method_pck_torso_0.20", "missing_raw_false_correction_rate",
        "low_conf_finite_raw_target_points", "low_conf_finite_raw_method_coverage",
        "low_conf_finite_raw_method_mean_error_torso_norm_detected",
        "low_conf_finite_raw_method_pck_torso_0.20",
        "low_conf_finite_raw_false_correction_rate",
        "initially_reliable_raw_target_points", "initially_reliable_raw_method_coverage",
        "initially_reliable_raw_method_mean_error_torso_norm_detected",
        "initially_reliable_raw_method_pck_torso_0.20",
        "initially_reliable_raw_false_correction_rate",
        "temporal_jitter_torso_norm",
        "gt_acceleration_error_torso_norm", "temporal_jitter_torso_norm_common_raw",
        "gt_acceleration_error_torso_norm_common_raw",
    ]
    summary = result.groupby(["condition", "method"], as_index=False).agg(
        n_sequences=("sequence", "nunique"),
        **{f"{c}_mean": (c, "mean") for c in metric_cols},
        **{f"{c}_sd_between_sequences": (c, "std") for c in metric_cols},
        **{f"n_{c}_valid": (c, "count") for c in metric_cols},
    )
    summary_path = out_dir / "condition_method_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Paired, sequence-level deltas relative to raw output. This avoids treating
    # correlated frames/joints as independent observations for the headline comparison.
    raw_ref = result[result["method"] == "raw"].set_index(["sequence", "condition"])
    deltas = []
    delta_metrics = [
        "prediction_coverage", "mean_error_torso_norm_detected", "pck_torso_0.20",
        "common_support_mean_error_torso_norm", "common_support_pck_torso_0.20",
        "repair_target_pck_torso_0.20", "false_correction_rate",
        "missing_raw_method_coverage", "missing_raw_method_pck_torso_0.20",
        "missing_raw_false_correction_rate",
        "low_conf_finite_raw_method_coverage",
        "low_conf_finite_raw_method_mean_error_torso_norm_detected",
        "low_conf_finite_raw_method_pck_torso_0.20",
        "low_conf_finite_raw_false_correction_rate",
        "initially_reliable_raw_method_coverage",
        "initially_reliable_raw_method_mean_error_torso_norm_detected",
        "initially_reliable_raw_method_pck_torso_0.20",
        "initially_reliable_raw_false_correction_rate",
        "temporal_jitter_torso_norm_common_raw",
        "gt_acceleration_error_torso_norm_common_raw",
    ]
    for _, row in result[result["method"] != "raw"].iterrows():
        ref = raw_ref.loc[(row["sequence"], row["condition"])]
        delta = {
            "sequence": row["sequence"], "action": row["action"],
            "condition": row["condition"], "method": row["method"],
        }
        for col in delta_metrics:
            delta[f"delta_{col}_vs_raw"] = row[col] - ref[col]
        deltas.append(delta)
    delta_df = pd.DataFrame(deltas)
    delta_path = out_dir / "paired_deltas_vs_raw.csv"
    delta_df.to_csv(delta_path, index=False)
    delta_cols = [c for c in delta_df.columns if c.startswith("delta_")]
    delta_summary = delta_df.groupby(["condition", "method"], as_index=False).agg(
        n_sequences=("sequence", "nunique"),
        **{f"{c}_mean": (c, "mean") for c in delta_cols},
        **{f"{c}_sd_between_sequences": (c, "std") for c in delta_cols},
    )

    # Action-stratified sequence bootstrap CIs for paired mean deltas. Resampling
    # whole sequences (never individual frames) respects temporal dependence and
    # preserves the action-stratified development design. These are exploratory
    # CIs on the development subset, not confirmatory test-set inference.
    boot_rng = np.random.default_rng(args.seed + 1)
    n_boot = 2000
    boot_rows: list[dict[str, Any]] = []
    for (condition, method), group in delta_df.groupby(["condition", "method"]):
        row: dict[str, Any] = {
            "condition": condition, "method": method,
            "n_sequences": int(group["sequence"].nunique()),
            "n_actions": int(group["action"].nunique()),
            "bootstrap_replicates": n_boot,
        }
        for metric in delta_cols:
            action_arrays = [
                sub[metric].dropna().to_numpy(dtype=float)
                for _, sub in group.groupby("action")
                if sub[metric].notna().any()
            ]
            action_arrays = [a for a in action_arrays if len(a)]
            row[f"n_{metric}_valid_sequences"] = int(group[metric].notna().sum())
            row[f"n_{metric}_valid_actions"] = len(action_arrays)
            if not action_arrays:
                row[f"{metric}_macro_mean"] = float("nan")
                row[f"{metric}_ci95_low"] = float("nan")
                row[f"{metric}_ci95_high"] = float("nan")
                continue
            macro_action_means = np.asarray([a.mean() for a in action_arrays], dtype=float)
            row[f"{metric}_macro_mean"] = float(macro_action_means.mean())
            draws = np.empty(n_boot, dtype=float)
            for b in range(n_boot):
                sampled_action_means = []
                for values in action_arrays:
                    sampled = boot_rng.choice(values, size=len(values), replace=True)
                    sampled_action_means.append(float(sampled.mean()))
                draws[b] = float(np.mean(sampled_action_means))
            row[f"{metric}_ci95_low"] = float(np.quantile(draws, 0.025))
            row[f"{metric}_ci95_high"] = float(np.quantile(draws, 0.975))
        boot_rows.append(row)
    bootstrap_summary = pd.DataFrame(boot_rows)
    delta_summary = delta_summary.merge(bootstrap_summary, on=["condition", "method", "n_sequences"], how="left")
    delta_summary_path = out_dir / "paired_delta_summary.csv"
    delta_summary.to_csv(delta_summary_path, index=False)

    action_summary = result.groupby(["action", "condition", "method"], as_index=False).agg(
        n_sequences=("sequence", "nunique"),
        **{f"{c}_mean": (c, "mean") for c in metric_cols},
        **{f"{c}_sd_between_sequences": (c, "std") for c in metric_cols},
        **{f"n_{c}_valid": (c, "count") for c in metric_cols},
    )
    action_summary_path = out_dir / "action_condition_method_summary.csv"
    action_summary.to_csv(action_summary_path, index=False)

    selected_manifest = manifest.copy()
    selected_manifest["sequence"] = selected_manifest["sequence"].astype(str).str.zfill(4)
    selected_manifest[selected_manifest["sequence"].isin(selected)].to_csv(
        out_dir / "selected_development_sequences.csv", index=False
    )

    metadata = {
        "dataset": "Penn Action",
        "experiment_stage": "action-stratified development subset; not final test evaluation",
        "official_split_used": "training only; each loaded annotation train flag must equal 1",
        "selected_sequences": selected,
        "action_stratification_max_per_action": args.max_per_action if not args.sequences.strip() else None,
        "random_seed": args.seed,
        "conditions": CONDITIONS,
        "methods": METHODS,
        "penn_joint_indices_zero_based": PENN_JOINTS.tolist(),
        "mediapipe_joint_indices_zero_based": MEDIAPIPE_JOINTS.tolist(),
        "joint_names": JOINT_NAMES,
        "head_joint_excluded": True,
        "media_pipe_smooth_landmarks": False,
        "confidence_threshold": args.confidence_threshold,
        "one_euro": {
            "fps_assumption": args.fps, "min_cutoff_hz": args.one_euro_min_cutoff,
            "beta": args.one_euro_beta, "derivative_cutoff_hz": args.one_euro_derivative_cutoff,
            "input_normalized_to_frame_dimensions": True,
            "causal": True, "parameters_tuned_on_this_subset": False,
        },
        "savgol": {
            "window_length_frames": args.savgol_window, "polyorder": args.savgol_polyorder,
            "mode": "interp", "offline_noncausal": True,
            "applied_after_linear_interpolation_of_low_confidence_or_missing_points": True,
        },
        "missing_observation_policy": f"skip pose inference when t>0 and t % {args.gap_every} == 0; retain GT frame/time step",
        "raw_observation_strata": "Disjoint groups: missing coordinates; finite coordinates below the confidence threshold; finite coordinates at or above the threshold.",
        "false_correction_rate_definition": "Among GT-valid raw predictions correct at torso-normalized PCK@0.20, fraction made incorrect or missing by the evaluated method.",
        "linear_missing_only_definition": "Fill only non-finite coordinate components using high-confidence, fully finite observations as interpolation anchors; preserve existing finite coordinate components exactly.",
        "primary_pck_definition": "For each frame, torso diameter is Euclidean distance between visible in-bounds Penn Action left shoulder and right hip (selected joint indices 0 and 7 after excluding head). Score visible in-bounds mapped ground-truth joints only in frames where this scale is valid. Prediction failures count as incorrect. Report PCK thresholds 0.05, 0.10, 0.20 of torso diameter.",
        "secondary_error_normalization": "per-frame Penn Action bounding-box diagonal, interpreted as [x_min,y_min,x_max,y_max] after inspecting overlays",
        "sequence_level_summary": True,
        "frame_level_observations_not_treated_as_independent_for_summary": True,
        "limitations": [
            "Fixed single severity per corruption in this development run.",
            "Action-stratified subset from official training split only.",
            "One Euro FPS assumed 30 because Penn Action MAT labels do not provide frame timestamps.",
            "One Euro and Savitzky-Golay parameters are preset baselines, not tuned on a separate validation split.",
            "No claim of method novelty or state of the art is made by this script.",
        ],
        "runtime_seconds": round(time.time() - started, 2),
    }
    metadata_path = out_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== Development summary: mean and SD are across sequences, not frames ===")
    summary_columns = [
        "condition", "method", "n_sequences", "prediction_coverage_mean",
        "mean_error_torso_norm_detected_mean", "pck_torso_0.20_mean",
        "common_support_mean_error_torso_norm_mean",
        "false_correction_rate_mean",
        "missing_raw_method_coverage_mean", "missing_raw_method_pck_torso_0.20_mean",
        "low_conf_finite_raw_method_pck_torso_0.20_mean",
        "low_conf_finite_raw_false_correction_rate_mean",
        "initially_reliable_raw_method_pck_torso_0.20_mean",
        "initially_reliable_raw_false_correction_rate_mean",
        "temporal_jitter_torso_norm_common_raw_mean",
        "gt_acceleration_error_torso_norm_common_raw_mean",
    ]
    print(summary[summary_columns].to_string(index=False))
    print("\n=== Paired deltas vs raw (mean across sequences; PCK/coverage positive is better; error/jitter/acceleration negative is better) ===")
    print(delta_summary.to_string(index=False))
    print(f"\nSaved per-sequence results: {sequence_path}")
    print(f"Saved condition/method summary: {summary_path}")
    print(f"Saved paired deltas: {delta_path}")
    print(f"Saved paired delta summary with action-stratified bootstrap CIs: {delta_summary_path}")
    print(f"Saved action-level summary: {action_summary_path}")
    print(f"Saved metadata: {metadata_path}")
    print("This is a development-set run only. Do not report these results as final test performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
