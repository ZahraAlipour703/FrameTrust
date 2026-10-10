"""Load manual 2D pose annotations and evaluate predictions against them."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"frame_index", "joint_index", "x", "y", "visible"}


def read_ground_truth_csv(path: str | Path) -> pd.DataFrame:
    """Read and validate FrameTrust's normalized-coordinate annotation CSV."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Ground-truth CSV not found: {source}")
    return validate_ground_truth(pd.read_csv(source))


def validate_ground_truth(annotations: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, indices, visibility, and normalized image coordinates.

    Required columns:
      frame_index, joint_index, x, y, visible
    x/y are normalized to [0, 1]. Invisible or unannotated joints have visible=0
    and may have blank x/y. Duplicate frame/joint annotations are rejected.
    """
    missing = REQUIRED_COLUMNS - set(annotations.columns)
    if missing:
        raise ValueError(f"Ground-truth CSV is missing columns: {sorted(missing)}")

    result = annotations.copy()
    for column in ("frame_index", "joint_index"):
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite integers")
        if ((values < 0) | (values != np.floor(values))).any():
            raise ValueError(f"{column} must contain non-negative integers")
        result[column] = values.astype(int)

    visible = pd.to_numeric(result["visible"], errors="coerce")
    if visible.isna().any() or not visible.isin([0, 1]).all():
        raise ValueError("visible must contain only 0 or 1")
    result["visible"] = visible.astype(int)
    result["x"] = pd.to_numeric(result["x"], errors="coerce")
    result["y"] = pd.to_numeric(result["y"], errors="coerce")

    active = result["visible"].eq(1)
    active_xy = result.loc[active, ["x", "y"]].to_numpy(dtype=float)
    if active_xy.size and not np.isfinite(active_xy).all():
        raise ValueError("Visible annotations must have finite x and y coordinates")
    if active_xy.size and ((active_xy < 0).any() or (active_xy > 1).any()):
        raise ValueError("Visible x/y coordinates must be normalized to [0, 1]")

    duplicates = result.duplicated(["frame_index", "joint_index"], keep=False)
    if duplicates.any():
        pairs = result.loc[duplicates, ["frame_index", "joint_index"]].drop_duplicates()
        raise ValueError(f"Duplicate frame/joint annotations found: {pairs.to_dict('records')}")
    return result


def evaluate_ground_truth(
    annotations: pd.DataFrame,
    frame_indices: np.ndarray | list[int],
    estimate: np.ndarray,
    pck_fraction: float = 0.1,
) -> dict[str, float | int]:
    """Score estimates against manual labels.

    The body scale for each frame is the diagonal of the bounding box of visible
    annotated points in that frame. PCK counts missing/non-finite predictions as
    failures. Mean errors are reported only for predictions with finite x/y;
    prediction coverage is therefore reported alongside error.

    These are project metrics, not a claim of compatibility with a named
    benchmark's official protocol.
    """
    if not 0 < pck_fraction <= 1:
        raise ValueError("pck_fraction must be in (0, 1]")
    coords = np.asarray(estimate, dtype=float)
    indices = np.asarray(frame_indices)
    if coords.ndim != 3 or coords.shape[-1] < 2:
        raise ValueError("estimate must have shape [frames, joints, >=2]")
    if indices.ndim != 1 or len(indices) != coords.shape[0]:
        raise ValueError("frame_indices must contain one index per estimate frame")
    if not np.isfinite(indices.astype(float)).all() or (indices < 0).any():
        raise ValueError("frame_indices must contain non-negative finite indices")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("frame_indices must be unique")

    gt = validate_ground_truth(annotations)
    if not gt.empty and (gt["joint_index"] >= coords.shape[1]).any():
        raise ValueError(
            f"Ground-truth joint_index must be less than the estimate's "
            f"{coords.shape[1]} joints"
        )

    row_for_frame = {int(frame_id): row for row, frame_id in enumerate(indices)}
    visible = gt[(gt["visible"] == 1) & gt["frame_index"].isin(row_for_frame)].copy()

    errors_image: list[float] = []
    errors_body: list[float] = []
    correct = 0
    evaluated = 0
    detected = 0
    evaluated_frames = 0
    skipped_scale_points = 0

    # A body scale is estimated independently for each annotated frame.
    for frame_id, group in visible.groupby("frame_index", sort=True):
        points = group[["x", "y"]].to_numpy(dtype=float)
        if len(points) < 2:
            skipped_scale_points += len(points)
            continue
        body_scale = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        if not np.isfinite(body_scale) or body_scale <= 1e-12:
            skipped_scale_points += len(points)
            continue
        evaluated_frames += 1
        estimate_row = row_for_frame[int(frame_id)]

        for item in group.itertuples(index=False):
            joint_id = int(item.joint_index)
            pred = coords[estimate_row, joint_id, :2]
            evaluated += 1
            if np.isfinite(pred).all():
                detected += 1
                error = float(np.linalg.norm(pred - np.array([item.x, item.y], dtype=float)))
                errors_image.append(error)
                errors_body.append(error / body_scale)
                if error / body_scale <= pck_fraction:
                    correct += 1
            # Missing predictions remain in the PCK denominator and count as misses.

    visible_rows_in_video = int(len(visible))
    valid_denominator = evaluated
    return {
        "gt_evaluable_points": int(valid_denominator),
        "gt_frames_evaluable": int(evaluated_frames),
        "gt_points_skipped_scale": int(skipped_scale_points),
        "gt_prediction_coverage": float(detected / valid_denominator) if valid_denominator else float("nan"),
        "gt_mean_error_normalized_image": float(np.mean(errors_image)) if errors_image else float("nan"),
        "gt_mean_error_body_diagonal": float(np.mean(errors_body)) if errors_body else float("nan"),
        "gt_pck": float(correct / valid_denominator) if valid_denominator else float("nan"),
        "gt_points_outside_run": int((gt["visible"] == 1).sum() - visible_rows_in_video),
    }
