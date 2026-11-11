#!/usr/bin/env python3
"""Independent, clean-video audit of direct vs left/right-swapped landmark mapping.

This runs clean MediaPipe inference on QC-approved Penn Action *training* sequences
that can be held out from a prior development run. It writes per-sequence and
per-joint metrics plus visual overlays for manual inspection. It never reads test
split sequences and it does not tune or change the mapping automatically.
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from penn_action_development import (  # noqa: E402
    MEDIAPIPE_JOINTS,
    JOINT_NAMES,
    extract_predictions,
    ground_truth_masks_and_scales,
    load_sequence,
)

PAIR_SWAP = np.array([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10], dtype=int)


def as_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def load_eligible_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"QC manifest not found: {path}")
    data = pd.read_csv(path, dtype={"sequence": str})
    required = {"sequence", "split", "include_primary", "action"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"QC manifest missing required columns: {sorted(missing)}")
    data["sequence"] = data["sequence"].astype(str).str.zfill(4)
    eligible = data[
        (data["split"].astype(str).str.lower() == "train")
        & as_bool(data["include_primary"])
    ].copy()
    if "visible_points" in eligible:
        eligible = eligible[pd.to_numeric(eligible["visible_points"], errors="coerce").fillna(0) > 0]
    return eligible


def choose_sequences(
    eligible: pd.DataFrame,
    explicit: str,
    max_per_action: int,
    seed: int,
    exclude_metadata: Path,
) -> list[str]:
    if explicit.strip():
        chosen = [x.strip().zfill(4) for x in explicit.split(",") if x.strip()]
        lookup = eligible.set_index("sequence")
        for seq in chosen:
            if seq not in lookup.index:
                raise ValueError(f"{seq} is not QC-approved in the official training split")
        return chosen

    excluded: set[str] = set()
    if exclude_metadata.is_file():
        with exclude_metadata.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        excluded = {str(x).zfill(4) for x in metadata.get("selected_sequences", [])}
        print(f"Excluding {len(excluded)} sequence IDs from {exclude_metadata}", flush=True)
    else:
        print(
            f"WARNING: exclusion metadata not found at {exclude_metadata}; "
            "sequence selection cannot guarantee independence from the previous run.",
            flush=True,
        )

    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for action in sorted(eligible["action"].astype(str).unique()):
        ids = sorted(
            eligible.loc[eligible["action"].astype(str) == action, "sequence"].astype(str).tolist()
        )
        ids = [seq for seq in ids if seq not in excluded]
        n_take = min(max_per_action, len(ids))
        if n_take:
            sampled = rng.choice(np.asarray(ids, dtype=object), size=n_take, replace=False)
            chosen.extend(sorted(str(x) for x in sampled))
    return chosen


def choose_winner(direct: float, swapped: float, higher_is_better: bool) -> str:
    direct_ok = np.isfinite(direct)
    swapped_ok = np.isfinite(swapped)
    if direct_ok and not swapped_ok:
        return "direct"
    if swapped_ok and not direct_ok:
        return "swapped"
    if not direct_ok and not swapped_ok:
        return "tie"
    if np.isclose(direct, swapped):
        return "tie"
    if higher_is_better:
        return "swapped" if swapped > direct else "direct"
    return "swapped" if swapped < direct else "direct"


def score_mapping(
    pred: np.ndarray,
    gt: np.ndarray,
    target: np.ndarray,
    scale: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return PCK@0.20, mean clipped normalized error, per-joint PCK and error."""
    finite = np.isfinite(pred).all(axis=-1)
    norm = np.linalg.norm(pred - gt, axis=-1) / np.maximum(scale[:, None], 1e-9)
    norm = np.where(finite & np.isfinite(norm), np.clip(norm, 0.0, 1.0), 1.0)
    if not target.any():
        return float("nan"), float("nan"), np.full(target.shape[1], np.nan), np.full(target.shape[1], np.nan)
    point_pck = (norm <= 0.20)
    pck = float(point_pck[target].mean())
    mean_error = float(norm[target].mean())
    per_joint_pck = np.full(target.shape[1], np.nan, dtype=float)
    per_joint_error = np.full(target.shape[1], np.nan, dtype=float)
    for j in range(target.shape[1]):
        m = target[:, j]
        if m.any():
            per_joint_pck[j] = float(point_pck[m, j].mean())
            per_joint_error[j] = float(norm[m, j].mean())
    return pck, mean_error, per_joint_pck, per_joint_error


def annotate_panel(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    valid_gt: np.ndarray,
    title: str,
) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (24, 24, 24), -1)
    cv2.putText(canvas, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    for j, joint_name in enumerate(JOINT_NAMES):
        if valid_gt[j] and np.isfinite(gt[j]).all():
            gx, gy = np.rint(gt[j]).astype(int)
            cv2.drawMarker(canvas, (gx, gy), (0, 220, 0), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
            cv2.putText(canvas, str(j), (gx + 4, gy - 4), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 220, 0), 1, cv2.LINE_AA)
        if np.isfinite(pred[j]).all():
            px, py = np.rint(pred[j]).astype(int)
            cv2.circle(canvas, (px, py), 5, (0, 0, 255), 2, cv2.LINE_AA)
            # Label prediction identities: otherwise direct/swapped panels render
            # the same unlabelled point set and are visually non-diagnostic.
            cv2.putText(canvas, f"P{j}", (px + 5, py + 12), cv2.FONT_HERSHEY_PLAIN,
                        0.9, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GT: green cross + index | prediction: red circle", (8, canvas.shape[0] - 9),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    parser.add_argument("--manifest", type=Path, default=Path("data/external/penn_action_qc_manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/mapping_audit_holdout"))
    parser.add_argument("--exclude-metadata", type=Path, default=Path("outputs/selective_repair_crossfit/run_metadata.json"))
    parser.add_argument("--sequences", default="", help="Optional comma-separated explicit QC-approved training IDs.")
    parser.add_argument("--max-per-action", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--gap-every", type=int, default=5, help="Unused for clean inference; retained for API consistency.")
    args = parser.parse_args()
    if args.max_per_action < 1:
        parser.error("--max-per-action must be >= 1")

    dataset_root = args.dataset_root.resolve()
    manifest_path = args.manifest.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    eligible = load_eligible_manifest(manifest_path)
    sequences = choose_sequences(
        eligible, args.sequences, args.max_per_action, args.seed, args.exclude_metadata.resolve()
    )
    if not sequences:
        raise ValueError("No eligible held-out sequences were selected.")
    if len(set(sequences)) != len(sequences):
        raise ValueError("Duplicate sequence IDs selected.")

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is required in the active environment") from exc

    print(f"Auditing {len(sequences)} clean official-training sequences.", flush=True)
    print("Direct = MediaPipe anatomical left/right order; swapped = pairwise left/right reversal.", flush=True)
    seq_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    contact_tiles: list[np.ndarray] = []
    for si, seq in enumerate(sequences, start=1):
        data = load_sequence(dataset_root, seq)  # rejects official test sequences
        frames = data["frames"]
        gt_xy, gt_vis, bbox = data["gt_xy"], data["gt_vis"], data["bbox_xyxy"]
        print(f"[{si}/{len(sequences)}] {seq} ({data['action']}), {len(frames)} frames", flush=True)
        # Existing extraction returns the currently configured swapped mapping.
        pred_swapped, conf, sizes = extract_predictions(
            frames, "clean", mp.solutions.pose, args.gap_every, seq
        )
        pred_direct = pred_swapped[:, PAIR_SWAP, :]
        masks = ground_truth_masks_and_scales(gt_xy, gt_vis, bbox, sizes)
        target = masks["valid_gt"]
        scale = masks["torso_scale"]
        target_bbox = masks["valid_gt_bbox"]
        scale_bbox = masks["bbox_diag"]

        # Keep the two normalization schemes separate. Torso-normalized metrics
        # require the left-shoulder/right-hip anchor; bbox metrics do not.
        direct = score_mapping(pred_direct, gt_xy, target, scale)
        swapped = score_mapping(pred_swapped, gt_xy, target, scale)
        direct_bbox = score_mapping(pred_direct, gt_xy, target_bbox, scale_bbox)
        swapped_bbox = score_mapping(pred_swapped, gt_xy, target_bbox, scale_bbox)
        seq_rows.append({
            "sequence": seq,
            "action": data["action"],
            "official_split": "train",
            "n_frames": len(frames),
            "n_scoreable_points_torso": int(target.sum()),
            "n_scoreable_points_bbox": int(target_bbox.sum()),
            "direct_pck_torso_0.20": direct[0],
            "swapped_pck_torso_0.20": swapped[0],
            "delta_swapped_minus_direct_pck_torso": swapped[0] - direct[0],
            "direct_mean_error_torso_norm": direct[1],
            "swapped_mean_error_torso_norm": swapped[1],
            "delta_swapped_minus_direct_error_torso": swapped[1] - direct[1],
            "pck_torso_winner": choose_winner(direct[0], swapped[0], higher_is_better=True),
            "mean_error_torso_winner": choose_winner(direct[1], swapped[1], higher_is_better=False),
            "direct_pck_bbox_diag_0.20": direct_bbox[0],
            "swapped_pck_bbox_diag_0.20": swapped_bbox[0],
            "delta_swapped_minus_direct_pck_bbox": swapped_bbox[0] - direct_bbox[0],
            "direct_mean_error_bbox_diag_norm": direct_bbox[1],
            "swapped_mean_error_bbox_diag_norm": swapped_bbox[1],
            "delta_swapped_minus_direct_error_bbox": swapped_bbox[1] - direct_bbox[1],
            "pck_bbox_winner": choose_winner(direct_bbox[0], swapped_bbox[0], higher_is_better=True),
            "mean_error_bbox_winner": choose_winner(direct_bbox[1], swapped_bbox[1], higher_is_better=False),
        })
        for j, name in enumerate(JOINT_NAMES):
            joint_rows.append({
                "sequence": seq, "action": data["action"], "joint_index": j, "joint": name,
                "n_scoreable_frames_torso": int(target[:, j].sum()),
                "direct_pck_torso_0.20": direct[2][j],
                "swapped_pck_torso_0.20": swapped[2][j],
                "delta_swapped_minus_direct_pck_torso": swapped[2][j] - direct[2][j],
                "direct_mean_error_torso_norm": direct[3][j],
                "swapped_mean_error_torso_norm": swapped[3][j],
                "delta_swapped_minus_direct_error_torso": swapped[3][j] - direct[3][j],
                "n_scoreable_frames_bbox": int(target_bbox[:, j].sum()),
                "direct_pck_bbox_diag_0.20": direct_bbox[2][j],
                "swapped_pck_bbox_diag_0.20": swapped_bbox[2][j],
                "delta_swapped_minus_direct_pck_bbox": swapped_bbox[2][j] - direct_bbox[2][j],
                "direct_mean_error_bbox_diag_norm": direct_bbox[3][j],
                "swapped_mean_error_bbox_diag_norm": swapped_bbox[3][j],
                "delta_swapped_minus_direct_error_bbox": swapped_bbox[3][j] - direct_bbox[3][j],
            })

        # Visualize all visible in-frame annotations, including sequences that lack
        # torso anchors. The metrics above continue to use their separate masks.
        visible_gt = masks["valid_gt_any_scale"]
        visible_count = visible_gt.sum(axis=1)
        best_frame = int(np.argmax(visible_count)) if len(visible_count) else 0
        img = cv2.imread(str(frames[best_frame]))
        if img is not None:
            valid_gt_frame = visible_gt[best_frame]
            panel_w = min(640, img.shape[1])
            panel_h = max(1, round(img.shape[0] * panel_w / img.shape[1]))
            base = cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            # Coordinates are in original pixel space; scale overlay coordinates to preview.
            gt_preview = gt_xy[best_frame].copy()
            direct_preview = pred_direct[best_frame].copy()
            swapped_preview = pred_swapped[best_frame].copy()
            sx = panel_w / img.shape[1]
            sy = panel_h / img.shape[0]
            for pts in (gt_preview, direct_preview, swapped_preview):
                pts[:, 0] *= sx
                pts[:, 1] *= sy
            direct_panel = annotate_panel(base, gt_preview, direct_preview, valid_gt_frame,
                                          f"{seq} {data['action']} | DIRECT")
            swapped_panel = annotate_panel(base, gt_preview, swapped_preview, valid_gt_frame,
                                           f"{seq} {data['action']} | SWAPPED")
            joined = cv2.hconcat([direct_panel, swapped_panel])
            cv2.imwrite(str(out / f"{seq}_mapping_comparison.jpg"), joined)
            thumb = cv2.resize(joined, (560, max(1, round(joined.shape[0] * 560 / joined.shape[1]))),
                               interpolation=cv2.INTER_AREA)
            contact_tiles.append(thumb)

    per_seq = pd.DataFrame(seq_rows)
    per_seq.to_csv(out / "mapping_audit_per_sequence.csv", index=False)
    per_joint = pd.DataFrame(joint_rows)
    per_joint.to_csv(out / "mapping_audit_per_joint.csv", index=False)

    action_rows: list[dict[str, Any]] = []
    for action, group in per_seq.groupby("action"):
        action_rows.append({
            "action": action,
            "n_sequences": int(group["sequence"].nunique()),
            "n_sequences_with_torso_scores": int(group["direct_pck_torso_0.20"].notna().sum()),
            "direct_pck_torso_mean_across_sequences": float(group["direct_pck_torso_0.20"].mean()),
            "swapped_pck_torso_mean_across_sequences": float(group["swapped_pck_torso_0.20"].mean()),
            "delta_swapped_minus_direct_pck_torso": float(group["delta_swapped_minus_direct_pck_torso"].mean()),
            "direct_error_torso_mean_across_sequences": float(group["direct_mean_error_torso_norm"].mean()),
            "swapped_error_torso_mean_across_sequences": float(group["swapped_mean_error_torso_norm"].mean()),
            "delta_swapped_minus_direct_error_torso": float(group["delta_swapped_minus_direct_error_torso"].mean()),
            "torso_sequences_won_by_direct_pck": int((group["pck_torso_winner"] == "direct").sum()),
            "torso_sequences_won_by_swapped_pck": int((group["pck_torso_winner"] == "swapped").sum()),
            "n_sequences_with_bbox_scores": int(group["direct_pck_bbox_diag_0.20"].notna().sum()),
            "direct_pck_bbox_mean_across_sequences": float(group["direct_pck_bbox_diag_0.20"].mean()),
            "swapped_pck_bbox_mean_across_sequences": float(group["swapped_pck_bbox_diag_0.20"].mean()),
            "delta_swapped_minus_direct_pck_bbox": float(group["delta_swapped_minus_direct_pck_bbox"].mean()),
            "direct_error_bbox_mean_across_sequences": float(group["direct_mean_error_bbox_diag_norm"].mean()),
            "swapped_error_bbox_mean_across_sequences": float(group["swapped_mean_error_bbox_diag_norm"].mean()),
            "delta_swapped_minus_direct_error_bbox": float(group["delta_swapped_minus_direct_error_bbox"].mean()),
            "bbox_sequences_won_by_direct_pck": int((group["pck_bbox_winner"] == "direct").sum()),
            "bbox_sequences_won_by_swapped_pck": int((group["pck_bbox_winner"] == "swapped").sum()),
        })
    pd.DataFrame(action_rows).to_csv(out / "mapping_audit_by_action.csv", index=False)

    if contact_tiles:
        cols = 2
        thumb_h = max(tile.shape[0] for tile in contact_tiles)
        padded = []
        for tile in contact_tiles:
            pad = np.zeros((thumb_h, tile.shape[1], 3), dtype=np.uint8)
            pad[:tile.shape[0], :tile.shape[1]] = tile
            padded.append(pad)
        rows = (len(padded) + cols - 1) // cols
        blank = np.zeros_like(padded[0])
        grid_rows = []
        for r in range(rows):
            row_tiles = []
            for c in range(cols):
                idx = r * cols + c
                row_tiles.append(padded[idx] if idx < len(padded) else blank)
            grid_rows.append(cv2.hconcat(row_tiles))
        cv2.imwrite(str(out / "mapping_audit_contact_sheet.jpg"), cv2.vconcat(grid_rows))

    meta = {
        "purpose": "Independent clean-frame comparison of direct and swapped left/right correspondence.",
        "dataset": "Penn Action",
        "official_split": "training only; load_sequence validates the MATLAB train flag",
        "sequences": sequences,
        "excluded_sequence_metadata": str(args.exclude_metadata.resolve()),
        "baseline_current_repository_mapping": MEDIAPIPE_JOINTS.tolist(),
        "direct_mapping_is_pairwise_swap_of_current": True,
        "primary_metric": "PCK@0.20 normalized by distance between Penn Action left shoulder and right hip; requires both torso anchors visible and in bounds",
        "secondary_metric": "PCK@0.20 normalized by Penn Action bounding-box diagonal; includes visible in-bounds annotated joints even when torso anchors are unavailable",
        "missing_predictions_count_as_incorrect": True,
        "mapping_is_not_automatically_changed": True,
        "limitations": [
            "This script compares two global left/right mappings; it does not infer camera view.",
            "The selected frame overlays are manual diagnostics, not a substitute for annotation documentation.",
            "A few sequences per action do not establish a universal mapping across viewpoints.",
            "No official test split is read.",
        ],
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n=== Mapping audit summary: development holdout only ===")
    print(per_seq[["sequence", "action", "n_scoreable_points_torso", "n_scoreable_points_bbox",
                   "direct_pck_torso_0.20", "swapped_pck_torso_0.20",
                   "delta_swapped_minus_direct_pck_torso",
                   "direct_pck_bbox_diag_0.20", "swapped_pck_bbox_diag_0.20",
                   "delta_swapped_minus_direct_pck_bbox"]].to_string(index=False))
    print(f"\nSaved audit outputs to {out}")
    print("Do not change the global mapping based solely on aggregate scores; inspect per-sequence metrics and overlays.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
