#!/usr/bin/env python3
"""Select a held-out Penn Action training subset for landmark-mapping validation.

Runs MediaPipe on QC-approved training sequences excluded from the previous
30-sequence development sample. It compares the corrected side mapping with the
prior mapping on identical GT points, frames, and normalization anchors.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CORRECTED_MAPPING = "[12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27]"
OPPOSITE = np.array([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10], dtype=int)


def mat_text(v: Any) -> str:
    a = np.asarray(v).squeeze()
    if a.dtype.kind in "US":
        return "".join(str(x) for x in a.reshape(-1)).strip()
    if a.size == 1:
        x = a.item()
        return x.decode().strip() if isinstance(x, bytes) else str(x)
    return str(a.tolist())


def read_seq_meta(root: Path, sequence: str) -> dict[str, str]:
    d = loadmat(root / "labels" / f"{sequence}.mat")
    return {"action": mat_text(d["action"]), "pose": mat_text(d["pose"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=Path("data/external/Penn_Action"))
    ap.add_argument("--manifest", type=Path, default=Path("data/external/penn_action_qc_manifest.csv"))
    ap.add_argument("--exclude-selection", type=Path, default=Path("outputs/penn_action_strata_dev30_mappingfix/selected_development_sequences.csv"))
    ap.add_argument("--per-action", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--out", type=Path, default=Path("outputs/penn_action_mapping_holdout"))
    args = ap.parse_args()

    if args.per_action < 1:
        ap.error("--per-action must be >= 1")
    data_root, manifest_path, excluded_path, out = (
        args.dataset_root.resolve(), args.manifest.resolve(),
        args.exclude_selection.resolve(), args.out.resolve()
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not excluded_path.is_file():
        raise FileNotFoundError(
            f"Original 30-sequence selection not found: {excluded_path}. "
            "Run the corrected development experiment first."
        )
    audit_script = ROOT / "scripts" / "penn_action_alignment_audit.py"
    audit_text = audit_script.read_text(encoding="utf-8")
    if EXPECTED_CORRECTED_MAPPING not in audit_text:
        raise RuntimeError("Expected corrected mapping was not found in the alignment audit; refusing to label results incorrectly.")

    manifest = pd.read_csv(manifest_path, dtype={"sequence": str})
    required = {"sequence", "split", "include_primary", "action"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    manifest["sequence"] = manifest["sequence"].astype(str).str.zfill(4)
    include = manifest["include_primary"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    eligible = manifest[(manifest["split"].astype(str).str.lower() == "train") & include].copy()
    if "visible_points" in eligible:
        eligible = eligible[pd.to_numeric(eligible["visible_points"], errors="coerce").fillna(0) > 0]
    prior = pd.read_csv(excluded_path, dtype={"sequence": str})
    excluded = set(prior["sequence"].astype(str).str.zfill(4))
    excluded.update({"1797", "1219", "1033"})
    eligible = eligible[~eligible["sequence"].isin(excluded)].copy()
    if eligible.empty:
        raise ValueError("No unused QC-approved training sequences remain.")

    rng = np.random.default_rng(args.seed)
    chosen: list[str] = []
    for action in sorted(eligible["action"].astype(str).unique()):
        ids = sorted(eligible.loc[eligible["action"].astype(str) == action, "sequence"].unique())
        take = min(args.per_action, len(ids))
        chosen.extend(sorted(rng.choice(np.asarray(ids, dtype=object), size=take, replace=False).tolist()))
    if not chosen:
        raise ValueError("Mapping-validation sequence selection is empty.")

    out.mkdir(parents=True, exist_ok=True)
    selection = eligible[eligible["sequence"].isin(chosen)].drop_duplicates("sequence").sort_values(["action", "sequence"])
    selection.to_csv(out / "selected_mapping_validation_sequences.csv", index=False)
    print(f"Selected {len(selection)} new QC-approved training sequences over {selection['action'].nunique()} actions.")
    print("Sequences:", ",".join(selection["sequence"].tolist()))

    audit_out = out / "alignment_audit"
    cmd = [
        sys.executable, str(audit_script),
        "--dataset-root", str(data_root),
        "--sequences", ",".join(selection["sequence"].tolist()),
        "--n-overlays", "1",
        "--out", str(audit_out),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    raw = pd.read_csv(audit_out / "per_joint_alignment.csv")
    for col in ["frame", "joint_index", "gt_x", "gt_y", "pred_x", "pred_y", "torso_scale_px"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    for col in ["gt_visible", "gt_in_bounds"]:
        raw[col] = raw[col].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    rows = []
    for seq, sg in raw.groupby("sequence", sort=True):
        seq = str(seq).zfill(4)
        current, prior_mapping = [], []
        valid_frames = 0
        for _, fg in sg.groupby("frame", sort=True):
            fg = fg.sort_values("joint_index")
            if len(fg) != 12:
                continue
            gt = fg[["gt_x", "gt_y"]].to_numpy(float)
            pred = fg[["pred_x", "pred_y"]].to_numpy(float)
            gt_ok = fg["gt_visible"].to_numpy(bool) & fg["gt_in_bounds"].to_numpy(bool) & np.isfinite(gt).all(axis=1)
            pred_ok = np.isfinite(pred).all(axis=1)
            scale = fg["torso_scale_px"].iloc[0]
            if not np.isfinite(scale) or scale <= 1e-6:
                continue
            paired = gt_ok & pred_ok & pred_ok[OPPOSITE]
            idx = np.flatnonzero(paired)
            if not len(idx):
                continue
            current.extend((np.linalg.norm(pred[idx] - gt[idx], axis=1) / scale).tolist())
            prior_mapping.extend((np.linalg.norm(pred[OPPOSITE[idx]] - gt[idx], axis=1) / scale).tolist())
            valid_frames += 1
        meta = read_seq_meta(data_root, seq)
        c, p = np.asarray(current, dtype=float), np.asarray(prior_mapping, dtype=float)
        rows.append({
            "sequence": seq, "action": meta["action"], "pose": meta["pose"],
            "paired_points": len(c), "valid_frames": valid_frames,
            "corrected_mean_error": float(c.mean()) if len(c) else np.nan,
            "prior_mean_error": float(p.mean()) if len(p) else np.nan,
            "corrected_pck_torso_0.20": float(np.mean(c <= 0.20)) if len(c) else np.nan,
            "prior_pck_torso_0.20": float(np.mean(p <= 0.20)) if len(p) else np.nan,
        })

    result = pd.DataFrame(rows)
    result["pck_gain_corrected_minus_prior"] = result["corrected_pck_torso_0.20"] - result["prior_pck_torso_0.20"]
    result.to_csv(out / "mapping_comparison_by_sequence.csv", index=False)
    valid = result[result["paired_points"] > 0].copy()
    if not valid.empty:
        def macro(group: pd.DataFrame) -> pd.Series:
            return pd.Series({
                "sequences": group["sequence"].nunique(),
                "paired_points": int(group["paired_points"].sum()),
                "corrected_mean_error_macro": group["corrected_mean_error"].mean(),
                "prior_mean_error_macro": group["prior_mean_error"].mean(),
                "corrected_pck_macro": group["corrected_pck_torso_0.20"].mean(),
                "prior_pck_macro": group["prior_pck_torso_0.20"].mean(),
                "pck_gain_macro": group["pck_gain_corrected_minus_prior"].mean(),
                "sequences_corrected_better": int((group["pck_gain_corrected_minus_prior"] > 0).sum()),
            })
        pd.DataFrame([macro(valid)]).to_csv(out / "mapping_comparison_overall.csv", index=False)
        valid.groupby("pose", dropna=False).apply(macro, include_groups=False).reset_index().to_csv(out / "mapping_comparison_by_viewpoint.csv", index=False)
        valid.groupby("action", dropna=False).apply(macro, include_groups=False).reset_index().to_csv(out / "mapping_comparison_by_action.csv", index=False)
        print("\n=== INDEPENDENT SUBSET OVERALL ===")
        print(pd.DataFrame([macro(valid)]).round(4).to_string(index=False))
        print("\n=== BY VIEWPOINT ===")
        print(pd.read_csv(out / "mapping_comparison_by_viewpoint.csv").round(4).to_string(index=False))
        print("\n=== BY ACTION ===")
        print(pd.read_csv(out / "mapping_comparison_by_action.csv").round(4).to_string(index=False))
    else:
        print("WARNING: no points were scorable with the primary torso anchor on this validation subset.")
    print(f"\nSaved mapping validation outputs under: {out}")
    print("Diagnostic uses the training split only and is independent of the original selected 30 sequences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
