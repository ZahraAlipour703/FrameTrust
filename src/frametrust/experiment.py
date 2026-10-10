"""Experiment orchestration and output reporting."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .corruptions import CORRUPTIONS
from .ground_truth import evaluate_ground_truth, read_ground_truth_csv
from .metrics import low_confidence_rate, mean_joint_displacement, temporal_jitter
from .pose import arrays_to_frame, extract_video_landmarks, frame_to_arrays
from .repair import confidence_aware_repair, interpolate_missing


def run_experiment(
    video_path: str | Path,
    out_dir: str | Path,
    corruptions: list[str] | None = None,
    max_frames: int | None = None,
    drop_every: int = 5,
    confidence_threshold: float = 0.35,
    ground_truth_path: str | Path | None = None,
    pck_fraction: float = 0.1,
) -> pd.DataFrame:
    """Compare pose estimates on degraded video, optionally scoring manual labels.

    Agreement with the clean model is reported for exploratory diagnostics only.
    When a ground-truth CSV is supplied, metrics include independently labeled
    accuracy measures for each corruption/method combination.
    """
    video_path = Path(video_path).resolve()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    corruption_list = list(corruptions) if corruptions else list(CORRUPTIONS)
    invalid = set(corruption_list) - set(CORRUPTIONS)
    if invalid:
        raise ValueError(f"Unknown corruptions: {sorted(invalid)}")
    if "clean" not in corruption_list:
        corruption_list = ["clean", *corruption_list]

    ground_truth = read_ground_truth_csv(ground_truth_path) if ground_truth_path else None
    if not 0 < pck_fraction <= 1:
        raise ValueError("pck_fraction must be in (0, 1]")

    extracted: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for corruption in dict.fromkeys(corruption_list):
        print(f"[FrameTrust] extracting poses: {corruption}", flush=True)
        df = extract_video_landmarks(video_path, corruption, max_frames, drop_every)
        df.to_csv(out / f"landmarks_{corruption}.csv", index=False)
        coords, conf = frame_to_arrays(df)
        extracted[corruption] = (df, coords, conf)

    _, reference_coords, reference_conf = extracted["clean"]
    # This agreement diagnostic is deliberately masked where clean-pass confidence
    # is low. It remains a pseudo-reference metric, never ground-truth accuracy.
    reference_for_scoring = reference_coords.copy()
    reference_for_scoring[reference_conf < confidence_threshold] = np.nan
    rows: list[dict[str, float | int | str]] = []
    for corruption, (df, coords, conf) in extracted.items():
        methods: dict[str, np.ndarray] = {
            "raw": coords,
            "linear_interpolation": interpolate_missing(
                coords, conf, threshold=confidence_threshold
            ),
            "confidence_aware": confidence_aware_repair(
                coords, conf, threshold=confidence_threshold
            ),
        }
        if corruption == "clean":
            methods = {"clean_reference": reference_coords}

        for method, estimate in methods.items():
            row: dict[str, float | int | str] = {
                "corruption": corruption,
                "method": method,
                "frames": int(len(df)),
                "mean_joint_displacement_vs_clean_model": mean_joint_displacement(
                    reference_for_scoring, estimate
                ),
                "low_confidence_rate_input": low_confidence_rate(conf, confidence_threshold),
                "temporal_jitter_normalized_per_frame2": temporal_jitter(estimate),
            }
            if ground_truth is not None:
                row.update(evaluate_ground_truth(
                    ground_truth,
                    df["frame_index"].to_numpy(dtype=int),
                    estimate,
                    pck_fraction=pck_fraction,
                ))
            rows.append(row)

            if method != "clean_reference":
                repaired = arrays_to_frame(df, estimate, method)
                repaired.to_csv(out / f"landmarks_{corruption}_{method}.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    _plot_metrics(metrics, out / "joint_error_by_condition.png")
    if ground_truth is not None:
        _plot_ground_truth_metrics(metrics, out, pck_fraction)

    metadata = {
        "video_path": str(video_path),
        "corruptions": list(dict.fromkeys(corruption_list)),
        "max_frames": max_frames,
        "drop_every": drop_every,
        "confidence_threshold": confidence_threshold,
        "frames_per_condition": {key: int(len(value[0])) for key, value in extracted.items()},
        "reference_caveat": "Clean MediaPipe predictions are a pseudo-reference, not ground-truth landmarks.",
        "frame_drop_caveat": "Every Nth frame is replaced by the previous clean frame to simulate a frozen update.",
        "ground_truth_path": str(Path(ground_truth_path).resolve()) if ground_truth_path else None,
        "ground_truth_protocol": (
            {
                "source": "manually labeled normalized 2D points",
                "pck_fraction_of_visible_body_box_diagonal": pck_fraction,
                "missing_predictions_count_as_pck_failures": True,
                "mean_error_uses_detected_points_only": True,
                "note": "Project-specific protocol; not an official benchmark metric."
            } if ground_truth is not None else None
        ),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[FrameTrust] wrote {out / 'metrics.csv'}", flush=True)
    return metrics


def _plot_metrics(metrics: pd.DataFrame, out_path: Path) -> None:
    plot_data = metrics[
        metrics["method"].isin(["raw", "linear_interpolation", "confidence_aware"])
    ].copy()
    if plot_data.empty:
        return
    pivot = plot_data.pivot(
        index="corruption", columns="method",
        values="mean_joint_displacement_vs_clean_model",
    )
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Landmark agreement with clean model predictions (pseudo-reference)")
    ax.set_ylabel("Mean XY displacement (normalized image coordinates)")
    ax.set_xlabel("Input video condition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Output method")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_ground_truth_metrics(metrics: pd.DataFrame, out_dir: Path, pck_fraction: float) -> None:
    """Create ground-truth accuracy plots only when labeled points are available."""
    plot_data = metrics[metrics["gt_evaluable_points"] > 0].copy()
    if plot_data.empty:
        print("[FrameTrust] no evaluable ground-truth points overlap this run; skipping GT plots.")
        return

    pck_pivot = plot_data.pivot(index="corruption", columns="method", values="gt_pck")
    ax = pck_pivot.plot(kind="bar", figsize=(10, 5), ylim=(0, 1))
    ax.set_title(f"Ground-truth PCK (threshold = {pck_fraction:.2f} × body-box diagonal)")
    ax.set_ylabel("PCK (higher is better)")
    ax.set_xlabel("Input video condition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Output method")
    plt.tight_layout()
    plt.savefig(out_dir / "ground_truth_pck_by_condition.png", dpi=160)
    plt.close()

    error_data = plot_data[np.isfinite(plot_data["gt_mean_error_body_diagonal"])].copy()
    if not error_data.empty:
        error_pivot = error_data.pivot(
            index="corruption", columns="method", values="gt_mean_error_body_diagonal"
        )
        ax = error_pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_title("Ground-truth mean joint error")
        ax.set_ylabel("Mean distance / annotated body-box diagonal (lower is better)")
        ax.set_xlabel("Input video condition")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Output method")
        plt.tight_layout()
        plt.savefig(out_dir / "ground_truth_error_by_condition.png", dpi=160)
        plt.close()
