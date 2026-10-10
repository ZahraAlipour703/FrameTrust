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
) -> pd.DataFrame:
    """Run clean-reference and degraded-input comparisons on one video."""
    video_path = Path(video_path).resolve()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    corruption_list = corruptions or list(CORRUPTIONS)
    invalid = set(corruption_list) - set(CORRUPTIONS)
    if invalid:
        raise ValueError(f"Unknown corruptions: {sorted(invalid)}")
    if "clean" not in corruption_list:
        corruption_list = ["clean", *corruption_list]
    extracted: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for corruption in dict.fromkeys(corruption_list):
        print(f"[FrameTrust] extracting poses: {corruption}", flush=True)
        df = extract_video_landmarks(video_path, corruption, max_frames, drop_every)
        df.to_csv(out / f"landmarks_{corruption}.csv", index=False)
        coords, conf = frame_to_arrays(df)
        extracted[corruption] = (df, coords, conf)

    _, reference_coords, reference_conf = extracted["clean"]
    # Only score joints that the clean reference pass itself considered visible.
    reference_for_scoring = reference_coords.copy()
    reference_for_scoring[reference_conf < confidence_threshold] = np.nan
    rows: list[dict[str, float | int | str]] = []
    for corruption, (df, coords, conf) in extracted.items():
        methods: dict[str, np.ndarray] = {"raw": coords}
        methods["linear_interpolation"] = interpolate_missing(coords, conf, threshold=confidence_threshold)
        methods["confidence_aware"] = confidence_aware_repair(
            coords, conf, threshold=confidence_threshold
        )
        if corruption == "clean":
            # Keep the clean baseline as a reference row; its agreement error is 0 by definition.
            methods = {"clean_reference": reference_coords}
        for method, estimate in methods.items():
            rows.append({
                "corruption": corruption,
                "method": method,
                "frames": int(len(df)),
                "mean_joint_displacement_vs_clean_model": mean_joint_displacement(reference_for_scoring, estimate),
                "low_confidence_rate_input": low_confidence_rate(conf, confidence_threshold),
                "temporal_jitter_normalized_per_frame2": temporal_jitter(estimate),
            })
            if method != "clean_reference":
                repaired = arrays_to_frame(df, estimate, method)
                repaired.to_csv(out / f"landmarks_{corruption}_{method}.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    plot_path = out / "joint_error_by_condition.png"
    _plot_metrics(metrics, plot_path)
    metadata = {
        "video_path": str(video_path),
        "corruptions": list(dict.fromkeys(corruption_list)),
        "max_frames": max_frames,
        "drop_every": drop_every,
        "confidence_threshold": confidence_threshold,
        "frames_per_condition": {key: int(len(value[0])) for key, value in extracted.items()},
        "reference_caveat": "Clean MediaPipe predictions are a pseudo-reference, not ground-truth landmarks.",
        "frame_drop_caveat": "Every Nth frame is replaced by the previous clean frame to simulate a frozen update.",
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[FrameTrust] wrote {out / 'metrics.csv'}", flush=True)
    return metrics


def _plot_metrics(metrics: pd.DataFrame, out_path: Path) -> None:
    plot_data = metrics[metrics["method"].isin(["raw", "linear_interpolation", "confidence_aware"])].copy()
    if plot_data.empty:
        return
    pivot = plot_data.pivot(index="corruption", columns="method", values="mean_joint_displacement_vs_clean_model")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Landmark agreement with clean model predictions")
    ax.set_ylabel("Mean XY displacement (normalized image coordinates)")
    ax.set_xlabel("Input video condition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Output method")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
