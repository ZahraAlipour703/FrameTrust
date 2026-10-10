"""Deterministic video degradations for controlled robustness experiments."""
from __future__ import annotations

import cv2
import numpy as np

CORRUPTIONS = ("clean", "frame_drop", "occlusion", "blur", "low_resolution")


def corrupt_frame(
    frame: np.ndarray,
    mode: str,
    frame_index: int,
    previous_clean_frame: np.ndarray | None = None,
    drop_every: int = 5,
) -> np.ndarray:
    """Return a degraded copy of a BGR frame.

    ``frame_drop`` repeats the previous frame at regular intervals. This keeps a
    fixed-length time series and models a frozen/missed video update; it is not
    a claim to reproduce every real camera's packet-loss behavior.
    """
    if mode not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption {mode!r}; choose from {CORRUPTIONS}")
    if frame is None or frame.ndim != 3:
        raise ValueError("frame must be a valid H×W×C image")
    out = frame.copy()
    h, w = out.shape[:2]

    if mode == "clean":
        return out
    if mode == "frame_drop":
        if drop_every < 2:
            raise ValueError("drop_every must be >= 2")
        if frame_index > 0 and frame_index % drop_every == 0 and previous_clean_frame is not None:
            return previous_clean_frame.copy()
        return out
    if mode == "occlusion":
        # Fixed, reproducible central torso-sized rectangle, scaled to frame size.
        x1, x2 = int(0.36 * w), int(0.64 * w)
        y1, y2 = int(0.25 * h), int(0.70 * h)
        cv2.rectangle(out, (x1, y1), (max(x1 + 1, x2), max(y1 + 1, y2)), (0, 0, 0), thickness=-1)
        return out
    if mode == "blur":
        # Kernel scales mildly with image dimensions, and is always odd.
        k = max(5, int(round(min(h, w) * 0.025)))
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(out, (k, k), sigmaX=0)
    if mode == "low_resolution":
        small_w, small_h = max(2, w // 4), max(2, h // 4)
        small = cv2.resize(out, (small_w, small_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise AssertionError("unreachable")
