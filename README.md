# FrameTrust: Reliable Human-Pose Tracking from Ordinary Video

**A camera-only, CPU-friendly computer-vision project. No wearable sensors or special hardware are required.**

FrameTrust studies how common video problems—partial occlusion, blur, low resolution, and repeated/dropped frames—affect monocular human-pose estimates, then compares a simple confidence-aware temporal repair method with uncorrected estimates and linear interpolation.

## Research question

> When pose estimation becomes unreliable because video is degraded, can a lightweight, confidence-aware temporal repair step recover more temporally stable landmarks without introducing excessive pose error?

This is an experimental project, not a claim of a novel state-of-the-art architecture. Video-based temporal pose estimation and occlusion robustness are active research areas; the defensible contribution must come from careful experimental design, transparent metrics, and a useful low-compute evaluation—not from claiming the broad problem is new.

## What it does

1. Loads a local video (`.mp4`, `.mov`, etc.) with OpenCV.
2. Runs MediaPipe Pose on a clean reference pass and on controlled degraded versions of the same frames.
3. Simulates frame repetition (a simple dropped-frame proxy), a central rectangular occlusion, Gaussian blur, and downsampling.
4. Compares raw landmarks, linear interpolation, and confidence-aware interpolation plus adaptive smoothing.
5. Reports landmark agreement against the clean model as an explicitly labeled diagnostic. With an optional manual annotation CSV, it also reports independently measured pose error, prediction coverage, and PCK.

## Quick start (Windows PowerShell)

Use Python 3.10 or 3.11 in a fresh virtual environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python scripts/run_experiment.py --video path\to\your_video.mp4 --out outputs\first_run --max-frames 300
```

To run the full clip, omit `--max-frames`. A 300-frame smoke run is a sensible first test. Pose inference runs several passes over the same video, so runtime depends on clip length and CPU speed.

If you already have an environment where `mediapipe` and OpenCV work, activate it and run `pip install -e .`.


**MediaPipe compatibility:** FrameTrust currently calls the legacy `mp.solutions.pose` API. MediaPipe removed that API from Python packages starting with 0.10.31, so this project deliberately pins MediaPipe to `>=0.10.14,<0.10.31`. A future migration to the MediaPipe Tasks API should remove this upper bound.


## Ground-truth evaluation (recommended before making accuracy claims)

The clean MediaPipe pass is not ground truth. To measure actual pose error without buying sensors or downloading a large dataset, annotate a sparse set of frames from the same video. The included OpenCV tool asks you to click 13 major landmarks on sampled frames.

First create annotations:

```powershell
python scripts/annotate_video.py --video path\to\your_video.mp4 --out data\annotations.csv --stride 15 --max-frames 300
```

In the annotation window, click the displayed joint; press `s` when that joint is not visible, `u` to undo the previous point, and `q` to save and quit. The CSV stores normalized image coordinates in [0, 1], original frame indices, MediaPipe joint indices, and visibility labels. Use a single-person clip and annotate the same original video that will be evaluated. Only manually labeled points are used; joints you did not reach before quitting are not treated as invisible.

Then run the experiment with those labels:

```powershell
python scripts/run_experiment.py --video path\to\your_video.mp4 --ground-truth data\annotations.csv --out outputs\annotated_run --max-frames 300
```

Ground-truth columns added to `metrics.csv` include:

- `gt_mean_error_normalized_image`: mean Euclidean x/y error in normalized image coordinates, on detected points.
- `gt_mean_error_body_diagonal`: mean error divided by the annotated visible-joint bounding-box diagonal for that frame.
- `gt_pck`: fraction of labeled points whose error is within the configured fraction of that frame's body-box diagonal (default 0.10).
- `gt_prediction_coverage`: fraction of visible labeled points for which the estimator returned finite coordinates.

Missing predictions count as PCK failures; coverage is reported so the error among detected points cannot conceal failures. Frames with fewer than two visible annotations cannot define the bounding-box scale and are omitted from these scale-normalized metrics. These are transparent, project-specific metrics—not a claim of compatibility with an official benchmark protocol. Repeated annotations, invalid coordinates, and joint indices outside the estimator's output are rejected.

## Outputs

- `landmarks_clean.csv`, `landmarks_frame_drop.csv`, etc.: extracted landmarks and confidence/visibility per frame.
- `metrics.csv`: one row per corruption × method.
- `joint_error_by_condition.png`: agreement diagnostic against the clean model predictions (not ground-truth accuracy).
- `ground_truth_pck_by_condition.png` and `ground_truth_error_by_condition.png`: created when valid manual annotations are supplied.
- `run_metadata.json`: configuration, frame counts, and explicit metric caveats.

## Experimental protocol

- First run on a short, clear clip with one person and a mostly visible body.
- Repeat with several clips and viewpoints; do not draw conclusions from one video.
- For every degradation, compare `raw`, `linear_interpolation`, and `confidence_aware` on exactly the same frames.
- Report mean and standard deviation across clips, not just pooled frame counts.
- Preserve a clip-level split if tuning thresholds: choose thresholds on development clips and report final metrics on held-out clips.
- If you need ground-truth accuracy claims, evaluate against manually annotated landmarks or a dataset with appropriate ground-truth labels. The clean model's outputs are **not** ground truth.

## Limitations

- The default occlusion is a synthetic black rectangle, not realistic object/person occlusion.
- Frame dropping is simulated by repeating the previous frame to preserve frame count; it approximates a frozen/missed update rather than physically removing timestamps.
- Results depend on the estimator, camera, clip, person scale, and motion.
- The manual annotation tool is a sparse single-person workflow; annotation quality and number of labeled frames constrain the strength of any conclusion.
- Interpolation can create unrealistic trajectories over long occlusions. Long-gap performance must be reported separately.
- This starter performs 2D monocular pose analysis; it does not measure true 3D anatomy or clinical correctness.

## Research positioning

Current video-pose work explicitly studies temporal consistency and failure under occlusion, blur, and difficult motion. FrameTrust should be positioned as a reproducible, low-compute robustness and recovery study. Do not claim that temporal pose estimation or occlusion robustness is itself novel. A stronger contribution would require a clear hypothesis, multi-clip evaluation, ablations, and evidence that the proposed repair method improves an agreed metric without hiding failure cases.

## Erasmus relevance

- **Computer vision / AI:** pose estimation, temporal modeling, perturbation benchmarks, reproducibility.
- **IMLEX / intelligent systems:** video perception, robustness, efficient inference, human-motion understanding.
- **Biomedical engineering / sensing:** reliability of motion measurements and sensitivity to missing or low-confidence observations.

This is a coherent transferable project, not a guarantee that it matches every programme's curriculum. Tailor the emphasis to each programme.

## Suggested next milestones

1. Verify extraction on your own short video.
2. Run at least 5–10 varied clips and aggregate metrics by clip.
3. Add realistic occlusion masks or annotated public video data.
4. Compare against a stronger temporal baseline and run ablations for confidence thresholds and smoothing strength.
5. Write results only after real video experiments have run; do not present synthetic unit-test inputs as experimental evidence.

## Citation and reproducibility

See [`SOURCES.md`](SOURCES.md) for research context and dataset notes. Cite MediaPipe and the papers relevant to any model and benchmark used in your final work. Respect licenses and privacy requirements for every video.
