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

The clean MediaPipe pass is not ground truth. To measure actual pose error without buying sensors or downloading a large dataset, annotate a sparse set of frames from the same video. The annotation tool supports video-specific landmark profiles. Its default `squat` profile asks for eight lower-body landmarks (left/right hips, knees, ankles, and foot indices). Use `upper_body` for the nose, shoulders, elbows, and wrists when lower-body landmarks are not reliably visible; use `full_body` only when the framing supports all 13 upper- and lower-body landmarks.


For an upper-body-only clip, pass `--profile upper_body`; the manual labels should match the landmarks you intend to evaluate.

First create annotations:

```powershell
python scripts/annotate_video.py --video path\to\your_video.mp4 --out data\annotations.csv --stride 15 --max-frames 300
```

In the annotation window, click the requested landmark if it is visible; press `s` only when that landmark is genuinely out of frame or occluded, `u` to undo the previous point, and `Enter` to accept a completed frame. Press `q` to save and quit early. To resume on later sampled frames without redoing earlier labels, use `--start-frame 60` with a separate output CSV; the start frame must align with the stride. The squat profile focuses evaluation on lower-body pose reliability, not overall full-body accuracy. The CSV stores normalized image coordinates in [0, 1], original frame indices, MediaPipe landmark indices, and visibility labels. The annotator uses a fixed-size preview and maps clicks into normalized image coordinates, including for high-resolution videos. If annotations were created with an earlier version, regenerate them after updating the repository; earlier window-scaling behavior could misplace points. Use the same original video for annotation and evaluation. Only manually labeled points are used; landmarks not reached before quitting are not treated as invisible.

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

## Smoothing-strength ablation (reuse existing predictions)

After an annotated run, compare the default confidence-aware smoother with weaker smoothing settings without re-running MediaPipe inference:

```powershell
python scripts/sweep_repair.py --landmarks-dir outputs/annotated_upper_body_v2 --ground-truth data/annotations_upper_body_v2.csv --profile upper_body --out outputs/repair_sweep_upper_body.csv
```

The sweep reports temporal jitter only for the selected anatomical profile, alongside ground-truth error, prediction coverage, and PCK.
The smoothing sweep also reports common-support localization error/PCK: these metrics compare each repair output only on labeled joints where the raw detector had finite coordinates. Overall ground-truth PCK and prediction coverage remain reported separately to account for repaired/missing predictions. It evaluates the current smoothing strengths (`alpha_low=0.20`, `alpha_high=0.75`) and three progressively weaker-smoothing settings. Treat this as an exploratory ablation; five labeled frames from one clip are not enough to claim general performance. Prefer settings that reduce profile-specific jitter without worsening manual-label error or PCK, then validate the selected setting on more labeled frames or other clips.

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


## Selective repair (research development)

A new experimental selector compares the existing candidate repairs per joint and frame. It uses only inference-time signals—reported confidence, missingness, local motion residuals, gap length, and candidate disagreement—to predict candidate localization error. It preserves the raw estimate unless the best repair's predicted torso-normalized error improves by at least a configured margin. The selector is deliberately allowed to abstain, including when the raw point is missing.

The cross-fitted evaluator trains a separate selector for each condition and holds out an entire sequence at a time. Ground-truth coordinates are used to make training targets for other sequences and to score the held-out sequence; they are not used as selector features. It runs only on the official Penn Action training split and is **not** a test-set evaluation.

Run from the repository root:

```powershell
python scripts/evaluate_selective_repair.py --max-per-action 2 --conditions clean,blur,low_resolution,occlusion,missing_pose_observations --min-predicted-gain 0.01 --out outputs/selective_repair_crossfit
```

Outputs include per-sequence metrics, per-point method choices, condition summaries, protocol metadata, and a nested margin-selection evaluation. By default, each outer held-out sequence is excluded while an action-balanced inner-fold evaluation selects the largest repair margin whose mean PCK is within 0.5 percentage points of the best feasible inner PCK, subject to mean coverage of at least 0.90 and false-correction rate no greater than 0.01. If those guardrails cannot both be met, the metadata records which guardrail was relaxed. This selection rule is a prespecified development choice; treat results as exploratory and check sensitivity to inner-fold assignments before locking a final protocol. The selector currently uses two-sided temporal features and is therefore an offline method, not a causal real-time filter.

The nested outputs are `nested_threshold_selection.csv`, `nested_inner_threshold_sweep.csv`, `nested_selective_sequence_metrics.csv`, `nested_selective_condition_summary.csv`, and `nested_selective_paired_comparison.csv`. The official test split is not read by the evaluator. The original `threshold_sweep_summary.csv` remains descriptive only and must not be used to report a selected optimum threshold.

Per-sequence inference features, candidate coordinates, and error targets are also saved under `feature_cache/` in the output directory. After an initial run, a threshold or inner-fold sensitivity analysis with the same sequence set and inference settings can use `--reuse-feature-cache` to avoid rerunning MediaPipe. The cache includes a configuration signature and refuses reuse if the selected sequences, conditions, landmark mapping, or inference settings differ.

**Important limitation:** a follow-up clean-frame audit on 15 additional QC-approved official-training sequences favored the pairwise-swapped mapping: with torso normalization, it had higher PCK on 9 of 10 scoreable sequences; with bounding-box-diagonal normalization, it had higher PCK on 13 of 15 sequences (sequence-macro PCK 0.696 direct vs 0.873 swapped). This supports retaining the swapped mapping for development, but does not prove a universal mapping across viewpoints; the `strum_guitar` sequence was a counterexample under the bbox metric. See [the recorded development results](docs/selective-repair-development-results.md). The selector remains a research hypothesis, not an established improvement. Report accuracy, coverage, false-correction rate, and repair fraction together; results on a development subset do not establish generalization.

## Independent landmark-mapping audit

Before further selector tuning, compare the direct MediaPipe left/right correspondence with the pairwise-swapped mapping on a separate, QC-approved subset of the Penn Action **training** split. By default, the audit excludes the sequence IDs recorded in the selective-repair development run metadata and samples up to one additional sequence per action.

```powershell
python scripts/audit_penn_action_mapping.py --out outputs/mapping_audit_holdout
```

Outputs include per-sequence and per-joint CSV metrics, action summaries, per-sequence overlay images, and a contact sheet. Each overlay compares direct and swapped predictions against the same visible Penn Action annotations. Inspect frontal-view examples and per-sequence disagreements rather than choosing a mapping from a pooled score alone. The script does not change the repository's mapping automatically and does not read the official test split.

If the selected sequences must be controlled explicitly, pass `--sequences 0001,0002,...`; these IDs must be QC-approved official-training sequences. The visual audit is diagnostic and does not replace checking the dataset's landmark-annotation convention.
