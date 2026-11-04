# FrameTrust Candidate-Transition Audit

This patch adds a diagnostic for the nested sequence-cross-fitted selector. It reconstructs the actual **nested-threshold** selections from the compatible feature cache and `nested_threshold_selection.csv`, then measures candidate switching, dwell lengths, and temporal error on transition-containing versus stable three-frame windows.

## Why this audit uses the feature cache

Do **not** use `per_point_choices.csv` for this analysis. That file contains choices from the fixed `--min-predicted-gain` margin. The temporal-fidelity results were produced from per-sequence margins selected by nested inner folds. This script reconstructs those same nested choices by refitting on all other selected sequences and applying each held-out sequence's entry in `nested_threshold_selection.csv`.

Ground-truth labels are not used to make the held-out choices. They are used only after reconstruction to score localization and temporal outcomes. The script validates the Penn Action training flag through the existing `load_sequence` function and does not read the official test split.

## Install

Extract this ZIP into the FrameTrust repository root, preserving the `scripts/` and `tests/` folders. It adds files only:

- `scripts/audit_candidate_transitions.py`
- `src/frametrust/transition_audit.py`
- `tests/test_candidate_transition_audit.py`
- `README_CANDIDATE_TRANSITION_AUDIT.md`

## Test

Activate the existing `frametrust310` environment, go to the FrameTrust repository root, then run:

```powershell
python -m pytest -q
```

## Run

The audit uses the existing output from `evaluate_selective_repair.py` and the existing Penn Action training data. No MediaPipe inference is rerun.

```powershell
python scripts/audit_candidate_transitions.py --out outputs/selective_repair_crossfit
```

Optional settings:

```powershell
python scripts/audit_candidate_transitions.py --out outputs/selective_repair_crossfit --fps 30 --bootstrap-samples 5000 --seed 20261104
```

## Outputs

All files are written into the supplied output directory:

- `candidate_transition_sequence_summary.csv`: per-sequence transition rate, repair entry/exit, repair-method switches, and candidate selection fractions.
- `candidate_transition_condition_summary.csv`: sequence-macro condition summaries.
- `candidate_transition_matrix_by_sequence.csv` and `candidate_transition_matrix_condition_summary.csv`: from/to candidate switch counts.
- `candidate_transition_events.csv`: each frame/joint switch and its type.
- `candidate_dwell_by_sequence.csv`: per-candidate run/dwell-length statistics.
- `candidate_transition_temporal_by_sequence.csv`: temporal metrics for transition-containing and stable three-frame windows.
- `candidate_transition_temporal_condition_summary.csv`: condition-level sequence-macro metrics by window group.
- `candidate_transition_temporal_paired.csv`: within-sequence transition-minus-stable three-frame deltas with sequence-bootstrap intervals.
- `candidate_transition_velocity_by_sequence.csv` and `candidate_transition_velocity_paired.csv`: one-step velocity error on switch intervals versus stable intervals, with paired sequence-bootstrap intervals.
- `candidate_transition_audit_metadata.json`: protocol, provenance, and limitations.

## Interpretation limits

A transition is a change in selected candidate identity for one joint between adjacent frames. A transition-containing temporal triple includes at least one such change across the two frame intervals in the triple. Stable triples use one selected candidate identity throughout the three frames.

A higher temporal error in transition windows is **not causal proof** that switching created the error. Occlusion, difficult motion, missing detections, or confidence fluctuations can cause both switching and temporal error. Interpret the audit as a mechanism diagnostic. Report sequence-level uncertainty and do not select a temporal penalty or tune hyperparameters on these same outcomes without a new nested validation layer.
