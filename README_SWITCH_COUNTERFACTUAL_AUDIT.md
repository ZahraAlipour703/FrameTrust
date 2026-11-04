# FrameTrust: Counterfactual Switch-Boundary Audit

This diagnostic follows the Candidate-Transition Audit. It asks a narrower question:
when the nested selector changes candidate identity between adjacent frames, does
its hybrid output incur more one-step motion error than continuing the previous or
new candidate across that *same pair of frames*?

## Files

- `scripts/audit_switch_counterfactuals.py`
- `src/frametrust/switch_counterfactual.py`
- `tests/test_switch_counterfactual.py`
- `README_SWITCH_COUNTERFACTUAL_AUDIT.md`

## Install

Extract this ZIP into the FrameTrust repository root, preserving folders. It adds
new files and does not overwrite existing files.

## Run

Activate the existing `frametrust310` environment and run:

```powershell
python -m pytest -q
python scripts/audit_switch_counterfactuals.py --out outputs/selective_repair_crossfit
```

This reads the existing compatible feature cache and nested threshold table. It
does not rerun MediaPipe. The script reconstructs the outer leave-one-sequence-out
selector exactly as used by the nested temporal evaluator, and checks each
sequence's Penn Action training flag through the existing loader. It does not read
the official Penn Action test split.

## Interpretation

For an observed switch `old -> new` at frames `t-1 -> t`, the actual hybrid interval
uses `old(t-1)` and `new(t)`. The audit compares it with `old(t-1) -> old(t)` and
`new(t-1) -> new(t)` on the same GT-valid interval. Positive
`hybrid_minus_*_continuation` means the observed hybrid interval has larger
one-step velocity error than that continuation.

The best-continuation comparator takes the lower error of the two continuation
alternatives using GT after the fact. It is an **oracle diagnostic only**, not an
inference-time method and not a fair deployable baseline. A continuation is scored
only when that candidate is finite at both endpoints. Support counts are therefore
essential.

Outputs:

- `candidate_switch_counterfactual_events.csv`
- `candidate_switch_counterfactual_by_sequence.csv`
- `candidate_switch_counterfactual_condition_paired.csv`
- `candidate_switch_counterfactual_type_paired.csv`
- `candidate_switch_counterfactual_event_support.csv`
- `candidate_switch_counterfactual_metadata.json`

All bootstrap intervals resample sequences after first averaging events within a
sequence. Results are exploratory, development-only, and do not establish
causality. A positive result would justify designing a temporal-consistency-aware
selector for a new nested development experiment; it would not itself validate one.
