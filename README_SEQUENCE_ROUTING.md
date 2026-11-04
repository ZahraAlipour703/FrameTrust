# FrameTrust Sequence-Level Candidate Routing Prototype

This patch adds a development-only prototype for choosing pose-repair candidates over complete joint trajectories using dynamic programming.

## What it compares

- `nested_selective`: the current independent per-frame selector, reconstructed with the existing nested margin for each outer-held-out sequence.
- `nested_constant_switch_cost`: dynamic programming with a fixed cost for changing candidate identity.
- `nested_disagreement_cost`: dynamic programming with a transition cost proportional to the disagreement between the two candidate coordinate outputs at the boundary.

Both routers use the existing Ridge risk predictor and the same outer nested margin. At zero penalty, the routing implementation must reproduce the current independent selector exactly. The new router uses no ground truth at inference time.

## Install

Extract the ZIP into the FrameTrust repository root, preserving `scripts/`, `src/`, and `tests/`. The patch adds files only.

## Tests

Activate the existing `frametrust310` virtual environment and run from the repository root:

```powershell
python -m pytest -q
```

## Run

The evaluator consumes the existing output from `evaluate_selective_repair.py`. It does not rerun MediaPipe inference.

```powershell
python scripts/evaluate_sequence_routing.py --out outputs/selective_repair_crossfit
```

Default penalty grid: `0, 0.0025, 0.005, 0.01, 0.02, 0.04`. You can explicitly change the grid for a sensitivity run:

```powershell
python scripts/evaluate_sequence_routing.py --out outputs/selective_repair_crossfit --switch-penalties 0,0.001,0.0025,0.005,0.01,0.02,0.04
```

Do not alter the grid after inspecting outer-held-out results and then report the same results as confirmatory. Any changed grid must be selected within a new nested development protocol.

## Outputs

- `sequence_routing_inner_sweep.csv`: inner validation metrics for every outer fold, routing mode and penalty.
- `sequence_routing_penalty_selection.csv`: penalty selected inside each outer fold, plus guardrail decisions.
- `sequence_routing_metrics.csv`: outer-held-out spatial and temporal metrics for all fixed candidates and the three selective methods.
- `sequence_routing_condition_summary.csv`: sequence-macro means by corruption condition and method.
- `sequence_routing_paired_per_sequence.csv`: outer-sequence paired deltas versus `nested_selective`.
- `sequence_routing_paired_comparison.csv`: sequence-bootstrap confidence intervals.
- `sequence_routing_metadata.json`: protocol and limitations.

## Selection rule

Within each outer training set, action-balanced inner sequence folds fit the risk predictor and evaluate each penalty. Feasible settings must satisfy spatial coverage, false-correction and temporal-support guardrails. Among settings within the configured PCK tolerance of the best feasible PCK, the evaluator minimizes acceleration error; velocity error and smaller penalty break ties. The outer held-out sequence is excluded from all fitting and penalty selection.

## Important limitations

This is a prototype, not a claim of novelty or superiority. The disagreement-weighted transition is a heuristic. The objective routes each joint independently and does not enforce skeletal consistency. The complete-sequence dynamic program is offline/non-causal. Continue to use only the official Penn Action training split; do not inspect the official test split during development.
