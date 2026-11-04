FrameTrust Mapping Validation v2 — corrected patch layout

This ZIP has repository-relative paths at its root:
  scripts/penn_action_mapping_validation_v2.py
  tests/test_penn_action_mapping_validation_v2.py

Install:
  Extract the ZIP directly into the FrameTrust repository root.
  It adds two new files and does not overwrite existing files.

From the FrameTrust repository root, with the frametrust310 virtual environment active:
  python -m pytest -q
  python scripts/penn_action_mapping_validation_v2.py --selection outputs/penn_action_mapping_holdout/selected_mapping_validation_sequences.csv --out outputs/penn_action_mapping_holdout_v2

The use of `python -m pytest` is intentional: it runs pytest through the active interpreter and keeps the repository root importable for existing tests that import modules under `scripts`.

Important methodological note:
  The script uses Penn Action ground-truth person bounding boxes to crop each frame before MediaPipe inference. This is an oracle ROI used only to isolate landmark mapping from person-association failures. It is NOT an end-to-end pose-estimation result and must not be presented as deployable camera-only performance.

Outputs:
  outputs/penn_action_mapping_holdout_v2/
    mapping_metrics_by_sequence.csv
    mapping_metrics_by_joint.csv
    mapping_common_support_by_sequence.csv
    mapping_summary_overall.csv
    mapping_summary_by_viewpoint.csv
    mapping_summary_by_action.csv
    mapping_audit_metadata.json
    overlays/ (five side-by-side comparisons per sequence by default)

Overlay legend:
  Ground-truth skeleton: red
  Mapping A predictions: green (left panel)
  Mapping B predictions: orange/blue (right panel)
  Original Penn Action bbox: magenta
  Padded inference ROI: yellow

The bbox coordinate convention is [x1,y1,x2,y2], inherited from the existing FrameTrust code. Visually verify both boxes against the intended person before interpreting scores.
