FrameTrust Mapping Validation v2 patch

Files:
  scripts/penn_action_mapping_validation_v2.py
  tests/test_penn_action_mapping_validation_v2.py

Install:
  Extract this ZIP into the FrameTrust repository root, preserving folders.
  It adds new files only; it does not overwrite existing scripts.

Run tests from the FrameTrust root using the frametrust310 environment:
  pytest -q

Run the v2 mapping audit using the existing 15-sequence selection:
  python scripts/penn_action_mapping_validation_v2.py --selection outputs/penn_action_mapping_holdout/selected_mapping_validation_sequences.csv --out outputs/penn_action_mapping_holdout_v2

Important methodological note:
  This v2 script uses Penn Action ground-truth person bounding boxes to crop each frame before MediaPipe inference. This is an oracle ROI used only to isolate landmark mapping from person-association failures. It is NOT an end-to-end pose-estimation result and must not be presented as deployable camera-only performance.

Outputs are written to:
  outputs/penn_action_mapping_holdout_v2/

Primary files:
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
  Original annotation bbox: magenta
  Padded inference ROI: yellow

The bbox coordinate convention is inherited from the existing FrameTrust code ([x1,y1,x2,y2]) and is explicitly recorded in metadata. Visually verify the magenta and yellow boxes against the intended person before interpreting scores.
