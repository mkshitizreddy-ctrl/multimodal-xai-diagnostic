# Raw Grad-CAM Test-Set Diagnostic

- Checkpoint: `checkpoints\vision_patient_disjoint_dedup_rho_v1_0\best_model.pth`
- Test CSV: `data\processed_patient_disjoint_dedup_rho_v1_0\test.csv`
- Device: `cuda`
- Test images analyzed: 1165

## Raw maps before ReLU and scaling

- Entirely negative raw maps: 268
- Raw maps with no positive pixels: 268
- Mean raw-map positive-pixel fraction: 65.7826%

## Interpretation

- An entirely negative raw map is expected to become a zero map after the standard Grad-CAM ReLU.
- A raw map with no positive pixels may contain negative and/or exactly zero values; inspect its minimum and maximum.
- These measurements describe Grad-CAM behavior and do not establish clinical validity or causal explanations.

Detailed measurements: `docs\gradcam_raw_testset_diagnostic_patient_disjoint_dedup_rho_v1_vision\gradcam_raw_testset_measurements.csv`
