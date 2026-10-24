# Full Test-Set Grad-CAM Diagnostic

- Checkpoint: `checkpoints\vision_patient_disjoint_dedup_rho_v1_0\best_model.pth`
- Test CSV: `data\processed_patient_disjoint_dedup_rho_v1_0\test.csv`
- Device: `cuda`
- Test images analyzed: 1165
- Zero heatmaps: 268 (23.00%)
- Nonzero heatmaps: 897

## Positive-pixel coverage

- Mean: 69.3920%
- Median: 95.4082%
- 90th percentile: 99.4898%

## Pneumonia probability by heatmap status

- Zero-map cases: n=268, mean probability=0.004164
- Nonzero-map cases: n=897, mean probability=0.943592

## Interpretation limits

- A zero map means the standard positive-evidence Grad-CAM map has no positive pixels after its ReLU; it does not prove that the model is insensitive to the image.
- Coverage is the fraction of pixels with positive heatmap values, not the fraction of lung tissue affected.
- This is a descriptive diagnostic, not evidence of clinical validity or causal explanation.

Detailed measurements: `docs\gradcam_testset_diagnostic_patient_disjoint_dedup_rho_v1_vision\gradcam_testset_measurements.csv`
