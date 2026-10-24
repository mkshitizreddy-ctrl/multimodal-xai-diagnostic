# Patient-Disjoint Deduplicated Rho Experiment — Results

## Objective
Compare image-only classification with image-plus-synthetic-vitals fusion as the synthetic class-dependent signal strength (rho) varies.

## Dataset and validation
- Source dataset: 5,856 records.
- Exact duplicate image records removed: 32.
- Deduplicated dataset: 5,824 images.
- Train: 3,727; validation: 932; test: 1,165.
- Exact image-content hashes and known pneumonia patient IDs were checked for cross-split overlap; none was found.
- Normal-patient identity could not be verified, so complete patient-level separation for normal images cannot be claimed.

## Test-set AUROC comparison

| rho | Image-only AUROC | Fusion AUROC | Difference (fusion - image) | Paired-bootstrap 95% CI | DeLong p-value |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.997782 | 0.998174 | +0.000391 | [-0.000612, 0.001465] | 0.455376 |
| 0.25 | 0.997782 | 0.999191 | +0.001409 | [0.000503, 0.002497] | 0.004231 |
| 0.50 | 0.997782 | 0.999825 | +0.002043 | [0.000917, 0.003425] | 0.001171 |
| 0.75 | 0.997782 | 0.999959 | +0.002177 | [0.000991, 0.003631] | 0.001002 |
| 1.00 | 0.997782 | 0.999981 | +0.002199 | [0.000991, 0.003687] | 0.001108 |

Each comparison used 1,165 test images and 2,000 paired-bootstrap replicates (seed 42). P-values are unadjusted; no multiple-comparison correction was applied.

## Interpretation
The image-only AUROC remained 0.997782 across the five conditions. Fusion AUROC increased with rho. The rho = 0 comparison was not statistically significant; the comparisons at rho = 0.25, 0.50, 0.75, and 1.00 had unadjusted DeLong p-values below 0.05.

## Limitations
Temperature and SpO2 are synthetic, with deliberately controlled class-dependent signal. These results demonstrate sensitivity to injected label-related information; they do not establish that real clinical measurements improve pneumonia diagnosis. The same test set was used across all five comparisons, and no multiple-comparison correction was applied. Normal-patient identity remains unverified. AUROC is near ceiling, so the absolute differences are small.

## Reproducibility
Combined results: `docs/master_auroc_comparison_patient_disjoint_dedup_rho_v1.csv`

The individual paired-comparison CSVs, model checkpoints, evaluation predictions, and logs were retained. This report summarizes the observed experiment and does not claim clinical validation.

## Explainability evaluation

### Grad-CAM test-set diagnostic

Grad-CAM was evaluated across all 1,165 images in the test set. A total of 268 images (23.00%) produced zero-valued heatmaps, while 897 produced nonzero heatmaps.

| Actual class | Images | Zero maps | Zero-map rate |
|---|---:|---:|---:|
| Normal | 316 | 266 | 84.18% |
| Pneumonia | 849 | 2 | 0.24% |
| **Total** | **1,165** | **268** | **23.00%** |

The zero-map cases were strongly associated with the Normal class and low predicted pneumonia probabilities. However, a zero heatmap does not establish that an input image has no influence on the model.

### Investigation of zero-valued heatmaps

The raw weighted Grad-CAM maps were inspected for six sampled zero-map cases. All six maps were entirely negative before the standard ReLU operation, which clips negative values to zero. This provides evidence for the mechanism in the inspected examples; the raw-map mechanism was not individually verified for every zero-map case.

The diagnostic results were repeatable in the inspected runs. Zero-valued heatmaps remain a limitation because they do not provide useful spatial localization for those images.

### Lung localization

Lung localization was measured by calculating the fraction of positive Grad-CAM activation falling inside the segmented lung masks. Of the 1,165 test images, 897 had valid nonzero heatmaps; the other 268 had undefined fractions because their heatmaps were zero-valued.

For the 897 valid maps, the mean lung-energy fraction was 0.499 (standard deviation 0.152).

| Actual class | Valid maps | Undefined maps | Mean lung-energy fraction | Median | Standard deviation |
|---|---:|---:|---:|---:|---:|
| Normal | 50 | 266 | 0.4646 | 0.5347 | 0.3604 |
| Pneumonia | 847 | 2 | 0.5006 | 0.5067 | 0.1300 |

This fraction is an overlap measure between Grad-CAM activation and the segmented lung region. It is not a measure of explanation correctness, clinical relevance, or diagnostic safety. The result is also conditional on having a valid nonzero heatmap, so it excludes the 268 zero-map cases.

### Counterfactual experiments

Counterfactual experiments were conducted on six sampled test images using the existing Grad-CAM-based mask and image inpainting procedure. The original, threshold-0.7, and no-dilation experiment outputs were retained. No prediction flips were observed in the six-image runs.

These experiments provide a limited sensitivity analysis, not evidence of causal importance or clinical validity. The threshold and dilation settings changed together in one comparison, so that comparison cannot isolate the effect of dilation. Probability changes after inpainting should be interpreted as responses to the chosen image intervention, not as causal effects.

### Explainability artifacts

The following outputs preserve the measurements and diagnostic evidence:

- `docs/gradcam_testset_diagnostic_patient_disjoint_dedup_rho_v1_vision/gradcam_testset_summary.md`
- `docs/gradcam_testset_diagnostic_patient_disjoint_dedup_rho_v1_vision/gradcam_testset_measurements.csv`
- `docs/gradcam_raw_zero_map_sample.csv`
- `docs/lung_localization_testset_rho0.csv`
- `docs/counterfactual_mask_comparison_measurements.csv`
- `docs/counterfactual_patient_disjoint_dedup_rho_v1_vision/`
- `docs/counterfactual_patient_disjoint_dedup_rho_v1_vision_threshold_070/`
- `docs/counterfactual_patient_disjoint_dedup_rho_v1_vision_no_dilation/`

The explainability results are diagnostic evidence for this model and test set. They do not establish clinical validity, causal reasoning, or reliable localization of pneumonia findings.