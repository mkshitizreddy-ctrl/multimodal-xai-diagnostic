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

Each comparison used 1,165 test images and 2,000 paired-bootstrap replicates (seed 42). The DeLong p-values in the table are unadjusted. Holm–Bonferroni-adjusted p-values are reported in the paragraph below. The paired-bootstrap confidence intervals are unadjusted for multiple comparisons.

To account for the five related comparisons, Holm–Bonferroni adjustment was applied to the DeLong p-values. The adjusted p-values for rho = 0.00, 0.25, 0.50, 0.75, and 1.00 were 0.455376, 0.008462, 0.005010, 0.005010, and 0.005010, respectively. The comparisons at rho = 0.25, 0.50, 0.75, and 1.00 remained statistically significant at the 0.05 level, while the rho = 0.00 comparison did not. These are comparisons within the same test-set experiment and should not be interpreted as independent replications.

## Interpretation
The image-only AUROC remained 0.997782 across all five conditions, while fusion AUROC increased as the synthetic signal strength (rho) increased. The rho = 0.00 comparison was not statistically significant (unadjusted DeLong p = 0.455376; Holm-adjusted p = 0.455376). The comparisons at rho = 0.25, 0.50, 0.75, and 1.00 remained statistically significant after Holm–Bonferroni correction (adjusted p-values = 0.008462, 0.005010, 0.005010, and 0.005010, respectively). Because Temperature and SpO2 were synthetic and deliberately class-dependent, these findings demonstrate the effect of injected label-related information under this experimental setup; they do not establish that real clinical measurements improve pneumonia diagnosis.

## Limitations
Temperature and SpO2 are synthetic, with deliberately controlled class-dependent signal. These results demonstrate sensitivity to injected label-related information; they do not establish that real clinical measurements improve pneumonia diagnosis. The same test set was used across all five comparisons, so the comparisons should not be interpreted as independent replications. Holm–Bonferroni correction was applied to the DeLong p-values; the paired-bootstrap confidence intervals remain unadjusted for multiple comparisons. Normal-patient identity remains unverified. AUROC is near ceiling, so the absolute differences are small.

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

Zero maps were concentrated among Normal images and were associated with low predicted pneumonia probabilities. This is a descriptive association; a zero heatmap does not establish that an input image has no influence on the model.

**Full-test-set investigation of zero Grad-CAM maps.** The raw Grad-CAM calculation was reproduced before ReLU and scaling for all 1,165 test images. The raw-map diagnostic was matched one-to-one with the original test-set diagnostic using the test index and image filename. All 268 zero heatmaps (23.00% of the test set) had entirely negative raw maps and were therefore clipped to zero by the standard Grad-CAM ReLU. None of the 897 nonzero heatmaps had an entirely negative raw map. Labels matched across both diagnostics, and the maximum difference in predicted pneumonia probability was 0.0.

These results verify the numerical mechanism behind the zero maps across the full test set. They do not establish that the resulting explanations are clinically valid or that the model lacks image sensitivity in zero-map cases.

### Investigation of zero-valued heatmaps

The raw weighted Grad-CAM maps were subsequently examined across all 1,165 test images before the standard ReLU operation. All 268 zero-valued heatmaps corresponded to raw maps whose values were entirely negative and were therefore clipped to zero. None of the 897 nonzero heatmaps came from an entirely negative raw map. The raw-map results matched the original diagnostic by test index and image filename, with no label mismatches or changes in predicted probabilities.

These findings confirm the numerical mechanism behind the zero-valued heatmaps across the full test set. However, zero-valued maps provide no useful spatial localization, and this analysis does not establish the clinical validity of the remaining explanations or show that images with zero-valued maps have no influence on the model.

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