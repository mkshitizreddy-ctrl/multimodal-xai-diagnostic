# Grad-CAM patch-deletion experiment

- Test CSV: `data/processed_patient_disjoint_dedup_rho_v1_0/test.csv`
- Checkpoint: `checkpoints/vision_baseline/best_model.pth`
- Images evaluated: 1165
- Nonzero Grad-CAM maps: 969
- Zero Grad-CAM maps: 196
- Random masks per image/fraction: 10
- Patch grid: 16 x 16
- Bootstrap replicates: 2000
- Random seed: 42

Each selected patch is replaced by zero in normalized input space, corresponding to mid-gray. The selected patch fraction is quantized to the nearest whole number of grid patches.

Grad-CAM and random-patch comparisons use the same nonzero-map images. Zero-map cases have no Grad-CAM ranking and are excluded from the primary paired comparison.

Random patches are spatially coherent individually but may be scattered across the image. This is a patch-level random baseline, not a matched contiguous-region baseline.

## Primary result: mean logit decrease

Positive paired difference means Grad-CAM deletion caused a larger logit decrease than the per-image mean random-patch deletion.

| Requested fraction | Actual patch fraction | n paired | Grad-CAM mean drop | Random mean drop (same images) | Paired difference | Bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 5.08% | 969 | 1.095916 | 0.451536 | 0.644380 | [0.573201, 0.717075] |
| 10% | 10.16% | 969 | 1.912190 | 1.185712 | 0.726477 | [0.654370, 0.807536] |
| 20% | 19.92% | 969 | 3.061148 | 2.341220 | 0.719928 | [0.634335, 0.806841] |
| 30% | 30.08% | 969 | 3.697545 | 2.958718 | 0.738826 | [0.654711, 0.824836] |
| 40% | 39.84% | 969 | 3.830365 | 3.172948 | 0.657417 | [0.570925, 0.744303] |

## Zero-map subgroup

- Zero-map images: 196 of 1165 (16.82%).
- Grad-CAM deletion metrics are undefined for these images; random patch deletion results are still recorded in the CSV.

## Class counts

| True label | Images | Zero maps |
|---:|---:|---:|
| 0 | 316 | 196 |
| 1 | 849 | 0 |

## Interpretation cautions

- Deletion can create out-of-distribution inputs; results measure sensitivity to this specific intervention.
- Grid patches are not lung-aware regions, and this experiment does not establish clinically correct localization or causal reasoning.
- Bootstrap intervals describe uncertainty in the mean paired difference across evaluated images; random-repeat SD is not a population confidence interval.
- This experiment must not be interpreted as clinical validation.
