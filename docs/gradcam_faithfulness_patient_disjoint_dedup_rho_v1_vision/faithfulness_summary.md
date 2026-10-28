# Grad-CAM patch-deletion experiment

- Test CSV: `data/processed_patient_disjoint_dedup_rho_v1_0/test.csv`
- Checkpoint: `checkpoints/vision_patient_disjoint_dedup_rho_v1_0/best_model.pth`
- Images evaluated: 1165
- Nonzero Grad-CAM maps: 897
- Zero Grad-CAM maps: 268
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
| 5% | 5.08% | 897 | 1.247706 | 0.044735 | 1.202971 | [1.136958, 1.267786] |
| 10% | 10.16% | 897 | 2.055965 | 0.635932 | 1.420033 | [1.331714, 1.507374] |
| 20% | 19.92% | 897 | 2.934244 | 2.024510 | 0.909734 | [0.806441, 1.014722] |
| 30% | 30.08% | 897 | 3.260296 | 3.076247 | 0.184049 | [0.078510, 0.297860] |
| 40% | 39.84% | 897 | 3.158397 | 3.559055 | -0.400658 | [-0.508866, -0.290252] |

## Zero-map subgroup

- Zero-map images: 268 of 1165 (23.00%).
- Grad-CAM deletion metrics are undefined for these images; random patch deletion results are still recorded in the CSV.

## Class counts

| True label | Images | Zero maps |
|---:|---:|---:|
| 0 | 316 | 266 |
| 1 | 849 | 2 |

## Interpretation cautions

- Deletion can create out-of-distribution inputs; results measure sensitivity to this specific intervention.
- Grid patches are not lung-aware regions, and this experiment does not establish clinically correct localization or causal reasoning.
- Bootstrap intervals describe uncertainty in the mean paired difference across evaluated images; random-repeat SD is not a population confidence interval.
- This experiment must not be interpreted as clinical validation.
