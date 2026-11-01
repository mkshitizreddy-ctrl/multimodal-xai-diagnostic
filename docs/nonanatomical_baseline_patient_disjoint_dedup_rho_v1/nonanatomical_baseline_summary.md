# Non-anatomical baseline

Classifiers that see only acquisition-level features, trained on the patient-disjoint, deduplicated
train split and scored on the test split. This shows how recoverable the label is from such
cues; it does not show that the CNN uses them.

| Features | Model (picked on val) | Val AUROC | Test AUROC [95% CI] | CNN AUROC | Change vs CNN [95% CI] |
|---|---|---|---|---|---|
| file_size_only | logistic | 0.9802 | 0.9801 [0.9715, 0.9876] | 0.9978 | -0.0177 [-0.0257, -0.0107] |
| dimensions_only | logistic | 0.9233 | 0.9163 [0.9005, 0.9320] | 0.9978 | -0.0814 [-0.0972, -0.0662] |
| pixel_mean_std_only | gradient_boosting | 0.7391 | 0.7474 [0.7167, 0.7763] | 0.9978 | -0.2504 [-0.2811, -0.2218] |
| size_meta | logistic | 0.9834 | 0.9812 [0.9727, 0.9887] | 0.9978 | -0.0166 [-0.0245, -0.0094] |
| intensity | gradient_boosting | 0.9314 | 0.9324 [0.9157, 0.9481] | 0.9978 | -0.0654 [-0.0822, -0.0500] |
| size_meta+intensity | logistic | 0.9877 | 0.9859 [0.9784, 0.9928] | 0.9978 | -0.0119 [-0.0190, -0.0055] |
| thumbnail_16x16 | gradient_boosting | 0.9840 | 0.9772 [0.9685, 0.9849] | 0.9978 | -0.0205 [-0.0288, -0.0134] |

## Strongest single features (test AUROC, direction-free)

| Feature | AUROC |
|---|---|
| file_kb | 0.9801 |
| bytes_per_pixel | 0.9247 |
| height | 0.9188 |
| width | 0.8931 |
| aspect | 0.8438 |
| p99 | 0.7790 |
| border_dark | 0.7333 |
| p01 | 0.7328 |

## Test-split acquisition properties by class (median)

| Class | Width | Height | File size (KB) | Bytes/pixel | Distinct (width, height) |
|---|---|---|---|---|---|
| Normal | 1644 | 1314 | 494.5 | 0.246 | 315 |
| Pneumonia | 1160 | 776 | 71.1 | 0.081 | 801 |
