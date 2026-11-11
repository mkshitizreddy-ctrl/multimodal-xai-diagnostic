# Patient-clustered bootstrap intervals

Frozen vision checkpoint, test split (1165 images, 651 patient or image units).
Pneumonia images are resampled by patient; Normal images are treated as independent because
their patient identity is unknown. 2000 resamples; percentile 95% intervals.
`Width ratio` is the patient-cluster interval width divided by the image-level width.

| Experiment | Quantity | Estimate | Image-level 95% CI | Patient-cluster 95% CI | Width ratio |
|---|---|---|---|---|---|
| audit | original | 0.9978 | [0.9963, 0.9990] | [0.9963, 0.9990] | 0.99 |
| audit | lung_only | 0.9746 | [0.9652, 0.9830] | [0.9648, 0.9832] | 1.03 |
| audit | background_only | 0.9759 | [0.9666, 0.9837] | [0.9661, 0.9839] | 1.04 |
| audit | displaced_lung_only | 0.9374 | [0.9226, 0.9512] | [0.9218, 0.9515] | 1.03 |
| audit | displaced_background_only | 0.9929 | [0.9891, 0.9960] | [0.9890, 0.9960] | 1.02 |
| audit | border_blank | 0.9971 | [0.9947, 0.9988] | [0.9949, 0.9988] | 0.96 |
| audit | lung_only - original | -0.0232 | [-0.0320, -0.0154] | [-0.0325, -0.0150] | 1.05 |
| audit | background_only - original | -0.0219 | [-0.0307, -0.0145] | [-0.0310, -0.0142] | 1.04 |
| audit | displaced_lung_only - original | -0.0604 | [-0.0749, -0.0469] | [-0.0757, -0.0468] | 1.03 |
| audit | displaced_background_only - original | -0.0049 | [-0.0080, -0.0024] | [-0.0082, -0.0023] | 1.04 |
| audit | border_blank - original | -0.0007 | [-0.0021, +0.0005] | [-0.0021, +0.0005] | 1.00 |
| audit | lung_only - displaced_lung_only | +0.0371 | [+0.0235, +0.0522] | [+0.0228, +0.0525] | 1.04 |
| audit | background_only - displaced_background_only | -0.0170 | [-0.0246, -0.0102] | [-0.0252, -0.0102] | 1.04 |
| cue | original | 0.9978 | [0.9963, 0.9990] | [0.9963, 0.9990] | 0.99 |
| cue | jpeg_q75 | 0.9978 | [0.9963, 0.9990] | [0.9963, 0.9990] | 1.00 |
| cue | jpeg_q25 | 0.9978 | [0.9962, 0.9990] | [0.9962, 0.9990] | 1.00 |
| cue | equalize | 0.9905 | [0.9861, 0.9943] | [0.9860, 0.9942] | 1.00 |
| cue | pad_square | 0.9907 | [0.9864, 0.9943] | [0.9864, 0.9943] | 1.00 |
| cue | down_up_2x | 0.9978 | [0.9963, 0.9990] | [0.9964, 0.9990] | 1.00 |
| cue | jpeg_q75 - original | +0.0000 | [-0.0001, +0.0001] | [-0.0001, +0.0001] | 0.97 |
| cue | jpeg_q25 - original | -0.0000 | [-0.0003, +0.0003] | [-0.0003, +0.0003] | 1.04 |
| cue | equalize - original | -0.0073 | [-0.0113, -0.0038] | [-0.0113, -0.0039] | 0.98 |
| cue | pad_square - original | -0.0071 | [-0.0107, -0.0039] | [-0.0109, -0.0040] | 1.01 |
| cue | down_up_2x - original | +0.0000 | [-0.0002, +0.0002] | [-0.0001, +0.0002] | 0.98 |
