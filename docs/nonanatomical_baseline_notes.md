# Non-anatomical baseline: what the label looks like without anatomy

Run 2026-11-01 on the patient-disjoint, deduplicated split (train 3,727 / val 932 / test 1,165 images,
849 Pneumonia in test). Code: `src/explain/nonanatomical_baseline.py`, `src/explain/baseline_discordance.py`.
Tables: `docs/nonanatomical_baseline_patient_disjoint_dedup_rho_v1/`.

## Result

Classifiers that never see lung detail recover the label almost as well as the CNN.

| Features | Test AUROC [95% CI] |
|---|---|
| File size (KB) alone, one feature | 0.980 [0.972, 0.988] |
| Width, height, aspect | 0.916 [0.901, 0.932] |
| Mean and SD of pixel intensity | 0.749 [0.718, 0.778] |
| 16x16 grey thumbnail (256 values) | 0.977 [0.968, 0.985] |
| Size, file size and intensity statistics (18 features) | 0.986 [0.979, 0.992] |
| CNN (DenseNet-121 + CBAM, vision only) | 0.998 |

The CNN is still better than every baseline (paired bootstrap difference excludes 0; smallest gap 0.012,
CI [0.006, 0.019] against the 18-feature model). The baselines are far from chance.

Why: the two classes were exported differently. In the test split the median Normal file is 1644x1314 px
and 495 KB, the median Pneumonia file is 1160x776 px and 71 KB. Normal images take 315 distinct
(width, height) values and Pneumonia images 801, and no (width, height) value shared by at least 15 test
images contains both classes, so stratifying on image size is not possible.

## What this does and does not show

- It shows the label is recoverable from acquisition properties. That is a precondition for shortcut
  learning, not evidence of it.
- The CNN receives 224x224 resized, normalised pixels, so it cannot read file size or original dimensions
  directly. It can still see aspect-ratio distortion, compression and noise texture, and global brightness
  layout, all of which correlate with the export pipeline.
- The thumbnail and intensity results are not pure shortcuts: consolidation changes brightness
  honestly. File size is the cleanest non-pathological cue, but compressibility could carry some disease signal too.
- The source of the export difference is not established here. Treat "different source pipelines" as a
  hypothesis.

## Exploratory link to the CNN (`baseline_discordance.md`)

| Subset (file-size baseline at its validation Youden threshold) | n | CNN AUROC [95% CI] | CNN error rate |
|---|---|---|---|
| Baseline right | 1,110 | 0.999 [0.999, 1.000] | 1.1% |
| Baseline wrong | 55 | 0.923 [0.845, 0.978] | 20.0% |

- 11 of the CNN's 23 test errors fall among the 55 images (4.7%) where file size points the wrong way.
- CNN false positives (Normal called Pneumonia) have a median file size of 121 KB versus 495 KB for Normal
  overall; false negatives have 138 KB versus 71 KB for Pneumonia overall.
- Within class, the CNN logit correlates weakly with the file-size score (Spearman +0.17 Pneumonia,
  +0.27 Normal; +0.28 and +0.36 for the 18-feature model).
- Even where file size misleads, CNN AUROC stays at 0.92, so the model is not simply reading the cue.

This is correlational: atypical images could be hard for both the baseline and the CNN. It makes partial
reliance plausible and untested, and justifies a causal test.

## Limits

One dataset, one split, bootstrap CIs at image level (patient clustering ignored), baselines trained on
the same train split as the CNN, 55 discordant images (wide CIs), no causal intervention yet.

## Next experiment

Done: see `docs/cue_perturbation_notes.md`.
