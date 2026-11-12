# Acquisition-cue perturbation test

Run 2026-11-02 on the frozen vision checkpoint (`checkpoints/vision_patient_disjoint_dedup_rho_v1_0`), patient-disjoint
deduplicated test split (1,165 images, 849 Pneumonia). Code: `src/explain/cue_perturbation.py`,
`cue_perturbation_utils.py`. Tables: `docs/cue_perturbation_patient_disjoint_dedup_rho_v1_vision/`.

Question: if the CNN leans on cues that separate the two classes' export pipelines (see
`nonanatomical_baseline_notes.md`), does perturbing those cues hurt it? Each perturbation is applied to the
original image before the usual resize and normalisation.

## Result

| Condition | AUROC, all [95% CI] | Change vs original | Error rate | False-negative rate | Mean logit shift |
|---|---|---|---|---|---|
| original | 0.9978 [0.9963, 0.9990] | - | 2.0% | 1.4% | 0 |
| JPEG q75 | 0.9978 [0.9963, 0.9990] | +0.0000 [-0.0001, +0.0001] | 2.1% | 1.4% | +0.01 |
| JPEG q25 | 0.9978 [0.9962, 0.9990] | -0.0000 [-0.0003, +0.0003] | 2.1% | 1.2% | +0.21 |
| Equalize histogram | 0.9905 [0.9861, 0.9943] | -0.0073 [-0.0113, -0.0038] | 5.2% | 5.5% | -1.01 |
| Pad to square | 0.9907 [0.9864, 0.9943] | -0.0071 [-0.0107, -0.0039] | 21.6% | 29.7% | -5.13 |
| Down-up 2x | 0.9978 [0.9963, 0.9990] | +0.0000 [-0.0002, +0.0002] | 2.1% | 1.3% | +0.22 |

On the 55 test images where a file-size-only classifier is wrong (20 Pneumonia, 35 Normal):

| Condition | AUROC [95% CI] | Change vs original | Error rate |
|---|---|---|---|
| original | 0.9229 [0.8450, 0.9780] | - | 20.0% |
| JPEG q25 | 0.9143 [0.8280, 0.9734] | -0.0086 [-0.0280, +0.0031] | 21.8% |
| Equalize histogram | 0.8100 [0.6610, 0.9298] | -0.1129 [-0.2353, -0.0108] | 25.5% |
| Pad to square | 0.9329 [0.8566, 0.9861] | +0.0100 [-0.0457, +0.0686] | 16.4% |
| Down-up 2x | 0.9186 [0.8375, 0.9760] | -0.0043 [-0.0160, +0.0000] | 23.6% |

## Reading

- **Compression and fine texture do not drive the ranking.** JPEG recompression to a common quality and
  removal of fine detail leave AUROC unchanged, overall and on the 55 discordant images.
- **Framing changes the calibration, not the ranking.** Padding to a square keeps AUROC at 0.991 but
  shifts logits by -5 and raises the false-negative rate from 1.4% to 29.7%. The lungs shrink inside the 224
  frame and flat borders appear, so the 0.5 threshold no longer fits. This is a distribution-shift sensitivity
  and affects all images, not specifically the discordant ones. It matters for deployment: the dashboard and
  evaluation assume the original framing.
- **Contrast equalisation costs a little overall and more on the discordant images** (-0.113, CI
  [-0.235, -0.011], n=55). This is the only result pointing toward reliance on global intensity. Equalisation
  also produces intensity distributions the model never saw, so it is not a clean removal of one cue.
- Combined with the baseline: the label is recoverable from acquisition properties (AUROC 0.98), but the
  CNN's ranking is insensitive to the compression and texture perturbations tested and sensitive to global
  contrast and framing. This is not evidence of a file-size shortcut, and does not rule out reliance on
  other cues.

## Limits

One dataset, one checkpoint, one seed. Image-level bootstrap CIs; a patient-clustered bootstrap (`docs/cluster_bootstrap/`) changes their widths by at most about 5%. Five conditions
were tested on three subsets without multiplicity correction. The discordant subset has 55 images. The
perturbations are not clean interventions on a single cue. Original file size and dimensions are removed by
the model's own resize and are not tested here.
