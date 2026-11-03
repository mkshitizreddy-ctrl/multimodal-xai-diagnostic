> Lung masks dilated by 8 px (model resolution).

# Shortcut audit: frozen-model interventions

Descriptive model-sensitivity experiment on the frozen checkpoint. It does not establish clinical
validity, and a drop under an intervention is not by itself proof of (or against) shortcut learning.

- Checkpoint SHA-256: `b622f8c91b43f39b4e89b34b680eaeba9ab65c99591dde0dd91d35436027abd7`
- Test images: 1165 (849 Pneumonia); lung-mask area mean 0.519, 1145 masks within [0.1, 0.7]
- Displaced-mask controls: 5 random shifts per image, mean logit; mean IoU with the true mask 0.373
- Bootstrap: 2000 image-level resamples (ignores within-patient clustering)

## Subset: all (n=1165, Pneumonia=849)

| Condition | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec | Mean P (Normal) |
|---|---|---|---|---|---|
| original | 0.9978 [0.9962, 0.9990] | +0.0000 [+0.0000, +0.0000] | 0.986 | 0.965 | 0.047 |
| lung_only | 0.9888 [0.9814, 0.9947] | -0.0090 [-0.0160, -0.0036] | 0.991 | 0.870 | 0.195 |
| background_only | 0.9638 [0.9514, 0.9740] | -0.0340 [-0.0459, -0.0240] | 0.995 | 0.554 | 0.464 |
| displaced_lung_only | 0.9837 [0.9774, 0.9890] | -0.0141 [-0.0200, -0.0090] | 0.989 | 0.693 | 0.331 |
| displaced_background_only | 0.9882 [0.9826, 0.9929] | -0.0096 [-0.0147, -0.0053] | 0.996 | 0.747 | 0.275 |
| border_blank | 0.9971 [0.9949, 0.9988] | -0.0007 [-0.0021, +0.0005] | 0.965 | 0.978 | 0.030 |

## Subset: valid_mask (n=1145, Pneumonia=830)

| Condition | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec | Mean P (Normal) |
|---|---|---|---|---|---|
| original | 0.9978 [0.9962, 0.9990] | +0.0000 [+0.0000, +0.0000] | 0.986 | 0.965 | 0.047 |
| lung_only | 0.9887 [0.9810, 0.9948] | -0.0091 [-0.0163, -0.0037] | 0.990 | 0.870 | 0.195 |
| background_only | 0.9641 [0.9523, 0.9746] | -0.0337 [-0.0452, -0.0233] | 0.995 | 0.556 | 0.462 |
| displaced_lung_only | 0.9843 [0.9785, 0.9897] | -0.0135 [-0.0189, -0.0086] | 0.990 | 0.692 | 0.332 |
| displaced_background_only | 0.9883 [0.9828, 0.9929] | -0.0095 [-0.0144, -0.0053] | 0.996 | 0.749 | 0.273 |
| border_blank | 0.9971 [0.9948, 0.9987] | -0.0007 [-0.0021, +0.0005] | 0.965 | 0.978 | 0.030 |

## How to read this

- `lung_only` keeps the segmented lungs and grays everything else; `background_only` grays the lungs.
- `displaced_*` apply the same operation with a shifted copy of the lung mask (same shape and area, little overlap with the lungs). They control for the artefact of graying a region of that size.
- `border_blank` grays the outer margin.
- Lung masks come from an automatic segmenter trained on adult radiographs; inspect `qc_lung_masks.png` before trusting them on these paediatric images.
- Gray fill is out of distribution for the model, so compare each condition with its displaced control, not only with the original images.
