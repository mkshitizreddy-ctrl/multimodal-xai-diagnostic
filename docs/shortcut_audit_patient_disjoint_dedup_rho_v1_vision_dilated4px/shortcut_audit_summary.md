> Lung masks dilated by 4 px (model resolution).

# Shortcut audit: frozen-model interventions

Descriptive model-sensitivity experiment on the frozen checkpoint. It does not establish clinical
validity, and a drop under an intervention is not by itself proof of (or against) shortcut learning.

- Checkpoint SHA-256: `b622f8c91b43f39b4e89b34b680eaeba9ab65c99591dde0dd91d35436027abd7`
- Test images: 1165 (849 Pneumonia); lung-mask area mean 0.454, 1156 masks within [0.1, 0.7]
- Displaced-mask controls: 5 random shifts per image, mean logit; mean IoU with the true mask 0.326
- Bootstrap: 2000 image-level resamples (ignores within-patient clustering)

## Subset: all (n=1165, Pneumonia=849)

| Condition | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec | Mean P (Normal) |
|---|---|---|---|---|---|
| original | 0.9978 [0.9962, 0.9990] | +0.0000 [+0.0000, +0.0000] | 0.986 | 0.965 | 0.047 |
| lung_only | 0.9816 [0.9723, 0.9891] | -0.0161 [-0.0247, -0.0092] | 0.979 | 0.870 | 0.211 |
| background_only | 0.9704 [0.9601, 0.9795] | -0.0273 [-0.0371, -0.0187] | 0.979 | 0.801 | 0.222 |
| displaced_lung_only | 0.9679 [0.9588, 0.9766] | -0.0298 [-0.0389, -0.0214] | 0.994 | 0.497 | 0.494 |
| displaced_background_only | 0.9905 [0.9861, 0.9944] | -0.0072 [-0.0110, -0.0040] | 0.988 | 0.842 | 0.188 |
| border_blank | 0.9971 [0.9949, 0.9988] | -0.0007 [-0.0021, +0.0005] | 0.965 | 0.978 | 0.030 |

## Subset: valid_mask (n=1156, Pneumonia=840)

| Condition | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec | Mean P (Normal) |
|---|---|---|---|---|---|
| original | 0.9978 [0.9962, 0.9989] | +0.0000 [+0.0000, +0.0000] | 0.986 | 0.965 | 0.047 |
| lung_only | 0.9815 [0.9717, 0.9892] | -0.0163 [-0.0253, -0.0093] | 0.979 | 0.870 | 0.211 |
| background_only | 0.9702 [0.9602, 0.9794] | -0.0276 [-0.0371, -0.0187] | 0.979 | 0.801 | 0.222 |
| displaced_lung_only | 0.9679 [0.9583, 0.9767] | -0.0298 [-0.0392, -0.0213] | 0.994 | 0.497 | 0.494 |
| displaced_background_only | 0.9905 [0.9859, 0.9944] | -0.0073 [-0.0110, -0.0039] | 0.988 | 0.842 | 0.188 |
| border_blank | 0.9970 [0.9946, 0.9987] | -0.0007 [-0.0022, +0.0005] | 0.964 | 0.978 | 0.030 |

## How to read this

- `lung_only` keeps the segmented lungs and grays everything else; `background_only` grays the lungs.
- `displaced_*` apply the same operation with a shifted copy of the lung mask (same shape and area, little overlap with the lungs). They control for the artefact of graying a region of that size.
- `border_blank` grays the outer margin.
- Lung masks come from an automatic segmenter trained on adult radiographs; inspect `qc_lung_masks.png` before trusting them on these paediatric images.
- Gray fill is out of distribution for the model, so compare each condition with its displaced control, not only with the original images.
