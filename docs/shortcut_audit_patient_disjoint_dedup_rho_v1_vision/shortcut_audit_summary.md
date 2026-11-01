# Shortcut audit: frozen-model interventions

Descriptive model-sensitivity experiment on the frozen checkpoint. It does not establish clinical
validity, and a drop under an intervention is not by itself proof of (or against) shortcut learning.

- Checkpoint SHA-256: `b622f8c91b43f39b4e89b34b680eaeba9ab65c99591dde0dd91d35436027abd7`
- Test images: 1165 (849 Pneumonia); lung-mask area mean 0.383, 1162 masks within [0.1, 0.7]
- Displaced-mask controls: 5 random shifts per image, mean logit; mean IoU with the true mask 0.263
- Bootstrap: 2000 image-level resamples (ignores within-patient clustering)

## Subset: all (n=1165, Pneumonia=849)

| Condition | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec | Mean P (Normal) |
|---|---|---|---|---|---|
| original | 0.9978 [0.9962, 0.9990] | +0.0000 [+0.0000, +0.0000] | 0.986 | 0.965 | 0.047 |
| lung_only | 0.9746 [0.9650, 0.9829] | -0.0232 [-0.0324, -0.0155] | 0.966 | 0.867 | 0.221 |
| background_only | 0.9759 [0.9672, 0.9839] | -0.0219 [-0.0304, -0.0144] | 0.932 | 0.911 | 0.105 |
| displaced_lung_only | 0.9374 [0.9229, 0.9509] | -0.0604 [-0.0748, -0.0468] | 0.994 | 0.313 | 0.631 |
| displaced_background_only | 0.9929 [0.9892, 0.9959] | -0.0049 [-0.0079, -0.0025] | 0.985 | 0.911 | 0.113 |
| border_blank | 0.9971 [0.9949, 0.9988] | -0.0007 [-0.0021, +0.0005] | 0.965 | 0.978 | 0.030 |

## Subset: valid_mask (n=1162, Pneumonia=846)

| Condition | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec | Mean P (Normal) |
|---|---|---|---|---|---|
| original | 0.9978 [0.9962, 0.9990] | +0.0000 [+0.0000, +0.0000] | 0.986 | 0.965 | 0.047 |
| lung_only | 0.9745 [0.9650, 0.9828] | -0.0233 [-0.0321, -0.0154] | 0.966 | 0.867 | 0.221 |
| background_only | 0.9758 [0.9666, 0.9846] | -0.0220 [-0.0307, -0.0135] | 0.931 | 0.911 | 0.105 |
| displaced_lung_only | 0.9372 [0.9229, 0.9510] | -0.0606 [-0.0745, -0.0471] | 0.994 | 0.313 | 0.631 |
| displaced_background_only | 0.9928 [0.9891, 0.9961] | -0.0049 [-0.0080, -0.0023] | 0.985 | 0.911 | 0.113 |
| border_blank | 0.9971 [0.9948, 0.9987] | -0.0007 [-0.0021, +0.0005] | 0.965 | 0.978 | 0.030 |

## How to read this

- `lung_only` keeps the segmented lungs and grays everything else; `background_only` grays the lungs.
- `displaced_*` apply the same operation with a shifted copy of the lung mask (same shape and area, little overlap with the lungs). They control for the artefact of graying a region of that size.
- `border_blank` grays the outer margin.
- Lung masks come from an automatic segmenter trained on adult radiographs; inspect `qc_lung_masks.png` before trusting them on these paediatric images.
- Gray fill is out of distribution for the model, so compare each condition with its displaced control, not only with the original images.
