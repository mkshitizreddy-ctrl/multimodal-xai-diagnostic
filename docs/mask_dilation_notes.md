# Mask dilation sensitivity

Run 2026-11-03 on the frozen vision checkpoint, patient-disjoint deduplicated test split (1,165 images,
849 Pneumonia). Code: `--mask-dilation` and `--masks-path` in `src/explain/shortcut_audit.py`. Tables:
`docs/shortcut_audit_patient_disjoint_dedup_rho_v1_vision_dilated4px/` and `..._dilated8px/`.

Question: is the strong "lungs grayed" result (AUROC 0.976) caused by lung tissue that the segmenter missed
and left in the background? If so, growing every mask should push it toward chance.

## Result (AUROC [95% CI], all 1,165 images; original 0.9978)

| Dilation | Mean mask area | Displaced IoU | Keep lungs | Keep displaced region | Gray the lungs | Gray displaced region |
|---|---|---|---|---|---|---|
| 0 px | 0.383 | 0.263 | 0.9746 [0.9650, 0.9829] | 0.9374 [0.9229, 0.9509] | 0.9759 [0.9672, 0.9839] | 0.9929 [0.9892, 0.9959] |
| 4 px | 0.454 | 0.326 | 0.9816 [0.9723, 0.9891] | 0.9679 [0.9588, 0.9766] | 0.9704 [0.9601, 0.9795] | 0.9905 [0.9861, 0.9944] |
| 8 px | 0.519 | 0.373 | 0.9888 [0.9814, 0.9947] | 0.9837 [0.9774, 0.9890] | 0.9638 [0.9514, 0.9740] | 0.9882 [0.9826, 0.9929] |

## Reading

- **Mask defects do not explain the non-lung signal.** With every mask grown by 8 px the model still reaches
  0.964 [0.951, 0.974] with the masked region grayed (0.976 before dilation). The intervals overlap.
- **At 8 px the mask is no longer "the lungs".** The two lung fields merge over the mediastinum and cover the
  heart, hilum, upper chest and part of the diaphragm (see `qc_lung_masks.png` in the 8 px folder), about 52% of
  the image. The model still ranks cases well from what is left: the chest-wall periphery, shoulders, neck,
  abdomen, image margins and any visible text markers. That points to cues outside the thorax, though which
  ones is not identified. Blanking only the outer 10% margin changes nothing (-0.0007), so margin content is
  not necessary; whether it is sufficient has not been tested.
- **The lung-specific advantage shrinks as masks grow** (keep lungs minus keep displaced region: 0.037, 0.014,
  0.005), but the control also degrades: its overlap with the lungs rises (IoU 0.263 to 0.373) and the kept
  area grows. The undilated comparison is the cleanest for the lung-specific claim.
- **Removing the true lungs keeps costing more than removing a displaced region** (gray lungs minus gray
  displaced: -0.017, -0.020, -0.024), so lung content still matters.
- Sensitivity and specificity move a lot with the gray area (specificity when graying the lungs falls
  0.911, 0.801, 0.554). Use AUROC.

## Consequence

The model's ranking is redundant: lung content is used, and so is information outside the thorax. This earlier
suspicion in `explanation_link_notes.md`, that lung remnants in the background inflate the gray-the-lungs
result, is largely not supported. Lung-aware training (Step 4) was run as the follow-up test; its results
(thorax-only, random-background and fixed-box variants, 3 seeds each) are in `lung_aware_notes.md`. They show that no single region is necessary for the headline AUROC.

## Limits

One dataset, one checkpoint, one seed. Image-level CIs ignore patient clustering. Dilation is isotropic and
cannot recover consolidated lung that the segmenter dropped inside the mask interior.
