# Mask QC, audit reading and the explanation link

Run 2026-11-02 on the frozen vision checkpoint, patient-disjoint deduplicated test split (1,165 images,
849 Pneumonia). Inputs: `docs/shortcut_audit_patient_disjoint_dedup_rho_v1_vision/` and
`docs/lung_localization_testset_rho0.csv`. Code: `src/explain/explanation_link.py`. Tables:
`docs/explanation_link_patient_disjoint_dedup_rho_v1_vision/`.

## Mask QC (visual, 24 random test images in `qc_lung_masks.png`)

- The segmenter follows lung borders well on aerated lungs. About 6-8 of the 24 tiles show visible defects:
  the outline cuts into a consolidated lower zone, leaves a notch, or drops a region. These cluster on
  dense lungs, i.e. Pneumonia.
- The mask excludes the mediastinum and hilum, so "background" contains heart, hilum, ribs, shoulders,
  abdomen and visible text markers.
- Verdict: usable for the intervention audit with the caveat below; not suitable as training supervision
  without repair, because it would teach the model to ignore consolidated lung.

## Audit reading (AUROC; original 0.9978)

| Condition | AUROC | Matched control |
|---|---|---|
| Keep lungs only | 0.9746 | Keep a displaced non-lung region: 0.9374 |
| Gray the lungs | 0.9759 | Gray a displaced region: 0.9929 |
| Gray the outer 10% margin | 0.9971 (-0.0007, CI includes 0) | none |

- Lung content is informative: keeping the lungs beats keeping a same-shape displaced region by 0.037, and
  removing the true lungs costs about 4 times more than removing a displaced region (0.022 versus 0.005).
- The non-lung part is also nearly sufficient (0.976), so the evidence is redundant. The model does not rely
  on the lungs alone.
- Exploratory, not scripted: on the 291 images with the smallest masks (likely under-segmented),
  gray-the-lungs AUROC is 0.993 versus 0.968 on the other 874, and within Pneumonia a smaller mask goes with a
  higher gray-the-lungs logit (Spearman -0.18). Consistent with lung remnants leaking into "background";
  small masks could also just be small chests. The later dilation run (`mask_dilation_notes.md`) did not
  support this as the main explanation.
- Sensitivity and specificity are not comparable across conditions: the gray fill pushes outputs toward
  Pneumonia. Use AUROC.

## Explanation link (Pneumonia only, n=847 nonzero maps)

The Grad-CAM lung-energy fraction depends strongly on mask area (Spearman 0.83 within Pneumonia), so all
correlations below control for mask area (partial Spearman, 95% image-level bootstrap CI).

| Logit | Partial rho |
|---|---|
| original | -0.415 [-0.476, -0.348] |
| lung only | -0.025 [-0.095, +0.046] |
| gray the lungs | -0.634 [-0.677, -0.582] |
| displaced, lung only | -0.282 [-0.343, -0.211] |
| displaced, gray the region | -0.472 [-0.529, -0.409] |
| true contrast (lung only - gray the lungs) | +0.536 [+0.479, +0.585] |
| displaced contrast (matched control) | +0.333 [+0.266, +0.393] |
| difference | +0.203 [+0.129, +0.281] |

- Grad-CAM mass is enriched in the lungs: median lung fraction / mask area is 1.26, with 97% of maps above 1.
- Images whose Grad-CAM mass sits more inside the lungs lose more of their Pneumonia margin when the lungs
  are grayed (rho -0.63), while keeping only the lungs is unrelated to the fraction (rho -0.03). So the
  fraction tracks how much the prediction survives without the lungs.
- Part of this is generic: the matched displaced-mask contrast also correlates (+0.33), and the original
  margin itself correlates with the fraction (-0.42), so confident images differ in map shape. The part
  specific to the true lung mask is +0.20 [+0.13, +0.28].
- Normal images are not analysed: only 50 have a nonzero map.

## Limits

One dataset, one checkpoint, one seed. Image-level CIs ignore patient clustering. Mask defects fall on
the Pneumonia images that matter most. The explanation link is correlational. The lung-fraction map
comes from the rho=0 run; its probabilities match the audit's to within 0.0011.

## Consequence for the plan

The model uses lung content but is not lung-exclusive, and Grad-CAM lung fraction only partly reflects the
intervention results. Lung-aware training (Step 4) is justified as a test, not as a fix for a proven shortcut.
The dilation check is done (`mask_dilation_notes.md`): the gray-the-lungs result barely shrinks.
