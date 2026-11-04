# Region probe: which parts of the image are enough?

Run 2026-11-04 on the frozen vision checkpoint, patient-disjoint deduplicated test split (1,165 images,
849 Pneumonia). Code: `src/explain/region_probe.py`, `region_probe_utils.py`. Tables:
`docs/region_probe_patient_disjoint_dedup_rho_v1_vision/`. It follows the mask-dilation run
(`mask_dilation_notes.md`), where the model still reached AUROC 0.964 with the whole thorax grayed.

Only the named region is kept and the rest is filled with mid-gray. The thorax is the lung mask grown by 8 px.
`margin_only_*` keeps an outer frame; `center_only_*` keeps a centred box of the same area.

## Result (AUROC [95% CI]; original 0.9978)

| Kept region | Kept area | AUROC | Change vs original |
|---|---|---|---|
| Thorax only | 0.52 | 0.9888 [0.9810, 0.9949] | -0.0090 [-0.0160, -0.0034] |
| Outside the thorax | 0.48 | 0.9638 [0.9516, 0.9743] | -0.0340 [-0.0458, -0.0237] |
| Outside the thorax and the 10% margin | 0.16 | 0.8463 [0.8173, 0.8726] | -0.1514 [-0.1802, -0.1259] |
| Outer 10% frame | 0.35 | 0.9340 [0.9174, 0.9495] | -0.0638 [-0.0803, -0.0485] |
| Centred box, same area | 0.36 | 0.9878 [0.9805, 0.9938] | -0.0099 [-0.0165, -0.0045] |
| Outer 20% frame | 0.64 | 0.9783 [0.9703, 0.9853] | -0.0195 [-0.0271, -0.0128] |
| Centred box, same area | 0.65 | 0.9971 [0.9947, 0.9988] | -0.0007 [-0.0021, +0.0005] |

## Reading

- **The label is decodable from the outer 10% frame alone** (AUROC 0.934). That frame is mostly
  non-anatomical: image borders, black bars, text markers and the edges of the body. The matched centred box does
  better (0.988), so the centre carries more, but the margin is far from chance.
- **Content between the thorax and the margin carries signal too:** with the thorax and the outer 10% both grayed
  (16% of the image left: chest wall, shoulders, neck, abdomen), AUROC is still 0.846.
- **Nothing tested is necessary.** Blanking the margin changes nothing (-0.0007, cue perturbation and audit),
  graying the whole thorax leaves 0.964, and JPEG recompression and texture removal change nothing. The signal is
  redundant across regions.
- Sensitivity and specificity are not comparable across rows: the gray fill drives outputs toward Pneumonia
  (specificity 0.003 for the margin-only frame). Compare AUROC.

## What the chain of results supports

1. The label is recoverable from acquisition properties alone (file size AUROC 0.980; `nonanatomical_baseline_notes.md`).
2. The CNN's ranking is redundant, with thorax content, outside-thorax content and the image margin each sufficient
   on its own. No single cue tested is necessary (`cue_perturbation_notes.md`, `mask_dilation_notes.md`).
3. Therefore the test AUROC of 0.998 overstates the anatomical diagnostic evidence the model uses on this
   dataset. This shows sufficiency and redundancy, not that the model uses the margin on normal inputs.
4. External validation on a different source would be the real test; none is available here.

## Limits

One dataset, one checkpoint, one seed. Image-level CIs ignore patient clustering. Gray-filled inputs are out of
distribution, so a region's AUROC says what the model can extract from it, not what it does extract from full
images. The cause of the margin signal (markers, black bars, framing, body edges) is not identified.
