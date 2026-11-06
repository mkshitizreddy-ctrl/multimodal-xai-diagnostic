# Lung-aware training: notes (Step 4)

Descriptive experiment on one dataset (patient-disjoint, deduplicated Kermany split), three seeds per
variant (42, 123, 2024). It asks whether training the DenseNet-121 + CBAM model on lung-masked images
changes how much of the image the model needs. Results: `docs/lung_aware_eval/`
(`lung_aware_summary.md`, per-checkpoint folders). Code: `src/data/lung_masking.py`,
`src/explain/cache_split_masks.py`, `src/explain/lung_aware_eval.py`.

## Variants

All variants use the same split, hyperparameters and early-stopping rule; only the data differ. Masks are
the automatic lung segmentation grown by 8 px (Euclidean), applied at 224 px before augmentation. Outside
the mask the image is mid-gray.

- `baseline`: full images.
- `random_background`: each training image is masked with probability 0.5; validation uses full images.
- `thorax_only`: every training and validation image is masked.

Best validation AUROC was 0.9965 to 0.9968 (baseline), 0.9958 to 0.9966 (random_background) and 0.9953 to
0.9966 (thorax_only). Validation does not separate the variants.

## Test-split region probe (AUROC, mean ± SD over 3 seeds, min to max)

| Condition | baseline | random_background | thorax_only |
|---|---|---|---|
| original | 0.9973 ± 0.0011 (0.996-0.998) | 0.9974 ± 0.0005 (0.997-0.998) | 0.9733 ± 0.0157 (0.963-0.991) |
| thorax_only | 0.9879 ± 0.0030 (0.985-0.990) | 0.9972 ± 0.0004 (0.997-0.997) | 0.9975 ± 0.0009 (0.997-0.999) |
| outside_thorax_only | 0.9672 ± 0.0033 (0.964-0.970) | 0.9572 ± 0.0053 (0.951-0.961) | 0.7441 ± 0.0959 (0.685-0.855) |
| outside_thorax_no_margin | 0.8479 ± 0.0286 (0.820-0.877) | 0.8893 ± 0.0195 (0.867-0.902) | 0.8628 ± 0.0343 (0.830-0.899) |
| margin_only_10 | 0.9419 ± 0.0069 (0.934-0.947) | 0.9355 ± 0.0208 (0.914-0.955) | 0.6789 ± 0.0722 (0.602-0.745) |
| center_only_60 | 0.9840 ± 0.0036 (0.981-0.988) | 0.9908 ± 0.0046 (0.986-0.994) | 0.9888 ± 0.0055 (0.983-0.994) |

Per-seed intervals are in the per-checkpoint folders (image-level bootstrap, ignoring patient clustering).
With three seeds the SD is a rough guide, not an interval. The gray fill pushes outputs toward Pneumonia,
so sensitivity and specificity are not comparable across rows; compare AUROC.

## What the numbers show

- A model that sees only the dilated thorax matches the full-image model on test AUROC. The thorax-only
  models reach 0.9975 on masked test images; the baseline reaches 0.9973 on full images. This does not show
  that the thorax-only models read lung tissue: see the silhouette result below.
- Random-background training makes the model read the thorax as well as the full image (0.9972 versus
  0.9879 for the baseline on thorax-only input) but does not remove the outside-thorax signal: outside-thorax
  AUROC is 0.957 versus 0.967, and the margin-only AUROC is 0.936 versus 0.942, both within seed spread.
  Training on full images half the time leaves the other cues available.
- Thorax-only training removes most of the outside signal: outside-thorax AUROC falls to 0.744 (seed
  range 0.685 to 0.855) and margin-only to 0.679. The one-seed spread is large (seed 2024 differs from the
  other two), so the size of the drop is uncertain.
- The `outside_thorax_no_margin` condition (16% of the image) stays near 0.85 to 0.89 for all three
  variants, including thorax-only models that never saw that region populated. This is not explained here.

## Silhouette baseline: the mask outline predicts the label

A thorax-only model sees the lungs through a gray cut-out. Classifiers that see only the cut-out's shape,
with no pixel intensities, reach (test AUROC, `src/explain/silhouette_baseline.py`,
`docs/silhouette_baseline_patient_disjoint_dedup_rho_v1/`):

| Mask dilation | Area only | Geometry (11 features) | Row/column profiles | 16x16 thumbnail |
|---|---|---|---|---|
| 0 px | 0.611 | 0.890 | 0.934 | 0.946 [0.932, 0.959] |
| 8 px (as in training) | 0.637 | 0.883 | 0.922 | 0.938 [0.923, 0.952] |

The outline therefore carries a strong label signal (0.94) although mask area alone does not (0.61).
Medians: mask area (0 px) 0.364 for Normal, 0.393 for Pneumonia.

Aspect ratio explains part of this. Images are resized to a square, so the original width/height changes
the lungs' apparent shape; the median aspect ratio is 1.236 for Normal and 1.484 for Pneumonia
(`src/explain/silhouette_aspect_check.py`, 0 px, test split):

| Features | Test AUROC |
|---|---|
| aspect ratio only | 0.844 |
| original width, height, aspect | 0.916 |
| silhouette thumbnail | 0.946 |
| silhouette thumbnail + aspect | 0.956 |

Inside aspect-ratio quintiles the silhouette classifier still reaches AUROC 0.81 to 0.96 (the two
highest-aspect bins hold only 16 and 6 Normal images, so those values are noisy). The within-class
correlation of its score with aspect ratio is about +0.3. So the outline carries label information beyond
aspect ratio, whose source (zoom, positioning, acquisition or anatomy) is not identified.

Consequence for the thorax-only result: its 0.9975 AUROC and its near-flat results on the outside-thorax
conditions cannot be attributed to lung tissue. The silhouette alone reaches 0.94, and the same cut-out
is visible in the outside-thorax inputs, so it may explain part of the residual 0.74 to 0.89 there. Both
claims are untested for the CNN: a silhouette-only baseline reaching 0.94 does not show the CNN uses it.

## What it does not show

- Thorax-only models are out of distribution on unmasked images, which is why their `original` AUROC is
  lower (0.973) and has high seed variance. That is not a defect of the models; it only says they were
  not trained for full images.
- The masks come from an automatic segmenter trained on adult radiographs; they were checked on a sample
  (`qc_lung_masks.png`) but not against expert annotation.
- Image-level bootstrap intervals ignore patient clustering. One dataset, one architecture, three seeds.
  Nothing here establishes clinical validity or shows which image evidence a clinician would use.

## Next checks

1. Remove the silhouette as a cue and retrain: a fixed central box (the same for every image, 3 seeds),
   or masks drawn from a different image of the same dilation, so the outline carries no per-image
   information. Compare test AUROC with the thorax-only models' 0.9975.
2. Run the frozen-model audit and the cue-perturbation test on the lung-aware checkpoints.
3. Aspect-preserving resizing (pad to square before the model) to see how much of the silhouette signal
   disappears.
