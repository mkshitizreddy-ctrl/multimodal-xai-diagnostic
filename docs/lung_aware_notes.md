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

- A model that sees only the dilated thorax matches the full-image model. The thorax-only models reach
  0.9975 on masked test images; the baseline reaches 0.9973 on full images. The label is therefore
  learnable from the thorax region at no cost in test AUROC on this dataset.
- Random-background training makes the model read the thorax as well as the full image (0.9972 versus
  0.9879 for the baseline on thorax-only input) but does not remove the outside-thorax signal: outside-thorax
  AUROC is 0.957 versus 0.967, and the margin-only AUROC is 0.936 versus 0.942, both within seed spread.
  Training on full images half the time leaves the other cues available.
- Thorax-only training removes most of the outside signal: outside-thorax AUROC falls to 0.744 (seed
  range 0.685 to 0.855) and margin-only to 0.679. The one-seed spread is large (seed 2024 differs from the
  other two), so the size of the drop is uncertain.
- The `outside_thorax_no_margin` condition (16% of the image) stays near 0.85 to 0.89 for all three
  variants, including thorax-only models that never saw that region populated. This is not explained here.

## What it does not show

- Thorax-only models are out of distribution on unmasked images, which is why their `original` AUROC is
  lower (0.973) and has high seed variance. That is not a defect of the models; it only says they were
  not trained for full images.
- A thorax-only model can still use the mask silhouette. The mask outline, its area and position reflect
  how the radiograph was framed, which may correlate with the label. The outside-thorax inputs also keep
  the gray silhouette of the lungs, which could explain part of the residual AUROC. A silhouette-only
  baseline has not been run yet.
- The masks come from an automatic segmenter trained on adult radiographs; they were checked on a sample
  (`qc_lung_masks.png`) but not against expert annotation.
- Image-level bootstrap intervals ignore patient clustering. One dataset, one architecture, three seeds.
  Nothing here establishes clinical validity or shows which image evidence a clinician would use.

## Next checks

1. Silhouette baseline: classify the label from mask area, bounding box and row/column profiles alone.
2. Run the frozen-model audit and the cue-perturbation test on the lung-aware checkpoints.
3. Test-set performance of the thorax-only model with a different dilation (0 and 16 px).
