# v3.0: Audit, consistency and provenance release

Research/portfolio code; not validated for clinical use.

## What this release is
A snapshot of the repository after the October 2026 patient-disjoint audit. The live demo serves the
patient-disjoint, deduplicated vision model, and its weights are verified by SHA-256 at load time.

## Evaluation (test set, 1,165 images; patient-disjoint, deduplicated split)
- Vision model AUROC 0.9978; accuracy 0.9803; sensitivity 0.9859; specificity 0.9652
  (TP 837 / FN 12 / TN 305 / FP 11). Full metrics with Wilson CIs: `docs/model_card.md`.
- Normal-patient identity cannot be verified in this dataset, so full patient-level separation is not claimed.
- The tabular features are synthetic. Fusion results reflect injected signal strength, not clinical benefit.

## Explainability caveats
- Grad-CAM gives a blank map for 268 of 1,165 test images (266 of 316 Normal); the dashboard says so
  instead of showing a heatmap.
- Patch-deletion faithfulness: Grad-CAM beats random deletion at 5-20% of patches, the gap shrinks at 30%
  and reverses at 40%. This is an intervention-specific proxy only.

## Included
- Model card, README quick start, `CITATION.cff` (MIT licence, author corrected), reconciled paper and docs.
- Informal GPU latency notes (`docs/benchmark_notes_2026-10-30.md`): no reliable fusion-vs-vision
  difference; no latency claim is made.

## Checkpoint
`best_model.pth` SHA-256: `b622f8c91b43f39b4e89b34b680eaeba9ab65c99591dde0dd91d35436027abd7`

Full history: `CHANGELOG.md`.
