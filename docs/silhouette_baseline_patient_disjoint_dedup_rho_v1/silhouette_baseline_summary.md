# Silhouette baseline

Classifiers that see only the shape of the lung mask (no pixel intensities), trained on the
patient-disjoint, deduplicated train split, model picked on validation, scored on the test split.
Intervals are image-level bootstrap and ignore patient clustering.

| Mask dilation (px) | Features | Model | Val AUROC | Test AUROC [95% CI] |
|---|---|---|---|---|
| 0 | area_only | gradient_boosting | 0.6188 | 0.6108 [0.5738, 0.6460] |
| 0 | geometry | gradient_boosting | 0.9275 | 0.8896 [0.8691, 0.9081] |
| 0 | profiles | gradient_boosting | 0.9527 | 0.9343 [0.9195, 0.9476] |
| 0 | thumbnail_16x16 | gradient_boosting | 0.9582 | 0.9463 [0.9323, 0.9587] |
| 0 | geometry+profiles | gradient_boosting | 0.9543 | 0.9353 [0.9200, 0.9488] |
| 8 | area_only | gradient_boosting | 0.6279 | 0.6368 [0.6026, 0.6721] |
| 8 | geometry | gradient_boosting | 0.9290 | 0.8830 [0.8613, 0.9021] |
| 8 | profiles | gradient_boosting | 0.9460 | 0.9217 [0.9043, 0.9376] |
| 8 | thumbnail_16x16 | gradient_boosting | 0.9552 | 0.9383 [0.9234, 0.9523] |
| 8 | geometry+profiles | gradient_boosting | 0.9497 | 0.9302 [0.9145, 0.9446] |
