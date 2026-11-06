# Region probe: which parts of the image are enough?

Frozen vision checkpoint on the test split. Only the named region is kept; the rest is filled with
mid-gray. The thorax is the lung mask grown by 8 px.
`margin_only_*` keeps an outer frame; `center_only_*` keeps a centred box of the same area.
Intervals are image-level bootstrap and ignore patient clustering. The gray fill
pushes outputs toward Pneumonia, so sensitivity and specificity are not comparable across rows;
compare AUROC.

| Condition | Kept area | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec |
|---|---|---|---|---|---|
| original | 1.00 | 0.9969 [0.9938, 0.9989] | +0.0000 [+0.0000, +0.0000] | 0.989 | 0.965 |
| thorax_only | 0.52 | 0.9968 [0.9939, 0.9987] | -0.0001 [-0.0010, +0.0007] | 0.989 | 0.953 |
| outside_thorax_only | 0.48 | 0.9511 [0.9371, 0.9635] | -0.0458 [-0.0598, -0.0331] | 1.000 | 0.155 |
| outside_thorax_no_margin | 0.16 | 0.8669 [0.8425, 0.8904] | -0.1300 [-0.1542, -0.1068] | 1.000 | 0.006 |
| margin_only_10 | 0.35 | 0.9551 [0.9414, 0.9676] | -0.0418 [-0.0548, -0.0299] | 1.000 | 0.000 |
| center_only_60 | 0.36 | 0.9856 [0.9750, 0.9939] | -0.0113 [-0.0200, -0.0046] | 0.988 | 0.899 |
| margin_only_20 | 0.64 | 0.9740 [0.9647, 0.9823] | -0.0229 [-0.0319, -0.0145] | 0.991 | 0.639 |
| center_only_80 | 0.65 | 0.9962 [0.9924, 0.9986] | -0.0007 [-0.0018, +0.0002] | 0.988 | 0.956 |
