# Region probe: which parts of the image are enough?

Frozen vision checkpoint on the test split. Only the named region is kept; the rest is filled with
mid-gray. The thorax is the lung mask grown by 8 px.
`margin_only_*` keeps an outer frame; `center_only_*` keeps a centred box of the same area.
Intervals are image-level bootstrap and ignore patient clustering. The gray fill
pushes outputs toward Pneumonia, so sensitivity and specificity are not comparable across rows;
compare AUROC.

| Condition | Kept area | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec |
|---|---|---|---|---|---|
| original | 1.00 | 0.9978 [0.9963, 0.9990] | +0.0000 [+0.0000, +0.0000] | 0.986 | 0.965 |
| thorax_only | 0.52 | 0.9888 [0.9810, 0.9949] | -0.0090 [-0.0160, -0.0034] | 0.991 | 0.870 |
| outside_thorax_only | 0.48 | 0.9638 [0.9516, 0.9743] | -0.0340 [-0.0458, -0.0237] | 0.995 | 0.554 |
| outside_thorax_no_margin | 0.16 | 0.8463 [0.8173, 0.8726] | -0.1514 [-0.1802, -0.1259] | 0.995 | 0.108 |
| margin_only_10 | 0.35 | 0.9340 [0.9174, 0.9495] | -0.0638 [-0.0803, -0.0485] | 1.000 | 0.003 |
| center_only_60 | 0.36 | 0.9878 [0.9805, 0.9938] | -0.0099 [-0.0165, -0.0045] | 0.989 | 0.864 |
| margin_only_20 | 0.64 | 0.9783 [0.9703, 0.9853] | -0.0195 [-0.0271, -0.0128] | 0.967 | 0.854 |
| center_only_80 | 0.65 | 0.9971 [0.9947, 0.9988] | -0.0007 [-0.0021, +0.0005] | 0.965 | 0.978 |
