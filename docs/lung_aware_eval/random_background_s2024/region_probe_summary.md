# Region probe: which parts of the image are enough?

Frozen vision checkpoint on the test split. Only the named region is kept; the rest is filled with
mid-gray. The thorax is the lung mask grown by 8 px.
`margin_only_*` keeps an outer frame; `center_only_*` keeps a centred box of the same area.
Intervals are image-level bootstrap and ignore patient clustering. The gray fill
pushes outputs toward Pneumonia, so sensitivity and specificity are not comparable across rows;
compare AUROC.

| Condition | Kept area | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec |
|---|---|---|---|---|---|
| original | 1.00 | 0.9978 [0.9963, 0.9990] | +0.0000 [+0.0000, +0.0000] | 0.961 | 0.991 |
| thorax_only | 0.52 | 0.9975 [0.9958, 0.9988] | -0.0003 [-0.0009, +0.0002] | 0.959 | 0.984 |
| outside_thorax_only | 0.48 | 0.9598 [0.9473, 0.9716] | -0.0380 [-0.0500, -0.0265] | 1.000 | 0.047 |
| outside_thorax_no_margin | 0.16 | 0.8996 [0.8796, 0.9183] | -0.0983 [-0.1183, -0.0797] | 1.000 | 0.003 |
| margin_only_10 | 0.35 | 0.9137 [0.8956, 0.9305] | -0.0841 [-0.1019, -0.0672] | 1.000 | 0.000 |
| center_only_60 | 0.36 | 0.9927 [0.9875, 0.9966] | -0.0052 [-0.0093, -0.0020] | 0.962 | 0.956 |
| margin_only_20 | 0.64 | 0.9592 [0.9472, 0.9700] | -0.0386 [-0.0498, -0.0281] | 0.994 | 0.380 |
| center_only_80 | 0.65 | 0.9970 [0.9949, 0.9986] | -0.0009 [-0.0017, -0.0002] | 0.943 | 0.991 |
