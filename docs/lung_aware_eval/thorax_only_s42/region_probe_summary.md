# Region probe: which parts of the image are enough?

Frozen vision checkpoint on the test split. Only the named region is kept; the rest is filled with
mid-gray. The thorax is the lung mask grown by 8 px.
`margin_only_*` keeps an outer frame; `center_only_*` keeps a centred box of the same area.
Intervals are image-level bootstrap and ignore patient clustering. The gray fill
pushes outputs toward Pneumonia, so sensitivity and specificity are not comparable across rows;
compare AUROC.

| Condition | Kept area | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec |
|---|---|---|---|---|---|
| original | 1.00 | 0.9654 [0.9539, 0.9751] | +0.0000 [+0.0000, +0.0000] | 1.000 | 0.000 |
| thorax_only | 0.52 | 0.9970 [0.9943, 0.9990] | +0.0316 [+0.0225, +0.0424] | 0.996 | 0.889 |
| outside_thorax_only | 0.48 | 0.6928 [0.6619, 0.7247] | -0.2726 [-0.3027, -0.2407] | 1.000 | 0.000 |
| outside_thorax_no_margin | 0.16 | 0.8301 [0.8023, 0.8555] | -0.1353 [-0.1658, -0.1081] | 1.000 | 0.000 |
| margin_only_10 | 0.35 | 0.6897 [0.6559, 0.7213] | -0.2757 [-0.3088, -0.2454] | 1.000 | 0.000 |
| center_only_60 | 0.36 | 0.9893 [0.9810, 0.9954] | +0.0239 [+0.0143, +0.0344] | 0.998 | 0.782 |
| margin_only_20 | 0.64 | 0.7783 [0.7499, 0.8059] | -0.1871 [-0.2148, -0.1599] | 1.000 | 0.000 |
| center_only_80 | 0.65 | 0.9936 [0.9886, 0.9972] | +0.0281 [+0.0198, +0.0376] | 1.000 | 0.646 |
