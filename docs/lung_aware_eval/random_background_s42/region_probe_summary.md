# Region probe: which parts of the image are enough?

Frozen vision checkpoint on the test split. Only the named region is kept; the rest is filled with
mid-gray. The thorax is the lung mask grown by 8 px.
`margin_only_*` keeps an outer frame; `center_only_*` keeps a centred box of the same area.
Intervals are image-level bootstrap and ignore patient clustering. The gray fill
pushes outputs toward Pneumonia, so sensitivity and specificity are not comparable across rows;
compare AUROC.

| Condition | Kept area | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec |
|---|---|---|---|---|---|
| original | 1.00 | 0.9975 [0.9958, 0.9988] | +0.0000 [+0.0000, +0.0000] | 0.986 | 0.946 |
| thorax_only | 0.52 | 0.9972 [0.9949, 0.9988] | -0.0003 [-0.0015, +0.0006] | 0.996 | 0.908 |
| outside_thorax_only | 0.48 | 0.9605 [0.9477, 0.9723] | -0.0370 [-0.0492, -0.0257] | 1.000 | 0.016 |
| outside_thorax_no_margin | 0.16 | 0.9015 [0.8806, 0.9214] | -0.0960 [-0.1168, -0.0766] | 1.000 | 0.000 |
| margin_only_10 | 0.35 | 0.9377 [0.9225, 0.9518] | -0.0599 [-0.0748, -0.0464] | 1.000 | 0.000 |
| center_only_60 | 0.36 | 0.9941 [0.9887, 0.9981] | -0.0034 [-0.0079, -0.0001] | 0.999 | 0.832 |
| margin_only_20 | 0.64 | 0.9738 [0.9652, 0.9816] | -0.0237 [-0.0320, -0.0162] | 0.998 | 0.383 |
| center_only_80 | 0.65 | 0.9971 [0.9949, 0.9987] | -0.0004 [-0.0014, +0.0004] | 0.992 | 0.934 |
