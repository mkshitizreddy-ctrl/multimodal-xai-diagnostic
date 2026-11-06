# Region probe: which parts of the image are enough?

Frozen vision checkpoint on the test split. Only the named region is kept; the rest is filled with
mid-gray. The thorax is the lung mask grown by 8 px.
`margin_only_*` keeps an outer frame; `center_only_*` keeps a centred box of the same area.
Intervals are image-level bootstrap and ignore patient clustering. The gray fill
pushes outputs toward Pneumonia, so sensitivity and specificity are not comparable across rows;
compare AUROC.

| Condition | Kept area | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec |
|---|---|---|---|---|---|
| original | 1.00 | 0.9982 [0.9963, 0.9994] | +0.0000 [+0.0000, +0.0000] | 0.991 | 0.965 |
| thorax_only | 0.52 | 0.9903 [0.9836, 0.9952] | -0.0079 [-0.0132, -0.0037] | 0.989 | 0.854 |
| outside_thorax_only | 0.48 | 0.9703 [0.9591, 0.9801] | -0.0279 [-0.0387, -0.0182] | 0.991 | 0.633 |
| outside_thorax_no_margin | 0.16 | 0.8772 [0.8554, 0.8985] | -0.1210 [-0.1425, -0.0999] | 0.913 | 0.646 |
| margin_only_10 | 0.35 | 0.9465 [0.9324, 0.9595] | -0.0517 [-0.0652, -0.0391] | 0.999 | 0.111 |
| center_only_60 | 0.36 | 0.9806 [0.9704, 0.9890] | -0.0176 [-0.0266, -0.0100] | 0.989 | 0.816 |
| margin_only_20 | 0.64 | 0.9832 [0.9766, 0.9890] | -0.0150 [-0.0213, -0.0092] | 0.941 | 0.927 |
| center_only_80 | 0.65 | 0.9964 [0.9922, 0.9989] | -0.0018 [-0.0043, -0.0002] | 0.985 | 0.968 |
