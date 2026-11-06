# Region probe: which parts of the image are enough?

Frozen vision checkpoint on the test split. Only the named region is kept; the rest is filled with
mid-gray. The thorax is the lung mask grown by 8 px.
`margin_only_*` keeps an outer frame; `center_only_*` keeps a centred box of the same area.
Intervals are image-level bootstrap and ignore patient clustering. The gray fill
pushes outputs toward Pneumonia, so sensitivity and specificity are not comparable across rows;
compare AUROC.

| Condition | Kept area | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec |
|---|---|---|---|---|---|
| original | 1.00 | 0.9914 [0.9863, 0.9955] | +0.0000 [+0.0000, +0.0000] | 1.000 | 0.544 |
| thorax_only | 0.52 | 0.9985 [0.9967, 0.9998] | +0.0071 [+0.0040, +0.0107] | 0.985 | 0.984 |
| outside_thorax_only | 0.48 | 0.8548 [0.8295, 0.8792] | -0.1366 [-0.1609, -0.1130] | 1.000 | 0.000 |
| outside_thorax_no_margin | 0.16 | 0.8597 [0.8350, 0.8820] | -0.1317 [-0.1561, -0.1086] | 1.000 | 0.000 |
| margin_only_10 | 0.35 | 0.7450 [0.7139, 0.7766] | -0.2464 [-0.2763, -0.2156] | 1.000 | 0.000 |
| center_only_60 | 0.36 | 0.9940 [0.9887, 0.9978] | +0.0026 [-0.0007, +0.0058] | 0.974 | 0.962 |
| margin_only_20 | 0.64 | 0.8551 [0.8287, 0.8795] | -0.1363 [-0.1617, -0.1130] | 1.000 | 0.000 |
| center_only_80 | 0.65 | 0.9975 [0.9946, 0.9993] | +0.0061 [+0.0035, +0.0089] | 0.985 | 0.975 |
