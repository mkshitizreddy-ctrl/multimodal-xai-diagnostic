# Region probe: which parts of the image are enough?

Frozen vision checkpoint on the test split. Only the named region is kept; the rest is filled with
mid-gray. The thorax is the lung mask grown by 8 px.
`margin_only_*` keeps an outer frame; `center_only_*` keeps a centred box of the same area.
Intervals are image-level bootstrap and ignore patient clustering. The gray fill
pushes outputs toward Pneumonia, so sensitivity and specificity are not comparable across rows;
compare AUROC.

| Condition | Kept area | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec |
|---|---|---|---|---|---|
| original | 1.00 | 0.9631 [0.9496, 0.9751] | +0.0000 [+0.0000, +0.0000] | 1.000 | 0.044 |
| thorax_only | 0.52 | 0.9969 [0.9943, 0.9989] | +0.0338 [+0.0227, +0.0460] | 0.996 | 0.946 |
| outside_thorax_only | 0.48 | 0.6847 [0.6525, 0.7181] | -0.2784 [-0.3101, -0.2454] | 1.000 | 0.000 |
| outside_thorax_no_margin | 0.16 | 0.8986 [0.8783, 0.9177] | -0.0646 [-0.0862, -0.0441] | 1.000 | 0.003 |
| margin_only_10 | 0.35 | 0.6019 [0.5666, 0.6374] | -0.3613 [-0.3961, -0.3273] | 1.000 | 0.000 |
| center_only_60 | 0.36 | 0.9830 [0.9730, 0.9915] | +0.0199 [+0.0080, +0.0319] | 0.996 | 0.772 |
| margin_only_20 | 0.64 | 0.7364 [0.7032, 0.7692] | -0.2267 [-0.2580, -0.1958] | 1.000 | 0.000 |
| center_only_80 | 0.65 | 0.9948 [0.9903, 0.9981] | +0.0316 [+0.0215, +0.0427] | 0.999 | 0.769 |
