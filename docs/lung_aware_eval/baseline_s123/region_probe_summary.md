# Region probe: which parts of the image are enough?

Frozen vision checkpoint on the test split. Only the named region is kept; the rest is filled with
mid-gray. The thorax is the lung mask grown by 8 px.
`margin_only_*` keeps an outer frame; `center_only_*` keeps a centred box of the same area.
Intervals are image-level bootstrap and ignore patient clustering. The gray fill
pushes outputs toward Pneumonia, so sensitivity and specificity are not comparable across rows;
compare AUROC.

| Condition | Kept area | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec |
|---|---|---|---|---|---|
| original | 1.00 | 0.9961 [0.9923, 0.9990] | +0.0000 [+0.0000, +0.0000] | 0.986 | 0.962 |
| thorax_only | 0.52 | 0.9846 [0.9758, 0.9921] | -0.0115 [-0.0183, -0.0057] | 1.000 | 0.766 |
| outside_thorax_only | 0.48 | 0.9674 [0.9552, 0.9776] | -0.0287 [-0.0398, -0.0196] | 0.973 | 0.772 |
| outside_thorax_no_margin | 0.16 | 0.8201 [0.7920, 0.8459] | -0.1760 [-0.2032, -0.1508] | 0.842 | 0.614 |
| margin_only_10 | 0.35 | 0.9451 [0.9316, 0.9577] | -0.0510 [-0.0640, -0.0389] | 1.000 | 0.044 |
| center_only_60 | 0.36 | 0.9837 [0.9731, 0.9921] | -0.0124 [-0.0209, -0.0054] | 0.982 | 0.911 |
| margin_only_20 | 0.64 | 0.9828 [0.9762, 0.9887] | -0.0132 [-0.0187, -0.0082] | 0.935 | 0.940 |
| center_only_80 | 0.65 | 0.9939 [0.9879, 0.9981] | -0.0022 [-0.0059, +0.0009] | 0.948 | 0.978 |
