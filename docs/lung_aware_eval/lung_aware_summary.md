# Lung-aware training: region probe across seeds

Test-split AUROC of each trained variant when only the named region is kept (rest filled with
mid-gray). Cells are mean ± SD over seeds (min to max). With three seeds the SD is a rough guide,
not an interval. Gray fill pushes outputs toward Pneumonia, so compare AUROC only.

| Condition | baseline | random_background | thorax_only |
|---|---|---|---|
| original | 0.9973 ± 0.0011 (0.996-0.998) | 0.9974 ± 0.0005 (0.997-0.998) | 0.9733 ± 0.0157 (0.963-0.991) |
| thorax_only | 0.9879 ± 0.0030 (0.985-0.990) | 0.9972 ± 0.0004 (0.997-0.997) | 0.9975 ± 0.0009 (0.997-0.999) |
| outside_thorax_only | 0.9672 ± 0.0033 (0.964-0.970) | 0.9572 ± 0.0053 (0.951-0.961) | 0.7441 ± 0.0959 (0.685-0.855) |
| outside_thorax_no_margin | 0.8479 ± 0.0286 (0.820-0.877) | 0.8893 ± 0.0195 (0.867-0.902) | 0.8628 ± 0.0343 (0.830-0.899) |
| margin_only_10 | 0.9419 ± 0.0069 (0.934-0.947) | 0.9355 ± 0.0208 (0.914-0.955) | 0.6789 ± 0.0722 (0.602-0.745) |
| center_only_60 | 0.9840 ± 0.0036 (0.981-0.988) | 0.9908 ± 0.0046 (0.986-0.994) | 0.9888 ± 0.0055 (0.983-0.994) |
