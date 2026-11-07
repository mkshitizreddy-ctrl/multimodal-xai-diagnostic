# Outline control: fixed box versus lung outline

Test-split AUROC of each trained variant when only the named region is kept (rest filled with
mid-gray). Cells are mean ± SD over seeds (min to max). With three seeds the SD is a rough guide,
not an interval. Gray fill pushes outputs toward Pneumonia, so compare AUROC only.

| Condition | baseline | random_background | thorax_only | fixed_box |
|---|---|---|---|---|
| original | 0.9973 ± 0.0011 (0.996-0.998) | 0.9974 ± 0.0005 (0.997-0.998) | 0.9733 ± 0.0157 (0.963-0.991) | 0.9937 ± 0.0006 (0.993-0.994) |
| thorax_only | 0.9879 ± 0.0030 (0.985-0.990) | 0.9972 ± 0.0004 (0.997-0.997) | 0.9975 ± 0.0009 (0.997-0.999) | 0.9949 ± 0.0014 (0.993-0.996) |
| fixed_box | 0.9933 ± 0.0019 (0.992-0.995) | 0.9956 ± 0.0016 (0.994-0.997) | 0.9944 ± 0.0019 (0.993-0.997) | 0.9965 ± 0.0012 (0.995-0.997) |
| outside_fixed_box | 0.9578 ± 0.0070 (0.951-0.965) | 0.9445 ± 0.0245 (0.916-0.961) | 0.7132 ± 0.0550 (0.654-0.763) | 0.8107 ± 0.0347 (0.780-0.848) |
