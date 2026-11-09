# Resize control: test AUROC by resize mode

Mean ± SD over seeds (individual seeds in the last column). Bins are equal-count aspect-ratio
quintiles of the original image, bin 1 the squarest. A bin with one class gives no value.

| Mode | Seeds | AUROC | bin 1 | bin 2 | bin 3 | bin 4 | bin 5 | Per seed |
|---|---|---|---|---|---|---|---|---|
| pad | 3 | 0.9976 ± 0.0007 | 0.9983 | 0.9988 | 0.9918 | 0.9911 | 0.9976 | 42: 0.9980, 123: 0.9967, 2024: 0.9979 |
| stretch | 3 | 0.9973 ± 0.0011 | 0.9979 | 0.9991 | 0.9921 | 0.9891 | 0.9980 | 42: 0.9978, 123: 0.9961, 2024: 0.9982 |
