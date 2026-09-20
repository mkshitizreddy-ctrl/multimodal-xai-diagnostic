# Rotation-and-zoom augmentation ablation

## Question

Does a training-only augmentation policy of horizontal flipping, small rotation,
and 1.00--1.10x random magnification improve held-out performance over the
existing CBAM vision checkpoint?

This is an augmentation experiment, not an attempt to change the dataset or
claim that synthetic transforms improve image quality. The validation and test
sets must remain unaugmented.

## Fixed conditions

- Dataset and train/validation/test CSVs: unchanged.
- Architecture: DenseNet-121 with CBAM.
- Seed: 42.
- Optimizer, learning schedule, epochs, and model selection: unchanged.
- Difference: the augmented run uses 7-degree random rotation and 1.00--1.10x
  random magnification in addition to the existing 0.5-probability horizontal
  flip. The original baseline already used 5-degree rotation.

## Run

```powershell
conda activate ai_env
python src/train.py --data-config configs/data.yaml --train-config configs/vision_rotation_zoom.yaml
python src/evaluate_full_metrics.py --checkpoint checkpoints/vision_rotation_zoom/best_model.pth --train-config configs/vision_rotation_zoom.yaml --n-bootstrap 2000 --output-csv docs/vision_rotation_zoom_metrics.csv
```

## Report table

| Model | Accuracy | F1 | AUROC | AUPR |
|---|---:|---:|---:|---:|
| Existing CBAM vision baseline | 0.8638 | 0.9015 | 0.9604 | 0.9628 |
| Rotation + zoom (seed 42) | pending | pending | pending | pending |

Report confidence intervals alongside the final table. Do not claim an
improvement unless the held-out metrics improve and the result is repeated
across the planned seeds (42, 123, and 2024).
