# Architecture

## Overview

Two models are trained and compared in this project:

1. **Vision-only baseline** (`src/models/vision_encoder.py`) — DenseNet-121 fine-tuned for multi-label classification directly on the X-ray image.
2. **Fusion model** (`src/models/fusion.py`) — the same DenseNet-121 vision branch, plus an MLP tabular branch encoding patient metadata, combined via late fusion.

The fusion model is the "real" architecture; the vision-only baseline exists
specifically as the control arm of an ablation study (see
`notebooks/02_fusion_ablation_results.ipynb`) to quantify what the tabular
metadata actually adds.

## Data flow

```
                ┌────────────────────┐
   X-ray image →│  DenseNet-121       │─────────────┐
   (224x224x3)  │  feature extractor  │  1024-dim    │
                └────────────────────┘  embedding    │
                                                       ▼
                ┌────────────────────┐         ┌───────────────┐      ┌─────────────────────┐
  Patient meta →│  Tabular MLP        │────────▶│ Concatenate +  │────▶│ 14 disease           │
   (age, sex,   │  encoder            │ 64-dim  │ classifier head│     │ probabilities         │
   view pos.)   └────────────────────┘ embedding└───────────────┘      │ (sigmoid, multi-label)│
                                                                        └─────────────────────┘
```

## Vision branch

- **Backbone:** DenseNet-121, ImageNet-pretrained (`torchvision.models.densenet121`).
- **Input:** 224×224 RGB (grayscale X-rays are replicated across 3 channels to match ImageNet-pretrained weights' expected input).
- **Output:** 1024-dim feature vector after global average pooling.
- **Grad-CAM target layer:** `features.denseblock4.denselayer16.conv2` — the last convolutional layer before pooling, standard choice for CNN-based Grad-CAM.

## Tabular branch

- **Input features:** Patient Age (numeric, standardized), Patient Gender and View Position (categorical, integer-encoded).
- **Architecture:** 2-layer MLP (`num_features → 128 → 64`) with BatchNorm + ReLU + Dropout.
- Deliberately kept simple (vs. e.g. TabTransformer) — with only 3 input features, a larger tabular model would be overkill and harder to justify.

## Fusion

- **Strategy:** late fusion — vision and tabular embeddings are concatenated (1024 + 64 = 1088-dim), then passed through a shared classifier head (`1088 → 256 → 14`).
- **Why late fusion over cross-attention:** with only 3 tabular features, a cross-attention mechanism between image patches and tabular tokens would add complexity without a clear benefit at this scale. Late fusion is simpler, faster to train, and easier to ablate cleanly (see `docs/ethics_statement.md` for why interpretability of the ablation matters here).

## Explainability pipeline

```
trained model + input image
        │
        ▼
  Grad-CAM (src/explain/gradcam.py)
        │
        ├──▶ heatmap overlay (visual explanation)
        │
        ▼
  Occlusion counterfactual (src/explain/counterfactual.py)
        │
        ├─ threshold heatmap → binary mask
        ├─ inpaint masked region (cv2.INPAINT_TELEA)
        ├─ re-run model on inpainted image
        └─▶ confidence before/after comparison
```

Both explainability modules operate on the vision-only baseline model
currently (they need a single-image forward pass; extending them to the
fusion model's dual-input forward pass is a natural next step — see
`ChestXrayFusionModel.get_target_layer()`, which already exposes the same
target layer name for this purpose).

## Training

- **Loss:** `BCEWithLogitsLoss` (multi-label; each class is an independent binary decision).
- **Optimizer:** AdamW, cosine LR schedule, early stopping on validation macro AUROC.
- **Splits:** patient-level train/val/test (70/15/15) — critical for chest X-ray datasets since patients have multiple scans, and image-level splitting would leak patient identity across splits.

## Config files

| File | Purpose |
|---|---|
| `configs/data.yaml` | Dataset paths, class list, tabular feature list, split ratios |
| `configs/vision_baseline.yaml` | Vision-only training hyperparameters |
| `configs/fusion.yaml` | Fusion model training hyperparameters |
