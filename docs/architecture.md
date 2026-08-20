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
  Patient meta →│  Tabular MLP        │────────▶│ Concatenate +  │────▶│ Pneumonia probability │
  (age, gender, │  encoder            │ 64-dim  │ classifier head│     │ (sigmoid, binary)     │
  temp, SpO2 —  └────────────────────┘ embedding└───────────────┘      └─────────────────────┘
  SYNTHETIC*)
```
*See `docs/ethics_statement.md` — temperature and SpO2 are simulated, not real measurements.

## Vision branch

- **Backbone:** DenseNet-121, ImageNet-pretrained (`torchvision.models.densenet121`).
- **Input:** 224×224 RGB (grayscale X-rays are replicated across 3 channels to match ImageNet-pretrained weights' expected input).
- **Output:** 1024-dim feature vector after global average pooling.
- **Grad-CAM target layer:** `features.denseblock4.denselayer16.conv2` — the last convolutional layer before pooling, standard choice for CNN-based Grad-CAM.

## Tabular branch

- **Input features:** Patient Age and Temperature/SpO2 (numeric, standardized), Patient Gender (categorical, integer-encoded). **Age, Temperature, and SpO2 are synthetically generated** — see `data/scripts/prepare_pneumonia_dataset.py` and `docs/ethics_statement.md`.
- **Architecture:** 2-layer MLP (`num_features → 128 → 64`) with BatchNorm + ReLU + Dropout.
- Deliberately kept simple (vs. e.g. TabTransformer) — with only 4 input features, a larger tabular model would be overkill and harder to justify.

## Fusion

- **Strategy:** late fusion — vision and tabular embeddings are concatenated (1024 + 64 = 1088-dim), then passed through a shared classifier head (`1088 → 256 → 1`).
- **Why late fusion over cross-attention:** with only 4 tabular features, a cross-attention mechanism between image patches and tabular tokens would add complexity without a clear benefit at this scale. Late fusion is simpler, faster to train, and easier to ablate cleanly (see `docs/ethics_statement.md` for why interpretability of the ablation matters here).

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

- **Loss:** `BCEWithLogitsLoss` (binary — the codebase is written generically for multi-label, so the same training loop handles this single-class binary case without modification).
- **Optimizer:** AdamW, cosine LR schedule, early stopping on validation macro AUROC.
- **Splits:** patient-level train/val split (original test split preserved exactly as the dataset curators provided it — already patient-disjoint from train). Pneumonia-positive filenames encode a person ID we group by; normal-class images lack an equivalent ID and are conservatively treated as individually unique patients — see `docs/ethics_statement.md`.

## Dataset history

This project originally targeted the full **NIH Chest X-ray14** dataset (14-class multi-label, ~45GB, real patient age/gender/view-position metadata). It was switched to the smaller **Chest X-ray Pneumonia** dataset (binary, ~2GB) to fit local disk/compute constraints, with synthetic clinical vitals added to preserve the multimodal fusion architecture. The original NIH config is kept at `configs/data_nih_legacy.yaml` and `data/scripts/download_nih.py` / `preprocess.py` for reference — the codebase's generic design (dataset, model, and training code are all parameterized by the class list and tabular feature list) meant switching datasets required no changes to `src/data/dataset.py`, `src/models/`, `src/explain/`, or the training loops themselves — only `configs/data.yaml` and a new data-preparation script.

## Config files

| File | Purpose |
|---|---|
| `configs/data.yaml` | Current dataset config (Chest X-ray Pneumonia): paths, class list, tabular feature list |
| `configs/data_nih_legacy.yaml` | Original NIH Chest X-ray14 config, kept for reference |
| `configs/vision_baseline.yaml` | Vision-only training hyperparameters |
| `configs/fusion.yaml` | Fusion model training hyperparameters |
