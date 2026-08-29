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

## Attention module (CBAM)

Added `src/models/attention.py` after reading the pneumonia-CXR literature (see the
list below) — CBAM (Convolutional Block Attention Module, Woo et al., ECCV 2018)
sits between the backbone and the classifier head, gated behind `use_cbam` in
`configs/vision_baseline.yaml` / `configs/fusion.yaml` so it's easy to A/B against
the original no-attention baseline.

- **Channel attention:** avg-pool + max-pool each of the 1024 feature channels,
  shared 2-layer MLP, sigmoid, rescale channels — "which channels matter."
- **Spatial attention:** pool across channels, 7×7 conv, sigmoid, rescale spatially
  — "which pixels matter." Applied after channel attention (the ordering CBAM's
  own ablation study found best).
- **Why this specifically, and not a different upgrade:** several 2024–2026 papers
  report CBAM/SE attention on this exact backbone + dataset combination (DenseNet-121
  on the Kaggle Chest X-ray Pneumonia set, the same one this project uses) improving
  both classification metrics *and* Grad-CAM localization quality — heatmaps
  concentrate more tightly on pathological lung regions instead of spreading across
  the whole image. That second point is what made this worth trying here
  specifically: it's a direct, literature-backed extension of the shortcut-learning
  finding in `docs/ethics_statement.md`, not just an accuracy play. Full reading
  notes in `docs/paper_notes.md`; papers read:
    - Dey, "CBAM-Enhanced DenseNet121 for Multi-Class Chest X-Ray Classification with Grad-CAM Explainability" (2026) — [arxiv.org/abs/2604.12305](https://arxiv.org/abs/2604.12305) — closest paper to this exact stack (DenseNet121 + CBAM + Grad-CAM on CXR pneumonia)
    - "Enhanced X-ray image classification for pneumonia detection using deep learning based CBAM and SE mechanisms," ScienceDirect (2025) — [doi link](https://www.sciencedirect.com/science/article/pii/S2666521225001036) — CBAM vs. SE comparison, argument for CBAM's spatial term specifically
    - "An Enhanced Deep Learning Framework for Pneumonia Detection in Chest X-rays," SN Computer Science (2025) — [link.springer.com](https://link.springer.com/article/10.1007/s42979-025-04017-x) — DenseNet-121+CBAM matching heavier ensembles at a fraction of the parameters
    - Shahi & Bagale, "Weakly Supervised Pneumonia Localization from Chest X-Rays Using Deep Neural Network and Grad-CAM Explanations" (2025) — [arxiv.org/pdf/2511.00456v1](https://arxiv.org/pdf/2511.00456v1) — the paper behind `src/explain/measure_lung_localization.py`; argues for measuring localization quality, not just eyeballing heatmaps
    - "Explainable Deep Learning in Medical Imaging: Brain Tumor and Pneumonia Detection" (2025) — [arxiv.org/html/2510.21823](https://arxiv.org/html/2510.21823) — DenseNet121 vs ResNet50 on the same ~5,860-image Kaggle set, independent confirmation of backbone choice
    - "Pneumonia Image Classification Using DenseNet Architecture," MDPI (2024) — [mdpi.com/2078-2489/15/10/611](https://www.mdpi.com/2078-2489/15/10/611) — DenseNet121/169/201 baseline accuracy comparison on the exact same dataset, no attention mechanism (used as a sanity check for our no-CBAM baseline numbers)
- **Why CBAM over plain SE:** SE-Net (the lighter-weight alternative some of the
  above papers use instead) only does channel attention. CBAM's extra
  spatial-attention term is what actually helps here, since the whole point is
  tightening *where* the model looks — a channel-only mechanism wouldn't touch that.
- **Cost:** negligible — CBAM adds ~2.1M parameters to a 121-layer, ~7M-parameter
  backbone, no architectural surgery needed since it's inserted between two
  existing modules.

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

Both explainability modules originally only operated on the vision-only
baseline model (they need a single-image forward pass; the fusion model
takes image + tabular). Closed via `src/explain/fusion_wrapper.py` —
`FusionModelImageWrapper` fixes one patient's tabular vector as a buffer
and exposes a single-input `forward(image)`, so `gradcam.py` and
`counterfactual.py` both work against the fusion model completely
unmodified. `src/explain/generate_fusion_examples.py` mirrors
`generate_examples.py` but produces both Grad-CAM and counterfactual output
per test image, using that image's real tabular vector (not a random one) —
7 tests in `tests/test_fusion_wrapper.py` cover shape correctness, that the
wrapper doesn't change the model's actual output, and that gradients
genuinely flow through it (not just a shape-compatible no-op).

`src/explain/measure_lung_localization.py` also now works on both model
types — it auto-detects which checkpoint it's given (fusion checkpoints
store a `tabular_features` key that vision-only ones don't, see
`is_fusion_checkpoint()`) rather than requiring a manual flag, and for
fusion builds a fresh `FusionModelImageWrapper` per test image so each
image is explained using that specific patient's real tabular vector, not
one shared/stale vector reused across the whole sample. Covered by a
subprocess-based integration test in `tests/test_measure_lung_localization.py`
that builds a real (tiny, untrained) fusion checkpoint from scratch and
runs the actual script end to end, not just its internal pieces in
isolation.

## Dashboard design

`dashboard/app.py` uses a dark reading-room palette instead of the default
Streamlit light theme — closer to how an actual PACS/DICOM viewer looks than
a generic SaaS dashboard, since that's the vernacular this tool actually
belongs to.

- **Palette:** near-black background (`#0B0D0F`), amber accent (`#F0A83C`)
  for interactive/in-range elements, clinical red (`#E4483C`) reserved
  *only* for an actual positive prediction — not used decoratively anywhere
  else, so its appearance always means something.
- **Type:** IBM Plex Mono for anything technical (probabilities, the study
  header bar, image captions), Inter for prose/explanations. The split is
  functional, not decorative — mono marks measured data, sans marks text
  written to be read.
- **Study header:** a technical readout bar (model name, CBAM on/off,
  checkpoint status, class count) in place of a generic app title, styled
  like a PACS viewer's study metadata strip.
- **Viewport framing:** every image the app shows (uploaded X-ray, Grad-CAM
  overlay, counterfactual comparison) renders through `render_viewport()`,
  which wraps it in a corner-bracket frame — like a viewfinder or DICOM
  viewport — instead of a plain `st.image()`. This is the one deliberately
  "designed" element; everything else stays quiet around it.

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
