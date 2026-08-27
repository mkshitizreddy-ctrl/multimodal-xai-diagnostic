# Changelog

All notable changes to this project are documented here, grouped by the
build phase they correspond to.

## [Unreleased]
- Deploy live demo to Hugging Face Spaces

## v0.9 — CBAM attention module
- `src/models/attention.py`: CBAM (channel + spatial attention, Woo et al.
  ECCV 2018), gated behind `use_cbam` config flag on both the vision-only
  and fusion models — added after reading the pneumonia-CXR literature
  (see `docs/architecture.md#attention-module` for the reading list)
- `src/explain/measure_lung_localization.py`: quantifies what fraction of
  a Grad-CAM heatmap's energy falls inside the segmented lung field —
  turns the manual shortcut-learning audit into a measurable number, used
  to check whether CBAM actually improves localization and not just accuracy
- Checkpoints now record `use_cbam` so evaluate/dashboard/explain scripts
  rebuild the right architecture automatically before loading weights
- 10 new tests (`test_attention.py`, `test_measure_lung_localization.py`,
  plus CBAM cases added to `test_vision_model.py`)

## v0.8 — Dataset pivot to Chest X-ray Pneumonia
- Switched primary dataset from NIH Chest X-ray14 (~45GB, 14-class) to Kaggle
  Chest X-ray Pneumonia (~2GB, binary) to fit local disk/compute constraints
- `data/scripts/prepare_pneumonia_dataset.py`: downloads images, builds
  patient-grouped train/val splits (test split preserved from source),
  generates synthetic clinical vitals (age, gender, temperature, SpO2)
  correlated with the Pneumonia label to keep the fusion architecture
  genuinely meaningful — clearly disclosed as synthetic throughout the docs
- `configs/data.yaml` updated for the new dataset; original NIH config
  preserved at `configs/data_nih_legacy.yaml` for reference
- No changes required to `src/data/dataset.py`, model code, training loops,
  or explainability modules — validates the original generic design
- Updated ethics statement, architecture doc, and README for the pivot

## v0.7 — Dashboard
- Streamlit dashboard: upload → probabilities → Grad-CAM → counterfactual
- Demo mode fallback when no trained checkpoint is present

## v0.6 — Tabular fusion
- MLP tabular encoder (age, gender, view position)
- Late-fusion multimodal model
- Fusion training/evaluation scripts and ablation notebook

## v0.5 — Counterfactual explainability
- Occlusion-based counterfactual explainer (Grad-CAM → mask → inpaint → re-predict)
- Example figure generation script

## v0.4 — Grad-CAM explainability
- Grad-CAM wrapper around the vision baseline model
- Example overlay generation script

## v0.3 — Vision baseline
- DenseNet-121 multi-label classifier
- Training loop with checkpointing, early stopping, per-epoch metrics logging
- Test-set evaluation with per-class AUROC

## v0.2 — Data pipeline
- NIH Chest X-ray14 download script (Kaggle mirror)
- Preprocessing: patient-level train/val/test splits, multi-hot label encoding
- PyTorch Dataset combining image + tabular features

## v0.1 — Project scaffold
- Repo structure, license, dependencies, initial README
