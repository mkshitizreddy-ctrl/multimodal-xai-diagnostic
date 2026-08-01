# Changelog

All notable changes to this project are documented here, grouped by the
build phase they correspond to.

## [Unreleased]
- Deploy live demo to Hugging Face Spaces

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
