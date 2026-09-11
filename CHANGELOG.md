# Changelog

All notable changes to this project are documented here, grouped by the
build phase they correspond to.

## [Unreleased]
- Deploy live demo to Hugging Face Spaces

## v0.10 — Fusion explainability, attention-consistency training, weight sweep
- `src/explain/fusion_wrapper.py`: `FusionModelImageWrapper` adapts the
  dual-input fusion model to the single-input interface Grad-CAM and the
  counterfactual explainer expect, so both work on the fusion model
  completely unmodified — closes the gap `docs/architecture.md` had
  flagged since v0.6. `src/explain/generate_fusion_examples.py` and an
  extended `measure_lung_localization.py` (auto-detects checkpoint type)
  followed the same pattern.
- Fusion CBAM result (3 seeds): a genuinely different outcome from vision
  — localization effect vanishes (mean diff ≈ 0.0003, p=0.995) while
  accuracy trends mildly *positive* (opposite direction from vision).
  CBAM's effect is architecture-dependent, not a property of the module
  alone — see [Results](README.md#results).
- `src/models/attention_consistency_loss.py` + `src/train_attention_consistency.py`:
  goes beyond CBAM by training the model's own spatial attention toward
  the segmented lung field directly (loss = `1 - lung_energy_fraction`),
  instead of only measuring localization after training. Required
  precomputing lung masks (`data/scripts/precompute_lung_masks.py`) and a
  dataset wrapper (`src/data/lung_mask_dataset.py`) kept fully separate
  from `ChestXrayDataset` to avoid touching 50+ existing tests.
- Attention-consistency result (3 seeds, weight=0.1): the most consistent
  effect in the project — all 3 seeds improved localization (p=0.051),
  with a real but seed-variable accuracy cost (p=0.164, not significant
  at n=3 alone).
- Weight-sensitivity sweep (3 seeds × 3 weights: 0.05/0.1/0.2): a clean,
  monotonic dose-response curve between localization gain and accuracy
  cost, with diminishing localization returns and growing instability at
  higher weights. Confirmed an earlier single unreplicated run at
  weight=0.03 was a genuine outlier, not a real data point.
- 23 new tests total across `test_vision_model.py`,
  `test_attention_consistency_loss.py`, `test_lung_mask_dataset.py`,
  `test_fusion_wrapper.py`, and the extended
  `test_measure_lung_localization.py`

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
- Retrained and benchmarked across 3 seeds (42, 123, 2024) per condition —
  caught and corrected a pseudo-replication mistake in an earlier single-seed
  analysis (image-level p=0.0013 was invalid; images from one model aren't
  independent replicates). Properly replicated result: a non-significant
  trend toward better localization (+0.060 ± 0.076, p=0.31, n=3 seeds) and
  a non-significant trend toward slightly worse accuracy (−0.012 ± 0.014,
  p=0.25) — see the [Results](README.md#results) section

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
