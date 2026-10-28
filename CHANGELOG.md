# Changelog

All notable changes to this project are documented here, grouped by the
build phase they correspond to.

## [Unreleased]
- Deploy live demo to Hugging Face Spaces

## v0.14 — Patient-disjoint, deduplicated evaluation and Grad-CAM diagnostics (Oct 24–27, 2026)
- New patient-disjoint split with exact-duplicate removal (`data/scripts/prepare_patient_disjoint_*`,
  `verify_patient_disjoint_dedup*.py`, `inspect_duplicate_images.py`): 5,856 → 5,824 unique image
  contents; 3,727 train / 932 val / 1,165 test. SHA-256 and known pneumonia patient-ID checks found
  no cross-split overlap; Normal-patient identity cannot be verified, so full patient-level
  separation is not claimed.
- Synthetic-vitals signal sweep (rho = 0, 0.25, 0.5, 0.75, 1.0) on the new split, with generated
  vision/fusion configs (`make_patient_disjoint_dedup_rho_configs.py`) and paired comparisons
  (DeLong, paired bootstrap, Holm–Bonferroni adjustment) — see
  `docs/patient_disjoint_dedup_rho_experiment_report_v1.md`. Vision-only AUROC is constant by
  design; fusion AUROC rises with rho. This reflects injected synthetic signal, not clinical benefit.
- Grad-CAM diagnostics on all 1,165 test images: 268 zero-valued maps (266/316 Normal, 2/849
  Pneumonia), traced to raw maps that are entirely negative before the standard ReLU
  (`src/explain/inspect_*gradcam*.py`, `analyze_gradcam_*`).
- Grad-CAM patch-deletion faithfulness evaluation (`src/explain/evaluate_gradcam_faithfulness.py`)
  against a random-patch baseline. A first run used the wrong default checkpoint and was replaced;
  the corrected run (dedup checkpoint, 897 nonzero maps) shows Grad-CAM deletion beating random
  deletion at 5–20% but the difference reversing at 40%. Intervention-specific proxy only.
- `docs/session_audit_2026-10-27.md` and preserved GPU latency utilities
  (`benchmark_latency.py`, `benchmark_latency_v2.py`). No latency results are claimed.
- CI now triggers on `master` and the CI-scoped ruff check passes.
- README shortened and rewritten around the latest experiment; Streamlit demo link fixed.

## v0.13 — Deterministic-seeding reruns; attention-consistency claim downgraded (Oct 3–4, 2026)
- Re-ran CBAM / no-attention (v2) and attention-consistency under consistent deterministic seeding
  (5 seeds). Attention comparisons: AUROC CBAM vs none p=0.147, SE vs none p=0.70, SE vs CBAM
  p=0.117; no significant localization differences (p ≥ 0.18).
- Attention-consistency: AUROC cost confirmed (−0.0317, p=0.036) but the localization gain is no
  longer significant (+0.063, p=0.196; was p=0.038 under mixed seeding). README claim downgraded
  accordingly.

## v0.12 — Reproducibility, SE attention, calibration and published-protocol comparison (Sep 20–29, 2026)
- `src/seeding.py`: deterministic seeding (Python/NumPy/torch, cuDNN deterministic, DataLoader
  workers). A same-seed rerun is now bit-for-bit identical; the earlier divergence was traced to
  unseeded randomness.
- SE attention (`src/models/attention.py`, `use_se` in `vision_encoder.py` / `train.py`) and a
  5-seed CBAM / SE / no-attention comparison.
- Calibration thread: baseline overconfidence (ECE ≈ 0.135), temperature scaling (little help),
  isotonic regression (large AUROC cost), and label smoothing (ECE 0.136 → 0.103 on vision; fusion
  0.135 → 0.112 with no detectable ranking cost). Smoothing sweep (0.05 / 0.1 / 0.2) and paired
  bootstrap comparisons added (`src/fit_temperature.py`, `src/fit_isotonic.py`,
  `src/evaluate_calibration.py`, `src/compare_auroc_paired.py`).
- Comparison with a published split protocol (`docs/potharaju_comparison.md`): on a patient-blind
  split our CBAM model's accuracy rose 86.38% → 96.88% and AUROC 0.960 → 0.997, with 63% of test
  images sharing a patient with training data — suggesting split methodology largely explains
  the gap to published numbers.
- Configurable augmentation and full-metrics evaluation pipeline (`src/evaluate_full_metrics.py`).
- Fixed a test fixture missing the `seed` key after the seeding change.
- CBAM and attention-consistency extended to 5 seeds; effects weakened or changed with more seeds
  (see v0.13 for the final seeded results).

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
