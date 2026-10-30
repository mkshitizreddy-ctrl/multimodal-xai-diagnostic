# Explainable Multimodal Pneumonia Classification

A research project exploring pediatric pneumonia classification from chest X-rays, multimodal fusion, and explainability. It uses DenseNet-121, optional CBAM/SE attention, Grad-CAM, lung-localization analysis, and occlusion-based counterfactuals.

> **Research prototype only.** This project is not clinically validated and must not be used for medical decisions. Tabular clinical features in the fusion experiments are synthetic, not real patient measurements.

[![Tests](https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic/actions/workflows/tests.yml/badge.svg)](https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic/actions)

**Python 3.11+ · MIT License · [Live demo](https://multimodal-xai-diagnostic-yhqvbbhkejld2b6jodcvh2.streamlit.app/)**

The live demo serves the vision-only model from the latest patient-disjoint, deduplicated experiment (no tabular or fusion input); its header shows whether the loaded checkpoint matches the expected SHA-256. When Grad-CAM has no positive evidence for an image (common for images scored as Normal), the demo says so instead of showing a heatmap or counterfactual.

---

## What this project investigates

- How well DenseNet-121 separates Normal and Pneumonia chest X-rays.
- How synthetic tabular features affect multimodal model performance.
- How attention mechanisms and attention-consistency training affect classification and localization.
- What Grad-CAM, lung-localization scores, and occlusion counterfactuals reveal—and what they cannot establish.
- How dataset split methodology affects reported performance.

## Latest experiment: patient-disjoint, deduplicated evaluation

The latest experiment uses **5,824 unique image contents**, after removing 32 duplicate records from the original 5,856-image dataset. The final split contains 3,727 training, 932 validation, and 1,165 test images. SHA-256 checks found no duplicate image contents across the finalized splits.

### Test AUROC

| Synthetic feature signal (ρ) | Vision-only AUROC | Fusion AUROC | Difference |
|---:|---:|---:|---:|
| 0.00 | 0.997782 | 0.998174 | +0.000391 |
| 0.25 | 0.997782 | 0.999191 | +0.001409 |
| 0.50 | 0.997782 | 0.999825 | +0.002043 |
| 0.75 | 0.997782 | 0.999959 | +0.002177 |
| 1.00 | 0.997782 | 0.999981 | +0.002199 |

Paired DeLong comparisons found no statistically significant difference at ρ = 0 (unadjusted p = 0.455; Holm–Bonferroni-adjusted p = 0.455). The comparisons at ρ = 0.25–1.00 remained significant after Holm–Bonferroni adjustment across the five comparisons (adjusted p ≈ 0.0085, 0.0050, 0.0050, 0.0050 for ρ = 0.25, 0.50, 0.75, 1.00). All five comparisons share the same test set and are not independent replications; the paired-bootstrap confidence intervals in the report are unadjusted.

**Interpretation:** vision-only AUROC stayed constant while fusion AUROC increased as the synthetic signal became more strongly associated with the label. This demonstrates model behavior under controlled synthetic inputs; it does **not** show that real clinical measurements improve diagnosis.

Full methodology, confidence intervals, statistical tests, and artifacts: [`docs/patient_disjoint_dedup_rho_experiment_report_v1.md`](docs/patient_disjoint_dedup_rho_experiment_report_v1.md).

## Explainability findings

- **Grad-CAM:** evaluated on all 1,165 test images. There were 268 zero-valued heatmaps (23.0%): 266/316 Normal images and 2/849 Pneumonia images. In a six-case investigation, zero maps arose because the weighted raw Grad-CAM maps were entirely negative before the standard ReLU step.
- **Lung localization:** among 897 nonzero maps, the mean fraction of positive Grad-CAM activation inside segmented lung masks was 0.499 (SD 0.152). This is a localization proxy, not a measure of whether the model focuses on pathology.
- **Counterfactual occlusion:** no prediction flips occurred in a six-image sample. Probability changes reflect sensitivity to the inpainting intervention, not causal proof. This small sample should not be generalized to the full test set.
- **Visualization caveat:** restricting a heatmap to a lung mask changes the displayed explanation; it does not establish that the classifier relied only on lung tissue.

The detailed report links to per-image CSVs and generated artifacts.

## Model and workflow

```text
Chest X-ray ──> DenseNet-121 (+ optional CBAM/SE) ──> image representation ──┐
                                                                            ├─> prediction
Synthetic tabular features ──> tabular encoder ─────────────────────────────┘

Analysis: Grad-CAM · lung localization · occlusion counterfactuals · calibration
```

## Repository layout

```text
data/       Dataset preparation scripts and split CSVs
src/        Models, training, evaluation, and explainability code
configs/    YAML experiment configurations
tests/      Automated tests
dashboard/  Streamlit demo
docs/       Experiment reports, metrics, and explainability artifacts
```

Model checkpoints and training logs are not committed to Git.

## Setup

```bash
git clone https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic.git
cd multimodal-xai-diagnostic

conda create -n ai_env python=3.11
conda activate ai_env
pip install -r requirements.txt
```

Dataset preparation scripts expect the source chest X-ray images to be available in the configured data location. Review `configs/data.yaml` and the relevant preparation script before running them. Note that `configs/data.yaml` supplies the label classes and tabular feature names, but the split CSV paths come from each experiment's train config. The latest patient-disjoint, deduplicated experiment uses `configs/*_patient_disjoint_dedup_rho_v1_*.yaml`, which point to their own split CSVs under `data/processed_patient_disjoint_dedup_rho_v1_*/`.

## Common commands

Train the vision baseline:

```bash
python src/train.py --data-config configs/data.yaml --train-config configs/vision_baseline.yaml
```

Train the fusion model:

```bash
python src/train_fusion.py --data-config configs/data.yaml --train-config configs/fusion.yaml
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Run the test suite:

```bash
python -m pytest -q
```

The most recently recorded full test run passed **84 tests**, with six third-party deprecation warnings. Run the tests locally to verify your environment. For multi-seed experiments, use separate `--checkpoint-dir` and `--log-dir` values to avoid overwriting outputs. Check the relevant YAML configuration and each script's `--help` before running other experiment variants.

## Reproducing the latest experiment

> **Status: derived from the scripts, not yet end-to-end tested.** The steps below follow the order and paths hard-coded in the repository's scripts. Run them in order and check each script's `--help` where it has one. Generated data (`data/processed*/`), checkpoints, and logs are not committed to Git.

1. **Base data.** Place the source images in `data/raw/images` and run `data/scripts/prepare_pneumonia_dataset.py` to create `data/processed/`.
2. **Patient-disjoint, deduplicated split** (reads `data/processed`, writes `data/processed_patient_disjoint_dedup_v1`):
   ```bash
   python data/scripts/prepare_patient_disjoint_dedup_dataset.py
   python data/scripts/verify_patient_disjoint_dedup.py
   ```
3. **Synthetic-signal variants** for ρ = 0, 0.25, 0.5, 0.75, 1.0 (writes `data/processed_patient_disjoint_dedup_rho_v1_<ρ>`):
   ```bash
   python data/scripts/make_patient_disjoint_dedup_rho_variants.py
   python data/scripts/verify_patient_disjoint_dedup_rho_variants.py
   ```
4. **Configs.** The per-ρ YAML files `configs/{vision,fusion}_patient_disjoint_dedup_rho_v1_<ρ>.yaml` are committed; `data/scripts/make_patient_disjoint_dedup_rho_configs.py` regenerates them. Each config sets its own split CSVs, checkpoint directory, and log directory.
5. **Train** (example: ρ = 0; repeat for each ρ). `configs/data.yaml` still supplies the class and feature names:
   ```bash
   python src/train.py --data-config configs/data.yaml --train-config configs/vision_patient_disjoint_dedup_rho_v1_0.yaml
   python src/train_fusion.py --data-config configs/data.yaml --train-config configs/fusion_patient_disjoint_dedup_rho_v1_0.yaml
   ```
   Vision-only training does not use the synthetic features, so the vision-only AUROC is the same across ρ in the reported table.
6. **Compare AUROCs** with `src/verify_auroc_comparison.py` (takes `--checkpoint-a`, `--checkpoint-b`, `--train-config`, `--output-csv`, and defaults to the older `rho_v2` config, so pass the dedup config explicitly).
7. **Explainability.** `src/explain/analyze_gradcam_testset.py` (zero-map diagnostic) and `src/explain/measure_lung_localization.py` take a checkpoint. `src/explain/evaluate_gradcam_faithfulness.py` **requires** `--checkpoint`; use `checkpoints/vision_patient_disjoint_dedup_rho_v1_0/best_model.pth` so it matches the split.

## Other research

The repository also contains experiments on CBAM versus SE versus no attention, attention-consistency training, probability calibration, and a comparison with a published paper's described split protocol. These experiments use different configurations and, in some cases, different split/seeding regimes. Their findings should not be treated as directly interchangeable with the latest deduplicated, patient-disjoint rho experiment.

See [`docs/potharaju_comparison.md`](docs/potharaju_comparison.md) for the published-protocol comparison and [`docs/architecture.md`](docs/architecture.md) for architecture details.

## Limitations

- Tabular features (including age, gender, temperature, and SpO₂) are synthetic and intentionally correlated with the target at different signal strengths.
- The dataset is a single pediatric cohort. External validation on independent hospitals and populations has not been performed.
- Patient IDs are available for pneumonia images but not reliably identifiable for Normal images; patient-level separation cannot be fully verified for the Normal class.
- AUROC values in the latest experiment are near ceiling, so small numerical differences require caution. DeLong p-values were Holm–Bonferroni-adjusted across the five ρ comparisons, but the bootstrap confidence intervals are unadjusted and all comparisons reuse one test set.
- Grad-CAM and lung-energy fractions are explanatory proxies. Zero heatmaps are a known limitation, and inpainting counterfactuals do not establish causality.
- This is a research prototype, not a medical device or clinical decision-support tool.

## Ethics and citation

See [`docs/ethics_statement.md`](docs/ethics_statement.md) for data and ethics notes. References are listed in [`docs/references.bib`](docs/references.bib).

**M. Kshitiz Reddy** · M.Tech (AI), Bennett University
