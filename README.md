# Explainable Multimodal Diagnostic Support System

An explainable multimodal deep learning pipeline for pediatric pneumonia detection from chest X-rays and clinical metadata. Built as part of my M.Tech (AI) work at Bennett University.

The core idea: it's not enough for a model to say "pneumonia". I wanted to know *where* it's looking, whether that changes with how the model is trained, and whether its confidence scores can be trusted. So this project combines a DenseNet-121 classifier with Grad-CAM, CBAM and SE attention, occlusion-based counterfactuals, lung-localization scoring, an attention-consistency loss that pushes the model to look inside the lungs, a calibration analysis with three attempted fixes, and a controlled comparison against a published paper that reports much higher accuracy on the same dataset.

[![Tests](https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic/actions/workflows/tests.yml/badge.svg)](https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic/actions)
Python 3.11+ · MIT License

[Live demo](https://multimodal-xai-diagnostic-yhqvbbhkejld2b6jodcvh2.streamlit.app)

![Dashboard demo](docs/screenshots/dashboard_demo.png)

---

## Summary of findings

- **Attention-consistency training works, at a cost.** Over 5 seeds it improves lung localization (+0.083, p = 0.038) and lowers AUROC (-0.064, p = 0.012). Both effects are significant.
- **CBAM alone does not reliably help.** An encouraging 3-seed result disappeared at 5 seeds (AUROC p = 0.67, localization p = 0.57). SE (the other attention block I tested) shows a mild improvement that is not statistically confirmed.
- **The models are overconfident** (ECE about 0.135, concentrated in one confidence bin). Of three fixes, only training-time label smoothing helped, on both the vision and fusion models, without a statistically confirmed AUROC cost.
- **A published paper's 98.6% accuracy is explained mostly by its split.** A patient-blind split on the same dataset leaks 63% of test images, and this project's unmodified model reaches 97.2% mean accuracy on it. On the honest patient-level split the same kind of model gets about 86%.
- **A real reproducibility bug was found and fixed:** `torch.manual_seed()` alone did not make training deterministic here. See "A note on reproducibility".

## Why this project

Medical imaging models can hit strong accuracy numbers while giving almost no insight into *why* they made a call, and a model that is right for the wrong reasons is still a liability. So alongside the classifier I built explainability tools and used them to interrogate the model instead of trusting heatmaps at face value.

**Note on the data:** the chest X-ray dataset has no real patient vitals, so the tabular features used for multimodal fusion (age, gender, temperature, SpO₂) are synthetic, generated with clinically plausible correlations to the label. They test the *architecture*, not any real diagnostic benefit. See `docs/ethics_statement.md`.

## Research questions

- Can DenseNet-121 classify pneumonia from chest X-rays accurately?
- Does adding tabular data help over image-only?
- Does Grad-CAM land on clinically relevant regions, or is the model cheating?
- Do CBAM or SE attention improve classification or localization, and does that hold up across seeds?
- Can I explicitly train attention to stay inside the lungs, and what does that cost?
- Are the model's confidence scores trustworthy, and can a standard fix repair them?
- Does a published paper's much higher accuracy on the same dataset hold up when the split methodology is controlled for?

## What's in here

**Vision model:** DenseNet-121, patient-level train/val splitting, binary classification, evaluated on accuracy, precision, recall, F1, AUROC and AUPR.

**Multimodal fusion:** image representation + tabular encoder + fusion head, to compare image-only against fused.

**Explainability:** Grad-CAM, occlusion-based counterfactuals, lung-field localization scoring, attention-map analysis.

**Attention modelling:** CBAM (channel + spatial) and SE (channel only), an attention-consistency loss trained against precomputed lung masks, all compared across multiple seeds.

**Calibration:** ECE and reliability diagrams, three attempted fixes (temperature scaling, isotonic regression, label smoothing), paired-bootstrap significance checks.

**External comparison:** a two-sided comparison against Potharaju et al. 2025 (their split with my model, their SE idea with my split).

**Engineering:** configurable training with CLI overrides, deterministic seeding, automated evaluation scripts, GitHub Actions CI.

## Architecture

```text
                    Chest X-ray Image
                            │
                            ▼
                     DenseNet-121
              (optional CBAM or SE attention)
                            │
                            ▼
                  Image Representation ──────┐
                                              │
              Synthetic Clinical Features     │
                            │                 │
                            ▼                 │
                    Tabular Encoder           │
                            │                 │
                            └────► Fusion Network
                                              │
                                              ▼
                                  Pneumonia Prediction

              Explainability layer:
              Grad-CAM · CBAM spatial attention ·
              Lung localization · Counterfactuals ·
              Attention-consistency analysis · Calibration
```

Full diagram and writeup: `docs/architecture.md`

## Dataset

Chest X-ray Pneumonia dataset (pediatric, ~1–5 years, Guangzhou Women and Children's Medical Center): 5,856 images, Normal vs. Pneumonia. I originally planned to use NIH ChestX-ray14 but switched given local compute and storage limits.

Patient-level splitting where possible (pneumonia filenames carry patient IDs; normal-class images don't, so I conservatively treated each as a unique patient):

- Train: ~4,434
- Val: ~798
- Test: ~624 (the original dataset test split, kept as-is)

This split matters more than it looks. See "Comparison against a published paper" for what happens when the split is not patient-level.

## Stack

Python 3.11+, PyTorch, Torchvision, DenseNet-121, CBAM, SE, scikit-learn, Pandas, NumPy, Grad-CAM, Streamlit, OpenCV/PIL, pytest

## Repo layout

```text
multimodal-xai-diagnostic/
├── data/
│   ├── scripts/              # dataset prep, lung-mask precomputation, Potharaju-style split
│   └── processed_potharaju_split/   # CSVs for the patient-blind comparison split
├── src/
│   ├── data/                 # dataset classes
│   ├── models/               # vision encoder, fusion, CBAM + SE (attention.py), attention-consistency loss
│   ├── explain/              # Grad-CAM, counterfactuals, lung segmentation/localization
│   ├── seeding.py            # deterministic seeding for all training scripts
│   ├── train.py, train_fusion.py, train_attention_consistency.py
│   ├── evaluate*.py, evaluate_calibration.py, fit_temperature.py, fit_isotonic.py,
│   │   compare_auroc_paired.py
├── dashboard/app.py          # Streamlit demo
├── notebooks/                # baseline + fusion ablation results
├── configs/                  # one YAML per experiment
├── tests/
├── docs/                     # architecture, ethics, per-run result CSVs, potharaju_comparison.md, paper drafts
└── CHANGELOG.md
```

Model checkpoints (`checkpoints/`) and training logs (`logs/`) are gitignored.

## Results

Test set n = 624, threshold 0.5 (never tuned on test). Every metric has a 95% bootstrap CI in the CSVs under `docs/`.

| Model / Experiment          | Accuracy | Precision | Recall |   F1   | AUROC  |  AUPR  |
| --------------------------- | -------: | --------: | -----: | -----: | -----: | -----: |
| Vision + CBAM (baseline)    |  86.38%  |   0.8224  | 0.9974 | 0.9015 | 0.9604 | 0.9628 |
| Vision + rotation/zoom      |  73.88%  |   0.7052  | 1.0000 | 0.8271 | 0.9651 | 0.9730 |
| Vision + label smoothing 0.05 | 87.98% |   0.8402  | 0.9974 | 0.9121 | 0.9574 | 0.9635 |
| Vision + label smoothing 0.1  | 86.06% |   0.8203  | 0.9949 | 0.8992 | 0.9459 | 0.9502 |
| Vision + label smoothing 0.2  | 91.03% |   0.8795  | 0.9923 | 0.9325 | 0.9652 | 0.9665 |
| Multimodal fusion           |  87.02%  |   0.8280  | 1.0000 | 0.9059 | 0.9899 | 0.9921 |
| Fusion + label smoothing 0.1 |  86.54% |   0.8255  | 0.9949 | 0.9023 | 0.9927 | 0.9959 |

These are single-seed (42) numbers. Baseline 95% CIs: vision AUROC [0.9412, 0.9757], fusion AUROC [0.9810, 0.9961]. The vision and fusion accuracy and F1 intervals overlap, so only the AUROC/AUPR gap is clearly beyond noise, and the tabular features are synthetic.

Rotation/zoom augmentation *raises* AUROC/AUPR but sharply lowers accuracy and F1, which is why I don't call something an improvement because one number went up. Recall is at or near 1.0 for almost every model at the 0.5 threshold, so precision and accuracy are what vary.

## Explainability findings

**Grad-CAM** sometimes landed on the lungs and sometimes didn't: a few cases lit up shoulders, image borders, or burned-in annotations. That motivated the localization work. A good accuracy number doesn't mean the model is looking at the right thing.

**Lung-restricted explanations:** using a pretrained segmentation model, Grad-CAM can be constrained to the lung field for visualization (`--restrict-to-lungs`). That constrains what you *see*, not necessarily what the classifier learned from.

**Counterfactuals:** occlude the highest-activation region and re-run the model. Across 6 borderline predictions, 1 flipped outright and 4 of the remaining 5 showed a real confidence drop. Not causal proof, but supporting evidence that the highlighted regions matter.

**Localization metric:** the lung-energy fraction is the share of Grad-CAM activation inside the lung mask, measured on a fixed random sample of 30 test images. Images with an all-zero heatmap are dropped, so each score is a mean over roughly 20–26 images.

## Attention mechanisms: CBAM vs. SE vs. none (5 seeds)

Seeds 42, 123, 2024, 7, 2025, vision model only.

Test AUROC:

| Seed | No attention | CBAM | SE |
|---|---:|---:|---:|
| 42 | 0.9592 | 0.9608 | 0.9663 |
| 123 | 0.9695 | 0.9445 | 0.9508 |
| 2024 | 0.9736 | 0.9604 | 0.9744 |
| 7 | 0.9280 | 0.9508 | 0.9657 |
| 2025 | 0.9673 | 0.9628 | 0.9588 |
| Mean ± SD | 0.9595 ± 0.0184 | 0.9559 ± 0.0079 | 0.9632 ± 0.0089 |

Lung-energy localization:

| Seed | No attention | CBAM | SE |
|---|---:|---:|---:|
| 42 | 0.4155 | 0.5110 | 0.406 |
| 123 | 0.4217 | 0.5717 | 0.505 |
| 2024 | 0.4624 | 0.4703 | 0.452 |
| 7 | 0.4720 | 0.3400 | 0.505 |
| 2025 | 0.4240 | 0.4480 | 0.526 |
| Mean ± SD | 0.439 ± 0.026 | 0.468 ± 0.086 | 0.479 ± 0.049 |

Paired t-tests (n = 5):

| Comparison | AUROC diff | p | Localization diff | p |
|---|---:|---:|---:|---:|
| CBAM vs. none | -0.0037 | 0.67 | +0.029 | 0.57 |
| SE vs. none | +0.0037 | 0.72 | +0.040 | 0.16 |
| SE vs. CBAM | +0.0073 | 0.099 | +0.011 | 0.84 |

**CBAM's story changed with more seeds.** At 3 seeds (42, 123, 2024) the AUROC difference was -0.0122 (p = 0.31) and the localization difference +0.060 (p = 0.31), which looked like a consistent positive trend for localization. Adding seeds 7 and 2025 shrank both, and seed 7 is the only seed where CBAM localized *worse* than no attention. An even earlier single-run result (p = 0.0013) turned out to be **pseudo-replication**: I had treated individual images as independent replicates when the real unit of replication is the training run. I reran it across seeds instead of keeping the flattering number.

**SE** has the best mean of the three on both metrics, but no difference is significant. I read it as a mild, real-looking, unconfirmed signal, not a win.

On the **fusion model** (3 seeds), CBAM's localization effect was essentially zero (+0.0003 ± 0.085, p = 0.995), so whatever CBAM does is architecture-dependent.

## Attention-consistency training

Instead of hoping CBAM's attention lands on the lungs, I added a loss term (`1 − lung_energy_fraction`) that pushes its spatial attention toward precomputed lung masks (weight 0.1 unless stated).

Five seeds, compared against the CBAM-only run at the same seed:

| Seed | CBAM AUROC | + attn.-consistency | Diff | CBAM loc. | + attn.-consistency | Diff |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.9608 | 0.9233 | -0.0375 | 0.5110 | 0.6253 | +0.1143 |
| 123 | 0.9445 | 0.9061 | -0.0384 | 0.5717 | 0.608 | +0.0363 |
| 2024 | 0.9604 | 0.8546 | -0.1058 | 0.4703 | 0.642 | +0.1717 |
| 7 | 0.9508 | 0.9045 | -0.0463 | 0.3400 | 0.361 | +0.0210 |
| 2025 | 0.9628 | 0.8713 | -0.0915 | 0.4480 | 0.521 | +0.0730 |
| Mean | 0.9559 | 0.8920 | **-0.0639 ± 0.0323** | 0.468 | 0.551 | **+0.0833 ± 0.0612** |

Paired t-tests: AUROC **p = 0.012**, localization **p = 0.038**. Every seed moves in the same direction on both metrics. This is the strongest, most consistently reproduced result in the project: a real, quantified trade-off between where the model looks and how well it ranks. It got *stronger* with more seeds, the opposite of what happened to CBAM alone.

A weight sweep (original 3-seed runs) shows the shape of the trade-off:

| Weight | Test AUROC | Localization |
|---:|---:|---:|
| 0.0 (CBAM only) | 0.9552 ± 0.0093 | 0.462 ± 0.062 |
| 0.05 | 0.9356 ± 0.0119 | 0.550 ± 0.022 |
| 0.10 | 0.9292 ± 0.0124 | 0.593 ± 0.048 |
| 0.20 | 0.9100 ± 0.0380 | 0.611 ± 0.028 |

Better localization, lower AUROC, diminishing returns as the weight rises. The sweep and the 5-seed table use different sets of runs (seeds 123 and 2024 were retrained after their original checkpoints were overwritten), so their weight-0.1 numbers differ and shouldn't be mixed. One earlier run at weight 0.03 (AUROC 0.8933 with a validation AUROC of exactly 1.0) didn't fit the trend and was excluded as an unreplicated outlier.

## Calibration

Accuracy and AUROC say nothing about whether confidence scores are trustworthy, so I measured Expected Calibration Error (ECE, 10 equal-width bins) and Brier score (`src/evaluate_calibration.py`).

| Model | ECE | Brier |
|---|---:|---:|
| Vision baseline | 0.136 (95% CI [0.113, 0.162]) | 0.116 |
| Fusion baseline | 0.135 (95% CI [0.110, 0.160]) | 0.113 |

Both models are overconfident, and it is concentrated in one place: **71% of the test set (442 of 624 images) sits in the 0.9–1.0 confidence bin**, where mean stated confidence is 0.995 but the observed pneumonia rate is 0.873. The lower bins show large gaps too, but with 4–10 images each I don't read much into them. Fusion didn't change calibration, so it looks like a property of the vision backbone and training setup. (A first version of my calibration script binned by probability but scored by decision accuracy, which gave a meaningless curve; I caught that from the reliability plot and fixed it before trusting any number.)

I tried three fixes, from least to most invasive. The first two are post-hoc, fit on validation data only.

### Attempt 1: temperature scaling

| Model | T | ECE before → after | Brier before → after |
|---|---:|---|---|
| Vision | 1.05 | 0.136 → 0.137 | 0.116 → 0.116 |
| Fusion | 1.21 | 0.135 → 0.136 | 0.113 → 0.110 |

No meaningful effect. AUROC is preserved as expected (fusion moved 0.9899 → 0.9902 from floating-point ties, not real re-ranking). A single global scalar can't fix miscalibration concentrated in one confidence region.

### Attempt 2: isotonic regression

| Model | ECE | Brier | AUROC |
|---|---|---|---|
| Vision | 0.136 → 0.178 | 0.116 → 0.149 | 0.960 → 0.932 |
| Fusion | 0.135 → 0.113 | 0.113 → 0.102 | 0.990 → 0.949 |

Not usable. On vision it overfit (only 798 validation images) and got worse on every metric. On fusion it improved calibration but cost 0.041 AUROC.

### Attempt 3: label smoothing (the one that worked)

Instead of patching a finished model, label smoothing changes the training targets (hard 0/1 toward ε/2 and 1 − ε/2), so the network is never trained to output near-certain logits. Same architecture, seed and split; only the loss changed.

| Metric | Vision baseline | Vision + LS 0.1 | Fusion baseline | Fusion + LS 0.1 |
|---|---:|---:|---:|---:|
| ECE | 0.136 | **0.103** | 0.135 | **0.112** |
| Brier | 0.116 | 0.108 | 0.113 | 0.101 |
| Accuracy | 86.38% | 86.06% | 87.02% | 86.54% |
| AUROC | 0.9604 | 0.9459 | 0.9899 | 0.9927 |

The AUROC changes were tested with a paired bootstrap on shared test indices (`src/compare_auroc_paired.py`): vision -0.0143 (95% CI [-0.0309, 0.0023], p = 0.087), fusion +0.0028 (95% CI [-0.0032, 0.0104], p = 0.42). Neither is significant. The vision result is borderline, so I'd call it a plausible small cost this test set can't confirm, not "no cost".

**Smoothing-value sweep** (vision, one seed per value):

| ε | ECE | Brier | Accuracy | AUROC |
|---:|---:|---:|---:|---:|
| 0 | 0.136 | 0.116 | 86.38% | 0.9604 |
| 0.05 | 0.101 | 0.097 | 87.98% | 0.9574 |
| 0.10 | 0.103 | 0.108 | 86.06% | 0.9459 |
| 0.20 | 0.106 | 0.078 | 91.03% | 0.9652 |

ECE improves at every value but not monotonically. The 0.2 run's higher accuracy and AUROC look surprising (smoothing is supposed to trade accuracy for calibration), so I tested it: AUROC +0.0047, 95% CI [-0.0089, 0.0190], p = 0.54, consistent with single-seed noise (that run also early-stopped at epoch 5 versus 7–9 for the others). With one seed per value, I can't say which ε is best.

**Net result:** the overconfidence was real and precisely located. Two of three fixes failed for understood reasons, and the one that worked changes training instead of patching outputs.

## A note on reproducibility

While rerunning seeds 123 and 2024 for the attention-consistency analysis, a same-seed rerun of seed 2024 gave a very different result (localization 0.642 vs. 0.537, a gap larger than the effect I was measuring). `torch.manual_seed()` alone was not making training deterministic here. Two causes:

- Training augmentation (`RandomHorizontalFlip`, `RandomRotation`) runs in DataLoader worker processes and draws from Python's `random` module, which nothing seeded.
- cuDNN's default convolution algorithms are non-deterministic on GPU.

`src/seeding.py` seeds Python's `random`, NumPy and torch, pins cuDNN to deterministic mode, and reseeds each worker. I verified it: two same-seed runs afterwards matched exactly in every epoch. All three training scripts use it.

The fix was applied partway through the project, so runs made before it (the CBAM, no-attention and attention-consistency comparisons) and after it (the SE and Potharaju-split runs) used different seeding. Any extra run-to-run noise in the older runs is not quantified. The training scripts also gained `--seed`, `--checkpoint-dir` and `--log-dir` overrides after I found that seed sweeps had been overwriting each other's checkpoints, which is how the original seed 123 and 2024 attention-consistency checkpoints were lost.

## Comparison against a published paper on the same dataset

Potharaju et al. 2025 ("Enhanced X-ray Image Classification for Pneumonia Detection Using Deep Learning Based CBAM and SE Mechanisms", *Intelligence-Based Medicine*) reports 98.6% accuracy on what its cited Kaggle source shows is the same Kermany dataset. Reasons I didn't take that at face value:

- Its own results table is inconsistent on that row: precision 98.4% and recall 98.3% imply an F1 near 98.3%, but F1 is printed as 94.5%. The abstract also gives SE+CNN accuracy as 96.25% while the conclusion says 96.17%.
- The reported split (5,216 / 160 / 480) sums to the whole dataset and never mentions patient-level grouping, though pneumonia filenames encode a patient ID with several images per patient.
- No code, no confidence intervals, no seeds, 10 training epochs, and the baseline CNN's architecture is never specified.

I ran the two-sided comparison my professor asked for: **my method on their split, and their method on my split.**

**My method, their split.** `data/scripts/prepare_dataset_potharaju_split.py` reproduces their split sizes as a patient-blind, class-stratified split. It reports the leakage directly: 272 pneumonia patients appear in both train/val and test, so **304 of 480 test images (63%) belong to a patient seen in training.** Training my existing DenseNet-121 + CBAM model on it, unchanged:

| Setting | Accuracy | Precision | Recall | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| Potharaju et al., CNN+CBAM (as reported) | 98.6% | 98.4% | 98.3% | 94.5% (inconsistent) | not reported |
| My model, their split, seed 42 | 96.88% | 98.83% | 96.86% | 97.84% | 0.9969 |
| My model, their split, seed 123 | 98.33% | 98.03% | 99.71% | 98.87% | 0.9984 |
| My model, their split, seed 2024 | 96.46% | 98.54% | 96.57% | 97.55% | 0.9967 |
| **My model, their split, 3-seed mean** | **97.22%** | 98.47% | 97.71% | 98.09% | 0.9973 |
| **My model, my honest split** | **86.38%** | 82.24% | 99.74% | 90.15% | 0.9604 |

On their split my model lands in the same range as their reported accuracy (97.2% mean, 98.3% best seed against their 98.6%) but does not exceed it. The point is the size of the jump: same model, same code, and about 11 points of accuracy come from the split alone. I can't prove what their actual pipeline did without their code. What I can say is that their described split would leak this much, and that a leaky split is enough to reach their reported range with no architecture changes.

**Their method, my split.** Their one precisely specified idea is SE. I added a standard SE block (global average pool, two FC layers, sigmoid channel rescale) to my DenseNet-121 backbone as an alternative to CBAM and evaluated it over 5 seeds on my honest split. Results are in the attention table above: best mean of the three settings, no significant differences. Their baseline CNN itself was not reconstructed, because the paper doesn't specify its architecture, and my SE block is a standard formulation, not a copy of their unpublished code.

**Takeaway:** the split explains most of the gap between their number and mine, and the honest, patient-separated 86% is the figure that estimates performance on unseen patients. Full writeup: `docs/potharaju_comparison.md`.

## Reproducing this

```bash
git clone https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic.git
cd multimodal-xai-diagnostic

conda create -n ai_env python=3.11
conda activate ai_env
pip install -r requirements.txt
```

Dataset prep:

```bash
python data/scripts/prepare_pneumonia_dataset.py
# patient-blind comparison split (see the Potharaju section):
python data/scripts/prepare_dataset_potharaju_split.py
```

Training (one config per experiment; pass `--seed`, `--checkpoint-dir`, `--log-dir` for multi-seed runs so nothing is overwritten):

```bash
# vision baseline (CBAM)
python src/train.py --data-config configs/data.yaml --train-config configs/vision_baseline.yaml

# vision + SE
python src/train.py --data-config configs/data.yaml --train-config configs/vision_se.yaml --seed 7 --checkpoint-dir checkpoints/vision_se_seed7 --log-dir logs/vision_se_seed7

# no attention, one seed (override the config's CBAM flag)
python src/train.py --data-config configs/data.yaml --train-config configs/vision_baseline.yaml --seed 7 --use-cbam false --checkpoint-dir checkpoints/vision_nocbam_seed7 --log-dir logs/vision_nocbam_seed7

# label smoothing (0.05 / 0.1 / 0.2)
python src/train.py --data-config configs/data.yaml --train-config configs/vision_label_smoothing.yaml

# fusion, and fusion + label smoothing
python src/train_fusion.py --data-config configs/data.yaml --train-config configs/fusion.yaml
python src/train_fusion.py --data-config configs/data.yaml --train-config configs/fusion_label_smoothing.yaml

# attention-consistency (precompute lung masks first)
python data/scripts/precompute_lung_masks.py --train-config configs/vision_attention_consistency.yaml
python src/train_attention_consistency.py --train-config configs/vision_attention_consistency.yaml --seed 7 --checkpoint-dir checkpoints/vision_attention_consistency_seed7 --log-dir logs/vision_attention_consistency_seed7

# my model on the Potharaju-style split
python src/train.py --data-config configs/data.yaml --train-config configs/vision_baseline_potharaju_split.yaml --seed 123 --checkpoint-dir checkpoints/vision_potharaju_split_seed123 --log-dir logs/vision_potharaju_split_seed123
```

Evaluation:

```bash
python src/evaluate_full_metrics.py --checkpoint checkpoints/vision_baseline/best_model.pth --output-csv docs/vision_full_metrics.csv
python src/evaluate_full_metrics.py --checkpoint checkpoints/vision_potharaju_split_seed123/best_model.pth --train-config configs/vision_baseline_potharaju_split.yaml
python src/explain/measure_lung_localization.py --checkpoint checkpoints/vision_se_seed7/best_model.pth --train-config configs/vision_se.yaml --output-csv docs/localization_se_seed7.csv
```

Calibration and significance testing:

```bash
python src/evaluate_calibration.py --checkpoint checkpoints/vision_baseline/best_model.pth --output-csv docs/calibration_vision.csv --output-plot docs/reliability_diagram_vision.png
python src/fit_temperature.py --checkpoint checkpoints/vision_baseline/best_model.pth --output-csv docs/temperature_scaling_vision.csv
python src/fit_isotonic.py --checkpoint checkpoints/vision_baseline/best_model.pth --output-csv docs/isotonic_vision.csv
python src/compare_auroc_paired.py --checkpoint-a checkpoints/vision_baseline/best_model.pth --checkpoint-b checkpoints/vision_label_smoothing/best_model.pth --label-a baseline --label-b label_smoothing --output-csv docs/paired_bootstrap_label_smoothing.csv
```

Dashboard (Streamlit demo with X-ray upload, prediction, Grad-CAM and counterfactual visualization):

```bash
streamlit run dashboard/app.py
```

Tests:

```bash
pytest
```

Currently: 84 passed. Also runs via GitHub Actions on push.

## Limitations

Worth being upfront about, since I'd rather someone find these here than in the viva:

- Tabular fusion features are synthetic. Fusion's gain shows the architecture can exploit a correlated signal, not that real vitals would help.
- The data is a single pediatric cohort from one center, and modest in size. Nothing here says anything about adults or other hospitals, and the model has not been clinically validated or intended as a diagnostic device.
- Grad-CAM and the lung-energy fraction are explanatory proxies. Lung-energy fraction says attention is inside the lung, not that it is on the pathology. Occlusion counterfactuals show sensitivity, not causality.
- Localization scores are means over about 20–26 images, so per-seed values are noisy.
- The weight sweep and the fusion-CBAM comparison are still at 3 seeds, and each label-smoothing value is a single seed, so those p-values and rankings are exploratory.
- The seeding fix was applied partway through, so SE and Potharaju-split runs (after) are not trained under exactly the same regime as the CBAM, no-attention and attention-consistency runs (before). The SE-vs-CBAM comparison mixes the two.
- Both models are overconfident by default (ECE about 0.135). Label smoothing improved this with no confirmed AUROC cost, but only at one seed per value.
- Threshold metrics use a fixed 0.5 cutoff. Recall is near 1.0 and precision is comparatively low, so accuracy depends on that choice.
- The Potharaju comparison rests on their paper's description. Their split is reproduced from that text, SE is a standard formulation, and their baseline CNN was not rebuilt.

## What I'd do differently / next

- Run the smoothing-value sweep over multiple seeds, and extend the fusion-CBAM comparison to 5 seeds, so nothing headline-level rests on 3 seeds or fewer.
- Retrain the older CBAM and no-attention runs under the deterministic seeding so all attention comparisons share one regime.
- Rerun the test suite after the SE and seeding changes.
- Replace the synthetic tabular features with real clinical data, and validate externally on a different hospital's data.
- Pathology-level localization annotations instead of just "inside the lung".

## Citation

If you use this repo, please cite it and the associated writeup in `docs/references.bib`.

---

**M. Kshitiz Reddy**, M.Tech (AI), Bennett University
