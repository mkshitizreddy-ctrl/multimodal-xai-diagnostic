# Explainable Multimodal Diagnostic Support System

An explainable multimodal deep learning pipeline for pediatric pneumonia detection from chest X-rays and clinical metadata. Built as part of my M.Tech (AI) work at Bennett University.

The core idea: it's not enough for a model to say "pneumonia" — I wanted to know *where* it's looking, whether that changes depending on how the model is trained, and whether its confidence scores can actually be trusted. So this project combines a DenseNet-121 classifier with Grad-CAM, CBAM attention, occlusion-based counterfactuals, lung-localization scoring, an attention-consistency loss that explicitly nudges the model to look inside the lungs, and a calibration analysis (with three attempted fixes, one of which actually worked) once I found the model's confidence wasn't trustworthy.

[![Tests](https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic/actions/workflows/tests.yml/badge.svg)](https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic/actions)
Python 3.11+ · MIT License

[Live demo](https://multimodal-xai-diagnostic-yhqvbbhkejld2b6jodcvh2.streamlit.app)

![Dashboard demo](docs/screenshots/dashboard_demo.png)

---

## Why this project

Medical imaging models can hit strong accuracy numbers while giving almost no insight into *why* they made a call. That's a real problem in a clinical setting — a model that's right for the wrong reasons is still a liability. So alongside the classifier, I built out a set of explainability tools and used them to actually interrogate the model instead of just trusting the heatmaps at face value.

**Note on the data:** the chest X-ray dataset doesn't come with real patient vitals, so the tabular features used for multimodal fusion (age, gender, temperature, SpO₂) are synthetically generated with clinically plausible correlations to the label. They're there to test the *architecture*, not to claim any real diagnostic benefit — see `docs/ethics_statement.md`.

## Research questions

- Can DenseNet-121 classify pneumonia from chest X-rays accurately?
- Does adding tabular data help over image-only?
- Does Grad-CAM actually land on clinically relevant regions, or is the model cheating?
- Does CBAM improve localization — and does that hold up across seeds?
- Can I explicitly train attention to stay inside the lungs, and what does that cost in accuracy?
- Are the model's confidence scores trustworthy, and if not, can a standard post-hoc fix repair them?
- Does label smoothing during training fix the overconfidence that post-hoc methods couldn't?

## What's in here

**Vision model** — DenseNet-121, patient-level train/val splitting, binary classification, evaluated on accuracy/precision/recall/F1/AUROC/AUPR.

**Multimodal fusion** — image representation + tabular encoder + fusion head, so I could compare image-only vs. fused.

**Explainability** — Grad-CAM, occlusion-based counterfactuals, lung-field localization scoring, attention-map analysis.

**Attention modelling** — CBAM (channel + spatial), evaluated across multiple seeds, on both vision-only and fusion models, plus an attention-consistency loss trained against precomputed lung masks.

**Calibration** — Expected Calibration Error and reliability diagrams, plus three attempted fixes (temperature scaling, isotonic regression, label smoothing), with a paired-bootstrap significance check on the trade-off.

**Engineering** — configurable training pipeline, automated eval scripts, GitHub Actions CI, 84 passing tests.

## Architecture

```text
                    Chest X-ray Image
                            │
                            ▼
                     DenseNet-121
                    (CBAM attention)
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

Full diagram/writeup: `docs/architecture.md`

## Dataset

Chest X-ray Pneumonia dataset (pediatric, ~1–5 years, Guangzhou Women and Children's Medical Center) — 5,856 images, Normal vs. Pneumonia. I originally planned to use NIH ChestX-ray14 but switched given local compute/storage limits.

Patient-level splitting where possible (pneumonia filenames carry patient IDs; normal-class images don't, so I conservatively treated each as a unique patient):

- Train: ~4,434
- Val: ~798
- Test: ~624 (original dataset test split, preserved as-is)

## Stack

Python 3.11+, PyTorch, Torchvision, DenseNet-121, CBAM, scikit-learn, Pandas, NumPy, Grad-CAM, Streamlit, OpenCV/PIL, pytest

## Repo layout

```text
multimodal-xai-diagnostic/
├── data/scripts/            # dataset prep, lung-mask precomputation
├── src/
│   ├── data/                 # dataset classes
│   ├── models/                # vision encoder, fusion, CBAM, attention-consistency loss
│   ├── explain/                # Grad-CAM, counterfactuals, lung segmentation/localization
│   ├── evaluate*.py, evaluate_calibration.py, fit_temperature.py, fit_isotonic.py,
│   │   compare_auroc_paired.py
│   ├── train*.py
├── dashboard/app.py          # Streamlit demo
├── notebooks/                # baseline + fusion ablation results
├── configs/
├── tests/
├── docs/                       # architecture, ethics, ablations, paper drafts
└── CHANGELOG.md
```

## Results

| Model / Experiment     | Accuracy | Precision | Recall |   F1   | AUROC  |  AUPR  |
| ----------------------- | -------: | --------: | -----: | -----: | -----: | -----: |
| Vision baseline         |  86.38%  |   0.8224  | 0.9974 | 0.9015 | 0.9604 | 0.9628 |
| Vision + rotation/zoom  |  73.88%  |   0.7052  | 1.0000 | 0.8271 | 0.9651 | 0.9730 |
| Vision + label smoothing|  86.06%  |   0.8203  | 0.9949 | 0.8992 | 0.9459 | 0.9502 |
| Multimodal fusion       |  87.02%  |   0.8280  | 1.0000 | 0.9059 | 0.9899 | 0.9921 |

(Full CSVs with bootstrap CIs in `docs/vision_full_metrics.csv`, `docs/vision_rotation_zoom_metrics.csv`, `docs/vision_label_smoothing_full_metrics.csv`, `docs/fusion_full_metrics.csv`.)

The rotation/zoom augmentation is a good example of why I look at more than one metric — it *improves* AUROC/AUPR but tanks accuracy and F1. Doesn't get called an improvement just because one number went up. The label-smoothing row trades a small (and, per the paired bootstrap below, not statistically significant) amount of AUROC for a real calibration improvement — see the Calibration section for details.

## Explainability findings

**Grad-CAM** sometimes landed on the lungs and sometimes didn't — a few cases lit up shoulders, image borders, burned-in annotations/timestamps instead. That's what motivated the localization work; a good accuracy number doesn't mean the model is looking at the right thing.

**Lung-restricted explanations** — using a pretrained segmentation model, Grad-CAM can be constrained to the lung field for visualization (`--restrict-to-lungs`). That constrains what you *see*, not necessarily what the classifier actually learned from — I keep that distinction deliberate throughout.

**Counterfactuals** — occlude the highest-activation region and re-run the model. Across 6 borderline predictions: 1 flipped outright, 4 of the remaining 5 showed a real confidence drop. Not causal proof, but decent supporting evidence that the highlighted regions matter.

## CBAM — and why the story isn't simple

CBAM (channel + spatial attention) was evaluated across 3 seeds (42, 123, 2024), on both the vision-only and fusion models.

Vision-only, test AUROC:

| Seed | No CBAM | CBAM | Diff |
|---|---:|---:|---:|
| 42 | 0.9592 | 0.9608 | +0.0016 |
| 123 | 0.9695 | 0.9445 | −0.0250 |
| 2024 | 0.9736 | 0.9604 | −0.0132 |
| Mean ± SD | 0.9674 ± 0.0074 | 0.9552 ± 0.0093 | −0.0122 ± 0.0139 |

Localization diff: +0.060 ± 0.076, p = 0.31 (exploratory, not significant).

An earlier single-seed run had suggested a much stronger effect — that turned out to be **pseudo-replication** (treating individual images as independent replicates when the actual unit of replication is the training run/seed). I reran it properly across 3 seeds rather than quietly keeping the flattering number. Full details in `CHANGELOG.md`.

On the fusion model, CBAM's localization effect basically vanished (mean diff +0.0003 ± 0.085, p = 0.995) — so whatever CBAM is doing, it's architecture-dependent and doesn't transfer cleanly from vision-only to fusion.

## Attention-consistency training

Instead of just hoping CBAM attention lands on the lungs, I added a loss term (`1 − lung_energy_fraction`) that explicitly pushes attention toward precomputed lung masks.

Across 3 seeds, localization improved consistently (+0.134 ± 0.055, p = 0.051 — borderline but consistent direction across all three seeds), at a cost to AUROC (−0.026 ± 0.021, p = 0.164).

A weight sweep makes the trade-off explicit:

| Weight | Test AUROC | Localization |
|---:|---:|---:|
| 0.0 (CBAM only) | 0.9552 ± 0.0093 | 0.462 ± 0.062 |
| 0.05 | 0.9356 ± 0.0119 | 0.550 ± 0.022 |
| 0.10 | 0.9292 ± 0.0124 | 0.593 ± 0.048 |
| 0.20 | 0.9100 ± 0.0380 | 0.611 ± 0.028 |

Better localization, worse AUROC, diminishing returns on localization as the weight climbs. I treat this as a tunable design decision, not a free win — and it's the honest way to present it.

One earlier run at weight 0.03 gave AUROC 0.8933 with a suspicious 1.0000 validation score — didn't fit the sweep trend, so I flagged it as an unreplicated outlier rather than cherry-picking it into the results.

## Calibration

Accuracy and AUROC say nothing about whether the model's confidence scores are trustworthy, so I ran a separate calibration check (`src/evaluate_calibration.py`): Expected Calibration Error (ECE) and a reliability diagram, on the held-out test set.

| Model | ECE | Brier |
|---|---:|---:|
| Vision baseline | 0.136 (95% CI [0.113, 0.162]) | 0.116 |
| Fusion | 0.135 (95% CI [0.110, 0.160]) | 0.113 |

Both models are meaningfully overconfident, and it's concentrated in one place: **71% of the test set (442/624 images) falls in the 0.9–1.0 confidence bin**, where the model's average stated confidence is 99.5% but its actual accuracy is only 87.3%. So for the large majority of predictions, "the model is nearly certain" overstates how often it's actually right — something worth knowing before treating a high confidence score as a proxy for correctness. The remaining bins (0.2–0.8) show large gaps too, but with only 4–10 samples each, I don't read much into their individual values — too few points for a reliable per-bin estimate.

Adding fusion features didn't meaningfully change calibration (0.136 vs. 0.135) — consistent with the earlier finding that CBAM's localization effect on fusion diverges from vision-only; calibration looks like a property of the underlying vision backbone/training setup rather than something fusion or attention changes.

I tried three fixes, in order of how invasive they are. The first two are post-hoc (applied after training, to an already-fixed model); the third changes training itself.

### Attempt 1: temperature scaling

Fit a single scalar T on validation logits only (Guo et al., 2017), apply it to test logits before the sigmoid.

| Model | T (fit on val) | ECE before → after | Brier before → after |
|---|---:|---|---|
| Vision baseline | 1.05 | 0.136 → 0.137 | 0.116 → 0.116 |
| Fusion | 1.21 | 0.135 → 0.136 | 0.113 → 0.110 |

Didn't work, in any meaningful sense — ECE was essentially unchanged (if anything, marginally worse) for both models, despite fusion needing a much larger correction than vision. AUROC was unaffected as expected (rank-preserving transform; fusion showed a ~0.0003 numerical wobble from floating-point tie-breaking on near-identical logits, not a real ranking change).

I read this as evidence that the miscalibration isn't simple global overconfidence that one scalar can absorb — it's concentrated specifically in the 0.9–1.0 confidence bin, and a single T fit on the whole validation distribution doesn't target that region well.

### Attempt 2: isotonic regression

Since temperature scaling's single global parameter couldn't target the localized problem, I tried isotonic regression instead — a non-decreasing step function fit on validation data, capable of correcting different probability ranges independently.

| Model | ECE before → after | Brier before → after | AUROC before → after |
|---|---|---|---|
| Vision baseline | 0.136 → 0.178 | 0.116 → 0.149 | 0.960 → 0.932 |
| Fusion | 0.135 → 0.113 | 0.113 → 0.102 | 0.990 → 0.949 |

Not a usable fix for either model, though for different reasons. For vision, every metric got worse — a sign of overfitting: isotonic regression is fit on only 798 validation examples, and with 71% of the test set concentrated in one confidence bin, there's likely not enough validation data in that region to fit a reliable step function, so it just memorizes validation-specific noise. For fusion, calibration genuinely improved, but at a real cost to ranking (AUROC dropped 0.041) — too large a trade to call it a win.

### Attempt 3: label smoothing (the one that actually worked)

Both fixes above tried to patch an already-overconfident model after training. Label smoothing works differently — it changes what the model is trained to predict in the first place: instead of pushing toward hard targets (0 or 1), training labels are softened toward 0.05/0.95 (`label_smoothing=0.1` in `configs/vision_label_smoothing.yaml`), which directly discourages the network from ever learning to output near-certain logits.

Retrained the vision model from scratch, same architecture/seed/split as the baseline, only the loss changed:

| Metric | Baseline | + Label smoothing |
|---|---:|---:|
| ECE | 0.136 | **0.103** |
| Brier | 0.116 | 0.108 |
| Accuracy | 86.38% | 86.06% |
| AUROC | 0.9604 | 0.9459 |

This is the one fix that actually helped — ECE dropped ~24% relative, Brier improved, and accuracy barely moved. There's a real-looking AUROC drop (0.960 → 0.946). I checked this properly with a paired bootstrap (`src/compare_auroc_paired.py`) — resampling the same test indices for both models each iteration and looking at the distribution of the difference directly, rather than comparing two separately-bootstrapped CIs. Result: mean difference −0.0143, 95% CI [−0.0309, 0.0023], p = 0.087. The CI includes 0, so the drop isn't statistically significant at the conventional 95% threshold — though p=0.087 is a borderline result, not a clean null, so I'd call this "a plausible small cost that this test set can't confirm" rather than "no cost at all."

**Net result of the calibration thread:** the overconfidence was real, precisely diagnosed (71% of test samples in one overconfident bin), and two of three fixes failed for well-understood reasons (wrong granularity for temperature scaling, overfitting for isotonic regression on vision). The one that worked changed training rather than patching the output, with a calibration gain that's solid and a possible AUROC cost that a proper paired test couldn't confirm on this test set — a reasonable, if modest, conclusion: **overconfidence baked in during training is better addressed during training than patched afterward.**

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
```

Training:

```bash
# vision baseline
python src/train.py --data-config configs/data.yaml --train-config configs/vision_baseline.yaml

# vision with label smoothing
python src/train.py --data-config configs/data.yaml --train-config configs/vision_label_smoothing.yaml

# fusion model
python src/train_fusion.py --data-config configs/data.yaml --train-config configs/fusion.yaml

# attention-consistency (precompute lung masks first)
python data/scripts/precompute_lung_masks.py --train-config configs/vision_attention_consistency.yaml
python src/train_attention_consistency.py --train-config configs/vision_attention_consistency.yaml
```

Evaluation:

```bash
python src/evaluate.py --checkpoint checkpoints/vision_baseline/best_model.pth
python src/evaluate_fusion.py --checkpoint checkpoints/fusion/best_model.pth
python src/evaluate_full_metrics.py --checkpoint checkpoints/vision_baseline/best_model.pth --output-csv docs/vision_full_metrics.csv
```

Lung localization:

```bash
python src/explain/measure_lung_localization.py --checkpoint checkpoints/vision_baseline/best_model.pth --output-csv docs/localization_cbam.csv
```

Calibration and fixes:

```bash
python src/evaluate_calibration.py --checkpoint checkpoints/vision_baseline/best_model.pth --output-csv docs/calibration_vision.csv --output-plot docs/reliability_diagram_vision.png
python src/fit_temperature.py --checkpoint checkpoints/vision_baseline/best_model.pth --output-csv docs/temperature_scaling_vision.csv --output-plot docs/reliability_diagram_vision_after_temp.png
python src/fit_isotonic.py --checkpoint checkpoints/vision_baseline/best_model.pth --output-csv docs/isotonic_vision.csv --output-plot docs/reliability_diagram_vision_after_isotonic.png
python src/evaluate_calibration.py --checkpoint checkpoints/vision_label_smoothing/best_model.pth --output-csv docs/calibration_vision_label_smoothing.csv --output-plot docs/reliability_diagram_vision_label_smoothing.png
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

Currently: 84 passed, 6 warnings (warnings are from dependency code, not test failures). Also runs automatically via GitHub Actions on push.

## Limitations

Worth being upfront about, since I'd rather someone find these in the README than in the viva:

- Tabular fusion features are synthetic — this shows the architecture can exploit a signal, not that real clinical vitals would help.
- Dataset is modest compared to large-scale medical imaging benchmarks.
- Several comparisons use only 3 seeds — reported p-values are exploratory, not confirmatory.
- Lung-energy fraction tells you attention is inside the lung, not that it's on the actual pathological region.
- Grad-CAM is an interpretation method, not a causal explanation. Same caveat for the occlusion counterfactuals — sensitivity isn't causality.
- The baseline model is meaningfully overconfident (ECE ~0.135); label smoothing improved this (ECE ~0.103) at a small AUROC cost that a paired bootstrap couldn't confirm as statistically significant (p=0.087) — but I've only validated this on the vision model with one smoothing value.
- This is a research/portfolio prototype. It has not been clinically validated and isn't a diagnostic device.

## What I'd do differently / next

- More seeds where compute allows — 3 is thin for the statistical claims I'd ideally want to make.
- Label smoothing worked for calibration but I only tried one value (0.1) and only on the vision model — a smoothing sweep (like the attention-consistency weight sweep) and checking it on the fusion model too would be the natural follow-up.
- Real clinical/EHR metadata instead of synthetic, if I ever get access to it.
- External validation on a different hospital/dataset.
- Pathology-level localization annotations instead of just "inside the lung."

## Citation

If you use this repo, please cite it and the associated writeup in `docs/references.bib`.

---

**M. Kshitiz Reddy** — M.Tech (AI), Bennett University