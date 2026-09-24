# Explainable Multimodal Diagnostic Support System

An explainable multimodal deep learning pipeline for pediatric pneumonia detection from chest X-rays and clinical metadata. Built as part of my M.Tech (AI) work at Bennett University.

The core idea: it's not enough for a model to say "pneumonia" — I wanted to know *where* it's looking, whether that changes depending on how the model is trained, and whether its confidence scores can actually be trusted. So this project combines a DenseNet-121 classifier with Grad-CAM, CBAM attention, occlusion-based counterfactuals, lung-localization scoring, an attention-consistency loss that explicitly nudges the model to look inside the lungs, and a calibration analysis (with three attempted fixes, one of which actually worked, and worked on both model variants) once I found the model's confidence wasn't trustworthy.

[![Tests](https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic/actions/workflows/tests.yml/badge.svg)](https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic/actions)

Python 3.11+ · MIT License

[Live demo](https://multimodal-xai-diagnostic-yhqvbbhkejld2b6jodcvh8.streamlit.app)

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
- Does label smoothing during training fix the overconfidence that post-hoc methods couldn't, and does it generalize to the fusion model?

## What's in here

**Vision model** — DenseNet-121, patient-level train/val splitting, binary classification, evaluated on accuracy/precision/recall/F1/AUROC/AUPR.

**Multimodal fusion** — image representation + tabular encoder + fusion head, so I could compare image-only vs. fused.

**Explainability** — Grad-CAM, occlusion-based counterfactuals, lung-field localization scoring, attention-map analysis.

**Attention modelling** — CBAM (channel + spatial), evaluated across multiple seeds, on both vision-only and fusion models, plus an attention-consistency loss trained against precomputed lung masks.

**Calibration** — Expected Calibration Error and reliability diagrams, plus three attempted fixes (temperature scaling, isotonic regression, label smoothing), with a paired-bootstrap significance check on the trade-off, tested on both vision and fusion models.

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

| Model / Experiment       | Accuracy | Precision | Recall |   F1   | AUROC  |  AUPR  |
| ------------------------ | -------: | --------: | -----: | -----: | -----: | -----: |
| Vision baseline           |  86.38%  |   0.8224  | 0.9974 | 0.9015 | 0.9604 | 0.9628 |
| Vision + rotation/zoom    |  73.88%  |   0.7052  | 1.0000 | 0.8271 | 0.9651 | 0.9730 |
| Vision + label smoothing  |  86.06%  |   0.8203  | 0.9949 | 0.8992 | 0.9459 | 0.9502 |
| Multimodal fusion         |  87.02%  |   0.8280  | 1.0000 | 0.9059 | 0.9899 | 0.9921 |
| Fusion + label smoothing  |  86.54%  |   0.8255  | 0.9949 | 0.9023 | 0.9927 | 0.9959 |

(Full CSVs with bootstrap CIs in `docs/vision_full_metrics.csv`, `docs/vision_rotation_zoom_metrics.csv`, `docs/vision_label_smoothing_full_metrics.csv`, `docs/fusion_full_metrics.csv`, `docs/fusion_label_smoothing_full_metrics.csv`.)

The rotation/zoom augmentation is a good example of why I look at more than one metric — it *improves* AUROC/AUPR but tanks accuracy and F1. Doesn't get called an improvement just because one number went up. The label-smoothing rows trade a small (and, per paired bootstrap testing below, not statistically significant in either direction) change in AUROC for a real calibration improvement — see the Calibration section for details.

## Explainability findings

**Grad-CAM** sometimes landed on the lungs and sometimes didn't — a few cases lit up shoulders, image borders, burned-in annotations/timestamps instead. That's what motivated the localization work; a good accuracy number doesn't mean the model is looking at the right thing.

**Lung-restricted explanations** — using a pretrained segmentation model, Grad-CAM can be constrained to the lung field for visualization (`--restrict-to-lungs`). That constrains what you *see*, not necessarily what the classifier actually learned from — I keep that distinction deliberate throughout.

**Counterfactuals** — occlude the highest-activation region and re-run the model. Across 6 borderline predictions: 1 flipped outright, 4 of the remaining 5 showed a real confidence drop. Not causal proof, but decent supporting evidence that the highlighted regions matter.

## CBAM — and why the story isn't simple

CBAM (channel + spatial attention) was evaluated across 5 seeds (42, 123, 2024, 7, 2025) on the vision-only model — extended from an initial 3 seeds after seeing how much the estimate moved as more seeds were added (see below). Also evaluated across 3 seeds on the fusion model.

Vision-only, test AUROC:

| Seed | No CBAM | CBAM | Diff |
|---|---:|---:|---:|
| 42 | 0.9592 | 0.9608 | +0.0016 |
| 123 | 0.9695 | 0.9445 | −0.0250 |
| 2024 | 0.9736 | 0.9604 | −0.0132 |
| 7 | 0.9280 | 0.9508 | +0.0228 |
| 2025 | 0.9673 | 0.9628 | −0.0045 |
| Mean ± SD | 0.9595 ± 0.0184 | 0.9559 ± 0.0079 | −0.0037 ± 0.0179 |

Paired t-test: p = 0.67 (was p = 0.31 at 3 seeds).

Vision-only, lung-energy localization:

| Seed | No CBAM | CBAM | Diff |
|---|---:|---:|---:|
| 42 | 0.4155 | 0.5110 | +0.0955 |
| 123 | 0.4217 | 0.5717 | +0.1499 |
| 2024 | 0.4624 | 0.4703 | +0.0079 |
| 7 | 0.4720 | 0.3400 | −0.1320 |
| 2025 | 0.4240 | 0.4480 | +0.0240 |
| Mean ± SD | 0.4391 ± 0.0261 | 0.4682 ± 0.0857 | +0.0291 ± 0.1066 |

Paired t-test: p = 0.57 (was p = 0.31 at 3 seeds).

**What changed going from 3 to 5 seeds, and why it matters:** the original 3-seed localization result looked like a fairly consistent positive trend (all 3 seeds favored CBAM, p=0.31). Adding seeds 7 and 2025 changed that — seed 7 is the only one of five where CBAM's localization is actually *worse* than without it (a large −0.132 swing, the opposite direction from every other seed), and it also has the weakest no-CBAM AUROC of any seed (0.928). That single seed roughly halved the mean localization effect and pushed AUROC's already-small diff even closer to zero. I don't have a principled reason to exclude seed 7 — no red flag like the validation-score inconsistency that got the weight=0.03 attention-consistency run flagged as an outlier — so it stays in.

**Honest conclusion:** with 5 seeds instead of 3, CBAM shows no reliable effect on AUROC (p=0.67) and no longer even shows a consistent positive-direction trend on localization (4 of 5 seeds favor CBAM, but the sizes range from +0.008 to +0.150, and one seed reverses the direction entirely, p=0.57). One consistent pattern that does hold across both seed counts: CBAM's AUROC is noticeably more stable run-to-run than no-CBAM's (SD 0.0079 vs. 0.0184) — even though it doesn't reliably improve the mean, it may reduce variance, which is a different and arguably more interesting property than the one I originally set out to test.

An earlier single-seed run had suggested a much stronger localization effect — that turned out to be **pseudo-replication** (treating individual images as independent replicates when the actual unit of replication is the training run/seed). Full details in `CHANGELOG.md`.

On the fusion model (3 seeds), CBAM's localization effect basically vanished (mean diff +0.0003 ± 0.085, p = 0.995) — so whatever CBAM is doing, it's architecture-dependent and doesn't transfer cleanly from vision-only to fusion, and even the vision-only effect doesn't hold up as seed count increases.

## Attention-consistency training

Instead of just hoping CBAM attention lands on the lungs, I added a loss term (`1 − lung_energy_fraction`) that explicitly pushes attention toward precomputed lung masks.

Evaluated across 5 seeds (42, 123, 2024, 7, 2025). Seeds 123 and 2024 were rerun partway through this analysis after I discovered their original checkpoints had been overwritten (no per-seed checkpoint directories were used at the time) — see the reproducibility note below for what that surfaced.

**Localization:**

| Seed | CBAM baseline | + Attention-consistency | Diff |
|---|---:|---:|---:|
| 42 | 0.5110 | 0.6253 | +0.1143 |
| 123 | 0.5717 | 0.6080 | +0.0363 |
| 2024 | 0.4703 | 0.6420 | +0.1717 |
| 7 | 0.3400 | 0.3610 | +0.0210 |
| 2025 | 0.4480 | 0.5210 | +0.0730 |
| Mean ± SD | | | **+0.0833 ± 0.0612** |

Paired t-test: **p = 0.038**.

**AUROC:**

| Seed | CBAM baseline | + Attention-consistency | Diff |
|---|---:|---:|---:|
| 42 | 0.9608 | 0.9233 | −0.0375 |
| 123 | 0.9445 | 0.9061 | −0.0384 |
| 2024 | 0.9604 | 0.8546 | −0.1058 |
| 7 | 0.9508 | 0.9045 | −0.0463 |
| 2025 | 0.9628 | 0.8713 | −0.0915 |
| Mean ± SD | 0.9559 ± 0.0079 | 0.8920 ± 0.0281 | **−0.0639 ± 0.0323** |

Paired t-test: **p = 0.0115**.

With a full, internally consistent 5-seed comparison (same checkpoints feeding both tables), attention-consistency training shows a **real, statistically significant trade-off**: it reliably improves lung localization and reliably costs AUROC. This is a stronger and more honest result than the original 3-seed estimate (localization p=0.051, AUROC p=0.164, both borderline) — the effect held up and became clearer with more data, not weaker, which is the opposite of what happened when I ran the same seed-extension exercise on the plain CBAM comparison above.

**A reproducibility finding, worth reporting on its own:** rerunning seed 2024 from scratch (same seed, same config) gave a localization value of 0.642 — nowhere near the original run's 0.537, a gap (+0.105) actually larger than the effect size I'm measuring (~0.08). Seed 123's rerun was much closer (0.608 vs. the original 0.616). So `torch.manual_seed` alone doesn't guarantee identical results in this setup — there's real run-to-run variance from something outside seed control, most likely cuDNN's non-deterministic convolution algorithms or DataLoader worker randomization that isn't fully pinned. I used the rerun's values in the tables above (matched to the checkpoints the AUROC numbers came from, for internal consistency) rather than the original, higher, now-orphaned number — the conservative choice, and it still left the result significant.

A weight sweep (at the original 3 seeds: 42, 123, 2024) makes the general trade-off shape explicit, independent of the seed-count analysis above:

| Weight | Test AUROC | Localization |
|---:|---:|---:|
| 0.0 (CBAM only) | 0.9552 ± 0.0093 | 0.462 ± 0.062 |
| 0.05 | 0.9356 ± 0.0119 | 0.550 ± 0.022 |
| 0.10 | 0.9292 ± 0.0124 | 0.593 ± 0.048 |
| 0.20 | 0.9100 ± 0.0380 | 0.611 ± 0.028 |

Better localization, worse AUROC, diminishing returns on localization as the weight climbs. I treat this as a tunable design decision, not a free win.

One earlier run at weight 0.03 gave AUROC 0.8933 with a suspicious 1.0000 validation score — didn't fit the sweep trend, so I flagged it as an unreplicated outlier rather than cherry-picking it into the results. In hindsight, given the seed-2024 non-determinism finding above, a perfect validation score alone isn't necessarily suspicious in this codebase — several of the 5-seed runs above also touched 1.0000 val AUROC without issue. What made weight=0.03 suspicious was specifically that its *test* AUROC didn't fit the surrounding trend, not the validation score by itself.

## A note on reproducibility

While rerunning seeds 123/2024 for the attention-consistency analysis above, I found that `torch.manual_seed()` alone didn't make training runs reproducible in this codebase — a same-seed rerun of the vision baseline gave different results (see seed 2024's original 0.537 vs. rerun 0.642 localization values above). Root cause: training augmentation (`RandomHorizontalFlip`, `RandomRotation`) draws from Python's built-in `random` module internally, which was never seeded in the DataLoader's worker processes; cuDNN's default convolution algorithms are also non-deterministic on GPU. `src/seeding.py` fixes both — seeds Python's `random`/NumPy/torch, pins cuDNN to deterministic mode, and reseeds each DataLoader worker. Verified with a controlled test: before the fix, two same-seed training runs diverged; after, they were bit-for-bit identical across every epoch. All three training scripts (`train.py`, `train_fusion.py`, `train_attention_consistency.py`) now use it.

**Bottom line:** this is now the strongest, most rigorously-tested finding in the project — a real, statistically significant trade-off between attention localization and classification performance, confirmed across 5 seeds with matched checkpoints throughout.

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
|---|---:|---:|---|
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

Both fixes above tried to patch an already-overconfident model after training. Label smoothing works differently — it changes what the model is trained to predict in the first place: instead of pushing toward hard targets (0 or 1), training labels are softened toward 0.05/0.95 (`label_smoothing=0.1`), which directly discourages the network from ever learning to output near-certain logits.

Retrained the vision model from scratch, same architecture/seed/split as the baseline, only the loss changed:

| Metric | Baseline | + Label smoothing |
|---|---:|---:|
| ECE | 0.136 | **0.103** |
| Brier | 0.116 | 0.108 |
| Accuracy | 86.38% | 86.06% |
| AUROC | 0.9604 | 0.9459 |

This is the one fix that actually helped — ECE dropped ~24% relative, Brier improved, and accuracy barely moved. There's a real-looking AUROC drop (0.960 → 0.946). I checked this properly with a paired bootstrap (`src/compare_auroc_paired.py`) — resampling the same test indices for both models each iteration and looking at the distribution of the difference directly, rather than comparing two separately-bootstrapped CIs. Result: mean difference −0.0143, 95% CI [−0.0309, 0.0023], p = 0.087. The CI includes 0, so the drop isn't statistically significant at the conventional 95% threshold — though p=0.087 is a borderline result, not a clean null, so I'd call this "a plausible small cost that this test set can't confirm" rather than "no cost at all."

**Does it generalize to fusion?** Retrained the fusion model the same way (`configs/fusion_label_smoothing.yaml`):

| Metric | Fusion baseline | + Label smoothing |
|---|---:|---:|
| ECE | 0.135 | **0.112** |
| Brier | 0.113 | 0.101 |
| Accuracy | 87.02% | 86.54% |
| AUROC | 0.9899 | 0.9927 |

Same calibration improvement (~17% relative ECE drop) as vision. Unlike vision, the point-estimate AUROC actually went *up* slightly — but a paired bootstrap (same method as above) put that at mean diff +0.0028, 95% CI [−0.0032, 0.0104], p = 0.42, clearly not significant. So the fusion result is arguably cleaner than vision's: **a real calibration improvement with no detectable cost — or benefit — to ranking.**

**Does the smoothing value matter?** Tried three values on the vision model (0.05, 0.1, 0.2), same seed/architecture:

| Smoothing | ECE | Brier | Accuracy | AUROC |
|---:|---:|---:|---:|---:|
| 0.0 (baseline) | 0.136 | 0.116 | 86.38% | 0.9604 |
| 0.05 | 0.101 | 0.097 | 87.98% | 0.9574 |
| 0.10 | 0.103 | 0.108 | 86.06% | 0.9459 |
| 0.20 | 0.106 | 0.078 | 91.03% | 0.9652 |

ECE improves at every value tried, but not monotonically — best at 0.05, not 0.2. The 0.2 run's accuracy and AUROC look like they beat baseline outright, which would be surprising (label smoothing is supposed to trade accuracy for calibration, not improve both), so I checked it with the same paired bootstrap as before: mean diff +0.0047, 95% CI [−0.0089, 0.0190], p = 0.54. Not significant — consistent with single-seed noise (the 0.2 run also early-stopped noticeably earlier than the others, at epoch 5 vs. 7–9), not a real effect of that smoothing value.

**Honest conclusion:** label smoothing reliably improves calibration somewhere in the 0.05–0.2 range with no confirmed accuracy/AUROC cost, but a single seed per value can't tell you which exact value is optimal — the differences between 0.05, 0.1, and 0.2 are themselves within noise of each other. A real answer to "what's the best smoothing value" would need multiple seeds per value, the same lesson the CBAM and attention-consistency experiments already taught.

**Net result of the calibration thread:** the overconfidence was real and precisely diagnosed, two of three fixes failed for well-understood reasons, and the one that worked — label smoothing — generalized cleanly from vision to fusion, in both cases with no statistically confirmed cost to ranking. Fusion's result is the strongest evidence: **overconfidence baked in during training is better addressed during training than patched afterward.**

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

# fusion with label smoothing
python src/train_fusion.py --data-config configs/data.yaml --train-config configs/fusion_label_smoothing.yaml

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
python src/evaluate_calibration.py --checkpoint checkpoints/fusion_label_smoothing/best_model.pth --train-config configs/fusion_label_smoothing.yaml --output-csv docs/calibration_fusion_label_smoothing.csv --output-plot docs/reliability_diagram_fusion_label_smoothing.png
python src/compare_auroc_paired.py --checkpoint-a checkpoints/vision_baseline/best_model.pth --checkpoint-b checkpoints/vision_label_smoothing/best_model.pth --label-a baseline --label-b label_smoothing --output-csv docs/paired_bootstrap_label_smoothing.csv
python src/compare_auroc_paired.py --checkpoint-a checkpoints/fusion/best_model.pth --checkpoint-b checkpoints/fusion_label_smoothing/best_model.pth --label-a fusion_baseline --label-b fusion_label_smoothing --train-config configs/fusion.yaml --output-csv docs/paired_bootstrap_fusion_label_smoothing.csv
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
- The CBAM and attention-consistency vision-only comparisons are now at 5 seeds; the weight sweep and the fusion-model CBAM comparison are still at 3 seeds, so those p-values remain exploratory rather than confirmatory.
- Lung-energy fraction tells you attention is inside the lung, not that it's on the actual pathological region.
- Grad-CAM is an interpretation method, not a causal explanation. Same caveat for the occlusion counterfactuals — sensitivity isn't causality.
- Both models are meaningfully overconfident by default (ECE ~0.135); label smoothing improved this on both (ECE ~0.10–0.11) with no statistically confirmed cost to ranking. A smoothing-value sweep (0.05/0.1/0.2) was run on vision, but only at a single seed per value — see the sweep result below.
- This is a research/portfolio prototype. It has not been clinically validated and isn't a diagnostic device.

## What I'd do differently / next

- The weight sweep and the fusion-model CBAM comparison are still at 3 seeds — the CBAM/attention-consistency 5-seed extension above showed this matters, so the same treatment would strengthen both.
- The smoothing-value sweep (0.05/0.1/0.2) showed calibration improves at every value but isn't clean or monotonic at a single seed - multiple seeds per value would be needed to say anything definitive about which value is actually best.
- Real clinical/EHR metadata instead of synthetic, if I ever get access to it.
- External validation on a different hospital/dataset.
- Pathology-level localization annotations instead of just "inside the lung."

## Citation

If you use this repo, please cite it and the associated writeup in `docs/references.bib`.

---

**M. Kshitiz Reddy** — M.Tech (AI), Bennett University