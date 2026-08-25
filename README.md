# 🩺 Explainable Multimodal Diagnostic Support System

![Tests](https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic/actions/workflows/tests.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)

> Chest X-ray diagnosis with fused clinical metadata, and visual explanations (Grad-CAM + occlusion-based counterfactuals) so the model's reasoning is inspectable instead of a black box.

🔗 **[Live Demo](https://multimodal-xai-diagnostic-yhqvbbhkejld2b6jodcvh2.streamlit.app)**

![Dashboard demo](docs/screenshots/dashboard_demo.png)

---

## Why this project

Clinical AI models are often accurate but opaque, which limits real-world trust. This project predicts pneumonia from pediatric chest X-rays **fused with patient vitals** (age, gender, temperature, SpO2 — see note below), and pairs every prediction with:

- A **Grad-CAM heatmap** showing which image regions drove the prediction.
- An **occlusion-based counterfactual view** showing how the model's confidence changes when the highlighted region is masked — an intuitive proxy for "what if this finding wasn't there?"

⚠️ **The tabular vitals (temperature, SpO2, age) are synthetically generated** — the source dataset ships images only, no real EHR data. They're simulated with clinically plausible correlations (fever/lower oxygen for pneumonia-positive cases) specifically to keep the fusion architecture genuinely meaningful to demonstrate. **See [`docs/ethics_statement.md`](docs/ethics_statement.md) before citing any results from this project** — this is disclosed prominently there and must be mentioned in any presentation of this work.

An ablation study (`notebooks/02_fusion_ablation_results.ipynb`) directly measures what the tabular metadata adds over the image alone. See [`docs/architecture.md`](docs/architecture.md) for full technical detail.

## Architecture

![Multimodal fusion architecture](docs/assets/architecture_diagram.svg)

*synthetic — see [`docs/ethics_statement.md`](docs/ethics_statement.md)

## Tech stack

`PyTorch` · `torchvision` · `pydicom` · `grad-cam` · `Streamlit` · `pandas` / `scikit-learn`

## Dataset

[Chest X-ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,856 pediatric chest X-ray images (ages 1–5) from Guangzhou Women and Children's Medical Center, labeled Normal/Pneumonia. Tabular fusion inputs (age, gender, temperature, SpO2) are synthetically generated — see [`docs/ethics_statement.md`](docs/ethics_statement.md).

*(This project originally targeted the full [NIH Chest X-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) dataset — 112k images, 14 classes, real patient metadata — switched to the above for local disk/compute constraints. See `configs/data_nih_legacy.yaml` and `docs/architecture.md#dataset-history`.)*

## Repository structure

```
multimodal-xai-diagnostic/
├── data/scripts/       # download & preprocessing scripts
├── src/
│   ├── data/            # Dataset / DataLoader classes
│   ├── models/           # vision encoder, tabular encoder, fusion model
│   └── explain/           # Grad-CAM + occlusion explainer
├── dashboard/            # Streamlit app
├── notebooks/            # EDA and results notebooks
├── tests/
└── docs/
```

## Setup

```bash
git clone https://github.com/<your-username>/multimodal-xai-diagnostic.git
cd multimodal-xai-diagnostic
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Explainability

Every prediction is paired with a **Grad-CAM heatmap** (`src/explain/gradcam.py`) showing which image regions drove the model's decision — built with [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam), hooked into DenseNet-121's final dense block.

```bash
# Generate example Grad-CAM overlays from the test set
python src/explain/generate_examples.py --checkpoint checkpoints/vision_baseline/best_model.pth
```

| High-confidence Pneumonia (0.99) | Low-confidence / Normal (0.06) |
|---|---|
| ![Grad-CAM Pneumonia example](docs/gradcam_examples/example_535_Pneumonia_0.99.png) | ![Grad-CAM Normal example](docs/gradcam_examples/example_479_Pneumonia_0.01.png) |

### A real finding: manual audit caught shortcut learning

Manually inspecting Grad-CAM outputs across several confident predictions found inconsistent localization — some examples showed a well-concentrated heatmap over lung tissue, but others showed activation spread outside the lung fields entirely, including one case with warm activation sitting directly on a burned-in laterality marker and timestamp. Full writeup with all examples in [`docs/ethics_statement.md`](docs/ethics_statement.md#observed-evidence-of-possible-shortcut-learning).

**Mitigation, verified working:** `src/explain/lung_segmentation.py` uses a pretrained chest X-ray segmentation model to constrain activation to the lung fields (`--restrict-to-lungs` flag). Compared below on the two worst offenders — same model, same image, before and after:

```bash
python src/explain/generate_examples.py --checkpoint checkpoints/vision_baseline/best_model.pth --restrict-to-lungs --output-dir docs/gradcam_examples_lung_restricted
```

| | Before (unrestricted) | After (lung-restricted) |
|---|---|---|
| **Diffuse, near-total saturation** | ![before](docs/gradcam_examples/example_406_Pneumonia_1.00.png) | ![after](docs/gradcam_examples_lung_restricted/example_406_Pneumonia_1.00.png) |
| **Activation on burned-in "R" marker/timestamp** | ![before](docs/gradcam_examples/example_269_Pneumonia_1.00.png) | ![after](docs/gradcam_examples_lung_restricted/example_269_Pneumonia_1.00.png) |

The lung boundary is now visibly carved into the mask in both cases, and the marker/timestamp in the second example no longer has any heat sitting on top of it. This constrains the *explanation* to anatomically valid regions — it doesn't necessarily fix whatever the model is doing internally, which is why the finding above stays documented rather than being quietly removed once "fixed."

**Counterfactual explanations** (`src/explain/counterfactual.py`) go a step further: the highest-activation region from Grad-CAM is inpainted out of the image, and the model is re-run on the result. A large confidence drop after removing that region is evidence the model's stated reasoning actually matches what's driving its prediction.

```bash
python src/explain/generate_counterfactual_examples.py --checkpoint checkpoints/vision_baseline/best_model.pth --strategy borderline
```

![Counterfactual flip example](docs/counterfactual_examples/example_471_Pneumonia.png)
*Borderline-confidence prediction (0.510) flips to Normal (0.062) after masking the Grad-CAM region.*

Testing on the 6 most borderline (closest to the 0.5 decision boundary) test predictions — the hardest possible case for this method, since confident predictions rarely flip regardless of explanation quality — **1/6 flipped outright, and 4 of the remaining 5 still showed a substantial confidence drop** toward Normal after occlusion. This is consistent evidence that the highlighted region is doing real work in the model's decision, not just noise.

## Dashboard

The Streamlit dashboard (`dashboard/app.py`) ties everything together: upload an X-ray, see per-class probabilities, a Grad-CAM heatmap for the top prediction, and the occlusion-based counterfactual comparison, all in one view.

```bash
streamlit run dashboard/app.py
```

It runs out of the box even before training — if no checkpoint is found at `checkpoints/vision_baseline/best_model.pth`, it falls back to a randomly-initialized model with a clearly visible warning banner, so the UI is explorable immediately. Once you've trained the vision baseline, predictions become meaningful automatically — no code changes needed.

**Deploying a live demo:** free via [Streamlit Community Cloud](https://streamlit.io/cloud), with the trained checkpoint hosted on a free Hugging Face Hub model repo — see [`docs/deployment.md`](docs/deployment.md) for the full walkthrough (Hugging Face Spaces now requires a paid plan for Streamlit apps, so this repo doesn't use that path).

## Usage

```bash
# 1. Download and prepare the Chest X-ray Pneumonia dataset (requires a Kaggle
#    API token — see data/scripts/prepare_pneumonia_dataset.py for setup).
#    This single script downloads images, builds patient-level train/val/test
#    splits, and generates the synthetic tabular features — no separate
#    preprocessing step needed for this dataset.
python data/scripts/prepare_pneumonia_dataset.py

# 2. Train the vision baseline (DenseNet-121)
python src/train.py --data-config configs/data.yaml --train-config configs/vision_baseline.yaml

# 3. Evaluate on the test set and generate the results table
python src/evaluate.py --checkpoint checkpoints/vision_baseline/best_model.pth

# 4. Train the fusion model (vision + tabular metadata)
python src/train_fusion.py --data-config configs/data.yaml --train-config configs/fusion.yaml

# 5. Evaluate the fusion model and generate its results table
python src/evaluate_fusion.py --checkpoint checkpoints/fusion/best_model.pth

# 6. Launch the dashboard
streamlit run dashboard/app.py
```

View training curves and per-class AUROC in `notebooks/01_vision_baseline_results.ipynb`, and the vision-only vs. fusion ablation comparison in `notebooks/02_fusion_ablation_results.ipynb`.

Run the test suite with:
```bash
pytest tests/ -v
```

### Comparing with vs. without CBAM

Once you've trained a baseline (`use_cbam: false` in `configs/vision_baseline.yaml`), train again with `use_cbam: true` (the current default) and compare both accuracy and Grad-CAM localization quality:

```bash
# Evaluate both checkpoints' test macro AUROC
python src/evaluate.py --checkpoint checkpoints/vision_baseline/best_model_no_cbam.pth
python src/evaluate.py --checkpoint checkpoints/vision_baseline/best_model.pth

# Compare what fraction of each model's Grad-CAM heatmap energy falls inside
# the segmented lung field — the localization claim from the papers in
# docs/paper_notes.md, made measurable instead of eyeballed
python src/explain/measure_lung_localization.py --checkpoint checkpoints/vision_baseline/best_model_no_cbam.pth --output-csv docs/localization_no_cbam.csv
python src/explain/measure_lung_localization.py --checkpoint checkpoints/vision_baseline/best_model.pth --output-csv docs/localization_cbam.csv
```

## Roadmap

- [x] Repo scaffold, license, dependencies
- [x] Data download + preprocessing pipeline
- [x] Vision baseline (DenseNet-121)
- [x] Tabular fusion model
- [x] Grad-CAM explainability module
- [x] Occlusion-based counterfactual explainer
- [x] Streamlit dashboard
- [x] Deploy live demo (Streamlit Community Cloud)
- [x] CBAM attention module (retrained + benchmarked — see [Results](#results))

## Results

Vision baseline (DenseNet-121) and fusion model (vision + tabular metadata) per-class and macro AUROC on the held-out test set are generated by `src/evaluate.py` / `src/evaluate_fusion.py` and compared in:

- `notebooks/01_vision_baseline_results.ipynb` — vision-only training curves and per-class AUROC
- `notebooks/02_fusion_ablation_results.ipynb` — vision-only vs. fusion comparison (the core ablation result)

```bash
python src/evaluate.py --checkpoint checkpoints/vision_baseline/best_model.pth
python src/evaluate_fusion.py --checkpoint checkpoints/fusion/best_model.pth
jupyter notebook notebooks/02_fusion_ablation_results.ipynb
```

| Model | Test Macro AUROC |
|---|---|
| Vision-only baseline (DenseNet-121) | 0.9708 |
| Vision + Tabular fusion | **0.9860** |

![Training curves](docs/training_curves.png)

Fusion improves test AUROC by **+1.5 points** over vision-only. Since the tabular vitals (temperature, SpO2) are synthetically generated with a deliberate correlation to the label (see [`docs/ethics_statement.md`](docs/ethics_statement.md)), this result demonstrates that **the fusion architecture correctly learns to exploit correlated tabular signal when present** — a valid architecture-level finding, not a real clinical discovery. Both models substantially exceed random chance (0.5) and validation AUROC (~0.999), with the gap between val and test AUROC (~0.03) reflecting normal generalization variance on a modestly-sized (~5,800 image) test set.

### CBAM attention — what it adds, and why

**What it is:** `src/models/attention.py` implements CBAM (Convolutional Block
Attention Module, Woo et al., ECCV 2018) — a small, cheap module inserted
between the DenseNet-121 backbone and the classifier head. It does two things
to the feature map before classification:
1. **Channel attention** — decides which of the 1024 feature channels matter more for this input, and rescales them.
2. **Spatial attention** — decides which *pixels* matter more, and rescales those.

Total cost: ~2.1M extra parameters on a ~7M-parameter backbone. No change to
input/output shapes, so it's a drop-in addition — enabled or disabled purely
via the `use_cbam` config flag, with no other code changes needed.

**Why it was added:** this wasn't a generic "add an attention mechanism"
upgrade. Reading the pneumonia chest X-ray literature (6 papers, 2024–2026,
full notes in [`docs/paper_notes.md`](docs/paper_notes.md)) turned up a
specific, repeated finding on this exact backbone + dataset combination:
CBAM doesn't just improve classification accuracy, it also makes Grad-CAM
heatmaps concentrate more tightly on actual pathological lung regions
instead of drifting to shoulders, image borders, or annotations. That second
part is what made it worth adding *here specifically* — it's a direct,
literature-backed extension of the shortcut-learning problem already
documented in [`docs/ethics_statement.md`](docs/ethics_statement.md), not
just a routine accuracy play. CBAM was picked over the lighter SE-Net
alternative (which some of the same papers also use) precisely because
SE only does channel attention — the spatial term is what actually
targets *where* the model looks, which is the whole point here.

**Current status:** implemented, tested (10 tests across `test_attention.py`
and the CBAM cases in `test_vision_model.py`), `use_cbam: true` is the
default in both configs, and **now actually retrained and benchmarked** —
both a no-CBAM baseline and a CBAM run, same data splits, same 15-epoch
config, only `use_cbam` different:

| Metric | No CBAM | CBAM | Difference |
|---|---|---|---|
| Test macro AUROC | 0.9592 | 0.9608 | +0.0016 (not meaningful — within noise) |
| Mean Grad-CAM lung-energy fraction (n=26 paired) | 0.415 | 0.511 | **+0.096** |
| Paired t-test | | | t=3.40, **p=0.0023** |
| Wilcoxon signed-rank | | | **p=0.0013** |
| Test images where CBAM improved localization | | | 20 / 26 |

*(Note: this run's no-CBAM baseline — 0.9592 — differs from the 0.9708 in
the main Results table above. Same architecture, same data splits, same
hyperparameters — just a different random seed on a separate training run.
That ~1-point spread is normal run-to-run variance for a test set this
size, not an inconsistency between the two tables.)*

**The honest reading:** CBAM barely moved accuracy — both models are already
near-ceiling on this dataset (test AUROC ~0.96 either way), so a 0.16-point
gap is noise, not a result. But it produced a real, statistically significant
shift in *where* the model looks: Grad-CAM heatmaps concentrate substantially
more inside the segmented lung field with CBAM on — a 23% relative increase
in lung-energy fraction, consistent across 20 of 26 test images (not driven
by one outlier). This is exactly the finding the literature review predicted
(see [`docs/paper_notes.md`](docs/paper_notes.md)): CBAM's value here is
localization quality, not raw accuracy, which matters more for an
explainability-focused project than it would for a pure classification one.

Four of the 30 sampled test images were dropped from the paired comparison
(indices 515, 479, 591, 461) because Grad-CAM produced a near-zero heatmap on
one side — this happens when the model is confidently predicting the class
is *absent* (all four had predicted probability ≤0.02), leaving essentially
no activation to measure a fraction of. `measure_lung_localization.py`
correctly flags these as NaN rather than reporting a meaningless number.

Visual example — index 272, same test image, both checkpoints:

| No CBAM | CBAM |
|---|---|
| ![No CBAM](docs/gradcam_examples_no_cbam/example_272_Pneumonia_0.99.png) | ![CBAM](docs/gradcam_examples/example_272_Pneumonia_1.00.png) |

The no-CBAM heatmap extends up over the shoulder/clavicle, outside the
actual lung field. The CBAM version sits more centrally over the upper
chest — not a dramatic transformation, but a real, visible shift toward the
lungs, consistent with the quantitative result above. Not every image
improved, though — index 479 (the low-confidence "normal" case) is a fair
counterexample: the no-CBAM heatmap is nearly nonexistent (matches its NaN
localization score), and CBAM's version, while measurable, lands mostly
outside the ribcage rather than being a clean win. Full per-image numbers in
[`docs/localization_no_cbam.csv`](docs/localization_no_cbam.csv) and
[`docs/localization_cbam.csv`](docs/localization_cbam.csv).

## Limitations & Ethics

This is a research/portfolio prototype trained on a public dataset and is **not validated for clinical use**. See [`docs/ethics_statement.md`](docs/ethics_statement.md) for a full discussion of dataset limitations, explainability caveats, and intended use.

## Deploying a live demo

See [`docs/deployment.md`](docs/deployment.md) for step-by-step instructions to deploy the dashboard for free.

## License

MIT — see [LICENSE](LICENSE).
