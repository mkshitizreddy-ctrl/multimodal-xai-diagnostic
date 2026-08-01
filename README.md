# 🩺 Explainable Multimodal Diagnostic Support System

![Tests](https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic/actions/workflows/tests.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)

> Chest X-ray diagnosis with fused clinical metadata, and visual explanations (Grad-CAM + occlusion-based counterfactuals) so the model's reasoning is inspectable instead of a black box.

<!-- 🔗 Live Demo link goes here once deployed — see docs/deployment.md -->

<!-- 🖼️ Dashboard demo GIF/screenshot goes here once built -->

---

## Why this project

Clinical AI models are often accurate but opaque, which limits real-world trust. This project predicts lung conditions from chest X-rays **fused with structured patient metadata** (age, sex, and view position), and pairs every prediction with:

- A **Grad-CAM heatmap** showing which image regions drove the prediction.
- An **occlusion-based counterfactual view** showing how the model's confidence changes when the highlighted region is masked — an intuitive proxy for "what if this finding wasn't there?"

An ablation study (`notebooks/02_fusion_ablation_results.ipynb`) directly measures what the tabular metadata adds over the image alone. See [`docs/architecture.md`](docs/architecture.md) for full technical detail and [`docs/ethics_statement.md`](docs/ethics_statement.md) for limitations and intended use.

## Architecture

```
                ┌────────────────────┐
   X-ray image →│  Vision Encoder     │──┐
                │  (DenseNet-121)     │  │
                └────────────────────┘  │      ┌───────────────┐      ┌─────────────────────┐
                                         ├─────▶│ Fusion Layer   │────▶│ Disease Probabilities │
                ┌────────────────────┐  │      └───────────────┘      └─────────────────────┘
  Patient meta →│  Tabular Encoder    │──┘              │
   (age, sex,   │  (MLP)              │                 ▼
   view, etc.)  └────────────────────┘        ┌───────────────────────┐
                                               │ Grad-CAM + Occlusion   │
                                               │ Explanation Module     │
                                               └───────────────────────┘
```

*(Diagram will be replaced with a proper Excalidraw export in `docs/` once the pipeline is finalized.)*

## Tech stack

`PyTorch` · `torchvision` · `pydicom` · `grad-cam` · `Streamlit` · `pandas` / `scikit-learn`

## Dataset

[NIH Chest X-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) — 112,120 X-ray images from 30,805 patients with 14 disease labels, plus patient age, gender, and view position metadata used as the tabular fusion input.

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

<!-- 🖼️ A couple of real Grad-CAM example images go here once generated from a trained model -->

**Counterfactual explanations** (`src/explain/counterfactual.py`) go a step further: the highest-activation region from Grad-CAM is inpainted out of the image, and the model is re-run on the result. A large confidence drop after removing that region is evidence the model's stated reasoning actually matches what's driving its prediction.

```bash
python src/explain/generate_counterfactual_examples.py --checkpoint checkpoints/vision_baseline/best_model.pth
```

<!-- 🖼️ A couple of real counterfactual side-by-side figures go here once generated -->

## Dashboard

The Streamlit dashboard (`dashboard/app.py`) ties everything together: upload an X-ray, see per-class probabilities, a Grad-CAM heatmap for the top prediction, and the occlusion-based counterfactual comparison, all in one view.

```bash
streamlit run dashboard/app.py
```

It runs out of the box even before training — if no checkpoint is found at `checkpoints/vision_baseline/best_model.pth`, it falls back to a randomly-initialized model with a clearly visible warning banner, so the UI is explorable immediately. Once you've trained the vision baseline, predictions become meaningful automatically — no code changes needed.

<!-- 🖼️ Dashboard screenshot/GIF goes here once you have a trained model to demo -->

**Deploying a live demo:** push this repo to a [Hugging Face Space](https://huggingface.co/new-space) (choose the Streamlit SDK) or connect it on [Streamlit Community Cloud](https://streamlit.io/cloud) — both work with `dashboard/app.py` and `requirements.txt` unmodified. Link the live demo at the top of this README once deployed.

## Usage

```bash
# 1. Download the NIH Chest X-ray14 dataset (requires a Kaggle API token — see
#    data/scripts/download_nih.py for one-time setup instructions)
python data/scripts/download_nih.py

# 2. Build clean, patient-level train/val/test splits
python data/scripts/preprocess.py --config configs/data.yaml

# 3. Train the vision baseline (DenseNet-121)
python src/train.py --data-config configs/data.yaml --train-config configs/vision_baseline.yaml

# 4. Evaluate on the test set and generate the results table
python src/evaluate.py --checkpoint checkpoints/vision_baseline/best_model.pth

# 5. Train the fusion model (vision + tabular metadata)
python src/train_fusion.py --data-config configs/data.yaml --train-config configs/fusion.yaml

# 6. Evaluate the fusion model and generate its results table
python src/evaluate_fusion.py --checkpoint checkpoints/fusion/best_model.pth

# 7. Launch the dashboard
streamlit run dashboard/app.py
```

View training curves and per-class AUROC in `notebooks/01_vision_baseline_results.ipynb`, and the vision-only vs. fusion ablation comparison in `notebooks/02_fusion_ablation_results.ipynb`.

Run the test suite with:
```bash
pytest tests/ -v
```

## Roadmap

- [x] Repo scaffold, license, dependencies
- [x] Data download + preprocessing pipeline
- [x] Vision baseline (DenseNet-121)
- [x] Tabular fusion model
- [x] Grad-CAM explainability module
- [x] Occlusion-based counterfactual explainer
- [x] Streamlit dashboard
- [ ] Deploy live demo (Hugging Face Spaces)

## Results

Vision baseline (DenseNet-121) and fusion model (vision + tabular metadata) per-class and macro AUROC on the held-out test set are generated by `src/evaluate.py` / `src/evaluate_fusion.py` and compared in:

- `notebooks/01_vision_baseline_results.ipynb` — vision-only training curves and per-class AUROC
- `notebooks/02_fusion_ablation_results.ipynb` — vision-only vs. fusion comparison (the core ablation result)

```bash
python src/evaluate.py --checkpoint checkpoints/vision_baseline/best_model.pth
python src/evaluate_fusion.py --checkpoint checkpoints/fusion/best_model.pth
jupyter notebook notebooks/02_fusion_ablation_results.ipynb
```

*(Results table and training curve images will be pasted here once a full
training run completes — the current epoch/batch-size config in
`configs/vision_baseline.yaml` targets a full run on the complete dataset;
reduce `epochs` or use a data subset for a faster smoke run.)*

## Limitations & Ethics

This is a research/portfolio prototype trained on a public dataset and is **not validated for clinical use**. See [`docs/ethics_statement.md`](docs/ethics_statement.md) for a full discussion of dataset limitations, explainability caveats, and intended use.

## Deploying a live demo

See [`docs/deployment.md`](docs/deployment.md) for step-by-step instructions to deploy the dashboard to Hugging Face Spaces.

## License

MIT — see [LICENSE](LICENSE).
