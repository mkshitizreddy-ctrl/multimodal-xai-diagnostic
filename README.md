# 🩺 Explainable Multimodal Diagnostic Support System

> Chest X-ray diagnosis with fused clinical metadata, and visual explanations (Grad-CAM + occlusion-based counterfactuals) so the model's reasoning is inspectable instead of a black box.

**Status:** 🚧 In active development — see [Roadmap](#roadmap) below.

<!-- 🖼️ Dashboard demo GIF/screenshot goes here once built -->

---

## Why this project

Clinical AI models are often accurate but opaque, which limits real-world trust. This project predicts lung conditions from chest X-rays **fused with structured patient metadata** (age, sex, view position, and — where available — clinical variables like smoking status and oxygen level), and pairs every prediction with:

- A **Grad-CAM heatmap** showing which image regions drove the prediction.
- An **occlusion-based counterfactual view** showing how the model's confidence changes when the highlighted region is masked — an intuitive proxy for "what if this finding wasn't there?"

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

## Usage

```bash
# 1. Download the NIH Chest X-ray14 dataset (requires a Kaggle API token — see
#    data/scripts/download_nih.py for one-time setup instructions)
python data/scripts/download_nih.py

# 2. Build clean, patient-level train/val/test splits
python data/scripts/preprocess.py --config configs/data.yaml

# 3. (coming next) Train baseline vision model
python src/train.py --config configs/vision_baseline.yaml

# 4. (coming next) Launch dashboard
streamlit run dashboard/app.py
```

Run the test suite with:
```bash
pytest tests/ -v
```

## Roadmap

- [x] Repo scaffold, license, dependencies
- [x] Data download + preprocessing pipeline
- [ ] Vision baseline (DenseNet-121)
- [ ] Tabular fusion model
- [ ] Grad-CAM explainability module
- [ ] Occlusion-based counterfactual explainer
- [ ] Streamlit dashboard
- [ ] Deploy live demo (Hugging Face Spaces)

## Results

*(Table added once the baseline is trained — AUROC per class, before/after fusion.)*

## Limitations & Ethics

This is a research/portfolio prototype trained on a public dataset and is **not validated for clinical use**. Public chest X-ray datasets carry known demographic and label-noise biases; predictions should not be treated as diagnostic ground truth.

## License

MIT — see [LICENSE](LICENSE).
