# Explainable Multimodal Diagnostic Support System

MTech project. Goal: chest X-ray diagnosis that fuses image + patient
metadata, with visual explanations for every prediction instead of a
black-box output.

Planning to start with the NIH Chest X-ray14 dataset (14-class, ~112k
images) since it has real patient metadata (age, gender, view position)
to fuse with the image features. Will revisit if the download/disk
situation doesn't work out.

## Planned stack

`PyTorch`, `torchvision`, Grad-CAM for explainability, Streamlit for a
demo dashboard. TBD on the tabular fusion approach.

## Setup

```bash
git clone https://github.com/<your-username>/multimodal-xai-diagnostic.git
cd multimodal-xai-diagnostic
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Roadmap

- [x] Repo scaffold, license, dependencies
- [ ] Data download + preprocessing pipeline
- [ ] Vision baseline
- [ ] Tabular fusion model
- [ ] Grad-CAM explainability module
- [ ] Occlusion-based counterfactual explainer
- [ ] Streamlit dashboard
- [ ] Deploy a live demo

## License

MIT — see [LICENSE](LICENSE).
