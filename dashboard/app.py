"""
Interactive clinician-facing dashboard.

Upload a chest X-ray -> see disease probabilities, a Grad-CAM heatmap for
the top prediction, and an occlusion-based counterfactual comparison.

Run locally:
    streamlit run dashboard/app.py

Deploy: push this repo to Streamlit Community Cloud (share.streamlit.io) —
free, and deploys directly from GitHub. The trained checkpoint (too large
for a normal git repo, and gitignored here) is instead hosted on a free
Hugging Face Hub MODEL repo and downloaded automatically at startup if not
already present locally — see docs/deployment.md. Set the HF_MODEL_REPO_ID
constant below (or the HF_MODEL_REPO_ID env var / Streamlit secret) to your
own repo once you've uploaded a checkpoint there.
"""

import os
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.counterfactual import OcclusionCounterfactualExplainer, make_side_by_side_figure
from src.explain.gradcam import ChestXrayExplainer
from src.models.fusion import ChestXrayFusionModel
from src.models.vision_encoder import ChestXrayVisionModel

st.set_page_config(page_title="Explainable Chest X-ray Diagnosis", layout="wide")

VISION_CHECKPOINT = "checkpoints/vision_baseline/best_model.pth"
FUSION_CHECKPOINT = "checkpoints/fusion/best_model.pth"
DATA_CONFIG_PATH = "configs/data.yaml"
IMAGE_SIZE = 224
NORM_MEAN, NORM_STD = 0.5, 0.25

# Set this to "<your-hf-username>/multimodal-xai-diagnostic-weights" once you've
# uploaded a checkpoint to a Hugging Face Hub model repo (see docs/deployment.md).
# Can be overridden without editing code via env var or a Streamlit secret.
def _get_hf_model_repo_id() -> str | None:
    """Reads HF_MODEL_REPO_ID from an env var first, then a Streamlit
    secret if configured — safely, since st.secrets raises (rather than
    returning None) when no secrets.toml file exists at all, which is the
    normal case for local runs and CI."""
    env_value = os.environ.get("HF_MODEL_REPO_ID")
    if env_value:
        return env_value
    try:
        return st.secrets.get("HF_MODEL_REPO_ID")
    except Exception:
        return None


HF_MODEL_REPO_ID = _get_hf_model_repo_id()
HF_CHECKPOINT_FILENAME = "vision_baseline_best_model.pth"


def _download_checkpoint_from_hf_hub() -> str | None:
    """Downloads the checkpoint from a public HF Hub model repo if
    HF_MODEL_REPO_ID is configured. Returns the local path, or None if not
    configured or the download fails (caller falls back to demo mode)."""
    if not HF_MODEL_REPO_ID:
        return None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=HF_CHECKPOINT_FILENAME)
    except Exception as e:
        st.warning(f"Could not download checkpoint from Hugging Face Hub ({HF_MODEL_REPO_ID}): {e}")
        return None


@st.cache_resource
def load_data_config():
    with open(DATA_CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_vision_model():
    """Loads the trained vision-only checkpoint if present locally, else
    tries downloading it from HF Hub (for the deployed demo), else falls
    back to an untrained model so the dashboard is still explorable
    (clearly labeled as demo mode in the UI)."""
    data_cfg = load_data_config()
    classes = data_cfg["labels"]["classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = VISION_CHECKPOINT if Path(VISION_CHECKPOINT).exists() else _download_checkpoint_from_hf_hub()

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = ChestXrayVisionModel(num_classes=len(checkpoint["classes"]), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device).eval(), checkpoint["classes"], device, True

    model = ChestXrayVisionModel(num_classes=len(classes), pretrained=False)
    return model.to(device).eval(), classes, device, False


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[NORM_MEAN] * 3, std=[NORM_STD] * 3),
        ]
    )
    return transform(pil_image.convert("RGB"))


def render_probability_chart(classes: list[str], probs: np.ndarray) -> go.Figure:
    order = np.argsort(probs)
    fig = go.Figure(
        go.Bar(
            x=probs[order],
            y=[classes[i] for i in order],
            orientation="h",
            marker_color=["crimson" if p >= 0.5 else "steelblue" for p in probs[order]],
        )
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="Predicted probability",
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def main():
    st.title("🩺 Explainable Chest X-ray Diagnosis")
    st.caption(
        "Research/portfolio prototype — not validated for clinical use. "
        "See the README for dataset, architecture, and limitations."
    )

    model, classes, device, is_trained = load_vision_model()

    if not is_trained:
        st.warning(
            "⚠️ No trained checkpoint found locally, and no Hugging Face Hub "
            "checkpoint is configured (`HF_MODEL_REPO_ID`) — running with "
            "**randomly initialized weights**. Predictions below are "
            "meaningless; this mode exists so the dashboard UI is explorable "
            "without a trained model. Run `python src/train.py` locally, or "
            "see `docs/deployment.md` to configure a hosted checkpoint."
        )

    uploaded_file = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])

    if uploaded_file is None:
        st.info("Upload a PNG/JPG chest X-ray to get started.")
        return

    pil_image = Image.open(uploaded_file)
    image_tensor = preprocess_image(pil_image)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(pil_image, caption="Uploaded X-ray", width='stretch')

    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    with col2:
        st.subheader("Predicted probabilities")
        st.plotly_chart(render_probability_chart(classes, probs), width='stretch')

    top_class_idx = int(np.argmax(probs))
    top_class_name = classes[top_class_idx]
    top_prob = float(probs[top_class_idx])

    st.divider()
    st.subheader(f"Explanation for top prediction: **{top_class_name}** ({top_prob:.1%})")

    explain_col1, explain_col2 = st.columns(2)

    gradcam_explainer = ChestXrayExplainer(model, device=device)
    overlay, _heatmap = gradcam_explainer.explain(image_tensor, top_class_idx)

    with explain_col1:
        st.markdown("**Grad-CAM** — which regions drove this prediction")
        st.image(overlay, width='stretch')

    cf_explainer = OcclusionCounterfactualExplainer(model, gradcam_explainer, device=device)
    result = cf_explainer.generate(image_tensor, top_class_idx, top_class_name)

    with explain_col2:
        st.markdown("**Counterfactual** — confidence after masking that region")
        figure = make_side_by_side_figure(result)
        st.image(figure, width='stretch')
        flip_msg = "🔻 Prediction flipped!" if result.flipped else "No flip at 0.5 threshold"
        st.caption(
            f"{result.original_probability:.1%} → {result.counterfactual_probability:.1%}  "
            f"({flip_msg})"
        )


if __name__ == "__main__":
    main()
