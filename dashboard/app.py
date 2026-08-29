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

import base64
import io
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

from src.explain.counterfactual import (
    OcclusionCounterfactualExplainer,
    make_side_by_side_figure,
)
from src.explain.gradcam import ChestXrayExplainer
from src.models.vision_encoder import ChestXrayVisionModel

st.set_page_config(page_title="Explainable Chest X-ray Diagnosis", layout="wide")

# Palette lives here as the single source of truth (also referenced in
# .streamlit/config.toml for Streamlit's own native widgets). Kept as a
# dict rather than scattered hex literals so the Grad-CAM/probability-bar
# colors below stay in sync with the CSS if this ever changes.
PALETTE = {
    "bg": "#0B0D0F",
    "panel": "#15181C",
    "text": "#E7E9EC",
    "muted": "#8B92A0",
    "accent": "#F0A83C",  # amber - interactive, in-range predictions
    "finding": "#E4483C",  # clinical red - reserved for an actual positive finding
    "hairline": "#22262C",
}


def _inject_theme_css():
    """Custom CSS on top of the base Streamlit dark theme (config.toml) -
    dark reading-room palette, IBM Plex Mono for technical readouts /
    Inter for prose, and the corner-bracket 'viewport' framing used
    throughout (see docs/architecture.md#dashboard-design for why)."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"], .stMarkdown, p, span, label {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background-color: {PALETTE["bg"]};
        }}

        /* Technical readout bar - model/checkpoint metadata, styled like
        a PACS viewer's study header rather than a generic app title. */
        .study-header {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            color: {PALETTE["muted"]};
            background: {PALETTE["panel"]};
            border: 1px solid {PALETTE["hairline"]};
            border-bottom: 2px solid {PALETTE["accent"]};
            padding: 10px 18px;
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px 24px;
            letter-spacing: 0.03em;
            margin-bottom: 1.6rem;
        }}
        .study-header .field .k {{
            color: {PALETTE["muted"]};
        }}
        .study-header .field .v {{
            color: {PALETTE["accent"]};
            font-weight: 600;
        }}

        h1.app-title {{
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            font-size: 1.7rem;
            color: {PALETTE["text"]};
            margin-bottom: 0.2rem;
        }}

        /* Corner-bracket image framing - the signature element. Every
        image the app shows (upload, Grad-CAM, counterfactual) goes
        through render_viewport() below so this stays consistent. */
        .viewport {{
            position: relative;
            background: #000;
            padding: 18px;
            border: 1px solid {PALETTE["hairline"]};
        }}
        .viewport img {{
            width: 100%;
            display: block;
        }}
        .viewport .corner {{
            position: absolute;
            width: 20px;
            height: 20px;
            border-color: {PALETTE["accent"]};
            border-style: solid;
            border-width: 0;
        }}
        .viewport .corner.tl {{ top: 5px; left: 5px; border-top-width: 2px; border-left-width: 2px; }}
        .viewport .corner.tr {{ top: 5px; right: 5px; border-top-width: 2px; border-right-width: 2px; }}
        .viewport .corner.bl {{ bottom: 5px; left: 5px; border-bottom-width: 2px; border-left-width: 2px; }}
        .viewport .corner.br {{ bottom: 5px; right: 5px; border-bottom-width: 2px; border-right-width: 2px; }}

        .viewport-caption {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: {PALETTE["muted"]};
            text-transform: uppercase;
            letter-spacing: 0.09em;
            margin-top: 8px;
        }}
        .viewport-caption .metric {{
            color: {PALETTE["accent"]};
        }}

        /* Section labels, mono like the study header, to keep technical
        readouts visually distinct from explanatory prose. */
        .section-label {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            color: {PALETTE["muted"]};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            border-bottom: 1px solid {PALETTE["hairline"]};
            padding-bottom: 6px;
            margin-bottom: 12px;
        }}

        div[data-testid="stFileUploader"] section {{
            background-color: {PALETTE["panel"]};
            border: 1px dashed {PALETTE["hairline"]};
        }}

        div[data-testid="stAlert"] {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _pil_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_viewport(image, caption: str, metric: str | None = None):
    """Renders an image inside the corner-bracket 'viewport' frame, with
    a mono caption underneath. `image` can be a PIL Image or a numpy
    array (uint8 HxWx3) - both come up in this file (uploaded PIL image
    vs. the Grad-CAM/counterfactual overlays, which are numpy arrays)."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    b64 = _pil_to_base64(image)
    metric_html = f' <span class="metric">{metric}</span>' if metric else ""
    st.markdown(
        f"""
        <div class="viewport">
            <span class="corner tl"></span><span class="corner tr"></span>
            <span class="corner bl"></span><span class="corner br"></span>
            <img src="data:image/png;base64,{b64}" />
        </div>
        <div class="viewport-caption">{caption}{metric_html}</div>
        """,
        unsafe_allow_html=True,
    )

VISION_CHECKPOINT = "checkpoints/vision_baseline/best_model.pth"
FUSION_CHECKPOINT = "checkpoints/fusion/best_model.pth"
DATA_CONFIG_PATH = "configs/data.yaml"
IMAGE_SIZE = 224
NORM_MEAN, NORM_STD = 0.5, 0.25

# Default HF Hub model repo hosting the trained checkpoint (see
# docs/deployment.md for how it got there). Env var / Streamlit secret
# below still take priority if set, so this can be overridden per-deploy
# without editing code — e.g. for a fork pointing at someone else's weights.
DEFAULT_HF_MODEL_REPO_ID = "Kshitiz151/multimodal-xai-diagnostic-weights"


def _get_hf_model_repo_id() -> str | None:
    """Reads HF_MODEL_REPO_ID from an env var first, then a Streamlit
    secret if configured — safely, since st.secrets raises (rather than
    returning None) when no secrets.toml file exists at all, which is the
    normal case for local runs and CI. Falls back to
    DEFAULT_HF_MODEL_REPO_ID rather than None, so the deployed app works
    out of the box without needing a secret set."""
    env_value = os.environ.get("HF_MODEL_REPO_ID")
    if env_value:
        return env_value
    try:
        secret_value = st.secrets.get("HF_MODEL_REPO_ID")
        if secret_value:
            return secret_value
    except Exception:
        pass
    return DEFAULT_HF_MODEL_REPO_ID


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

    checkpoint_path = (
        VISION_CHECKPOINT if Path(VISION_CHECKPOINT).exists() else _download_checkpoint_from_hf_hub()
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = ChestXrayVisionModel(
            num_classes=len(checkpoint["classes"]),
            pretrained=False,
            use_cbam=checkpoint.get("use_cbam", False),
        )
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
            marker_color=[
                PALETTE["finding"] if p >= 0.5 else PALETTE["accent"] for p in probs[order]
            ],
        )
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color=PALETTE["muted"])
    fig.update_layout(
        xaxis_title="Predicted probability",
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        font=dict(family="IBM Plex Mono, monospace", color=PALETTE["muted"], size=12),
        xaxis=dict(gridcolor=PALETTE["hairline"], zerolinecolor=PALETTE["hairline"]),
        yaxis=dict(gridcolor=PALETTE["hairline"]),
    )
    return fig


def main():
    _inject_theme_css()

    model, classes, device, is_trained = load_vision_model()

    checkpoint_status = "trained checkpoint" if is_trained else "random weights (demo mode)"
    cbam_status = "on" if getattr(model, "use_cbam", False) else "off"

    st.markdown(
        f"""
        <div class="study-header">
            <div class="field"><span class="k">MODEL&nbsp;</span><span class="v">densenet121</span></div>
            <div class="field"><span class="k">CBAM&nbsp;</span><span class="v">{cbam_status}</span></div>
            <div class="field"><span class="k">WEIGHTS&nbsp;</span><span class="v">{checkpoint_status}</span></div>
            <div class="field"><span class="k">CLASSES&nbsp;</span><span class="v">{len(classes)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h1 class="app-title">Explainable Chest X-ray Diagnosis</h1>', unsafe_allow_html=True)
    st.caption(
        "Research/portfolio prototype — not validated for clinical use. "
        "See the README for dataset, architecture, and limitations."
    )

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
        render_viewport(
            pil_image.convert("RGB"),
            "uploaded study",
            metric=f"{pil_image.size[0]}×{pil_image.size[1]}",
        )

    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    with col2:
        st.markdown('<div class="section-label">Predicted probabilities</div>', unsafe_allow_html=True)
        st.plotly_chart(render_probability_chart(classes, probs), width="stretch")

    top_class_idx = int(np.argmax(probs))
    top_class_name = classes[top_class_idx]
    top_prob = float(probs[top_class_idx])

    st.divider()
    st.markdown(
        f'<div class="section-label">Explanation for top prediction: '
        f'<span style="color:{PALETTE["accent"]}">{top_class_name}</span> ({top_prob:.1%})</div>',
        unsafe_allow_html=True,
    )

    explain_col1, explain_col2 = st.columns(2)

    gradcam_explainer = ChestXrayExplainer(model, device=device)
    overlay, _heatmap = gradcam_explainer.explain(image_tensor, top_class_idx)

    with explain_col1:
        render_viewport(overlay, "grad-cam — regions driving this prediction")

    cf_explainer = OcclusionCounterfactualExplainer(model, gradcam_explainer, device=device)
    result = cf_explainer.generate(image_tensor, top_class_idx, top_class_name)

    with explain_col2:
        figure = make_side_by_side_figure(result)
        flip_msg = "flipped" if result.flipped else "no flip @ 0.5"
        render_viewport(
            figure,
            "counterfactual — confidence after masking top region",
            metric=f"{result.original_probability:.0%} → {result.counterfactual_probability:.0%} ({flip_msg})",
        )


if __name__ == "__main__":
    main()
