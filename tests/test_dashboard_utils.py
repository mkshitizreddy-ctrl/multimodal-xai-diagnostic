"""
Tests for the pure-logic helper functions in dashboard/app.py
(preprocessing and chart building). Doesn't test Streamlit's rendering
itself — that's covered by manually running `streamlit run dashboard/app.py`
and confirming it boots (see README) — just the functions that don't
depend on Streamlit's script-run context.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dashboard.app as dashboard_app


def test_preprocess_image_shape_and_normalization():
    fake_image = Image.fromarray(
        (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
    )
    tensor = dashboard_app.preprocess_image(fake_image)

    assert tensor.shape == (3, dashboard_app.IMAGE_SIZE, dashboard_app.IMAGE_SIZE)
    assert isinstance(tensor, torch.Tensor)


def test_preprocess_image_handles_grayscale_input():
    fake_image = Image.fromarray((np.random.rand(224, 224) * 255).astype(np.uint8), mode="L")
    tensor = dashboard_app.preprocess_image(fake_image)
    assert tensor.shape == (3, dashboard_app.IMAGE_SIZE, dashboard_app.IMAGE_SIZE)


def test_render_probability_chart_returns_figure():
    classes = [f"class_{i}" for i in range(14)]
    probs = np.random.rand(14)

    fig = dashboard_app.render_probability_chart(classes, probs)

    assert fig.data[0].x.tolist() == sorted(probs.tolist())


def test_load_vision_model_demo_mode_when_no_checkpoint(monkeypatch):
    """With no checkpoint present (the state of a fresh clone before
    training), the app should fall back to an untrained model rather
    than crashing.

    Uses monkeypatch to force this regardless of the actual filesystem
    state — on a machine that HAS actually trained a real checkpoint
    (e.g. after following the training guide), the real checkpoint would
    legitimately be found and is_trained would be True, which isn't a bug,
    just a different (better!) environment than a fresh clone. Testing the
    fallback behavior shouldn't depend on which state the machine is in.
    """
    monkeypatch.setattr(dashboard_app, "VISION_CHECKPOINT", "checkpoints/does_not_exist.pth")
    dashboard_app.load_vision_model.clear()  # bypass @st.cache_resource across test runs

    model, classes, device, is_trained = dashboard_app.load_vision_model()

    expected_classes = dashboard_app.load_data_config()["labels"]["classes"]
    assert model is not None
    assert len(classes) == len(expected_classes)
    assert is_trained is False


def test_load_vision_model_loads_real_checkpoint_when_present(monkeypatch, tmp_path):
    """Complementary test: when a checkpoint genuinely exists (e.g. after
    real training), it should actually be loaded and is_trained should be
    True — locks in the behavior your machine just demonstrated."""
    import torch

    from src.models.vision_encoder import ChestXrayVisionModel

    classes = dashboard_app.load_data_config()["labels"]["classes"]
    fake_checkpoint_path = tmp_path / "fake_checkpoint.pth"
    fake_model = ChestXrayVisionModel(num_classes=len(classes), pretrained=False)
    torch.save({"model_state_dict": fake_model.state_dict(), "classes": classes}, fake_checkpoint_path)

    monkeypatch.setattr(dashboard_app, "VISION_CHECKPOINT", str(fake_checkpoint_path))
    dashboard_app.load_vision_model.clear()

    model, loaded_classes, device, is_trained = dashboard_app.load_vision_model()

    assert is_trained is True
    assert loaded_classes == classes
