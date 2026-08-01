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


def test_load_vision_model_demo_mode_when_no_checkpoint():
    """With no checkpoint present (the state of a fresh clone before
    training), the app should fall back to an untrained model rather
    than crashing."""
    model, classes, device, is_trained = dashboard_app.load_vision_model()

    expected_classes = dashboard_app.load_data_config()["labels"]["classes"]
    assert model is not None
    assert len(classes) == len(expected_classes)
    assert is_trained is False  # no checkpoint exists in this test environment
