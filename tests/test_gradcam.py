"""
Tests for ChestXrayExplainer. Uses an untrained model and random input —
we're testing that Grad-CAM produces correctly shaped, valid output, not
that the heatmaps are semantically meaningful (that requires a trained
model and real data, verified manually via generate_examples.py).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.gradcam import ChestXrayExplainer, denormalize_to_rgb
from src.models.vision_encoder import ChestXrayVisionModel

NUM_CLASSES = 14
CLASSES = [f"class_{i}" for i in range(NUM_CLASSES)]


def _build_explainer():
    model = ChestXrayVisionModel(num_classes=NUM_CLASSES, pretrained=False)
    return ChestXrayExplainer(model, device="cpu")


def test_denormalize_range():
    image = torch.randn(3, 224, 224)
    rgb = denormalize_to_rgb(image)
    assert rgb.shape == (224, 224, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_explain_output_shapes():
    explainer = _build_explainer()
    image = torch.randn(3, 224, 224)

    overlay, heatmap = explainer.explain(image, class_idx=0)

    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-5


def test_explain_top_k_returns_k_results():
    explainer = _build_explainer()
    image = torch.randn(3, 224, 224)

    results = explainer.explain_top_k(image, CLASSES, k=3)

    assert len(results) == 3
    for r in results:
        assert r["class"] in CLASSES
        assert 0.0 <= r["probability"] <= 1.0
        assert r["overlay"].shape == (224, 224, 3)


def test_explain_top_k_sorted_by_probability():
    explainer = _build_explainer()
    image = torch.randn(3, 224, 224)

    results = explainer.explain_top_k(image, CLASSES, k=5)
    probs = [r["probability"] for r in results]

    assert probs == sorted(probs, reverse=True)
