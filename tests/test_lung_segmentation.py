"""
Tests for the lung segmentation module and its integration into Grad-CAM.
Uses an untrained model and synthetic input — verifying the mechanics
(shapes, masking behavior) rather than real anatomical accuracy, which
requires a genuine chest X-ray to evaluate.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.gradcam import ChestXrayExplainer
from src.explain.lung_segmentation import get_lung_mask
from src.models.vision_encoder import ChestXrayVisionModel

NUM_CLASSES = 14


def test_get_lung_mask_returns_correct_shape_and_dtype():
    image = torch.randn(3, 224, 224) * 0.3
    mask = get_lung_mask(image, output_size=224)

    assert mask.shape == (224, 224)
    assert mask.dtype == np.float32
    assert set(np.unique(mask)).issubset({0.0, 1.0})


def test_get_lung_mask_resizes_to_requested_output_size():
    image = torch.randn(3, 224, 224) * 0.3
    mask = get_lung_mask(image, output_size=64)
    assert mask.shape == (64, 64)


def test_explain_without_lung_restriction_is_unaffected():
    """Default behavior (restrict_to_lungs=False) must be byte-for-byte
    identical to before this feature was added — no regression for
    existing callers."""
    model = ChestXrayVisionModel(num_classes=NUM_CLASSES, pretrained=False)
    explainer = ChestXrayExplainer(model, device="cpu")
    image = torch.randn(3, 224, 224)

    torch.manual_seed(0)
    _, heatmap_a = explainer.explain(image, class_idx=0, restrict_to_lungs=False)
    torch.manual_seed(0)
    _, heatmap_b = explainer.explain(image, class_idx=0)  # restrict_to_lungs defaults to False

    assert np.allclose(heatmap_a, heatmap_b)


def test_explain_with_lung_restriction_zeroes_activation_outside_mask():
    model = ChestXrayVisionModel(num_classes=NUM_CLASSES, pretrained=False)
    explainer = ChestXrayExplainer(model, device="cpu")
    image = torch.randn(3, 224, 224)

    overlay, heatmap = explainer.explain(image, class_idx=0, restrict_to_lungs=True)

    lung_mask = get_lung_mask(image, output_size=heatmap.shape[0])
    # Everywhere the lung mask is 0, the constrained heatmap must also be 0
    assert np.all(heatmap[lung_mask == 0] == 0)
    assert overlay.shape == (224, 224, 3)


def test_explain_top_k_passes_restrict_to_lungs_through():
    model = ChestXrayVisionModel(num_classes=NUM_CLASSES, pretrained=False)
    explainer = ChestXrayExplainer(model, device="cpu")
    image = torch.randn(3, 224, 224)
    classes = [f"class_{i}" for i in range(NUM_CLASSES)]

    results = explainer.explain_top_k(image, classes, k=2, restrict_to_lungs=True)

    lung_mask = get_lung_mask(image, output_size=results[0]["heatmap"].shape[0])
    for r in results:
        assert np.all(r["heatmap"][lung_mask == 0] == 0)
