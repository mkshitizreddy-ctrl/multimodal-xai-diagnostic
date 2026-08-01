"""
Tests for OcclusionCounterfactualExplainer. Uses an untrained model and
random input — verifies the pipeline (heatmap -> mask -> inpaint ->
re-predict) runs correctly and produces valid, correctly-shaped output.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.counterfactual import (
    CounterfactualResult,
    OcclusionCounterfactualExplainer,
    heatmap_to_mask,
    make_side_by_side_figure,
    uint8_to_normalized_tensor,
)
from src.explain.gradcam import ChestXrayExplainer, denormalize_to_rgb
from src.models.vision_encoder import ChestXrayVisionModel

NUM_CLASSES = 14


def _build_explainers():
    model = ChestXrayVisionModel(num_classes=NUM_CLASSES, pretrained=False)
    gradcam_explainer = ChestXrayExplainer(model, device="cpu")
    cf_explainer = OcclusionCounterfactualExplainer(model, gradcam_explainer, device="cpu")
    return cf_explainer


def test_heatmap_to_mask_binary():
    heatmap = np.random.rand(224, 224).astype(np.float32)
    mask = heatmap_to_mask(heatmap, threshold=0.6, dilate_px=0)
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask.shape == (224, 224)


def test_roundtrip_normalization_is_stable():
    # Simulate a real image: valid pixel values in [0,1], then normalized —
    # matching what ChestXrayDataset actually produces. (Using raw
    # torch.randn() here would generate normalized values outside the range
    # any real image can produce, which the intentional clamp in
    # denormalize_to_rgb would legitimately clip — not a bug, just an unfair
    # test input.)
    pixels = torch.rand(3, 224, 224)  # valid [0, 1] pixel values
    image = (pixels - 0.5) / 0.25     # matches ChestXrayDataset's Normalize

    rgb = denormalize_to_rgb(image)
    uint8_img = (rgb * 255).astype(np.uint8)
    recovered = uint8_to_normalized_tensor(uint8_img)

    assert recovered.shape == image.shape
    assert torch.allclose(recovered, image, atol=0.05)


def test_generate_returns_valid_result():
    cf_explainer = _build_explainers()
    image = torch.randn(3, 224, 224)

    result = cf_explainer.generate(image, class_idx=0, class_name="TestClass")

    assert isinstance(result, CounterfactualResult)
    assert 0.0 <= result.original_probability <= 1.0
    assert 0.0 <= result.counterfactual_probability <= 1.0
    assert result.original_image.shape == (224, 224, 3)
    assert result.counterfactual_image.shape == (224, 224, 3)
    assert result.mask.shape == (224, 224)


def test_probability_drop_is_consistent():
    cf_explainer = _build_explainers()
    image = torch.randn(3, 224, 224)
    result = cf_explainer.generate(image, class_idx=1, class_name="TestClass")

    expected_drop = result.original_probability - result.counterfactual_probability
    assert abs(result.probability_drop - expected_drop) < 1e-6


def test_side_by_side_figure_shape():
    cf_explainer = _build_explainers()
    image = torch.randn(3, 224, 224)
    result = cf_explainer.generate(image, class_idx=2, class_name="TestClass")

    figure = make_side_by_side_figure(result)
    # 3 images side by side (224*3 wide) + caption strip below
    assert figure.shape[1] == 224 * 3
    assert figure.shape[0] == 224 + 40
