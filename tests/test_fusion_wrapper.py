"""
Tests for FusionModelImageWrapper. Uses an untrained model and random
input — testing that the adapter correctly bridges the dual-input fusion
model to the single-input interface Grad-CAM/counterfactual expect, and
that the tabular vector is genuinely held fixed (not silently dropped or
leaking a gradient it shouldn't). Semantic correctness of the resulting
heatmaps needs a trained model and real data — verified manually via
generate_fusion_examples.py once a fusion checkpoint exists.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.counterfactual import OcclusionCounterfactualExplainer
from src.explain.fusion_wrapper import FusionModelImageWrapper
from src.explain.gradcam import ChestXrayExplainer
from src.models.fusion import ChestXrayFusionModel

NUM_CLASSES = 1
NUM_TABULAR_FEATURES = 4


def _build_wrapper():
    model = ChestXrayFusionModel(
        num_classes=NUM_CLASSES, num_tabular_features=NUM_TABULAR_FEATURES, pretrained=False
    )
    tabular = torch.randn(NUM_TABULAR_FEATURES)
    return FusionModelImageWrapper(model, tabular), model, tabular


def test_rejects_batched_tabular_input():
    """Constructor should catch a common mistake — passing a [1, N] batch
    instead of a [N] single-sample vector — with a clear error rather than
    a confusing shape mismatch three calls later inside forward()."""
    model = ChestXrayFusionModel(
        num_classes=NUM_CLASSES, num_tabular_features=NUM_TABULAR_FEATURES, pretrained=False
    )
    batched_tabular = torch.randn(1, NUM_TABULAR_FEATURES)
    with pytest.raises(ValueError, match="1-D"):
        FusionModelImageWrapper(model, batched_tabular)


def test_forward_output_shape():
    wrapper, _, _ = _build_wrapper()
    image = torch.randn(2, 3, 224, 224)  # batch of 2
    logits = wrapper(image)
    assert logits.shape == (2, NUM_CLASSES)


def test_forward_matches_direct_fusion_call():
    """The wrapper shouldn't change the model's actual output — same
    image + same tabular through the wrapper should equal calling the
    fusion model directly with that tabular broadcast to the batch.

    Uses .eval() on both paths — without it, the model's dropout layers
    are active and randomly differ between the two calls even for
    identical input, which would fail this comparison for a reason
    that has nothing to do with the wrapper being correct or not.
    """
    wrapper, model, tabular = _build_wrapper()
    wrapper.eval()
    image = torch.randn(3, 3, 224, 224)  # batch of 3

    with torch.no_grad():
        wrapped_out = wrapper(image)
        direct_out = model(image, tabular.unsqueeze(0).expand(3, -1))

    assert torch.allclose(wrapped_out, direct_out, atol=1e-6)


def test_get_target_layer_delegates_to_fusion_model():
    wrapper, model, _ = _build_wrapper()
    assert wrapper.get_target_layer() is model.get_target_layer()


def test_tabular_buffer_is_not_a_trainable_parameter():
    """Fixed at construction, not something a training loop would
    accidentally update — it should show up as a buffer, not in
    wrapper.parameters()."""
    wrapper, _, tabular = _build_wrapper()
    param_names = [name for name, _ in wrapper.named_parameters()]
    assert "fixed_tabular" not in param_names
    assert torch.equal(wrapper.fixed_tabular, tabular)


def test_gradcam_runs_against_wrapped_fusion_model():
    """Seed pinned deliberately — Grad-CAM on a randomly-initialized
    (untrained) network produces an all-zero heatmap for a real but
    unlucky reason on roughly 1 in 10 random seeds (dead gradient at the
    target layer under that particular random init, not a wrapper bug —
    verified separately across 10 seeds). Pinning one known-good seed
    keeps this test deterministic instead of occasionally failing for a
    reason unrelated to what it's actually checking.
    """
    torch.manual_seed(1)
    wrapper, _, _ = _build_wrapper()
    explainer = ChestXrayExplainer(wrapper, device="cpu")
    image = torch.randn(3, 224, 224)

    overlay, heatmap = explainer.explain(image, class_idx=0)

    assert overlay.shape == (224, 224, 3)
    assert heatmap.shape == (224, 224)
    # a genuinely dead/all-zero heatmap would mean gradients aren't
    # actually flowing back through the wrapper into the fusion model
    assert heatmap.std() > 0


def test_counterfactual_runs_against_wrapped_fusion_model():
    torch.manual_seed(1)  # same reasoning as the gradcam test above
    wrapper, _, _ = _build_wrapper()
    gradcam_explainer = ChestXrayExplainer(wrapper, device="cpu")
    cf_explainer = OcclusionCounterfactualExplainer(wrapper, gradcam_explainer, device="cpu")
    image = torch.randn(3, 224, 224)

    result = cf_explainer.generate(image, class_idx=0, class_name="Pneumonia")

    assert 0.0 <= result.original_probability <= 1.0
    assert 0.0 <= result.counterfactual_probability <= 1.0
    assert result.mask.shape == (224, 224)
