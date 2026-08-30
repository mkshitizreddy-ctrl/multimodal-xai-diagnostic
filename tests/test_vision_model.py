"""
Smoke tests for ChestXrayVisionModel. Doesn't require the real dataset —
just confirms the model builds, runs a forward pass, and is trainable
(loss backpropagates without shape errors).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.vision_encoder import ChestXrayVisionModel

NUM_CLASSES = 14
BATCH_SIZE = 2


def test_forward_pass_output_shape():
    model = ChestXrayVisionModel(num_classes=NUM_CLASSES, pretrained=False)
    x = torch.randn(BATCH_SIZE, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


def test_backward_pass_runs():
    model = ChestXrayVisionModel(num_classes=NUM_CLASSES, pretrained=False)
    x = torch.randn(BATCH_SIZE, 3, 224, 224)
    labels = torch.randint(0, 2, (BATCH_SIZE, NUM_CLASSES)).float()

    logits = model(x)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()

    # Confirm gradients actually flowed into the classifier head
    grad_norms = [p.grad.norm().item() for p in model.classifier.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g >= 0 for g in grad_norms)


def test_gradcam_target_layer_resolves():
    model = ChestXrayVisionModel(num_classes=NUM_CLASSES, pretrained=False)
    layer = model.get_target_layer()
    assert isinstance(layer, nn.Module)


def test_cbam_forward_pass_output_shape():
    """use_cbam=True shouldn't change the output shape, just add attention
    in the middle of the forward pass."""
    model = ChestXrayVisionModel(num_classes=NUM_CLASSES, pretrained=False, use_cbam=True)
    x = torch.randn(BATCH_SIZE, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


def test_cbam_backward_pass_runs():
    model = ChestXrayVisionModel(num_classes=NUM_CLASSES, pretrained=False, use_cbam=True)
    x = torch.randn(BATCH_SIZE, 3, 224, 224)
    labels = torch.randint(0, 2, (BATCH_SIZE, NUM_CLASSES)).float()

    logits = model(x)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()

    # Confirm gradients flowed into the CBAM module itself, not just
    # around it — otherwise it'd be dead weight sitting in the graph.
    grad_norms = [p.grad.norm().item() for p in model.cbam.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g >= 0 for g in grad_norms)


def test_cbam_off_by_default_is_identity():
    """use_cbam=False (the old default) should still produce a plain
    nn.Identity so existing no-CBAM checkpoints keep loading correctly."""
    model = ChestXrayVisionModel(num_classes=NUM_CLASSES, pretrained=False)
    assert isinstance(model.cbam, nn.Identity)
    assert list(model.cbam.parameters()) == []


def test_forward_with_attention_map_matches_plain_forward():
    """forward_with_attention_map() must be a strict superset of
    forward() — same logits, just with the attention map alongside. If
    these diverged, the training script using this method would be
    optimizing a different model path than what actually gets deployed."""
    model = ChestXrayVisionModel(num_classes=NUM_CLASSES, pretrained=False, use_cbam=True)
    model.eval()
    x = torch.randn(BATCH_SIZE, 3, 224, 224)

    with torch.no_grad():
        plain_logits = model(x)
        augmented_logits, attention_map = model.forward_with_attention_map(x)

    assert torch.allclose(plain_logits, augmented_logits, atol=1e-6)
    assert attention_map.shape == (BATCH_SIZE, 1, 7, 7)
    assert attention_map.min() >= 0.0 and attention_map.max() <= 1.0


def test_forward_with_attention_map_returns_none_when_cbam_off():
    model = ChestXrayVisionModel(num_classes=NUM_CLASSES, pretrained=False, use_cbam=False)
    x = torch.randn(BATCH_SIZE, 3, 224, 224)
    _logits, attention_map = model.forward_with_attention_map(x)
    assert attention_map is None


def test_forward_with_attention_map_gradients_flow_to_cbam():
    model = ChestXrayVisionModel(num_classes=NUM_CLASSES, pretrained=False, use_cbam=True)
    x = torch.randn(BATCH_SIZE, 3, 224, 224)
    logits, attention_map = model.forward_with_attention_map(x)

    # A loss that only depends on the attention map (not the classification
    # logits at all) should still be able to backprop into CBAM's
    # parameters — this is the actual usage pattern in
    # src/train_attention_consistency.py, where the consistency loss is
    # computed purely from the attention map against a lung mask.
    fake_lung_mask = torch.ones_like(attention_map)
    consistency_loss = ((attention_map - fake_lung_mask) ** 2).mean()
    consistency_loss.backward()

    grad_norms = [p.grad.norm().item() for p in model.cbam.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert any(g > 0 for g in grad_norms)
