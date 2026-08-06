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
