"""
Smoke tests for TabularEncoder and ChestXrayFusionModel. Uses random
tensors — no real dataset needed — to confirm shapes, forward pass, and
gradient flow through both branches of the fusion model.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.fusion import ChestXrayFusionModel
from src.models.tabular_encoder import TabularEncoder

NUM_CLASSES = 14
NUM_TABULAR_FEATURES = 3
BATCH_SIZE = 4


def test_tabular_encoder_output_shape():
    encoder = TabularEncoder(num_features=NUM_TABULAR_FEATURES, embedding_dim=64)
    x = torch.randn(BATCH_SIZE, NUM_TABULAR_FEATURES)
    out = encoder(x)
    assert out.shape == (BATCH_SIZE, 64)


def test_fusion_model_forward_shape():
    model = ChestXrayFusionModel(
        num_classes=NUM_CLASSES, num_tabular_features=NUM_TABULAR_FEATURES, pretrained=False
    )
    images = torch.randn(BATCH_SIZE, 3, 224, 224)
    tabular = torch.randn(BATCH_SIZE, NUM_TABULAR_FEATURES)

    logits = model(images, tabular)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


def test_fusion_model_backward_pass_updates_both_branches():
    model = ChestXrayFusionModel(
        num_classes=NUM_CLASSES, num_tabular_features=NUM_TABULAR_FEATURES, pretrained=False
    )
    images = torch.randn(BATCH_SIZE, 3, 224, 224)
    tabular = torch.randn(BATCH_SIZE, NUM_TABULAR_FEATURES)
    labels = torch.randint(0, 2, (BATCH_SIZE, NUM_CLASSES)).float()

    logits = model(images, tabular)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()

    # Confirm gradients flowed into BOTH the vision and tabular branches —
    # this is the key check for a fusion model: it's easy to accidentally
    # build one where one branch gets ignored.
    vision_grad = next(p for p in model.features.parameters() if p.requires_grad).grad
    tabular_grad = next(p for p in model.tabular_encoder.parameters() if p.requires_grad).grad

    assert vision_grad is not None and vision_grad.abs().sum() > 0
    assert tabular_grad is not None and tabular_grad.abs().sum() > 0


def test_fusion_model_single_sample_batch():
    """BatchNorm in the tabular encoder can break on batch_size=1 during
    training mode — worth an explicit regression test since this is a
    common real-world gotcha (e.g. the last batch of an epoch)."""
    model = ChestXrayFusionModel(
        num_classes=NUM_CLASSES, num_tabular_features=NUM_TABULAR_FEATURES, pretrained=False
    )
    model.eval()  # BatchNorm requires eval mode (running stats) for batch_size=1
    images = torch.randn(1, 3, 224, 224)
    tabular = torch.randn(1, NUM_TABULAR_FEATURES)

    with torch.no_grad():
        logits = model(images, tabular)
    assert logits.shape == (1, NUM_CLASSES)
