"""
Tests for src/models/attention_consistency_loss.py. Pure arithmetic tests
on synthetic attention maps and masks — no model, no GPU, no
torchxrayvision — mirroring the style of
tests/test_measure_lung_localization.py's tests for the closely related
lung_energy_fraction() function (this loss is deliberately its
differentiable, batched, trainable counterpart).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.attention_consistency_loss import attention_consistency_loss


def test_perfect_attention_gives_zero_loss():
    """All attention inside the lung mask -> loss should be ~0."""
    attention_map = torch.zeros(2, 1, 4, 4)
    attention_map[:, :, 1:3, 1:3] = 1.0  # attention concentrated in the center
    lung_mask = torch.ones(2, 1, 4, 4)  # "lungs" cover the whole image

    loss = attention_consistency_loss(attention_map, lung_mask)
    assert loss.item() < 1e-4


def test_attention_entirely_outside_lungs_gives_loss_near_one():
    attention_map = torch.zeros(2, 1, 4, 4)
    attention_map[:, :, 0, 0] = 1.0  # corner, outside the mask below
    lung_mask = torch.zeros(2, 1, 4, 4)
    lung_mask[:, :, 2:, 2:] = 1.0

    loss = attention_consistency_loss(attention_map, lung_mask)
    assert loss.item() > 0.999


def test_half_and_half_gives_loss_near_half():
    attention_map = torch.ones(2, 1, 4, 4)
    lung_mask = torch.zeros(2, 1, 4, 4)
    lung_mask[:, :, :, :2] = 1.0  # left half is "lung"

    loss = attention_consistency_loss(attention_map, lung_mask)
    assert abs(loss.item() - 0.5) < 1e-4


def test_accepts_3d_lung_mask_without_channel_dim():
    """measure_lung_localization.py's get_lung_mask() returns [H,W] per
    image (no channel dim); a batched version of that is [B,H,W], not
    [B,1,H,W] — this loss should handle both without the caller needing
    to remember to unsqueeze."""
    attention_map = torch.rand(3, 1, 5, 5)
    lung_mask_3d = torch.randint(0, 2, (3, 5, 5)).float()

    loss_3d = attention_consistency_loss(attention_map, lung_mask_3d)
    loss_4d = attention_consistency_loss(attention_map, lung_mask_3d.unsqueeze(1))

    assert torch.allclose(loss_3d, loss_4d)


def test_mismatched_spatial_resolution_raises_clear_error():
    """A precomputed lung mask at the wrong resolution (e.g. someone
    accidentally used a different backbone's feature map size) should
    fail loudly and specifically, not silently broadcast-mismatch or
    produce a confusing shape error deep inside a sum() call."""
    attention_map = torch.rand(2, 1, 7, 7)
    wrong_size_mask = torch.rand(2, 1, 14, 14)

    with pytest.raises(ValueError, match="doesn't match"):
        attention_consistency_loss(attention_map, wrong_size_mask)


def test_gradients_flow_back_to_attention_map():
    attention_map = torch.rand(2, 1, 4, 4, requires_grad=True)
    lung_mask = torch.randint(0, 2, (2, 1, 4, 4)).float()

    loss = attention_consistency_loss(attention_map, lung_mask)
    loss.backward()

    assert attention_map.grad is not None
    assert attention_map.grad.abs().sum().item() > 0


def test_degenerate_zero_attention_map_does_not_produce_nan():
    """An all-zero attention map (the same degenerate case
    measure_lung_localization.py guards against with its own eps) should
    not produce NaN and break the training loop — the eps term should
    keep this finite, unlike lung_energy_fraction()'s deliberate NaN
    return (that function reports "couldn't measure this", which makes
    sense for a one-off eval script; a training loss instead needs a
    real, finite gradient signal even in this edge case, since it runs
    every batch of every epoch)."""
    attention_map = torch.zeros(2, 1, 4, 4, requires_grad=True)
    lung_mask = torch.ones(2, 1, 4, 4)

    loss = attention_consistency_loss(attention_map, lung_mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(attention_map.grad).all()
