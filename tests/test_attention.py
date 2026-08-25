"""
Unit tests for src/models/attention.py (CBAM). Separate from
test_vision_model.py's integration-style tests — these check the
module in isolation: shapes, gradient flow, and that it actually
does something (isn't accidentally a no-op).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.attention import CBAM, ChannelAttention, SpatialAttention

BATCH_SIZE = 2
CHANNELS = 32
H, W = 7, 7  # matches DenseNet-121's feature map size at 224x224 input


def test_channel_attention_preserves_shape():
    attn = ChannelAttention(CHANNELS)
    x = torch.randn(BATCH_SIZE, CHANNELS, H, W)
    out = attn(x)
    assert out.shape == x.shape


def test_spatial_attention_preserves_shape():
    attn = SpatialAttention()
    x = torch.randn(BATCH_SIZE, CHANNELS, H, W)
    out = attn(x)
    assert out.shape == x.shape


def test_cbam_preserves_shape():
    cbam = CBAM(CHANNELS)
    x = torch.randn(BATCH_SIZE, CHANNELS, H, W)
    out = cbam(x)
    assert out.shape == x.shape


def test_cbam_is_not_a_no_op():
    """Sanity check against a silent bug where attention weights collapse
    to all-ones and CBAM just passes the input through unchanged."""
    torch.manual_seed(0)
    cbam = CBAM(CHANNELS)
    x = torch.randn(BATCH_SIZE, CHANNELS, H, W)
    out = cbam(x)
    assert not torch.allclose(out, x)


def test_cbam_gradients_flow_to_input():
    cbam = CBAM(CHANNELS)
    x = torch.randn(BATCH_SIZE, CHANNELS, H, W, requires_grad=True)
    out = cbam(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


def test_spatial_attention_map_shape_and_range():
    """get_spatial_attention_map is what we'll actually save as a PNG
    for the Friday demo, so its output shape/range matters more than
    the internals — should be a single-channel map in [0, 1]."""
    cbam = CBAM(CHANNELS)
    x = torch.randn(BATCH_SIZE, CHANNELS, H, W)
    attn_map = cbam.get_spatial_attention_map(x)
    assert attn_map.shape == (BATCH_SIZE, 1, H, W)
    assert attn_map.min() >= 0.0
    assert attn_map.max() <= 1.0
