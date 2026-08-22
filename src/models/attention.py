"""
CBAM (Convolutional Block Attention Module) — Woo et al., ECCV 2018,
https://arxiv.org/abs/1807.06521

Added on top of the DenseNet-121 backbone based on the pneumonia-CXR
literature review (docs/paper_notes.md, docs/architecture.md#attention-module):
several 2024-2026 papers report CBAM on this exact backbone + dataset
combination improving both classification metrics and Grad-CAM localization
quality (heatmaps concentrate more tightly on pathological lung regions
instead of spreading across the whole image). That second point is what
makes it worth trying here specifically — it's a direct extension of the
shortcut-learning finding in docs/ethics_statement.md, not just an
accuracy play.

Two sub-modules, applied in sequence (channel attention, then spatial):
  - Channel attention: "which feature channels matter" — squeeze each
    channel via both avg-pool and max-pool, run through a shared MLP,
    sigmoid, and rescale the channels.
  - Spatial attention: "which pixels matter" — pool across channels
    (avg + max), 7x7 conv, sigmoid, rescale spatially.

Reference: Woo, S., Park, J., Lee, J.Y., Kweon, I.S. "CBAM: Convolutional
Block Attention Module." ECCV 2018. https://arxiv.org/abs/1807.06521
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(in_channels // reduction_ratio, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        scale = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        scale = torch.sigmoid(self.conv(pooled))
        return x * scale


class CBAM(nn.Module):
    """Applies channel attention, then spatial attention, in sequence
    (the ordering CBAM's original ablation study found best)."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

    def get_spatial_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Returns just the spatial attention map (post channel-attention,
        pre spatial-rescale) — useful for visualizing what CBAM itself is
        attending to, separately from the Grad-CAM heatmap. Shape: [B,1,H,W].
        """
        x = self.channel_attention(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.spatial_attention.conv(pooled))
