"""
Multimodal fusion model: DenseNet-121 vision branch + MLP tabular branch,
combined via late fusion (concatenate embeddings -> shared classifier head).

This is compared against the vision-only baseline (src/models/vision_encoder.py)
in the ablation study — see notebooks/02_fusion_ablation_results.ipynb.
"""

import torch
import torch.nn as nn

from src.models.tabular_encoder import TabularEncoder
from src.models.vision_encoder import build_densenet_backbone


class ChestXrayFusionModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_tabular_features: int,
        pretrained: bool = True,
        tabular_embedding_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.features, vision_dim = build_densenet_backbone(pretrained)
        self.vision_pool = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.tabular_encoder = TabularEncoder(
            num_features=num_tabular_features,
            embedding_dim=tabular_embedding_dim,
            dropout=dropout,
        )

        fused_dim = vision_dim + tabular_embedding_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Same target layer name as the vision-only model, so Grad-CAM
        # could in principle be pointed at the fusion model's vision branch too.
        self.gradcam_target_layer = "features.denseblock4.denselayer16.conv2"

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        vision_feats = self.vision_pool(self.features(image))
        tabular_feats = self.tabular_encoder(tabular)
        fused = torch.cat([vision_feats, tabular_feats], dim=1)
        return self.classifier(fused)

    def get_target_layer(self) -> nn.Module:
        module = self
        for attr in self.gradcam_target_layer.split("."):
            module = getattr(module, attr)
        return module
