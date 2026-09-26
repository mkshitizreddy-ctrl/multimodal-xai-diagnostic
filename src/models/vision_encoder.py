"""
Vision baseline model: DenseNet-121 fine-tuned for multi-label chest X-ray
classification. This is the vision-only baseline the fusion model will later
be compared against.
"""

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

from src.models.attention import CBAM, SEBlock


def build_densenet_backbone(pretrained: bool = True) -> tuple[nn.Module, int]:
    """Builds a DenseNet-121 feature extractor (everything except the final
    classifier). Shared by both the vision-only baseline and the fusion
    model (src/models/fusion.py) so both branches use identical vision
    encoding.

    Returns:
        features: the convolutional feature extractor
        in_features: output embedding dimension (needed to size classifier heads)
    """
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = densenet121(weights=weights)
    return backbone.features, backbone.classifier.in_features


class ChestXrayVisionModel(nn.Module):
    """DenseNet-121 backbone with a multi-label classification head.

    Supports three configurations:
        - Plain DenseNet-121 baseline
        - DenseNet-121 + CBAM
        - DenseNet-121 + SE

    CBAM and SE are mutually exclusive because they are alternative
    attention mechanisms being compared rather than stacked.

    Also exposes `features` and the final conv layer name, which the
    Grad-CAM module (src/explain/gradcam.py) hooks into later.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.2,
        use_cbam: bool = False,
        use_se: bool = False,
    ):
        super().__init__()

        if use_cbam and use_se:
            raise ValueError(
                "use_cbam and use_se are mutually exclusive in this model - they're "
                "alternative attention mechanisms being compared (see "
                "docs/potharaju_comparison.md), not meant to be stacked. Potharaju et "
                "al. 2025 also evaluate them as separate models, not combined."
            )

        self.features, in_features = build_densenet_backbone(pretrained)

        self.use_cbam = use_cbam
        self.use_se = use_se

        self.cbam = CBAM(in_features) if use_cbam else nn.Identity()
        self.se = SEBlock(in_features) if use_se else nn.Identity()

        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        # Name of the layer Grad-CAM should hook into (last conv block).
        # Deliberately kept as the backbone's last conv, not the CBAM/SE output.
        #
        # Attention modules sit between this layer and the classifier, so
        # Grad-CAM's gradients already flow back through the selected
        # attention mechanism. Hooking here gives spatially meaningful
        # activations and keeps the target-layer name valid across plain,
        # CBAM, and SE configurations.
        self.gradcam_target_layer = "features.denseblock4.denselayer16.conv2"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits — apply sigmoid outside for probabilities,
        or use nn.BCEWithLogitsLoss directly during training (more stable).
        """
        feats = self.features(x)
        feats = self.cbam(feats)
        feats = self.se(feats)
        return self.classifier(feats)

    def forward_with_attention_map(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Like forward(), but also returns CBAM's spatial attention map.

        For CBAM models:
            Returns (logits, spatial_attention_map).

        For plain and SE models:
            Returns (logits, None), because SE does not provide a spatial
            attention map suitable for the existing attention-consistency loss.

        This method is primarily used by
        src/train_attention_consistency.py.

        Returns:
            logits: Model output logits.
            attention_map: CBAM spatial attention map, or None when CBAM
                is not enabled.
        """
        feats = self.features(x)

        attention_map = (
            self.cbam.get_spatial_attention_map(feats)
            if self.use_cbam
            else None
        )

        feats = self.cbam(feats)
        feats = self.se(feats)

        logits = self.classifier(feats)

        return logits, attention_map

    def get_target_layer(self) -> nn.Module:
        """Resolve gradcam_target_layer string into the actual module,
        for use with the Grad-CAM explainability module.
        """
        module = self

        for attr in self.gradcam_target_layer.split("."):
            module = getattr(module, attr)

        return module