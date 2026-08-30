"""
Vision baseline model: DenseNet-121 fine-tuned for multi-label chest X-ray
classification. This is the vision-only baseline the fusion model will later
be compared against.
"""

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

from src.models.attention import CBAM


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

    Also exposes `features` and the final conv layer name, which the
    Grad-CAM module (src/explain/gradcam.py) hooks into later.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.2,
        use_cbam: bool = False,
    ):
        super().__init__()

        self.features, in_features = build_densenet_backbone(pretrained)
        self.use_cbam = use_cbam
        self.cbam = CBAM(in_features) if use_cbam else nn.Identity()

        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        # Name of the layer Grad-CAM should hook into (last conv block).
        # Deliberately kept as the backbone's last conv, not the CBAM output —
        # CBAM sits between this layer and the classifier, so Grad-CAM's
        # gradients already flow back through it. Hooking here still gives
        # spatially meaningful activations either way, and keeps the
        # target-layer name valid whether or not use_cbam is set (matters
        # for comparing CBAM vs. no-CBAM checkpoints with the same script).
        self.gradcam_target_layer = "features.denseblock4.denselayer16.conv2"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits — apply sigmoid outside for probabilities,
        or use nn.BCEWithLogitsLoss directly during training (more stable).
        """
        feats = self.features(x)
        feats = self.cbam(feats)
        return self.classifier(feats)

    def forward_with_attention_map(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Like forward(), but also returns CBAM's own spatial attention
        map (shape [B,1,7,7] for 224x224 input) — used by
        src/train_attention_consistency.py to regularize the model's
        attention toward the segmented lung field, not just measure it
        after the fact like measure_lung_localization.py does. Returns
        (logits, None) when use_cbam=False, since there's no attention map
        to regularize in that case — callers should skip the consistency
        loss term when this happens, not treat it as an error.

        Duplicates the channel-attention computation once (also done
        inside self.cbam(feats) below) — a minor inefficiency, acceptable
        for training-time overhead, kept separate from forward() so the
        normal inference path stays untouched and exactly as fast as before.
        """
        feats = self.features(x)
        attention_map = self.cbam.get_spatial_attention_map(feats) if self.use_cbam else None
        feats = self.cbam(feats)
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
