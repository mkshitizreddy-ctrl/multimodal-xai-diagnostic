"""
Adapts ChestXrayFusionModel's dual-input forward(image, tabular) to the
single-input forward(image) interface that src/explain/gradcam.py and
src/explain/counterfactual.py both assume (pytorch_grad_cam's GradCAM calls
`model(input_tensor)` internally with one positional argument, and
OcclusionCounterfactualExplainer does the same).

Design: the tabular vector is fixed at construction time and held as a
registered buffer, not a parameter — it's the *patient's* vitals for this
one explanation, not something we want a gradient for or that should move
under optimization. Grad-CAM only ever needs image-space gradients (it's
inherently a "which pixels mattered" method), so fixing tabular this way
loses nothing for that purpose while letting both existing, already-tested
explainability modules run against the fusion model completely unmodified —
no changes needed to gradcam.py or counterfactual.py themselves.

Usage:
    wrapper = FusionModelImageWrapper(fusion_model, tabular_tensor)
    explainer = ChestXrayExplainer(wrapper, device="cuda")
    overlay, heatmap = explainer.explain(image_tensor, class_idx=0)
"""

import torch
import torch.nn as nn


class FusionModelImageWrapper(nn.Module):
    def __init__(self, fusion_model: nn.Module, tabular_tensor: torch.Tensor):
        """
        Args:
            fusion_model: a ChestXrayFusionModel instance (or anything with
                the same forward(image, tabular) -> logits signature and a
                get_target_layer() method).
            tabular_tensor: 1-D tensor, shape [num_tabular_features] — the
                single patient's vitals to hold fixed for this explanation.
                Unsqueezed and expanded to match the image batch size inside
                forward(), so callers just pass the per-sample vector as-is
                (matching what ChestXrayDataset.__getitem__ returns).
        """
        super().__init__()
        if tabular_tensor.dim() != 1:
            raise ValueError(
                f"tabular_tensor must be 1-D [num_features], got shape {tuple(tabular_tensor.shape)}. "
                "Pass a single sample's tabular vector, not a batch."
            )
        self.fusion_model = fusion_model
        self.register_buffer("fixed_tabular", tabular_tensor)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]
        tabular = self.fixed_tabular.unsqueeze(0).expand(batch_size, -1)
        return self.fusion_model(image, tabular)

    def get_target_layer(self) -> nn.Module:
        """Delegates to the wrapped fusion model — both gradcam.py and
        counterfactual.py call this on whatever model they're given, so
        the wrapper needs to expose it too, not just forward()."""
        return self.fusion_model.get_target_layer()
