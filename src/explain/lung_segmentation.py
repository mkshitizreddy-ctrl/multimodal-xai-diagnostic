"""
Lung segmentation, used to constrain Grad-CAM (and, downstream, the
occlusion-based counterfactual module) to only activate inside the lung
fields — directly addressing the shortcut-learning evidence documented in
docs/ethics_statement.md, where some Grad-CAM outputs showed activation on
shoulders, image borders, or burned-in annotations rather than lung tissue.

Uses the pretrained PSPNet segmentation model from torchxrayvision
(https://github.com/mlmed/torchxrayvision), trained specifically on chest
X-rays to output per-pixel probabilities for 14 anatomical structures,
including left/right lung.
"""

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

SEGMENTATION_INPUT_SIZE = 512  # torchxrayvision's PSPNet expects 512x512 input
NORM_MEAN = 0.5
NORM_STD = 0.25


@lru_cache(maxsize=1)
def _load_segmentation_model():
    """Cached singleton loader — the model is ~100MB and downloaded once
    on first use, then reused for every subsequent call in the process."""
    import torchxrayvision as xrv

    model = xrv.baseline_models.chestx_det.PSPNet()
    model.eval()
    return model


def get_lung_mask(image_tensor: torch.Tensor, output_size: int, threshold: float = 0.3) -> np.ndarray:
    """
    Args:
        image_tensor: normalized image, shape [3, H, W], as produced by
            ChestXrayDataset (mean=0.5, std=0.25). Only one channel is used
            since the dataset replicates grayscale to 3 identical channels.
        output_size: the mask is resized to (output_size, output_size) to
            match the resolution of the Grad-CAM heatmap it will constrain.
        threshold: per-pixel probability threshold (from the segmentation
            model's sigmoid output) for a pixel to count as "lung".

    Returns:
        Binary mask, shape (output_size, output_size), dtype float32,
        values in {0.0, 1.0}. 1.0 = inside left or right lung.
    """
    import torchxrayvision as xrv

    model = _load_segmentation_model()

    # Denormalize back to a raw [0, 255]-ish single-channel image, then
    # apply torchxrayvision's own normalization convention (roughly
    # [-1024, 1024]) — required for their pretrained model to behave as
    # trained; skipping this step produces degenerate output.
    single_channel = image_tensor[0].detach().cpu().numpy()  # [H, W], in original normalized space
    pixel_0_255 = np.clip((single_channel * NORM_STD + NORM_MEAN) * 255.0, 0, 255).astype(np.float32)
    xrv_normalized = xrv.datasets.normalize(pixel_0_255, 255)

    resized = F.interpolate(
        torch.from_numpy(xrv_normalized).unsqueeze(0).unsqueeze(0).float(),
        size=(SEGMENTATION_INPUT_SIZE, SEGMENTATION_INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    with torch.no_grad():
        logits = model(resized)
    probs = torch.sigmoid(logits)[0]  # [14, 512, 512]

    left_lung_idx = model.targets.index("Left Lung")
    right_lung_idx = model.targets.index("Right Lung")
    lung_prob = torch.maximum(probs[left_lung_idx], probs[right_lung_idx])

    lung_mask = (lung_prob >= threshold).float()

    mask_resized = F.interpolate(
        lung_mask.unsqueeze(0).unsqueeze(0),
        size=(output_size, output_size),
        mode="nearest",
    )[0, 0]

    return mask_resized.numpy()
