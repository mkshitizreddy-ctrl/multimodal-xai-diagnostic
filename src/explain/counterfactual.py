"""
Occlusion-based counterfactual explainer.

Instead of generating a new "healthy" image via a generative model (cVAE /
diffusion — high-risk, research-grade), this module answers the same
question ("what if this finding wasn't there?") by:

    1. Taking the Grad-CAM heatmap for the predicted class.
    2. Thresholding it into a binary mask of the highest-activation region.
    3. Inpainting that region (removing the visual evidence there).
    4. Re-running the model on the inpainted image.
    5. Reporting how much the model's confidence dropped.

A large confidence drop after removing the highlighted region is evidence
the model's attention genuinely aligns with its stated reasoning — which is
the core idea a full counterfactual-diffusion module would also be
demonstrating, at a fraction of the engineering cost.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.explain.gradcam import ChestXrayExplainer, denormalize_to_rgb

NORM_MEAN = 0.5
NORM_STD = 0.25


@dataclass
class CounterfactualResult:
    class_name: str
    original_probability: float
    counterfactual_probability: float
    mask: np.ndarray            # HxW uint8, 0/255
    original_image: np.ndarray  # HxWx3 uint8
    counterfactual_image: np.ndarray  # HxWx3 uint8

    @property
    def probability_drop(self) -> float:
        return self.original_probability - self.counterfactual_probability

    @property
    def flipped(self) -> bool:
        """True if the drop was large enough to change the prediction
        (crossing the standard 0.5 decision threshold)."""
        return self.original_probability >= 0.5 > self.counterfactual_probability


def heatmap_to_mask(heatmap: np.ndarray, threshold: float = 0.6, dilate_px: int = 5) -> np.ndarray:
    """Binarize a Grad-CAM heatmap (values in [0,1]) into an inpainting mask."""
    mask = (heatmap >= threshold).astype(np.uint8) * 255
    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def rgb_float_to_uint8(rgb_img: np.ndarray) -> np.ndarray:
    return (rgb_img * 255).astype(np.uint8)


def uint8_to_normalized_tensor(image_uint8: np.ndarray) -> torch.Tensor:
    """Inverse of denormalize_to_rgb — converts an HWC uint8 image back into
    the normalized CHW tensor the model expects."""
    img = image_uint8.astype(np.float32) / 255.0
    img = (img - NORM_MEAN) / NORM_STD
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return tensor


class OcclusionCounterfactualExplainer:
    def __init__(self, model: nn.Module, explainer: ChestXrayExplainer, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.explainer = explainer
        self.device = device

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: int,
        class_name: str,
        threshold: float = 0.6,
    ) -> CounterfactualResult:
        # 1. Get the original prediction + Grad-CAM heatmap for this class
        with torch.no_grad():
            logits = self.model(image_tensor.unsqueeze(0).to(self.device))
            original_prob = torch.sigmoid(logits)[0, class_idx].item()

        _, heatmap = self.explainer.explain(image_tensor, class_idx)

        # 2. Build an inpainting mask from the highest-activation region
        mask = heatmap_to_mask(heatmap, threshold=threshold)

        # 3. Inpaint that region out of the image
        rgb_img = rgb_float_to_uint8(denormalize_to_rgb(image_tensor))
        counterfactual_img = cv2.inpaint(rgb_img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        # 4. Re-run the model on the counterfactual image
        cf_tensor = uint8_to_normalized_tensor(counterfactual_img)
        with torch.no_grad():
            cf_logits = self.model(cf_tensor.unsqueeze(0).to(self.device))
            cf_prob = torch.sigmoid(cf_logits)[0, class_idx].item()

        return CounterfactualResult(
            class_name=class_name,
            original_probability=original_prob,
            counterfactual_probability=cf_prob,
            mask=mask,
            original_image=rgb_img,
            counterfactual_image=counterfactual_img,
        )


def make_side_by_side_figure(result: CounterfactualResult) -> np.ndarray:
    """Stitches original | mask | counterfactual into one image with a
    probability caption strip — the exact figure the dashboard (Part 7)
    and README will display."""
    h, w = result.original_image.shape[:2]
    mask_rgb = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2RGB)

    caption_h = 40
    strip = np.ones((caption_h, w * 3, 3), dtype=np.uint8) * 255
    cv2.putText(
        strip,
        f"Original: {result.original_probability:.2f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        strip,
        "Masked region",
        (w + 10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        strip,
        f"Counterfactual: {result.counterfactual_probability:.2f}",
        (2 * w + 10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    row = np.concatenate([result.original_image, mask_rgb, result.counterfactual_image], axis=1)
    figure = np.concatenate([row, strip], axis=0)
    return figure
