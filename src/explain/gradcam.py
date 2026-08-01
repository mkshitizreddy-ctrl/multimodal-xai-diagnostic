"""
Grad-CAM explainability module.

Wraps `pytorch_grad_cam` around ChestXrayVisionModel to produce a heatmap
showing which image regions drove the prediction for a given disease class.

Usage:
    explainer = ChestXrayExplainer(model, device="cuda")
    overlay, heatmap = explainer.explain(image_tensor, class_idx=3)
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Must match src/data/dataset.py's transforms.Normalize call
NORM_MEAN = 0.5
NORM_STD = 0.25


def denormalize_to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    """Reverse the dataset's Normalize transform and return an HWC float
    image in [0, 1], as required by pytorch_grad_cam's show_cam_on_image."""
    img = image_tensor.detach().cpu().clone()
    img = img * NORM_STD + NORM_MEAN
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()  # CHW -> HWC
    return img


class ChestXrayExplainer:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.device = device
        self.model = model.to(device).eval()
        target_layer = model.get_target_layer()
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])

    def explain(self, image_tensor: torch.Tensor, class_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image_tensor: single image, shape [3, H, W], normalized (as
                returned by ChestXrayDataset).
            class_idx: index into the model's class list to explain.

        Returns:
            overlay: HWC uint8 RGB image with the heatmap blended in —
                ready to display or save.
            grayscale_cam: HxW float array in [0, 1], the raw heatmap —
                useful downstream for the occlusion-based counterfactual
                module (Part 5), which masks the highest-activation region.
        """
        input_tensor = image_tensor.unsqueeze(0).to(self.device)
        targets = [ClassifierOutputTarget(class_idx)]

        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0]

        rgb_img = denormalize_to_rgb(image_tensor)
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        return overlay, grayscale_cam

    def explain_top_k(
        self, image_tensor: torch.Tensor, classes: list[str], k: int = 3
    ) -> list[dict]:
        """Runs the model once, then generates Grad-CAM overlays for the
        top-k predicted classes. Convenient for the dashboard (Part 7),
        which shows the model's top findings side by side with their
        explanations."""
        with torch.no_grad():
            logits = self.model(image_tensor.unsqueeze(0).to(self.device))
            probs = torch.sigmoid(logits)[0].cpu()

        top_indices = torch.argsort(probs, descending=True)[:k].tolist()

        results = []
        for idx in top_indices:
            overlay, heatmap = self.explain(image_tensor, idx)
            results.append(
                {
                    "class": classes[idx],
                    "probability": probs[idx].item(),
                    "overlay": overlay,
                    "heatmap": heatmap,
                }
            )
        return results


def save_overlay(overlay: np.ndarray, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
