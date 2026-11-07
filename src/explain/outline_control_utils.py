"""Torch-free helpers for the outline-control evaluation (see outline_control_eval.py)."""

import numpy as np

from src.explain.shortcut_audit_utils import keep_region

CONDITIONS = ["original", "thorax_only", "fixed_box", "outside_fixed_box"]


def outline_images(image: np.ndarray, thorax_mask: np.ndarray, box_mask: np.ndarray) -> tuple[dict, dict]:
    """Return ({condition: image}, {condition: kept area fraction}) for one [C, H, W] image.

    `thorax_only` keeps the (dilated) lung mask, `fixed_box` keeps the same centred box for every image,
    and `outside_fixed_box` keeps everything else. Gray fill is 0.0 in the dataset's normalised space.
    """
    thorax = np.asarray(thorax_mask) > 0.5
    box = np.asarray(box_mask) > 0.5
    if thorax.shape != box.shape or image.shape[-2:] != thorax.shape:
        raise ValueError("image, thorax mask and box mask must share the same height and width")
    keeps = {
        "original": np.ones(thorax.shape, dtype=bool),
        "thorax_only": thorax,
        "fixed_box": box,
        "outside_fixed_box": ~box,
    }
    return {name: keep_region(image, keep) for name, keep in keeps.items()}, {
        name: float(keep.mean()) for name, keep in keeps.items()
    }
