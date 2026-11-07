"""Lung-aware training data: gray out everything outside the (dilated) lung mask.

Wraps a ChestXrayDataset without changing it. Two modes:

- ``thorax_only``: every image is masked, at train and validation time (the model can only see the thorax).
- ``random_background``: each training image is masked with probability ``prob``; validation images are left
  untouched so model selection still uses full images.
- ``fixed_box``: control for the lung outline. Every image, at train and validation time, keeps the same
  centred box (side ``box_fraction`` of the image, default 0.72, so its area matches the mean thorax mask).
  The cut-out is identical for all images, so it carries no per-image information.

The mask is applied after the resize to the model resolution and before any augmentation, so flips and
rotations move the gray fill with the image. Fill is mid-gray (pixel 128), matching the audit's gray.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.explain.shortcut_audit_utils import dilate_mask

GRAY_PIXEL = 128
MODES = ("thorax_only", "random_background", "fixed_box")


def load_split_masks(
    masks_dir, split: str, names: list[str], size: int, threshold: float, dilation_px: float
) -> np.ndarray:
    """Load cached lung masks [N, size, size] for one split, verify they match `names`, and dilate them."""
    path = Path(masks_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run src/explain/cache_split_masks.py first")
    cached = np.load(path, allow_pickle=False)
    if cached["names"].tolist() != list(names):
        raise ValueError(f"{path} was built for a different image list; rerun cache_split_masks.py")
    if int(cached["size"]) != size or float(cached["threshold"]) != threshold:
        raise ValueError(f"{path} was built with a different size or threshold; rerun cache_split_masks.py")
    bits = np.unpackbits(cached["packed"], axis=1, count=size * size)
    masks = bits.reshape(len(names), size, size).astype(np.float32)
    if dilation_px > 0:
        masks = np.stack([dilate_mask(m, dilation_px) for m in masks])
    return masks > 0.5


def centered_box(size: int, fraction: float) -> np.ndarray:
    """Boolean [size, size] mask of a centred square whose side is `fraction` of the image side."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("box_fraction must be in (0, 1)")
    margin = int(round(size * (1.0 - fraction) / 2.0))
    box = np.zeros((size, size), dtype=bool)
    box[margin : size - margin, margin : size - margin] = True
    return box


def fill_outside(image: Image.Image, keep: np.ndarray, fill: int = GRAY_PIXEL) -> Image.Image:
    """Return a copy of `image` with every pixel outside the boolean `keep` mask set to `fill`."""
    array = np.asarray(image).copy()
    if array.shape[:2] != keep.shape:
        raise ValueError(f"mask shape {keep.shape} does not match image shape {array.shape[:2]}")
    array[~keep] = fill
    return Image.fromarray(array)


class LungMaskedDataset(Dataset):
    def __init__(self, base, masks: np.ndarray, mode: str, prob: float = 0.5, always_mask: bool = False):
        """`masks` is [N, size, size] (one per image) or [size, size] (shared by every image)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        masks = np.asarray(masks)
        self.shared = masks.ndim == 2
        if not self.shared and len(masks) != len(base):
            raise ValueError("one mask per image is required")
        self.base = base
        # Bit-packed (~8x smaller): DataLoader workers on Windows receive this dataset by pickling, and a
        # full boolean array (>100 MB for the training split) made that pickle fail with OSError 22.
        self.size = masks.shape[-1]
        self.packed = np.packbits(masks.reshape(1 if self.shared else len(masks), -1), axis=1)
        self.mode = mode
        self.prob = prob
        self.always_mask = always_mask

    def __len__(self) -> int:
        return len(self.base)

    def mask_at(self, idx: int) -> np.ndarray:
        """Boolean keep-mask [size, size] for image `idx`."""
        bits = np.unpackbits(self.packed[0 if self.shared else idx], count=self.size * self.size)
        return bits.reshape(self.size, self.size).astype(bool)

    def _should_mask(self) -> bool:
        if self.always_mask or self.mode in ("thorax_only", "fixed_box"):
            return True
        return random.random() < self.prob

    def __getitem__(self, idx: int):
        base = self.base
        row = base.df.iloc[idx]
        image = Image.open(base.image_dir / row["Image Index"]).convert("RGB")
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        if self._should_mask():
            image = fill_outside(image, self.mask_at(idx))
        image = base.transform(image)
        tabular = base._encode_tabular(row)
        labels = torch.tensor(row[base.classes].values.astype(np.float32), dtype=torch.float32)
        return image, tabular, labels


def wrap_with_lung_masking(base, masking_cfg: dict, split: str, size: int):
    """Wrap `base` according to the `data.lung_masking` config block; `split` is 'train' or 'val'."""
    mode = masking_cfg["mode"]
    if mode == "fixed_box":
        box = centered_box(size, float(masking_cfg.get("box_fraction", 0.72)))
        return LungMaskedDataset(base, box, mode)
    masks = load_split_masks(
        masking_cfg.get("masks_dir", "data/lung_masks"),
        split,
        [str(name) for name in base.df["Image Index"]],
        size,
        float(masking_cfg.get("threshold", 0.3)),
        float(masking_cfg.get("dilation_px", 8)),
    )
    if mode == "thorax_only":
        return LungMaskedDataset(base, masks, mode)
    if split == "train":
        return LungMaskedDataset(base, masks, mode, prob=float(masking_cfg.get("prob", 0.5)))
    return base  # random_background: validate on full images
