"""
Tests for src/data/lung_mask_dataset.py — builds a tiny real synthetic
dataset (fake images + CSV + pickled masks) rather than mocking, so this
actually exercises ChestXrayDataset's real __getitem__ underneath the
wrapper, not just the wrapper's own logic in isolation.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.lung_mask_dataset import LungMaskAugmentedDataset

CLASSES = ["Pneumonia"]
TABULAR_FEATURES = ["Patient Age", "Patient Gender", "Temperature", "SpO2"]


def _build_fake_dataset(tmp_path, num_images=4, with_masks=True):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    masks = {}
    for i in range(num_images):
        fname = f"fake_{i}.png"
        arr = (np.random.rand(64, 64) * 255).astype("uint8")
        Image.fromarray(arr, mode="L").save(image_dir / fname)
        rows.append(
            {
                "Image Index": fname,
                "Pneumonia": float(i % 2),
                "Patient Age": 20 + i,
                "Patient Gender": "M" if i % 2 == 0 else "F",
                "Temperature": 37.0,
                "SpO2": 96.0,
            }
        )
        if with_masks:
            masks[fname] = np.random.randint(0, 2, (7, 7)).astype("float32")

    csv_path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    masks_path = tmp_path / "masks.pkl"
    with open(masks_path, "wb") as f:
        pickle.dump(masks, f)

    return str(csv_path), str(image_dir), str(masks_path)


def test_returns_four_tuple_with_correct_mask_shape(tmp_path):
    csv_path, image_dir, masks_path = _build_fake_dataset(tmp_path)
    ds = LungMaskAugmentedDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        classes=CLASSES,
        tabular_features=TABULAR_FEATURES,
        image_size=224,
        train=False,
        lung_masks_path=masks_path,
    )

    image, tabular, labels, lung_mask = ds[0]
    assert image.shape == (3, 224, 224)
    assert lung_mask.shape == (7, 7)
    assert isinstance(lung_mask, torch.Tensor)


def test_underlying_chestxraydataset_behavior_is_unaffected(tmp_path):
    """The wrapper should return the exact same image/tabular/labels
    ChestXrayDataset itself would — only adding the mask, not changing
    anything about the existing three values."""
    from src.data.dataset import ChestXrayDataset

    csv_path, image_dir, masks_path = _build_fake_dataset(tmp_path)
    kwargs = dict(
        csv_path=csv_path,
        image_dir=image_dir,
        classes=CLASSES,
        tabular_features=TABULAR_FEATURES,
        image_size=224,
        train=False,
    )
    plain_ds = ChestXrayDataset(**kwargs)
    wrapped_ds = LungMaskAugmentedDataset(**kwargs, lung_masks_path=masks_path)

    plain_image, plain_tabular, plain_labels = plain_ds[0]
    wrapped_image, wrapped_tabular, wrapped_labels, _mask = wrapped_ds[0]

    assert torch.allclose(plain_image, wrapped_image)
    assert torch.allclose(plain_tabular, wrapped_tabular)
    assert torch.allclose(plain_labels, wrapped_labels)


def test_missing_mask_raises_clear_error_at_construction(tmp_path):
    """Should fail loudly when the dataset is BUILT (before any training
    starts), not partway through epoch 3 when __getitem__ happens to hit
    the one image without a mask — that's a much worse failure mode."""
    csv_path, image_dir, masks_path = _build_fake_dataset(tmp_path, with_masks=False)

    with pytest.raises(ValueError, match="no precomputed lung mask"):
        LungMaskAugmentedDataset(
            csv_path=csv_path,
            image_dir=image_dir,
            classes=CLASSES,
            tabular_features=TABULAR_FEATURES,
            image_size=224,
            train=False,
            lung_masks_path=masks_path,
        )
