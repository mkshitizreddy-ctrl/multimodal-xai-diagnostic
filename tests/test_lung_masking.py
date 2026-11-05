"""Tests for lung-aware training data (mask loading, filling, wrapper modes, train.py wiring)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import ChestXrayDataset  # noqa: E402
from src.data.lung_masking import (  # noqa: E402
    GRAY_PIXEL,
    LungMaskedDataset,
    fill_outside,
    load_split_masks,
    wrap_with_lung_masking,
)

SIZE = 32
GRAY_NORMALISED = (GRAY_PIXEL / 255.0 - 0.5) / 0.25


def _make_split(tmp_path, n=6, name="train"):
    rng = np.random.default_rng(0)
    names = []
    for i in range(n):
        file_name = f"{name}_{i}.png"
        array = rng.integers(0, 255, (48, 40), dtype=np.uint8)  # non-square, resized by the wrapper
        Image.fromarray(array).save(tmp_path / file_name)
        names.append(file_name)
    csv = tmp_path / f"{name}.csv"
    pd.DataFrame({"Image Index": names, "Pneumonia": [i % 2 for i in range(n)], "age": range(n)}).to_csv(
        csv, index=False
    )
    masks = np.zeros((n, SIZE, SIZE), dtype=bool)
    masks[:, 8:24, 8:24] = True
    np.savez_compressed(
        tmp_path / f"{name}_masks.npz",
        packed=np.packbits(masks.reshape(n, -1), axis=1),
        names=np.array(names),
        size=SIZE,
        threshold=0.3,
    )
    return csv, names, masks


def _base(csv, tmp_path):
    return ChestXrayDataset(
        csv_path=str(csv), image_dir=str(tmp_path), classes=["Pneumonia"], tabular_features=["age"],
        image_size=SIZE, train=False,
    )  # fmt: skip


def _stage_masks(tmp_path, name):
    """load_split_masks expects <masks_dir>/<split>.npz."""
    (tmp_path / "masks").mkdir(exist_ok=True)
    (tmp_path / f"{name}_masks.npz").rename(tmp_path / "masks" / f"{name}.npz")


def test_fill_outside_keeps_the_mask_and_grays_the_rest():
    image = Image.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8))
    keep = np.zeros((8, 8), dtype=bool)
    keep[2:6, 2:6] = True
    out = np.asarray(fill_outside(image, keep))
    assert (out[2:6, 2:6] == 200).all() and (out[0] == GRAY_PIXEL).all()
    with pytest.raises(ValueError):
        fill_outside(image, np.ones((4, 4), dtype=bool))


def test_load_split_masks_checks_names_and_dilates(tmp_path):
    csv, names, masks = _make_split(tmp_path)
    _stage_masks(tmp_path, "train")
    loaded = load_split_masks(tmp_path / "masks", "train", names, SIZE, 0.3, 0)
    assert np.array_equal(loaded, masks)
    grown = load_split_masks(tmp_path / "masks", "train", names, SIZE, 0.3, 3)
    assert grown.sum() > masks.sum() and (grown >= masks).all()
    with pytest.raises(ValueError):
        load_split_masks(tmp_path / "masks", "train", names[::-1], SIZE, 0.3, 0)
    with pytest.raises(ValueError):
        load_split_masks(tmp_path / "masks", "train", names, SIZE, 0.5, 0)
    with pytest.raises(FileNotFoundError):
        load_split_masks(tmp_path / "masks", "val", names, SIZE, 0.3, 0)


def test_thorax_only_always_masks_and_random_background_follows_prob(tmp_path):
    csv, names, masks = _make_split(tmp_path)
    base = _base(csv, tmp_path)
    always = LungMaskedDataset(base, masks, "thorax_only")
    image, _tab, label = always[0]
    assert tuple(image.shape) == (3, SIZE, SIZE) and label.shape == (1,)
    assert np.allclose(image[:, 0, 0].numpy(), GRAY_NORMALISED, atol=1e-5)  # outside the mask
    assert not np.allclose(image[:, 16, 16].numpy(), GRAY_NORMALISED, atol=1e-3)  # inside the mask

    never = LungMaskedDataset(base, masks, "random_background", prob=0.0)
    full, _t, _l = never[0]
    assert np.array_equal(full.numpy(), base[0][0].numpy())
    sure = LungMaskedDataset(base, masks, "random_background", prob=1.0)
    assert np.allclose(sure[0][0][:, 0, 0].numpy(), GRAY_NORMALISED, atol=1e-5)

    with pytest.raises(ValueError):
        LungMaskedDataset(base, masks[:-1], "thorax_only")
    with pytest.raises(ValueError):
        LungMaskedDataset(base, masks, "nonsense")


def test_wrap_with_lung_masking_validation_behaviour(tmp_path):
    csv, names, _masks = _make_split(tmp_path)
    _stage_masks(tmp_path, "train")
    (tmp_path / "masks" / "val.npz").write_bytes((tmp_path / "masks" / "train.npz").read_bytes())
    base = _base(csv, tmp_path)
    cfg = {"mode": "random_background", "prob": 0.5, "dilation_px": 0, "masks_dir": str(tmp_path / "masks")}
    assert isinstance(wrap_with_lung_masking(base, cfg, "train", SIZE), LungMaskedDataset)
    assert wrap_with_lung_masking(base, cfg, "val", SIZE) is base  # validated on full images
    cfg["mode"] = "thorax_only"
    assert isinstance(wrap_with_lung_masking(base, cfg, "val", SIZE), LungMaskedDataset)


def test_train_build_dataloaders_uses_the_masking_config(tmp_path):
    from src.train import build_dataloaders

    csv, _names, _masks = _make_split(tmp_path)
    _stage_masks(tmp_path, "train")
    (tmp_path / "masks" / "val.npz").write_bytes((tmp_path / "masks" / "train.npz").read_bytes())
    train_cfg = {
        "data": {
            "train_csv": str(csv), "val_csv": str(csv), "image_dir": str(tmp_path), "image_size": SIZE,
            "lung_masking": {"mode": "thorax_only", "dilation_px": 0, "masks_dir": str(tmp_path / "masks")},
        },
        "train": {"batch_size": 2, "num_workers": 0, "seed": 1},
    }  # fmt: skip
    data_cfg = {"labels": {"classes": ["Pneumonia"]}, "tabular_features": ["age"]}
    train_loader, val_loader, classes, _stats = build_dataloaders(data_cfg, train_cfg)
    images, _tab, labels = next(iter(train_loader))
    assert tuple(images.shape) == (2, 3, SIZE, SIZE) and classes == ["Pneumonia"]
    # Augmentation runs after masking (rotation fills corners black), so check the gray fraction.
    gray_fraction = np.isclose(images.numpy(), GRAY_NORMALISED, atol=1e-3).mean()
    assert isinstance(train_loader.dataset, LungMaskedDataset) and gray_fraction > 0.3
    del train_cfg["data"]["lung_masking"]
    plain_loader, _v, _c, _s = build_dataloaders(data_cfg, train_cfg)
    assert not isinstance(plain_loader.dataset, LungMaskedDataset)


def test_masked_dataset_pickles_small_and_round_trips(tmp_path):
    import pickle

    csv, _names, masks = _make_split(tmp_path, n=6)
    dataset = LungMaskedDataset(_base(csv, tmp_path), masks, "thorax_only")
    assert all(np.array_equal(dataset.mask_at(i), masks[i]) for i in range(len(masks)))
    assert dataset.packed.nbytes * 8 <= masks.size + 8 * len(masks)
    assert len(pickle.dumps(dataset)) < 200_000  # workers receive the dataset by pickling on Windows
