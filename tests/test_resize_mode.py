"""Tests for the resize-mode option (stretch / pad / center_crop) and the aspect-quintile helper."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import PAD_FILL, ChestXrayDataset, to_square  # noqa: E402
from src.explain.resize_mode_eval import aspect_quintile_auroc  # noqa: E402


def _wide(w=40, h=20):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, : w // 2] = 255  # left half white, right half black
    return Image.fromarray(arr)


def test_stretch_returns_image_unchanged():
    image = _wide()
    assert to_square(image, "stretch") is image


def test_pad_makes_square_keeps_all_content_and_gray_bands():
    out = to_square(_wide(40, 20), "pad")
    assert out.size == (40, 40)
    arr = np.asarray(out)
    assert (arr[:5] == PAD_FILL).all() and (arr[-5:] == PAD_FILL).all()  # bands above and below
    assert arr[20, 5, 0] == 255 and arr[20, 35, 0] == 0  # both halves survive, unscaled


def test_center_crop_removes_sides_not_bands():
    out = to_square(_wide(40, 20), "center_crop")
    assert out.size == (20, 20)
    arr = np.asarray(out)
    assert PAD_FILL not in np.unique(arr)  # no gray band, so no aspect cue


def test_square_input_is_unchanged_by_pad_and_crop():
    sq = _wide(20, 20)
    for mode in ("pad", "center_crop"):
        assert np.array_equal(np.asarray(to_square(sq, mode)), np.asarray(sq))


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        to_square(_wide(), "zoom")
    with pytest.raises(ValueError):
        ChestXrayDataset(
            csv_path="x.csv", image_dir=".", classes=["Pneumonia"], tabular_features=[], resize_mode="zoom"
        )


def test_aspect_quintiles_cover_everything_and_flag_single_class_bins():
    rng = np.random.default_rng(0)
    n = 100
    aspect = rng.uniform(0.8, 1.6, n)
    y = (aspect > 1.3).astype(float)  # label determined by aspect: outer bins single-class
    y[::7] = 1 - y[::7]
    scores = y + rng.normal(0, 0.1, n)
    table = aspect_quintile_auroc(y, scores, aspect)
    assert table["n"].sum() == n and len(table) == 5
    assert (table["aspect_min"].diff().dropna() >= 0).all()
    assert table["auroc"].dropna().between(0.0, 1.0).all()


def test_aspect_quintile_nan_when_bin_has_one_class():
    aspect = np.arange(50, dtype=float)
    y = (aspect >= 10).astype(float)  # first bin (aspect 0-9) is all class 0
    table = aspect_quintile_auroc(y, y + 0.0, aspect)
    assert np.isnan(table.loc[0, "auroc"])
