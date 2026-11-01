"""Tests for the non-anatomical baseline helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.nonanatomical_baseline import (  # noqa: E402
    GROUPS,
    INTENSITY_FEATURES,
    SIZE_FEATURES,
    THUMB_FEATURES,
    evaluate_group,
    image_features,
)


def _write_image(path, width, height, level):
    array = np.full((height, width), level, dtype=np.uint8)
    array[:, : width // 10] = 0  # dark left strip
    Image.fromarray(array).save(path, format="JPEG")


def test_image_features_have_expected_keys_and_values(tmp_path):
    path = tmp_path / "a.jpg"
    _write_image(path, 400, 200, 200)
    f = image_features(path)
    for key in SIZE_FEATURES + INTENSITY_FEATURES + THUMB_FEATURES:
        assert key in f
    assert f["width"] == 400.0 and f["height"] == 200.0 and abs(f["aspect"] - 2.0) < 1e-9
    assert f["file_kb"] > 0 and f["bytes_per_pixel"] > 0
    assert 0.55 < f["mean"] < 0.85  # mostly bright image with a dark strip
    assert f["frac_dark"] > 0.05
    assert len(THUMB_FEATURES) == 256 and all(g for g in GROUPS.values())


def test_evaluate_group_finds_an_informative_feature_and_ignores_noise():
    rng = np.random.default_rng(0)

    def make(n):
        y = rng.integers(0, 2, n).astype(float)
        return pd.DataFrame(
            {"label": y, "signal": y * 2.0 + rng.normal(size=n), "noise": rng.normal(size=n)}
        )

    train, val, test = make(400), make(200), make(300)
    informative = evaluate_group(train, val, test, ["signal"])
    noise_only = evaluate_group(train, val, test, ["noise"])
    assert informative["test_auroc"] > 0.85
    assert abs(noise_only["test_auroc"] - 0.5) < 0.12
    assert len(informative["test_scores"]) == 300
