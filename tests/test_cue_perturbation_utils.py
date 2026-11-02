"""Tests for the acquisition-cue perturbation helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.cue_perturbation_utils import (  # noqa: E402
    PERTURBATIONS,
    down_up,
    equalize_histogram,
    pad_to_square,
    recompress_jpeg,
    summarize_perturbations,
)


def _image(width=120, height=80, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(40, 160, width)[None, :].repeat(height, axis=0)
    return Image.fromarray(np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8))


def test_all_perturbations_return_grayscale_images_of_expected_size():
    image = _image()
    for name, fn in PERTURBATIONS.items():
        out = fn(image)
        assert out.mode == "L", name
        expected = (120, 120) if name == "pad_square" else (120, 80)
        assert out.size == expected, name


def test_recompress_changes_pixels_and_stronger_quality_changes_more():
    image = _image()
    original = np.asarray(image, dtype=float)
    mild = np.abs(np.asarray(recompress_jpeg(image, 90), dtype=float) - original).mean()
    strong = np.abs(np.asarray(recompress_jpeg(image, 10), dtype=float) - original).mean()
    assert 0 < mild < strong


def test_equalize_spreads_a_narrow_histogram():
    narrow = Image.fromarray(np.random.default_rng(1).integers(100, 120, (64, 64)).astype(np.uint8))
    out = np.asarray(equalize_histogram(narrow))
    assert out.std() > 2 * np.asarray(narrow).std()


def test_pad_to_square_keeps_content_centred():
    image = _image(120, 80)
    out = pad_to_square(image)
    assert out.size == (120, 120)
    assert np.array_equal(np.asarray(out)[20:100], np.asarray(image))


def test_down_up_removes_high_frequency_noise():
    image = _image()
    smooth = np.asarray(down_up(image, 2), dtype=float)
    rough = np.asarray(image, dtype=float)
    assert np.abs(np.diff(smooth, axis=1)).mean() < np.abs(np.diff(rough, axis=1)).mean()


def test_summarize_perturbations_flags_a_shift_only_in_the_perturbed_condition():
    rng = np.random.default_rng(0)
    n = 200
    label = rng.integers(0, 2, n).astype(float)
    logit = (label * 2 - 1) * 3 + rng.normal(0, 1, n)
    frame = pd.DataFrame({"label": label})
    for name in PERTURBATIONS:
        frame[f"logit_{name}"] = logit
    frame["logit_equalize"] = rng.normal(0, 1, n)  # destroys the signal
    discordant = np.zeros(n, dtype=bool)
    discordant[:40] = True
    summary = summarize_perturbations(frame, discordant, seed=1, replicates=200)
    assert set(summary["subset"]) == {"all", "baseline_right", "baseline_wrong"}
    row = summary[(summary.subset == "all") & (summary.condition == "equalize")].iloc[0]
    same = summary[(summary.subset == "all") & (summary.condition == "jpeg_q75")].iloc[0]
    assert row["delta_vs_original"] < -0.2
    assert same["delta_vs_original"] == 0.0 and same["mean_logit_shift"] == 0.0


def test_summarize_perturbations_skips_an_empty_subset():
    frame = pd.DataFrame({"label": [0.0, 1.0, 0.0, 1.0]})
    for name in PERTURBATIONS:
        frame[f"logit_{name}"] = [-2.0, 2.0, -1.0, 1.0]
    summary = summarize_perturbations(frame, np.zeros(4, dtype=bool), seed=1, replicates=100)
    assert "baseline_wrong" not in set(summary["subset"])
