"""Tests for the region-probe helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.region_probe_utils import (  # noqa: E402
    CONDITIONS,
    center_keep_mask,
    margin_keep_mask,
    region_images,
    summarize_regions,
)


def test_margin_and_center_masks_have_matched_area():
    shape = (100, 100)
    margin10, center60 = margin_keep_mask(shape, 0.10), center_keep_mask(shape, 0.60)
    margin20, center80 = margin_keep_mask(shape, 0.20), center_keep_mask(shape, 0.80)
    assert abs(margin10.mean() - 0.36) < 1e-9 and abs(center60.mean() - 0.36) < 1e-9
    assert abs(margin20.mean() - 0.64) < 1e-9 and abs(center80.mean() - 0.64) < 1e-9
    assert not np.any(margin10 & center60)  # disjoint; the 20/80 pair only matches in area


def test_region_images_partition_and_fill():
    rng = np.random.default_rng(0)
    image = rng.normal(1.0, 0.1, (3, 40, 40)).astype(np.float32)
    thorax = np.zeros((40, 40), dtype=np.float32)
    thorax[10:30, 10:30] = 1.0
    images, areas = region_images(image, thorax)
    assert set(images) == set(CONDITIONS) == set(areas)
    assert np.array_equal(images["original"], image)
    assert abs(areas["thorax_only"] + areas["outside_thorax_only"] - 1.0) < 1e-9
    assert areas["outside_thorax_no_margin"] < areas["outside_thorax_only"]
    assert np.all(images["thorax_only"][:, :10, :] == 0.0)  # gray outside the thorax
    assert np.all(images["thorax_only"][:, 10:30, 10:30] == image[:, 10:30, 10:30])
    assert np.all(images["margin_only_10"][:, 20, 20] == 0.0)  # interior grayed
    assert np.all(images["margin_only_10"][:, 0, 0] == image[:, 0, 0])


def test_summarize_regions_reports_every_condition():
    rng = np.random.default_rng(1)
    n = 120
    label = rng.integers(0, 2, n).astype(float)
    frame = pd.DataFrame({"label": label})
    for name in CONDITIONS:
        frame[f"logit_{name}"] = (label * 2 - 1) * 2 + rng.normal(0, 1, n)
        frame[f"area_{name}"] = 0.5
    frame["logit_margin_only_10"] = rng.normal(0, 1, n)  # no signal
    summary = summarize_regions(frame, seed=1, replicates=100)
    assert list(summary["condition"]) == CONDITIONS
    margin = summary[summary.condition == "margin_only_10"].iloc[0]
    assert margin["auroc"] < 0.7 and margin["delta_vs_original"] < -0.2
    assert summary[summary.condition == "original"].iloc[0]["delta_vs_original"] == 0.0
