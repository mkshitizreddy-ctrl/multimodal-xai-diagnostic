"""Tests for the torch-free shortcut-audit helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.shortcut_audit_utils import (  # noqa: E402
    CONDITIONS,
    audit_image,
    auroc,
    blank_border,
    displace_mask,
    keep_region,
    mask_area_fraction,
    paired_bootstrap_auroc,
    remove_region,
    summarize,
    summary_markdown,
    threshold_metrics,
)


def _disk_mask(size=64, cy=32, cx=24, radius=10):
    yy, xx = np.mgrid[:size, :size]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2).astype(np.float32)


def test_auroc_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    scores = np.round(rng.normal(size=200) + y, 1)  # rounding creates ties
    assert abs(auroc(y, scores) - roc_auc_score(y, scores)) < 1e-12


def test_auroc_single_class_is_nan():
    assert np.isnan(auroc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_keep_and_remove_are_complementary():
    rng = np.random.default_rng(1)
    image = rng.normal(size=(3, 64, 64)).astype(np.float32)
    mask = _disk_mask()
    kept = keep_region(image, mask, fill=0.0)
    removed = remove_region(image, mask, fill=0.0)
    assert np.allclose(kept + removed, image)
    assert np.all(kept[:, mask < 0.5] == 0.0)
    assert np.all(removed[:, mask > 0.5] == 0.0)


def test_blank_border_keeps_center_and_fills_edge():
    image = np.ones((3, 100, 100), dtype=np.float32)
    out = blank_border(image, 0.1, fill=0.0)
    assert np.all(out[:, 10:90, 10:90] == 1.0)
    assert np.all(out[:, :10, :] == 0.0) and np.all(out[:, :, 90:] == 0.0)


def test_displace_mask_preserves_area_and_reduces_overlap():
    mask = _disk_mask()
    shifted, iou = displace_mask(mask, np.random.default_rng(3))
    assert mask_area_fraction(shifted) == mask_area_fraction(mask)
    assert iou <= 0.5
    assert not np.array_equal(shifted, mask)


def test_displace_mask_is_deterministic_for_a_seed():
    mask = _disk_mask()
    a, _ = displace_mask(mask, np.random.default_rng(7))
    b, _ = displace_mask(mask, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_paired_bootstrap_identical_scores_gives_zero_delta():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 150)
    s = rng.normal(size=150) + y
    out = paired_bootstrap_auroc(y, s, s, seed=1, replicates=200)
    assert out["delta"] == 0.0
    assert out["delta_ci"] == (0.0, 0.0)


def test_paired_bootstrap_detects_worse_scores():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    good = y + rng.normal(scale=0.3, size=300)
    bad = rng.normal(size=300)
    out = paired_bootstrap_auroc(y, good, bad, seed=1, replicates=300)
    assert out["delta"] < -0.2
    assert out["delta_ci"][1] < 0.0
    assert out["auroc_ref_ci"][0] < out["auroc_ref"] < out["auroc_ref_ci"][1]


def test_threshold_metrics():
    y = [1, 1, 0, 0]
    p = [0.9, 0.4, 0.1, 0.6]
    m = threshold_metrics(y, p)
    assert m["accuracy"] == 0.5 and m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert abs(m["mean_prob_normal"] - 0.35) < 1e-12


def _mean_pixel_infer(batch):
    return batch.mean(axis=(1, 2, 3))


def test_audit_image_conditions_with_fake_model():
    image = np.full((3, 64, 64), 2.0, dtype=np.float32)
    mask = _disk_mask()
    out = audit_image(image, mask, _mean_pixel_infer, np.random.default_rng(0), repeats=4, border_frac=0.1)
    area = mask_area_fraction(mask)
    assert abs(out["logit_original"] - 2.0) < 1e-6
    assert abs(out["logit_lung_only"] - 2.0 * area) < 1e-6
    assert abs(out["logit_background_only"] - 2.0 * (1 - area)) < 1e-6
    # displaced masks have identical area, so the controls match the real conditions here
    assert abs(out["logit_displaced_lung_only"] - out["logit_lung_only"]) < 1e-6
    assert abs(out["logit_displaced_background_only"] - out["logit_background_only"]) < 1e-6
    assert out["displaced_iou_mean"] <= 0.5


def test_audit_image_rejects_wrong_infer_shape():
    image = np.zeros((3, 32, 32), dtype=np.float32)
    try:
        audit_image(image, _disk_mask(32, 16, 16, 5), lambda b: np.zeros(3), np.random.default_rng(0), 2, 0.1)
    except ValueError as err:
        assert "expected" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_summarize_and_markdown_end_to_end():
    rng = np.random.default_rng(5)
    n = 200
    y = rng.integers(0, 2, n).astype(float)
    df = pd.DataFrame({"label": y, "mask_area": rng.uniform(0.05, 0.6, n)})
    df["logit_original"] = 3 * y + rng.normal(size=n)
    for condition in CONDITIONS[1:]:
        df[f"logit_{condition}"] = rng.normal(size=n)  # uninformative scores
    summary = summarize(df, seed=1, replicates=100, min_area=0.1, max_area=0.7)
    assert set(summary["subset"]) == {"all", "valid_mask"}
    assert len(summary) == 2 * len(CONDITIONS)
    row = summary[(summary["subset"] == "all") & (summary["condition"] == "original")].iloc[0]
    assert row["delta_auroc_vs_original"] == 0.0 and row["auroc"] > 0.9
    worse = summary[(summary["subset"] == "all") & (summary["condition"] == "lung_only")].iloc[0]
    assert worse["delta_auroc_vs_original"] < -0.2
    markdown = summary_markdown(summary, {"n_images": n, "n_pneumonia": int(y.sum())})
    assert "## Subset: all" in markdown and "lung_only" in markdown and "How to read this" in markdown
