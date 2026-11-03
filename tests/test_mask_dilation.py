"""Tests for the lung-mask dilation helper."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.shortcut_audit_utils import dilate_mask, mask_area_fraction  # noqa: E402


def _square(side=64, lo=24, hi=40):
    mask = np.zeros((side, side), dtype=np.float32)
    mask[lo:hi, lo:hi] = 1.0
    return mask


def test_zero_dilation_returns_the_same_mask():
    mask = _square()
    assert np.array_equal(dilate_mask(mask, 0), mask)


def test_dilation_is_a_superset_and_area_grows_with_pixels():
    mask = _square()
    small, large = dilate_mask(mask, 2), dilate_mask(mask, 5)
    assert np.all(small >= mask) and np.all(large >= small)
    assert mask_area_fraction(mask) < mask_area_fraction(small) < mask_area_fraction(large)


def test_dilation_radius_is_euclidean():
    mask = np.zeros((21, 21), dtype=np.float32)
    mask[10, 10] = 1.0
    out = dilate_mask(mask, 3)
    assert out[10, 13] == 1.0 and out[10, 14] == 0.0
    assert out[12, 12] == 1.0  # distance 2.83
    assert out[13, 13] == 0.0  # distance 4.24


def test_all_foreground_mask_stays_full():
    mask = np.ones((8, 8), dtype=np.float32)
    assert dilate_mask(mask, 3).all()
