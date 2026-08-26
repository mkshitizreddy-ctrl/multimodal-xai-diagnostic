"""
Unit test for src/explain/measure_lung_localization.py's scoring function.
Deliberately doesn't touch the model, dataset, or torchxrayvision's
segmentation network — just checks the arithmetic on synthetic heatmaps,
so this test runs fast and without a checkpoint or GPU.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.measure_lung_localization import lung_energy_fraction


def test_all_activation_inside_lungs_gives_fraction_one():
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[3:7, 3:7] = 1.0
    lung_mask = np.ones((10, 10), dtype=np.float32)
    assert lung_energy_fraction(heatmap, lung_mask) == 1.0


def test_all_activation_outside_lungs_gives_fraction_zero():
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[0:2, 0:2] = 1.0  # corner, outside the mask below
    lung_mask = np.zeros((10, 10), dtype=np.float32)
    lung_mask[5:10, 5:10] = 1.0
    assert lung_energy_fraction(heatmap, lung_mask) == 0.0


def test_half_and_half_gives_fraction_half():
    heatmap = np.ones((10, 10), dtype=np.float32)
    lung_mask = np.zeros((10, 10), dtype=np.float32)
    lung_mask[:, :5] = 1.0  # left half is "lung"
    assert abs(lung_energy_fraction(heatmap, lung_mask) - 0.5) < 1e-6


def test_degenerate_zero_heatmap_returns_nan():
    heatmap = np.zeros((10, 10), dtype=np.float32)
    lung_mask = np.ones((10, 10), dtype=np.float32)
    assert np.isnan(lung_energy_fraction(heatmap, lung_mask))
