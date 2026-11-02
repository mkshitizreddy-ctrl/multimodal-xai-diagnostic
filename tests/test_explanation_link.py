"""Tests for the explanation-link statistics."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.explanation_link import bootstrap_partial, partial_spearman  # noqa: E402


def test_partial_spearman_removes_a_shared_confounder():
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    x = z + 0.3 * rng.normal(size=500)
    y = z + 0.3 * rng.normal(size=500)
    raw = np.corrcoef(x, y)[0, 1]
    assert raw > 0.8
    assert abs(partial_spearman(x, y, z)) < 0.15


def test_partial_spearman_keeps_a_direct_relationship():
    rng = np.random.default_rng(1)
    z = rng.normal(size=500)
    x = rng.normal(size=500)
    y = x + 0.5 * z + 0.3 * rng.normal(size=500)
    assert partial_spearman(x, y, z) > 0.7


def test_bootstrap_partial_interval_contains_the_point_estimate():
    rng = np.random.default_rng(2)
    z = rng.normal(size=200)
    x = rng.normal(size=200)
    y = x + rng.normal(size=200)
    out = bootstrap_partial(x, {"y": y}, z, seed=3, replicates=200)
    point, low, high, draws = out["y"]
    assert low <= point <= high and len(draws) == 200
