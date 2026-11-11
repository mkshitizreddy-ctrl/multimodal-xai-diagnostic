"""Tests for the patient-clustered bootstrap."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.cluster_bootstrap import bootstrap_indices, cluster_bootstrap, compare_levels  # noqa: E402


def test_bootstrap_indices_resample_whole_groups():
    groups = np.array(["a", "a", "a", "b", "c", "c"])
    for idx in bootstrap_indices(groups, seed=0, replicates=20):
        counts = {g: int((groups[idx] == g).sum()) for g in "abc"}
        assert counts["a"] % 3 == 0 and counts["c"] % 2 == 0  # groups come in whole


def test_singleton_groups_match_image_level_bootstrap():
    rng = np.random.default_rng(0)
    y = (rng.random(80) > 0.5).astype(float)
    s = {"a": y + rng.normal(0, 1, 80)}
    one = cluster_bootstrap(y, s, {}, np.arange(80), seed=1, replicates=200)
    two = cluster_bootstrap(y, s, {}, np.arange(80), seed=1, replicates=200)
    assert one.equals(two)  # deterministic given the seed
    assert 0.5 < one.loc[0, "estimate"] <= 1.0


def test_correlated_patients_widen_the_cluster_interval():
    rng = np.random.default_rng(0)
    n_patients, per = 40, 6
    patient = np.repeat(np.arange(n_patients), per)
    y = np.repeat((rng.random(n_patients) > 0.5).astype(float), per)
    shift = np.repeat(rng.normal(0, 1.5, n_patients), per)  # shared within a patient
    score = y + shift + rng.normal(0, 0.3, len(y))
    table = compare_levels(y, {"s": score}, {}, patient, seed=2, replicates=300)
    assert table.loc[0, "width_ratio"] > 1.2
