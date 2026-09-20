"""Unit tests for the standalone full-metrics evaluator."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_full_metrics import (
    bootstrap_ci,
    compute_metrics_for_class,
    metric_function,
    select_validation_threshold,
)


def test_all_metrics_are_calculated_for_perfect_predictions():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_metrics_for_class(labels, probabilities)

    assert metrics == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "auroc": 1.0,
        "aupr": 1.0,
    }


def test_threshold_is_respected_for_thresholded_metrics():
    labels = np.array([0, 1])
    probabilities = np.array([0.4, 0.6])

    metrics = compute_metrics_for_class(labels, probabilities, threshold=0.7)

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["auroc"] == 1.0


def test_bootstrap_ci_is_deterministic_and_in_metric_bounds():
    labels = np.array([0, 0, 1, 1, 1, 0])
    probabilities = np.array([0.1, 0.7, 0.6, 0.9, 0.8, 0.2])
    metric = metric_function("f1", threshold=0.5)

    first = bootstrap_ci(labels, probabilities, metric, n_bootstrap=100, seed=7)
    second = bootstrap_ci(labels, probabilities, metric, n_bootstrap=100, seed=7)

    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0


def test_ranking_metrics_and_their_cis_are_nan_for_single_class_labels():
    labels = np.array([0, 0, 0])
    probabilities = np.array([0.1, 0.2, 0.3])

    metrics = compute_metrics_for_class(labels, probabilities)
    interval = bootstrap_ci(labels, probabilities, metric_function("auroc", 0.5), n_bootstrap=10)

    assert np.isnan(metrics["auroc"])
    assert np.isnan(metrics["aupr"])
    assert all(np.isnan(value) for value in interval)


def test_validation_threshold_selects_f1_optimum_without_using_test_data():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    assert select_validation_threshold(labels, probabilities) == 0.6
