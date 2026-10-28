
"""Independent verification of paired AUROC comparisons.

Implements DeLong's test for two correlated ROC AUCs, following
Sun & Xu (2014), plus a stratified paired-bootstrap percentile CI.

Existing project scripts and results are not modified.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compute_midrank(values: np.ndarray) -> np.ndarray:
    """Return one-based midranks, correctly handling tied scores."""
    values = np.asarray(values)
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)
    sorted_ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        sorted_ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j

    ranks = np.empty(n, dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def fast_delong(
    predictions_sorted: np.ndarray,
    n_positive: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate AUCs and their covariance using the fast DeLong algorithm.

    predictions_sorted has shape (2, n_samples), with positive-label
    observations first and negative-label observations second.
    """
    n_models, n_samples = predictions_sorted.shape
    m = n_positive
    n = n_samples - m

    if m < 2 or n < 2:
        raise ValueError("DeLong analysis requires at least two examples per class.")

    positive = predictions_sorted[:, :m]
    negative = predictions_sorted[:, m:]

    tx = np.empty((n_models, m))
    ty = np.empty((n_models, n))
    tz = np.empty((n_models, n_samples))

    for r in range(n_models):
        tx[r] = compute_midrank(positive[r])
        ty[r] = compute_midrank(negative[r])
        tz[r] = compute_midrank(predictions_sorted[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    covariance = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(covariance)


def delong_test(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
) -> dict:
    """Two-sided DeLong test for two models evaluated on the same cases."""
    y_true = np.asarray(y_true).reshape(-1)
    prob_a = np.asarray(prob_a).reshape(-1)
    prob_b = np.asarray(prob_b).reshape(-1)

    if not (len(y_true) == len(prob_a) == len(prob_b)):
        raise ValueError("Labels and prediction arrays must have equal lengths.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Ground-truth labels must be binary values 0 and 1.")
    if not (np.isfinite(prob_a).all() and np.isfinite(prob_b).all()):
        raise ValueError("Predictions contain NaN or infinite values.")

    n_positive = int(np.sum(y_true == 1))
    n_negative = int(np.sum(y_true == 0))

    if n_positive < 2 or n_negative < 2:
        raise ValueError("Need at least two positive and two negative examples.")

    # DeLong requires positives to appear first.
    order = np.argsort(-y_true)
    predictions = np.vstack([prob_a, prob_b])[:, order]

    aucs, covariance = fast_delong(predictions, n_positive)
    contrast = np.array([1.0, -1.0])
    variance_diff = float(contrast @ covariance @ contrast)

    observed_diff = float(aucs[1] - aucs[0])

    if variance_diff <= 0 or not np.isfinite(variance_diff):
        raise ValueError(
            "DeLong variance of the AUC difference is non-positive or invalid; "
            "the test cannot provide a reliable p-value for these predictions."
        )

    z_statistic = observed_diff / np.sqrt(variance_diff)
    p_value = float(2.0 * norm.sf(abs(z_statistic)))

    return {
        "auroc_a": float(aucs[0]),
        "auroc_b": float(aucs[1]),
        "observed_diff": observed_diff,
        "delong_z": float(z_statistic),
        "delong_p_value": p_value,
        "delong_se_diff": float(np.sqrt(variance_diff)),
    }


def stratified_paired_bootstrap_ci(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict:
    """Percentile CI for B-A using paired, class-stratified resampling."""
    y_true = np.asarray(y_true).reshape(-1)
    prob_a = np.asarray(prob_a).reshape(-1)
    prob_b = np.asarray(prob_b).reshape(-1)

    positive_idx = np.flatnonzero(y_true == 1)
    negative_idx = np.flatnonzero(y_true == 0)

    if len(positive_idx) < 2 or len(negative_idx) < 2:
        raise ValueError("Bootstrap requires examples from both classes.")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sampled_positive = rng.choice(
            positive_idx, size=len(positive_idx), replace=True
        )
        sampled_negative = rng.choice(
            negative_idx, size=len(negative_idx), replace=True
        )
        indices = np.concatenate([sampled_positive, sampled_negative])

        diffs[i] = (
            roc_auc_score(y_true[indices], prob_b[indices])
            - roc_auc_score(y_true[indices], prob_a[indices])
        )

    low, high = np.percentile(diffs, [2.5, 97.5])

    return {
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_mean_diff": float(np.mean(diffs)),
        "n_bootstrap": n_bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a paired test-set AUROC comparison."
    )
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--label-a", default="image_only")
    parser.add_argument("--label-b", default="image_plus_vitals_rho0")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--train-config", default="configs/vision_baseline_rho_v2_0.yaml"
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv",
        default="docs/auroc_verification_vision_vs_fusion_rho_v2_0.csv",
    )
    args = parser.parse_args()

    if args.n_bootstrap < 100:
        parser.error("--n-bootstrap must be at least 100.")

    from src.compare_auroc_paired import load_predictions
    from src.evaluate_full_metrics import load_config

    data_config = load_config(args.data_config)
    train_config = load_config(args.train_config)

    print(f"Loading {args.label_a} predictions...")
    true_a, prob_a = load_predictions(
        args.checkpoint_a, data_config, train_config
    )

    print(f"Loading {args.label_b} predictions...")
    true_b, prob_b = load_predictions(
        args.checkpoint_b, data_config, train_config
    )

    if not np.array_equal(true_a, true_b):
        raise ValueError(
            "The checkpoints do not have identical test labels in identical "
            "order. Refusing to compare mismatched test sets."
        )

    y_true = np.asarray(true_a).reshape(-1)
    prob_a = np.asarray(prob_a).reshape(-1)
    prob_b = np.asarray(prob_b).reshape(-1)

    result = delong_test(y_true, prob_a, prob_b)
    bootstrap = stratified_paired_bootstrap_ci(
        y_true, prob_a, prob_b,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    print(f"\nTest images: {len(y_true)}")
    print(f"AUROC {args.label_a}: {result['auroc_a']:.6f}")
    print(f"AUROC {args.label_b}: {result['auroc_b']:.6f}")
    print(f"Observed difference (B - A): {result['observed_diff']:+.6f}")
    print(
        "Paired-bootstrap 95% CI: "
        f"[{bootstrap['ci_low']:.6f}, {bootstrap['ci_high']:.6f}]"
    )
    print(f"DeLong z-statistic: {result['delong_z']:.4f}")
    print(f"DeLong two-sided p-value: {result['delong_p_value']:.8g}")

    output_path = Path(args.output_csv)
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. Choose a new --output-csv path "
            "to avoid overwriting previous results."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "n": len(y_true),
        **result,
        **bootstrap,
        "bootstrap_seed": args.seed,
    }

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"\nSaved verification results to {output_path}")


if __name__ == "__main__":
    main()
