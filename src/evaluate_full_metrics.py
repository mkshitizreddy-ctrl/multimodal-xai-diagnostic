"""Report thresholded and ranking metrics for vision or fusion checkpoints.

Unlike :mod:`src.evaluate` and :mod:`src.evaluate_fusion`, which preserve
the project's cited AUROC-only result tables, this script reports Accuracy,
Precision, Recall, F1, AUROC, and AUPR.  Every reported metric has a
percentile-bootstrap 95% confidence interval over the held-out test set.

The decision threshold defaults to 0.5 and is never fitted on the test set.

Examples:
    python src/evaluate_full_metrics.py --checkpoint checkpoints/vision_baseline/best_model.pth
    python src/evaluate_full_metrics.py --checkpoint checkpoints/fusion/best_model.pth \\
        --train-config configs/fusion.yaml
"""

import argparse
import csv
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "auroc", "aupr")


def load_config(path: str) -> dict:
    import yaml

    with open(path) as file:
        return yaml.safe_load(file)


def get_predictions(model, loader, device, is_fusion: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return true labels and sigmoid probabilities, both shaped ``[N, C]``."""
    import torch
    from tqdm import tqdm

    all_labels, all_probabilities = [], []
    for batch in tqdm(loader, desc="Predicting"):
        images, tabular, labels = batch
        images = images.to(device)
        logits = model(images, tabular.to(device)) if is_fusion else model(images)
        all_probabilities.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels)
    return torch.cat(all_labels).numpy(), torch.cat(all_probabilities).numpy()


def compute_metrics_for_class(
    y_true: np.ndarray, y_probability: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute all supported metrics for one binary label.

    AUROC and AUPR are undefined for a single-class sample and are returned
    as NaN. Thresholded metrics remain defined in that case.
    """
    y_prediction = (y_probability >= threshold).astype(int)
    result = {
        "accuracy": float(accuracy_score(y_true, y_prediction)),
        "precision": float(precision_score(y_true, y_prediction, zero_division=0)),
        "recall": float(recall_score(y_true, y_prediction, zero_division=0)),
        "f1": float(f1_score(y_true, y_prediction, zero_division=0)),
    }
    if np.unique(y_true).size < 2:
        result.update(auroc=float("nan"), aupr=float("nan"))
    else:
        result.update(
            auroc=float(roc_auc_score(y_true, y_probability)),
            aupr=float(average_precision_score(y_true, y_probability)),
        )
    return result


def metric_function(metric_name: str, threshold: float) -> Callable[[np.ndarray, np.ndarray], float]:
    """Build a metric callable suitable for bootstrapping."""
    if metric_name not in METRIC_NAMES:
        raise ValueError(f"Unknown metric: {metric_name}")
    return lambda y_true, y_probability: compute_metrics_for_class(y_true, y_probability, threshold)[
        metric_name
    ]


def bootstrap_ci(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a percentile-bootstrap 95% CI for one metric.

    Degenerate resamples are excluded only when their metric is undefined
    (for example, AUROC on an all-negative resample). This permits CIs for
    thresholded metrics even on a single-class test set.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if len(y_true) == 0:
        raise ValueError("Cannot bootstrap an empty test set")

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        value = metric(y_true[indices], y_probability[indices])
        if not np.isnan(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def add_confidence_intervals(
    y_true: np.ndarray, y_probability: np.ndarray, threshold: float, n_bootstrap: int
) -> dict[str, float]:
    """Compute CIs for every metric from the same bootstrap resamples.

    Computing the complete metric dictionary once per resample is both
    statistically coherent (the intervals use an identical resampling plan)
    and six times faster than independently resampling each metric.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if len(y_true) == 0:
        raise ValueError("Cannot bootstrap an empty test set")

    rng = np.random.default_rng(42)
    samples = {name: [] for name in METRIC_NAMES}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        metrics = compute_metrics_for_class(y_true[indices], y_probability[indices], threshold)
        for metric_name, value in metrics.items():
            if not np.isnan(value):
                samples[metric_name].append(value)

    intervals = {}
    for metric_name, values in samples.items():
        if values:
            low, high = np.percentile(values, [2.5, 97.5])
        else:
            low = high = float("nan")
        intervals[f"{metric_name}_ci_low"] = float(low)
        intervals[f"{metric_name}_ci_high"] = float(high)
    return intervals


def select_validation_threshold(
    y_true: np.ndarray, y_probability: np.ndarray, metric_name: str = "f1"
) -> float:
    """Select a binary threshold on validation data only.

    Candidate thresholds are the observed probabilities plus 0 and 1. Ties
    are resolved toward 0.5, preserving the conventional threshold whenever
    it performs equally well. This function must never receive test labels.
    """
    if metric_name not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError("Threshold selection supports accuracy, precision, recall, or f1")
    candidates = np.unique(np.concatenate(([0.0], y_probability, [1.0])))
    scores = np.array(
        [compute_metrics_for_class(y_true, y_probability, float(value))[metric_name] for value in candidates]
    )
    best_candidates = candidates[np.isclose(scores, scores.max())]
    return float(best_candidates[np.argmin(np.abs(best_candidates - 0.5))])


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.dataset import ChestXrayDataset
    from src.explain.measure_lung_localization import is_fusion_checkpoint
    from src.models.fusion import ChestXrayFusionModel
    from src.models.vision_encoder import ChestXrayVisionModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_baseline.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--calibrate-threshold-on-validation",
        action="store_true",
        help="Select the threshold that maximizes validation F1, then evaluate once on test.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")

    data_config = load_config(args.data_config)
    train_config = load_config(args.train_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    is_fusion = is_fusion_checkpoint(checkpoint)
    use_cbam = checkpoint.get("use_cbam", False)
    use_se = checkpoint.get("use_se", False)

    if is_fusion:
        tabular_features = checkpoint["tabular_features"]
        model = ChestXrayFusionModel(
            num_classes=len(classes),
            num_tabular_features=len(tabular_features),
            pretrained=False,
            use_cbam=use_cbam,
        )
    else:
        tabular_features = data_config["tabular_features"]
        model = ChestXrayVisionModel(
            num_classes=len(classes), pretrained=False, use_cbam=use_cbam, use_se=use_se
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    test_dataset = ChestXrayDataset(
        csv_path=train_config["data"].get("test_csv", "data/processed/test.csv"),
        image_dir=train_config["data"]["image_dir"],
        classes=classes,
        tabular_features=tabular_features,
        image_size=train_config["data"]["image_size"],
        train=False,
        tabular_stats=checkpoint.get("tabular_stats"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_config["train"]["batch_size"], shuffle=False, num_workers=0
    )
    threshold = args.threshold
    if args.calibrate_threshold_on_validation:
        validation_dataset = ChestXrayDataset(
            csv_path=train_config["data"]["val_csv"],
            image_dir=train_config["data"]["image_dir"],
            classes=classes,
            tabular_features=tabular_features,
            image_size=train_config["data"]["image_size"],
            train=False,
            tabular_stats=checkpoint.get("tabular_stats"),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=train_config["train"]["batch_size"],
            shuffle=False,
            num_workers=0,
        )
        with torch.no_grad():
            validation_true, validation_probability = get_predictions(
                model, validation_loader, device, is_fusion
            )
        if len(classes) != 1:
            parser.error("Validation threshold calibration currently supports binary single-label models only")
        threshold = select_validation_threshold(validation_true[:, 0], validation_probability[:, 0])
        print(f"Selected threshold = {threshold:.4f} by maximizing validation F1.")
    with torch.no_grad():
        y_true, y_probability = get_predictions(model, test_loader, device, is_fusion)

    print(f"model_type = {'fusion' if is_fusion else 'vision'}")
    print(f"use_cbam = {use_cbam}")
    print(f"use_se = {use_se}")
    print(
        f"n = {len(y_true)} test images; threshold = {threshold:.4f}; "
        f"bootstrap samples = {args.n_bootstrap}\n"
    )

    rows = []
    for index, class_name in enumerate(classes):
        metrics = compute_metrics_for_class(y_true[:, index], y_probability[:, index], threshold)
        metrics.update(
            add_confidence_intervals(
                y_true[:, index], y_probability[:, index], threshold, args.n_bootstrap
            )
        )
        row = {"class": class_name, "threshold": threshold, **metrics}
        rows.append(row)

        print(f"Class: {class_name}")
        for metric_name in METRIC_NAMES:
            print(
                f"  {metric_name.upper():<9} {metrics[metric_name]:.4f}"
                f"  (95% CI: [{metrics[f'{metric_name}_ci_low']:.4f}, "
                f"{metrics[f'{metric_name}_ci_high']:.4f}])"
            )
        print()

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
