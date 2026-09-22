"""Paired bootstrap comparison of test-set AUROC between two checkpoints.

The README's label-smoothing writeup originally compared two independently-
bootstrapped 95% CIs (baseline vs. label smoothing) and noted that as a
real but slightly conservative way to judge the AUROC difference, since it
ignores that both models are scored on the *same* 624 test images - some of
which are just easier or harder regardless of which model is used.

A paired bootstrap fixes this: the same resampled indices are used to score
both models on each iteration, and the CI is built directly on the
difference (AUROC_b - AUROC_a), which is more sensitive and the more
standard way to test "is model B's AUROC actually different from model A's
on this test set."

Example:
    python src/compare_auroc_paired.py \\
        --checkpoint-a checkpoints/vision_baseline/best_model.pth \\
        --checkpoint-b checkpoints/vision_label_smoothing/best_model.pth \\
        --label-a baseline --label-b label_smoothing \\
        --output-csv docs/paired_bootstrap_label_smoothing.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_predictions(checkpoint_path: str, data_config: dict, train_config: dict):
    """Load a checkpoint and return (y_true, y_probability) on the test set.

    Mirrors the model-reconstruction logic in evaluate_full_metrics.py /
    evaluate_calibration.py so this stays consistent with how the rest of
    the project builds a model from a checkpoint.
    """
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import ChestXrayDataset
    from src.evaluate_full_metrics import get_predictions
    from src.explain.measure_lung_localization import is_fusion_checkpoint
    from src.models.fusion import ChestXrayFusionModel
    from src.models.vision_encoder import ChestXrayVisionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    is_fusion = is_fusion_checkpoint(checkpoint)
    use_cbam = checkpoint.get("use_cbam", False)

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
        model = ChestXrayVisionModel(num_classes=len(classes), pretrained=False, use_cbam=use_cbam)
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
    with torch.no_grad():
        y_true, y_probability = get_predictions(model, test_loader, device, is_fusion)
    return y_true[:, 0], y_probability[:, 0]


def paired_bootstrap_auroc(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap the AUROC difference (B - A) using shared resample indices.

    Both models are scored on the identical resampled indices each
    iteration, so the resulting CI is on the paired difference itself, not
    on two separately-resampled quantities. A resample that is single-class
    (all one label) is skipped, since AUROC is undefined there.
    """
    if len(y_true) == 0:
        raise ValueError("Cannot bootstrap an empty test set")

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        resampled_true = y_true[indices]
        if np.unique(resampled_true).size < 2:
            continue
        auroc_a = roc_auc_score(resampled_true, prob_a[indices])
        auroc_b = roc_auc_score(resampled_true, prob_b[indices])
        diffs.append(auroc_b - auroc_a)

    diffs = np.array(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # Two-sided bootstrap p-value: proportion of resamples where the sign
    # of the difference flips relative to the observed direction, doubled.
    # Standard nonparametric approach when a closed-form test isn't
    # applicable (diffs need not be normally distributed).
    observed_diff = float(np.mean(diffs))
    p_value = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    p_value = min(p_value, 1.0)

    return {
        "mean_diff": observed_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "n_valid_resamples": len(diffs),
    }


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.evaluate_full_metrics import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", required=True, help="Baseline / reference checkpoint")
    parser.add_argument("--checkpoint-b", required=True, help="Comparison checkpoint")
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_baseline.yaml")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    train_config = load_config(args.train_config)

    print(f"Loading predictions for {args.label_a} ({args.checkpoint_a}) ...")
    true_a, prob_a = load_predictions(args.checkpoint_a, data_config, train_config)
    print(f"Loading predictions for {args.label_b} ({args.checkpoint_b}) ...")
    true_b, prob_b = load_predictions(args.checkpoint_b, data_config, train_config)

    if not np.array_equal(true_a, true_b):
        raise ValueError(
            "The two checkpoints produced different true-label orderings on the test set - "
            "they may be using different test CSVs/splits. Refusing to pair mismatched labels."
        )
    y_true = true_a

    auroc_a = roc_auc_score(y_true, prob_a)
    auroc_b = roc_auc_score(y_true, prob_b)
    result = paired_bootstrap_auroc(y_true, prob_a, prob_b, args.n_bootstrap)

    print(f"\nn = {len(y_true)} test images; paired bootstrap samples = {result['n_valid_resamples']}\n")
    print(f"AUROC {args.label_a} = {auroc_a:.4f}")
    print(f"AUROC {args.label_b} = {auroc_b:.4f}")
    print(f"Observed difference ({args.label_b} - {args.label_a}) = {auroc_b - auroc_a:.4f}")
    print(
        f"Paired bootstrap: mean diff = {result['mean_diff']:.4f}, "
        f"95% CI = [{result['ci_low']:.4f}, {result['ci_high']:.4f}], "
        f"p = {result['p_value']:.4f}"
    )
    if result["ci_low"] <= 0 <= result["ci_high"]:
        print("-> CI includes 0: the difference is NOT statistically significant at the 95% level.")
    else:
        print("-> CI excludes 0: the difference IS statistically significant at the 95% level.")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["label_a", "label_b", "auroc_a", "auroc_b", "mean_diff", "ci_low", "ci_high", "p_value", "n"]
            )
            writer.writerow(
                [
                    args.label_a,
                    args.label_b,
                    auroc_a,
                    auroc_b,
                    result["mean_diff"],
                    result["ci_low"],
                    result["ci_high"],
                    result["p_value"],
                    len(y_true),
                ]
            )
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
