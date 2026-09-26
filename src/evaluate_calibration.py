"""Report calibration quality for vision or fusion checkpoints.

A model can have strong AUROC while still being badly miscalibrated - i.e.
when it says "90% confident", it isn't actually right 90% of the time. This
matters for a clinical-facing tool where a confidence number might inform a
decision, so it's evaluated separately from the ranking/thresholded metrics
in :mod:`src.evaluate_full_metrics`.

Reports:
    * Expected Calibration Error (ECE), equal-width binning, with a
      percentile-bootstrap 95% CI over the held-out test set.
    * Brier score (mean squared error between confidence and outcome).
    * A reliability diagram (predicted confidence vs. observed positive rate
      per bin) saved as a PNG.

The threshold used elsewhere in the project (for accuracy/precision/recall)
is irrelevant here - calibration is evaluated over the full probability
range, not at a single cutoff.

Examples:
    python src/evaluate_calibration.py --checkpoint checkpoints/vision_baseline/best_model.pth
    python src/evaluate_calibration.py --checkpoint checkpoints/fusion/best_model.pth \\
        --train-config configs/fusion.yaml --output-csv docs/calibration_fusion.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

METRIC_NAMES = ("ece", "brier")


def compute_ece(
    y_true: np.ndarray, y_probability: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict]]:
    """Compute Expected Calibration Error with equal-width probability bins.

    Returns the scalar ECE plus per-bin detail (bin range, sample count,
    mean confidence, observed positive rate) for building the reliability
    diagram. Empty bins are skipped in both the ECE sum and the returned
    detail, since they carry no information about the model's calibration.

    The observed value in each probability bin is the empirical frequency
    of the positive class, not thresholded decision accuracy. This follows
    the standard probability-calibration convention used by
    sklearn.calibration.calibration_curve.
    """
    if not 1 <= n_bins <= len(y_true):
        raise ValueError("n_bins must be between 1 and the number of samples")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    bins = []
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        # Include the right edge only for the final bin, so every sample
        # falls in exactly one bin.
        if high == bin_edges[-1]:
            in_bin = (y_probability >= low) & (y_probability <= high)
        else:
            in_bin = (y_probability >= low) & (y_probability < high)
        count = int(in_bin.sum())
        if count == 0:
            continue

        confidence = float(y_probability[in_bin].mean())
        # Observed frequency of the positive class in this bin - not
        # decision accuracy. Binning by raw P(positive) and then comparing
        # against decision accuracy mixes two different calibration
        # conventions and gives a meaningless curve. This is the same
        # convention as sklearn.calibration.calibration_curve.
        observed_rate = float(y_true[in_bin].mean())
        weight = count / len(y_true)
        ece += weight * abs(confidence - observed_rate)
        bins.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "count": count,
                "confidence": confidence,
                "observed_rate": observed_rate,
            }
        )
    return float(ece), bins


def compute_brier(y_true: np.ndarray, y_probability: np.ndarray) -> float:
    """Mean squared error between predicted probability and true label."""
    return float(np.mean((y_probability - y_true) ** 2))


def get_logits(model, loader, device, is_fusion: bool):
    """Return true labels and raw logits, both shaped [N, C].

    Temperature scaling needs the pre-sigmoid logits - dividing a
    probability by T is not equivalent to dividing the logit by T before
    applying sigmoid, so this can't reuse evaluate_full_metrics.get_predictions
    directly.
    """
    import torch
    from tqdm import tqdm

    all_labels, all_logits = [], []
    for batch in tqdm(loader, desc="Predicting (logits)"):
        images, tabular, labels = batch
        images = images.to(device)
        logits = model(images, tabular.to(device)) if is_fusion else model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_labels).numpy(), torch.cat(all_logits).numpy()


def bootstrap_calibration_ci(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    n_bins: int,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Percentile-bootstrap 95% CIs for ECE and Brier score.

    Mirrors the resampling approach in evaluate_full_metrics.add_confidence_intervals:
    both metrics are computed from the same resample each iteration so the
    intervals share a single, statistically coherent resampling plan.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if len(y_true) == 0:
        raise ValueError("Cannot bootstrap an empty test set")

    rng = np.random.default_rng(seed)
    ece_samples, brier_samples = [], []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        resampled_true, resampled_probability = y_true[indices], y_probability[indices]
        # A resample can land all-empty in a bin edge case; ECE just skips
        # empty bins, so this never raises.
        ece_value, _ = compute_ece(resampled_true, resampled_probability, n_bins)
        ece_samples.append(ece_value)
        brier_samples.append(compute_brier(resampled_true, resampled_probability))

    ece_low, ece_high = np.percentile(ece_samples, [2.5, 97.5])
    brier_low, brier_high = np.percentile(brier_samples, [2.5, 97.5])
    return {
        "ece": (float(ece_low), float(ece_high)),
        "brier": (float(brier_low), float(brier_high)),
    }


def plot_reliability_diagram(bins: list[dict], ece: float, output_path: Path, title: str) -> None:
    """Save a reliability diagram: observed positive rate vs. predicted confidence per bin."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    bin_centers = [(entry["bin_low"] + entry["bin_high"]) / 2 for entry in bins]
    observed_rates = [entry["observed_rate"] for entry in bins]
    counts = [entry["count"] for entry in bins]
    bar_width = 1.0 / max(len(bins), 1) * 0.9

    ax.bar(
        bin_centers,
        observed_rates,
        width=bar_width,
        edgecolor="black",
        alpha=0.7,
        label="Observed rate",
    )
    for center, count in zip(bin_centers, counts):
        ax.annotate(str(count), (center, 0.02), ha="center", fontsize=7, color="dimgray")

    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.dataset import ChestXrayDataset
    from src.evaluate_full_metrics import get_predictions, load_config
    from src.explain.measure_lung_localization import is_fusion_checkpoint
    from src.models.fusion import ChestXrayFusionModel
    from src.models.vision_encoder import ChestXrayVisionModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_baseline.yaml")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Path to save the reliability diagram PNG. Skipped if omitted.",
    )
    args = parser.parse_args()

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
    with torch.no_grad():
        y_true, y_probability = get_predictions(model, test_loader, device, is_fusion)

    print(f"model_type = {'fusion' if is_fusion else 'vision'}")
    print(f"use_cbam = {use_cbam}")
    print(f"use_se = {use_se}")
    print(f"n = {len(y_true)} test images; bins = {args.n_bins}; bootstrap samples = {args.n_bootstrap}\n")

    rows = []
    for index, class_name in enumerate(classes):
        class_true = y_true[:, index]
        class_probability = y_probability[:, index]

        ece, bins = compute_ece(class_true, class_probability, args.n_bins)
        brier = compute_brier(class_true, class_probability)
        intervals = bootstrap_calibration_ci(class_true, class_probability, args.n_bins, args.n_bootstrap)

        row = {
            "class": class_name,
            "n_bins": args.n_bins,
            "ece": ece,
            "ece_ci_low": intervals["ece"][0],
            "ece_ci_high": intervals["ece"][1],
            "brier": brier,
            "brier_ci_low": intervals["brier"][0],
            "brier_ci_high": intervals["brier"][1],
        }
        rows.append(row)

        print(f"Class: {class_name}")
        print(f"  ECE     {ece:.4f}  (95% CI: [{intervals['ece'][0]:.4f}, {intervals['ece'][1]:.4f}])")
        print(f"  BRIER   {brier:.4f}  (95% CI: [{intervals['brier'][0]:.4f}, {intervals['brier'][1]:.4f}])")
        print()

        if args.output_plot:
            plot_path = Path(args.output_plot)
            if len(classes) > 1:
                plot_path = plot_path.with_name(f"{plot_path.stem}_{class_name}{plot_path.suffix}")
            plot_reliability_diagram(
                bins, ece, plot_path, title=f"{'Fusion' if is_fusion else 'Vision'} - {class_name}"
            )
            print(f"  Saved reliability diagram to {plot_path}")

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
