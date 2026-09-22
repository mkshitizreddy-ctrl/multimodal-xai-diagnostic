"""Fit a temperature-scaling scalar to fix the overconfidence found in
evaluate_calibration.py, and report before/after calibration on the test set.

Temperature scaling (Guo et al., 2017) divides logits by a single learned
scalar T > 1 before the sigmoid, softening overconfident probabilities. T is
fit by minimizing binary cross-entropy on the validation set only - the test
set is used exclusively to report the before/after comparison, never to fit
T, matching the validation/test discipline already used for threshold
selection in evaluate_full_metrics.py.

Because dividing every logit by the same constant preserves their relative
order, temperature scaling changes confidence but not ranking: AUROC/AUPR
are unaffected. Only ECE/Brier/reliability are expected to change.

Example:
    python src/fit_temperature.py --checkpoint checkpoints/vision_baseline/best_model.pth \\
        --output-csv docs/temperature_scaling_vision.csv \\
        --output-plot docs/reliability_diagram_vision_after_temp.png
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Fit a single scalar T minimizing BCE(labels, sigmoid(logits / T)) on the given data.

    Uses LBFGS on a single parameter, the standard approach from Guo et al.
    T is initialized at 1.0 (i.e. no scaling) and constrained positive by
    optimizing log(T) instead of T directly, since T <= 0 is undefined.
    """
    import torch

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter)
    criterion = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = criterion(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).item())


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.dataset import ChestXrayDataset
    from src.evaluate_calibration import compute_brier, compute_ece, get_logits, plot_reliability_diagram
    from src.evaluate_full_metrics import load_config
    from src.explain.measure_lung_localization import is_fusion_checkpoint
    from src.models.fusion import ChestXrayFusionModel
    from src.models.vision_encoder import ChestXrayVisionModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_baseline.yaml")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Path to save the post-scaling reliability diagram PNG. Skipped if omitted.",
    )
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    train_config = load_config(args.train_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    is_fusion = is_fusion_checkpoint(checkpoint)
    use_cbam = checkpoint.get("use_cbam", False)
    if len(classes) != 1:
        parser.error("Temperature scaling here supports binary single-label models only")

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

    def make_loader(csv_path: str) -> DataLoader:
        dataset = ChestXrayDataset(
            csv_path=csv_path,
            image_dir=train_config["data"]["image_dir"],
            classes=classes,
            tabular_features=tabular_features,
            image_size=train_config["data"]["image_size"],
            train=False,
            tabular_stats=checkpoint.get("tabular_stats"),
        )
        return DataLoader(dataset, batch_size=train_config["train"]["batch_size"], shuffle=False, num_workers=0)

    validation_loader = make_loader(train_config["data"]["val_csv"])
    test_loader = make_loader(train_config["data"].get("test_csv", "data/processed/test.csv"))

    with torch.no_grad():
        validation_true, validation_logits = get_logits(model, validation_loader, device, is_fusion)
        test_true, test_logits = get_logits(model, test_loader, device, is_fusion)

    validation_true = validation_true[:, 0]
    validation_logits = validation_logits[:, 0]
    test_true = test_true[:, 0]
    test_logits = test_logits[:, 0]

    temperature = fit_temperature(validation_logits, validation_true)
    print(f"model_type = {'fusion' if is_fusion else 'vision'}")
    print(f"use_cbam = {use_cbam}")
    print(f"Fitted temperature (on validation, n={len(validation_true)}) = {temperature:.4f}\n")

    probability_before = 1 / (1 + np.exp(-test_logits))
    probability_after = 1 / (1 + np.exp(-test_logits / temperature))

    ece_before, bins_before = compute_ece(test_true, probability_before, args.n_bins)
    ece_after, bins_after = compute_ece(test_true, probability_after, args.n_bins)
    brier_before = compute_brier(test_true, probability_before)
    brier_after = compute_brier(test_true, probability_after)

    print(f"n = {len(test_true)} test images; bins = {args.n_bins}\n")
    print(f"{'':10}{'ECE':>10}{'Brier':>10}")
    print(f"{'before':10}{ece_before:>10.4f}{brier_before:>10.4f}")
    print(f"{'after':10}{ece_after:>10.4f}{brier_after:>10.4f}")

    # Sanity check: temperature scaling must not change ranking. Any
    # difference in AUROC here would indicate a bug (e.g. rank-breaking
    # numerical issues), not a real effect of temperature scaling.
    from sklearn.metrics import roc_auc_score

    auroc_before = roc_auc_score(test_true, probability_before)
    auroc_after = roc_auc_score(test_true, probability_after)
    print(f"\nAUROC before = {auroc_before:.4f}, after = {auroc_after:.4f} (expected to match - ranking is unaffected)")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["stage", "temperature", "n", "ece", "brier", "auroc"])
            writer.writerow(["before", 1.0, len(test_true), ece_before, brier_before, auroc_before])
            writer.writerow(["after", temperature, len(test_true), ece_after, brier_after, auroc_after])
        print(f"\nSaved results to {output_path}")

    if args.output_plot:
        plot_reliability_diagram(
            bins_after,
            ece_after,
            Path(args.output_plot),
            title=f"{'Fusion' if is_fusion else 'Vision'} - after temperature scaling (T={temperature:.2f})",
        )
        print(f"Saved post-scaling reliability diagram to {args.output_plot}")


if __name__ == "__main__":
    main()
