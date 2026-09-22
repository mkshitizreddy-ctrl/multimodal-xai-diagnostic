"""Fit isotonic regression calibration, since temperature scaling
(fit_temperature.py) failed to meaningfully reduce ECE.

Unlike a single global temperature, isotonic regression fits a non-decreasing
step function mapping raw probability -> calibrated probability, learned on
validation data. It can correct one region of the probability range (e.g.
the 0.9-1.0 bin, where evaluate_calibration.py found the overconfidence
concentrated) independently of the rest, which a single scalar cannot do.

Caveat: isotonic regression is monotonic but, unlike temperature scaling, is
not mathematically guaranteed to exactly preserve AUROC - it can in
principle re-order predictions that were extremely close together. This
script reports AUROC before/after explicitly rather than assuming it's
unchanged.

With ~800 validation examples this is a reasonable amount of data for
isotonic fitting, but on the smaller side - the reliability diagram after
fitting is worth a visual sanity check for a jagged, overfit-looking curve.

Example:
    python src/fit_isotonic.py --checkpoint checkpoints/vision_baseline/best_model.pth \\
        --output-csv docs/isotonic_vision.csv \\
        --output-plot docs/reliability_diagram_vision_after_isotonic.png
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    import torch
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score
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
        help="Path to save the post-calibration reliability diagram PNG. Skipped if omitted.",
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
        parser.error("Isotonic calibration here supports binary single-label models only")

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

    validation_probability = 1 / (1 + np.exp(-validation_logits))
    test_probability_before = 1 / (1 + np.exp(-test_logits))

    # out_of_bounds="clip" avoids NaN/extrapolation surprises for any test
    # probability that falls outside the validation set's observed range.
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(validation_probability, validation_true)
    test_probability_after = calibrator.predict(test_probability_before)

    print(f"model_type = {'fusion' if is_fusion else 'vision'}")
    print(f"use_cbam = {use_cbam}")
    print(f"Isotonic calibrator fit on validation, n={len(validation_true)}\n")

    ece_before, _ = compute_ece(test_true, test_probability_before, args.n_bins)
    ece_after, bins_after = compute_ece(test_true, test_probability_after, args.n_bins)
    brier_before = compute_brier(test_true, test_probability_before)
    brier_after = compute_brier(test_true, test_probability_after)
    auroc_before = roc_auc_score(test_true, test_probability_before)
    auroc_after = roc_auc_score(test_true, test_probability_after)

    print(f"n = {len(test_true)} test images; bins = {args.n_bins}\n")
    print(f"{'':10}{'ECE':>10}{'Brier':>10}{'AUROC':>10}")
    print(f"{'before':10}{ece_before:>10.4f}{brier_before:>10.4f}{auroc_before:>10.4f}")
    print(f"{'after':10}{ece_after:>10.4f}{brier_after:>10.4f}{auroc_after:>10.4f}")
    if abs(auroc_after - auroc_before) > 0.005:
        print(
            "\nNOTE: AUROC shifted by more than 0.005 - unlike temperature scaling, isotonic "
            "regression is not guaranteed to preserve ranking. Worth checking this isn't masking "
            "a real degradation before reporting the ECE improvement alone."
        )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["stage", "n", "ece", "brier", "auroc"])
            writer.writerow(["before", len(test_true), ece_before, brier_before, auroc_before])
            writer.writerow(["after", len(test_true), ece_after, brier_after, auroc_after])
        print(f"\nSaved results to {output_path}")

    if args.output_plot:
        plot_reliability_diagram(
            bins_after,
            ece_after,
            Path(args.output_plot),
            title=f"{'Fusion' if is_fusion else 'Vision'} - after isotonic calibration",
        )
        print(f"Saved post-calibration reliability diagram to {args.output_plot}")


if __name__ == "__main__":
    main()
