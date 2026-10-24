
"""Measure occlusion mask coverage and probability changes for two settings.

This script does not modify existing figures or model checkpoints.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ChestXrayDataset
from src.explain.counterfactual import OcclusionCounterfactualExplainer
from src.explain.gradcam import ChestXrayExplainer
from src.models.vision_encoder import ChestXrayVisionModel


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Measure counterfactual mask coverage and probability changes."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[760, 898, 504, 1000, 510, 103])
    args = parser.parse_args()

    output_path = Path(args.output_csv)
    if output_path.exists():
        parser.error(f"Refusing to overwrite existing CSV: {output_path}")

    data_cfg = load_config(args.data_config)
    train_cfg = load_config(args.train_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    classes = checkpoint["classes"]

    model = ChestXrayVisionModel(
        num_classes=len(classes),
        pretrained=False,
        use_cbam=checkpoint.get("use_cbam", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    gradcam = ChestXrayExplainer(model, device=str(device))
    explainer = OcclusionCounterfactualExplainer(
        model, gradcam, device=str(device)
    )

    dataset = ChestXrayDataset(
        csv_path=train_cfg["data"].get("test_csv", "data/processed/test.csv"),
        image_dir=train_cfg["data"]["image_dir"],
        classes=classes,
        tabular_features=data_cfg["tabular_features"],
        image_size=train_cfg["data"]["image_size"],
        train=False,
    )

    if any(i < 0 or i >= len(dataset) for i in args.indices):
        parser.error(f"All indices must be in [0, {len(dataset) - 1}].")

    # Each row is measured directly from the raw in-memory mask, not a PNG.
    settings = [
        {"name": "threshold_070_dilation_5", "threshold": 0.7, "dilate_px": 5},
        {"name": "threshold_060_dilation_0", "threshold": 0.6, "dilate_px": 0},
    ]

    rows = []
    for index in args.indices:
        image, _tabular, _labels = dataset[index]

        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            class_idx = torch.sigmoid(logits)[0].argmax().item()

        for setting in settings:
            result = explainer.generate(
                image,
                class_idx=class_idx,
                class_name=classes[class_idx],
                threshold=setting["threshold"],
                dilate_px=setting["dilate_px"],
            )

            coverage = 100.0 * np.count_nonzero(result.mask) / result.mask.size
            delta = (
                result.counterfactual_probability - result.original_probability
            )

            row = {
                "test_index": index,
                "class_name": result.class_name,
                "setting": setting["name"],
                "threshold": setting["threshold"],
                "dilate_px": setting["dilate_px"],
                "original_probability": f"{result.original_probability:.10f}",
                "counterfactual_probability": f"{result.counterfactual_probability:.10f}",
                "probability_change_cf_minus_original": f"{delta:.10f}",
                "probability_drop_original_minus_cf": f"{-delta:.10f}",
                "mask_coverage_percent": f"{coverage:.6f}",
                "prediction_flipped": result.flipped,
            }
            rows.append(row)

            print(
                f"Index {index} | {setting['name']} | "
                f"P: {result.original_probability:.8f} -> "
                f"{result.counterfactual_probability:.8f} | "
                f"change: {delta:+.8f} | mask: {coverage:.4f}% | "
                f"flip: {result.flipped}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMeasurements saved to: {output_path}")
    for setting in settings:
        subset = [
            row for row in rows if row["setting"] == setting["name"]
        ]
        coverages = [float(row["mask_coverage_percent"]) for row in subset]
        changes = [
            float(row["probability_change_cf_minus_original"])
            for row in subset
        ]
        print(
            f"{setting['name']}: mean raw mask coverage = "
            f"{np.mean(coverages):.4f}%; "
            f"mean probability change = {np.mean(changes):+.8f}"
        )


if __name__ == "__main__":
    main()
