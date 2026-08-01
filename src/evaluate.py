"""
Evaluate a trained checkpoint on the test set and print/save a per-class
AUROC results table (used in the README Results section and the results
notebook).

Usage:
    python src/evaluate.py \
        --checkpoint checkpoints/vision_baseline/best_model.pth \
        --data-config configs/data.yaml \
        --train-config configs/vision_baseline.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import ChestXrayDataset
from src.models.vision_encoder import ChestXrayVisionModel
from src.train import compute_macro_auroc


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_evaluation(checkpoint_path: str, data_cfg: dict, train_cfg: dict) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]

    model = ChestXrayVisionModel(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    test_ds = ChestXrayDataset(
        csv_path=train_cfg["data"].get("test_csv", "data/processed/test.csv"),
        image_dir=train_cfg["data"]["image_dir"],
        classes=classes,
        tabular_features=checkpoint.get("tabular_features", data_cfg["tabular_features"]),
        image_size=train_cfg["data"]["image_size"],
        train=False,
        tabular_stats=checkpoint.get("tabular_stats"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg["train"]["batch_size"], shuffle=False
    )

    all_preds, all_labels = [], []
    for images, _tabular, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        all_preds.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    macro_auroc, per_class_auroc = compute_macro_auroc(all_labels, all_preds, classes)

    results = pd.DataFrame(
        [{"class": cls, "auroc": score} for cls, score in per_class_auroc.items()]
    ).sort_values("auroc", ascending=False)

    print(results.to_string(index=False))
    print(f"\nMacro AUROC: {macro_auroc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_baseline.yaml")
    parser.add_argument("--output-csv", default="docs/vision_baseline_results.csv")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    train_cfg = load_config(args.train_config)

    results = run_evaluation(args.checkpoint, data_cfg, train_cfg)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
