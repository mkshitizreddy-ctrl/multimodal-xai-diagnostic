"""
Generate a handful of example Grad-CAM overlays from the test set, saved to
docs/gradcam_examples/ for use in the README.

Usage:
    python src/explain/generate_examples.py \
        --checkpoint checkpoints/vision_baseline/best_model.pth \
        --num-examples 6
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ChestXrayDataset
from src.explain.gradcam import ChestXrayExplainer, save_overlay
from src.models.vision_encoder import ChestXrayVisionModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_baseline.yaml")
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--output-dir", default="docs/gradcam_examples")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    train_cfg = load_config(args.train_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint["classes"]

    model = ChestXrayVisionModel(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    explainer = ChestXrayExplainer(model, device=str(device))

    test_ds = ChestXrayDataset(
        csv_path=train_cfg["data"].get("test_csv", "data/processed/test.csv"),
        image_dir=train_cfg["data"]["image_dir"],
        classes=classes,
        tabular_features=data_cfg["tabular_features"],
        image_size=train_cfg["data"]["image_size"],
        train=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.num_examples, len(test_ds))):
        image, _tabular, _labels = test_ds[i]
        top_results = explainer.explain_top_k(image, classes, k=1)
        top = top_results[0]

        filename = f"example_{i}_{top['class']}_{top['probability']:.2f}.png"
        save_overlay(top["overlay"], str(output_dir / filename))
        print(f"Saved {filename}")

    print(f"\n{args.num_examples} example Grad-CAM overlays saved to {output_dir}/")
    print("Pick a few of these to embed directly in the README's Explainability section.")


if __name__ == "__main__":
    main()
