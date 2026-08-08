"""
Generate example occlusion-based counterfactual figures from the test set,
saved to docs/counterfactual_examples/ for use in the README.

Usage:
    python src/explain/generate_counterfactual_examples.py \
        --checkpoint checkpoints/vision_baseline/best_model.pth \
        --num-examples 6
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ChestXrayDataset
from src.explain.counterfactual import OcclusionCounterfactualExplainer, make_side_by_side_figure
from src.explain.gradcam import ChestXrayExplainer
from src.models.vision_encoder import ChestXrayVisionModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def select_example_indices(
    model, test_ds, device, num_examples: int, strategy: str, seed: int = 42
) -> list[int]:
    """
    strategy="random": a representative, unbiased sample across the whole
        test set. The dataset script lists all Pneumonia-positive rows
        before Normal rows, so taking the first N (the old behavior) always
        picked the same class — this fixes that.
    strategy="borderline": picks the N test examples whose predicted
        probability is CLOSEST to the 0.5 decision boundary. Confident
        predictions (e.g. 0.999) rarely flip under occlusion regardless of
        method quality; borderline cases are where masking the
        highest-activation region is most likely to demonstrably change the
        decision, making for a much more informative demo figure.
    """
    if strategy == "random":
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(test_ds), size=min(num_examples, len(test_ds)), replace=False)
        return indices.tolist()

    if strategy == "borderline":
        probs = []
        with torch.no_grad():
            for i in range(len(test_ds)):
                image, _tabular, _labels = test_ds[i]
                logits = model(image.unsqueeze(0).to(device))
                prob = torch.sigmoid(logits)[0, 0].item()
                probs.append((i, abs(prob - 0.5)))
        probs.sort(key=lambda x: x[1])
        return [i for i, _ in probs[:num_examples]]

    raise ValueError(f"Unknown strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_baseline.yaml")
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument(
        "--strategy",
        choices=["random", "borderline"],
        default="random",
        help="'random' for a representative sample; 'borderline' to specifically "
        "showcase examples near the decision boundary, where flips are most likely.",
    )
    parser.add_argument("--output-dir", default="docs/counterfactual_examples")
    parser.add_argument(
        "--restrict-to-lungs",
        action="store_true",
        help="Zero out Grad-CAM activation outside segmented lung fields "
        "before building the occlusion mask (mitigates shortcut-learning "
        "risk — see docs/ethics_statement.md). Requires torchxrayvision.",
    )
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    train_cfg = load_config(args.train_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]

    model = ChestXrayVisionModel(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    gradcam_explainer = ChestXrayExplainer(model, device=str(device))
    cf_explainer = OcclusionCounterfactualExplainer(model, gradcam_explainer, device=str(device))

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

    indices = select_example_indices(model, test_ds, device, args.num_examples, args.strategy)

    flip_count = 0
    for i in indices:
        image, _tabular, labels = test_ds[i]

        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            top_class_idx = torch.sigmoid(logits)[0].argmax().item()

        result = cf_explainer.generate(
            image,
            class_idx=top_class_idx,
            class_name=classes[top_class_idx],
            threshold=args.threshold,
            restrict_to_lungs=args.restrict_to_lungs,
        )

        figure = make_side_by_side_figure(result)
        filename = f"example_{i}_{result.class_name}.png"
        cv2.imwrite(str(output_dir / filename), cv2.cvtColor(figure, cv2.COLOR_RGB2BGR))

        flip_status = "FLIPPED" if result.flipped else "no flip"
        print(
            f"Saved {filename} | {result.class_name}: "
            f"{result.original_probability:.3f} -> {result.counterfactual_probability:.3f} "
            f"({flip_status})"
        )
        if result.flipped:
            flip_count += 1

    print(f"\n{flip_count}/{len(indices)} examples flipped the prediction after occlusion.")
    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
