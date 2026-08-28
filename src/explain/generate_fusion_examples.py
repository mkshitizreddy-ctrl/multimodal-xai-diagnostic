"""
Generate Grad-CAM overlays and counterfactual comparisons from the FUSION
model (image + tabular), using FusionModelImageWrapper to bridge it to the
single-input explainability tools in gradcam.py and counterfactual.py.

Each example uses that specific test patient's REAL tabular vector (not a
random one) — the whole point is explaining what the model actually did for
that image+vitals combination, not a generic image-only explanation.

Usage:
    python src/explain/generate_fusion_examples.py \
        --checkpoint checkpoints/fusion/best_model.pth \
        --num-examples 6
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ChestXrayDataset
from src.explain.counterfactual import OcclusionCounterfactualExplainer, make_side_by_side_figure
from src.explain.fusion_wrapper import FusionModelImageWrapper
from src.explain.gradcam import ChestXrayExplainer, save_overlay
from src.models.fusion import ChestXrayFusionModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/fusion.yaml")
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="docs/gradcam_examples_fusion")
    parser.add_argument(
        "--restrict-to-lungs",
        action="store_true",
        help="Zero out Grad-CAM activation outside segmented lung fields "
        "(mitigates shortcut-learning risk — see docs/ethics_statement.md). "
        "Requires torchxrayvision.",
    )
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    train_cfg = load_config(args.train_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    tabular_features = checkpoint["tabular_features"]

    model = ChestXrayFusionModel(
        num_classes=len(classes),
        num_tabular_features=len(tabular_features),
        pretrained=False,
        use_cbam=checkpoint.get("use_cbam", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    tabular_stats = checkpoint.get("tabular_stats")
    if tabular_stats is None:
        print(
            "WARNING: checkpoint has no saved tabular_stats (trained with an "
            "older version of train_fusion.py) — fitting normalizers fresh "
            "from the test split instead of reusing train's. This will give "
            "inconsistent tabular scaling vs. what the model was trained on."
        )

    test_ds = ChestXrayDataset(
        csv_path=train_cfg["data"].get("test_csv", "data/processed/test.csv"),
        image_dir=train_cfg["data"]["image_dir"],
        classes=classes,
        tabular_features=tabular_features,
        image_size=train_cfg["data"]["image_size"],
        train=False,
        tabular_stats=tabular_stats,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    num_examples = min(args.num_examples, len(test_ds))
    indices = rng.choice(len(test_ds), size=num_examples, replace=False)

    for i in indices:
        image, tabular, _labels = test_ds[int(i)]

        # Fresh wrapper per example — the whole point is fixing THIS
        # patient's tabular vector for THIS explanation, not sharing one
        # wrapper (and its stale fixed_tabular buffer) across examples.
        wrapper = FusionModelImageWrapper(model, tabular).to(device)
        gradcam_explainer = ChestXrayExplainer(wrapper, device=str(device))

        top_results = gradcam_explainer.explain_top_k(
            image, classes, k=1, restrict_to_lungs=args.restrict_to_lungs
        )
        top = top_results[0]

        gradcam_filename = f"example_{i}_{top['class']}_{top['probability']:.2f}_gradcam.png"
        save_overlay(top["overlay"], str(output_dir / gradcam_filename))

        cf_explainer = OcclusionCounterfactualExplainer(wrapper, gradcam_explainer, device=str(device))
        top_class_idx = classes.index(top["class"])
        cf_result = cf_explainer.generate(image, top_class_idx, top["class"])
        cf_figure = make_side_by_side_figure(cf_result)
        cf_filename = f"example_{i}_{top['class']}_{top['probability']:.2f}_counterfactual.png"
        save_overlay(cf_figure, str(output_dir / cf_filename))

        print(
            f"Saved {gradcam_filename} and {cf_filename} "
            f"(tabular: {dict(zip(tabular_features, tabular.tolist()))})"
        )

    print(f"\n{num_examples} example pairs saved to {output_dir}/")
    print("Note: tabular values are normalized (per tabular_stats in the checkpoint), not raw units.")


if __name__ == "__main__":
    main()
