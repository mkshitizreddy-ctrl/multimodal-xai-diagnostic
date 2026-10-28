
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ChestXrayDataset
from src.explain.generate_counterfactual_examples import load_config
from src.explain.gradcam import ChestXrayExplainer
from src.models.vision_encoder import ChestXrayVisionModel


def main():
    data_cfg = load_config("configs/data.yaml")
    train_cfg = load_config(
        "configs/vision_patient_disjoint_dedup_rho_v1_0.yaml"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        "checkpoints/vision_patient_disjoint_dedup_rho_v1_0/best_model.pth",
        map_location=device,
        weights_only=False,
    )
    classes = checkpoint["classes"]

    model = ChestXrayVisionModel(
        num_classes=len(classes),
        pretrained=False,
        use_cbam=checkpoint.get("use_cbam", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    explainer = ChestXrayExplainer(model, device=str(device))

    dataset = ChestXrayDataset(
        csv_path=train_cfg["data"].get(
            "test_csv", "data/processed/test.csv"
        ),
        image_dir=train_cfg["data"]["image_dir"],
        classes=classes,
        tabular_features=data_cfg["tabular_features"],
        image_size=train_cfg["data"]["image_size"],
        train=False,
    )

    print("index | logit | probability | run 1 max | run 2 max | repeatable")

    for index in [760, 898, 1000, 504, 510, 103]:
        image, _, _ = dataset[index]

        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))[0]
            probabilities = torch.sigmoid(logits)
            class_idx = probabilities.argmax().item()

        _, heatmap1 = explainer.explain(image, class_idx)
        _, heatmap2 = explainer.explain(image, class_idx)

        max1 = float(np.max(heatmap1))
        max2 = float(np.max(heatmap2))
        repeatable = np.allclose(heatmap1, heatmap2, atol=1e-6)

        print(
            f"{index} | {logits[class_idx].item():.8f} | "
            f"{probabilities[class_idx].item():.8f} | "
            f"{max1:.8f} | {max2:.8f} | {repeatable}"
        )


if __name__ == "__main__":
    main()
