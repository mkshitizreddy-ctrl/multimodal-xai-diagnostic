
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explain.generate_counterfactual_examples import load_config
from src.data.dataset import ChestXrayDataset
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
        use_se=checkpoint.get("use_se", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    captured = {}

    def forward_hook(_module, _inputs, output):
        captured["activation"] = output

    handle = model.get_target_layer().register_forward_hook(forward_hook)

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

    print("index | raw min | raw max | positive % | post-ReLU max")

    try:
        for index in [760, 898, 1000, 504, 510, 103]:
            captured.clear()
            model.zero_grad(set_to_none=True)

            image, _, _ = dataset[index]
            logits = model(image.unsqueeze(0).to(device))
            target_idx = torch.sigmoid(logits)[0].argmax().item()
            score = logits[0, target_idx]

            activation = captured["activation"]
            gradient = torch.autograd.grad(score, activation)[0]

            weights = gradient.mean(dim=(2, 3), keepdim=True)
            raw_cam = (weights * activation).sum(dim=1)[0].detach().cpu().numpy()

            positive_pct = 100.0 * np.mean(raw_cam > 0)
            print(
                f"{index} | {raw_cam.min():+.6e} | "
                f"{raw_cam.max():+.6e} | {positive_pct:.4f}% | "
                f"{np.maximum(raw_cam, 0).max():.6e}"
            )
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
