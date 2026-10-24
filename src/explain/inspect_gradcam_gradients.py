
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    target_layer = model.get_target_layer()
    captured = {}

    def forward_hook(_module, _inputs, output):
        captured["activation"] = output

    handle = target_layer.register_forward_hook(forward_hook)

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

    print(f"Target layer: {model.gradcam_target_layer}")
    print("index | logit | sigmoid | activation abs max | gradient abs max")

    try:
        for index in [760, 898, 1000, 504, 510, 103]:
            image, _, _ = dataset[index]
            captured.clear()

            model.zero_grad(set_to_none=True)
            logits = model(image.unsqueeze(0).to(device))
            class_idx = torch.sigmoid(logits)[0].argmax().item()
            target_logit = logits[0, class_idx]

            activation = captured["activation"]
            gradient = torch.autograd.grad(
                target_logit,
                activation,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]

            activation_max = activation.detach().abs().max().item()
            gradient_max = (
                gradient.detach().abs().max().item()
                if gradient is not None
                else float("nan")
            )

            print(
                f"{index} | {target_logit.item():.8f} | "
                f"{torch.sigmoid(target_logit).item():.8f} | "
                f"{activation_max:.8e} | {gradient_max:.8e}"
            )
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
