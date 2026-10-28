
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ChestXrayDataset  # noqa: E402
from src.explain.generate_counterfactual_examples import load_config  # noqa: E402
from src.models.vision_encoder import ChestXrayVisionModel  # noqa: E402


def main():
    config = load_config(
        str(ROOT / "configs/vision_patient_disjoint_dedup_rho_v1_0.yaml")
    )
    data_config = load_config(str(ROOT / "configs/data.yaml"))

    checkpoint_path = ROOT / config["checkpoint"]["dir"] / "best_model.pth"
    test_csv = ROOT / config["data"]["test_csv"]
    image_dir = ROOT / config["data"]["image_dir"]
    measurements_path = ROOT / (
        "docs/gradcam_testset_diagnostic_patient_disjoint_dedup_rho_v1_vision/"
        "gradcam_testset_measurements.csv"
    )

    for path in (checkpoint_path, test_csv, image_dir, measurements_path):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    measurements = pd.read_csv(measurements_path)
    zero_indices = (
        measurements.loc[measurements["zero_map"], "test_index"]
        .astype(int)
        .head(6)
        .tolist()
    )
    if not zero_indices:
        raise RuntimeError("No zero-map cases were found in the diagnostic CSV.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    classes = checkpoint["classes"]
    model_config = config["model"]

    model = ChestXrayVisionModel(
        num_classes=len(classes),
        pretrained=False,
        dropout=model_config["dropout"],
        use_cbam=model_config["use_cbam"],
        use_se=model_config.get("use_se", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dataset = ChestXrayDataset(
        csv_path=str(test_csv),
        image_dir=str(image_dir),
        classes=classes,
        tabular_features=data_config["tabular_features"],
        image_size=config["data"]["image_size"],
        train=False,
    )

    target_layer = model.get_target_layer()
    rows = []

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for index in zero_indices:
            image, _, labels = dataset[index]
            input_tensor = image.unsqueeze(0).to(device)

            # Reproduce the raw CAM calculation used by pytorch-grad-cam,
            # before compute_cam_per_layer applies ReLU.
            with torch.no_grad():
                logit = model(input_tensor)[0, 0].item()
                probability = float(torch.sigmoid(torch.tensor(logit)))

            cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(0)],
            )

            activations = (
                cam.activations_and_grads.activations[0]
                .detach()
                .cpu()
                .numpy()
            )
            gradients = (
                cam.activations_and_grads.gradients[0]
                .detach()
                .cpu()
                .numpy()
            )

            weights = cam.get_cam_weights(
                input_tensor,
                target_layer,
                [ClassifierOutputTarget(0)],
                activations,
                gradients,
            )

            weighted = weights[:, :, None, None] * activations
            raw_map = weighted.sum(axis=1)[0]

            rows.append({
                "test_index": index,
                "image": dataset.df.iloc[index]["Image Index"],
                "actual_label": float(labels[0].item()),
                "pneumonia_probability": probability,
                "raw_min": float(raw_map.min()),
                "raw_max": float(raw_map.max()),
                "raw_positive_pct": float(np.mean(raw_map > 0) * 100),
                "raw_negative_pct": float(np.mean(raw_map < 0) * 100),
                "raw_zero_pct": float(np.mean(raw_map == 0) * 100),
            })

    result = pd.DataFrame(rows)
    print(f"Device: {device}")
    print("Raw weighted Grad-CAM maps before ReLU")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    output_path = ROOT / "docs/gradcam_raw_zero_map_sample.csv"
    result.to_csv(output_path, index=False)
    print(f"\nSaved sample measurements to: {output_path}")


if __name__ == "__main__":
    main()
