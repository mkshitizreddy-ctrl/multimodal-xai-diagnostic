
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ChestXrayDataset
from src.explain.generate_counterfactual_examples import load_config
from src.models.vision_encoder import ChestXrayVisionModel


def main():
    root = Path(__file__).resolve().parents[2]
    config = load_config(
        str(root / "configs/vision_patient_disjoint_dedup_rho_v1_0.yaml")
    )
    data_config = load_config(str(root / "configs/data.yaml"))

    checkpoint_path = root / config["checkpoint"]["dir"] / "best_model.pth"
    test_csv = root / config["data"]["test_csv"]
    image_dir = root / config["data"]["image_dir"]

    for path in (checkpoint_path, test_csv, image_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

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

    output_dir = root / (
        "docs/gradcam_raw_testset_diagnostic_patient_disjoint_dedup_rho_v1_vision"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gradcam_raw_testset_measurements.csv"
    summary_path = output_dir / "gradcam_raw_testset_summary.md"

    if csv_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"Output already exists in {output_dir}. "
            "Preserve previous results and choose a new directory."
        )

    target_layer = model.get_target_layer()
    rows = []

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for index in tqdm(range(len(dataset)), desc="Analyzing raw Grad-CAM"):
            image, _, labels = dataset[index]
            input_tensor = image.unsqueeze(0).to(device)

            with torch.no_grad():
                logit = model(input_tensor)[0, 0].item()
                probability = float(torch.sigmoid(torch.tensor(logit)))

            # Run Grad-CAM to populate the same activations and gradients
            # used by the installed pytorch-grad-cam implementation.
            cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(0)],
            )

            activations = cam.activations_and_grads.activations[0]
            gradients = cam.activations_and_grads.gradients[0]

            if isinstance(activations, torch.Tensor):
                activations = activations.detach().cpu().numpy()
            if isinstance(gradients, torch.Tensor):
                gradients = gradients.detach().cpu().numpy()

            # Standard Grad-CAM weights: mean gradients over spatial axes.
            weights = np.mean(gradients, axis=(2, 3))

            # Raw weighted activation sum, BEFORE ReLU and CAM scaling.
            weighted_activations = (
                weights[:, :, None, None] * activations
            )
            raw_map = weighted_activations.sum(axis=1)[0]

            raw_min = float(raw_map.min())
            raw_max = float(raw_map.max())
            positive_fraction = float(np.mean(raw_map > 0))
            negative_fraction = float(np.mean(raw_map < 0))
            zero_fraction = float(np.mean(raw_map == 0))

            rows.append({
                "test_index": index,
                "image": dataset.df.iloc[index]["Image Index"],
                "true_pneumonia_label": float(labels[0].item()),
                "pneumonia_probability": probability,
                "raw_map_min": raw_min,
                "raw_map_max": raw_max,
                "raw_positive_pixel_fraction": positive_fraction,
                "raw_negative_pixel_fraction": negative_fraction,
                "raw_zero_pixel_fraction": zero_fraction,
                "raw_map_entirely_negative": bool(raw_max < 0),
                "raw_map_no_positive_pixels": bool(raw_max <= 0),
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    all_negative = df["raw_map_entirely_negative"]
    no_positive = df["raw_map_no_positive_pixels"]

    summary = [
        "# Raw Grad-CAM Test-Set Diagnostic",
        "",
        f"- Checkpoint: `{checkpoint_path.relative_to(root)}`",
        f"- Test CSV: `{test_csv.relative_to(root)}`",
        f"- Device: `{device}`",
        f"- Test images analyzed: {len(df)}",
        "",
        "## Raw maps before ReLU and scaling",
        "",
        f"- Entirely negative raw maps: {int(all_negative.sum())}",
        f"- Raw maps with no positive pixels: {int(no_positive.sum())}",
        f"- Mean raw-map positive-pixel fraction: "
        f"{df['raw_positive_pixel_fraction'].mean():.4%}",
        "",
        "## Interpretation",
        "",
        "- An entirely negative raw map is expected to become a zero map "
        "after the standard Grad-CAM ReLU.",
        "- A raw map with no positive pixels may contain negative and/or "
        "exactly zero values; inspect its minimum and maximum.",
        "- These measurements describe Grad-CAM behavior and do not "
        "establish clinical validity or causal explanations.",
        "",
        f"Detailed measurements: `{csv_path.relative_to(root)}`",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"\nAnalyzed {len(df)} test images on {device}.")
    print(f"Entirely negative raw maps: {int(all_negative.sum())}")
    print(f"Raw maps with no positive pixels: {int(no_positive.sum())}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
