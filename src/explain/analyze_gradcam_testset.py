
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
        "docs/gradcam_testset_diagnostic_patient_disjoint_dedup_rho_v1_vision"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gradcam_testset_measurements.csv"
    report_path = output_dir / "gradcam_testset_summary.md"

    if csv_path.exists() or report_path.exists():
        raise FileExistsError(
            f"Output already exists in {output_dir}. "
            "Choose a new output directory to preserve prior results."
        )

    target_layer = model.get_target_layer()
    rows = []

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for index in tqdm(range(len(dataset)), desc="Analyzing test images"):
            image, _, labels = dataset[index]
            input_tensor = image.unsqueeze(0).to(device)

            with torch.no_grad():
                logit = model(input_tensor)[0, 0].item()
                probability = float(torch.sigmoid(torch.tensor(logit)))

            # The model has one Pneumonia logit. Explain that output directly.
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(0)],
            )[0]

            heatmap = np.asarray(grayscale_cam)
            positive_coverage = float(np.mean(heatmap > 0))
            maximum = float(heatmap.max())
            zero_map = bool(maximum == 0.0)

            rows.append({
                "test_index": index,
                "image": dataset.df.iloc[index]["Image Index"],
                "true_pneumonia_label": float(labels[0].item()),
                "pneumonia_probability": probability,
                "heatmap_max": maximum,
                "positive_pixel_coverage": positive_coverage,
                "zero_map": zero_map,
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    zero_df = df[df["zero_map"]]
    nonzero_df = df[~df["zero_map"]]

    def coverage_stats(series):
        return {
            "mean": float(series.mean()) if len(series) else float("nan"),
            "median": float(series.median()) if len(series) else float("nan"),
            "p90": float(series.quantile(0.9)) if len(series) else float("nan"),
        }

    all_coverage = coverage_stats(df["positive_pixel_coverage"])
    report = [
        "# Full Test-Set Grad-CAM Diagnostic",
        "",
        f"- Checkpoint: `{checkpoint_path.relative_to(root)}`",
        f"- Test CSV: `{test_csv.relative_to(root)}`",
        f"- Device: `{device}`",
        f"- Test images analyzed: {len(df)}",
        f"- Zero heatmaps: {len(zero_df)} ({len(zero_df) / len(df):.2%})",
        f"- Nonzero heatmaps: {len(nonzero_df)}",
        "",
        "## Positive-pixel coverage",
        "",
        f"- Mean: {all_coverage['mean']:.4%}",
        f"- Median: {all_coverage['median']:.4%}",
        f"- 90th percentile: {all_coverage['p90']:.4%}",
        "",
        "## Pneumonia probability by heatmap status",
        "",
        f"- Zero-map cases: n={len(zero_df)}, "
        f"mean probability="
        f"{zero_df['pneumonia_probability'].mean():.6f}"
        if len(zero_df) else "- Zero-map cases: none",
        f"- Nonzero-map cases: n={len(nonzero_df)}, "
        f"mean probability="
        f"{nonzero_df['pneumonia_probability'].mean():.6f}"
        if len(nonzero_df) else "- Nonzero-map cases: none",
        "",
        "## Interpretation limits",
        "",
        "- A zero map means the standard positive-evidence Grad-CAM map "
        "has no positive pixels after its ReLU; it does not prove that "
        "the model is insensitive to the image.",
        "- Coverage is the fraction of pixels with positive heatmap values, "
        "not the fraction of lung tissue affected.",
        "- This is a descriptive diagnostic, not evidence of clinical "
        "validity or causal explanation.",
        "",
        f"Detailed measurements: `{csv_path.relative_to(root)}`",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nAnalyzed {len(df)} test images on {device}.")
    print(f"Zero maps: {len(zero_df)} ({len(zero_df) / len(df):.2%})")
    print(f"Nonzero maps: {len(nonzero_df)}")
    print(f"Mean positive-pixel coverage: {all_coverage['mean']:.4%}")
    print(f"Median positive-pixel coverage: {all_coverage['median']:.4%}")
    print(f"90th percentile coverage: {all_coverage['p90']:.4%}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {report_path}")


if __name__ == "__main__":
    main()
