"""
Quantifies how much of a Grad-CAM heatmap's energy falls inside the
segmented lung fields vs. outside — turns the manual audit described in
docs/ethics_statement.md into a number, and gives a concrete way to check
whether CBAM actually helps localization (not just accuracy) as the
literature review in docs/architecture.md#attention-module claimed it should.

Works on both the vision-only and fusion models — auto-detects which one a
checkpoint is from (fusion checkpoints store a "tabular_features" key,
vision-only ones don't; see src/train.py vs src/train_fusion.py's save
calls). For fusion, each test image gets its own FusionModelImageWrapper
built with that image's real tabular vector (see
src/explain/fusion_wrapper.py and generate_fusion_examples.py, which uses
the same per-image-wrapper pattern) — reusing one wrapper across images
would explain every image using some OTHER patient's fixed vitals, which
would silently produce meaningless results.

Run once per checkpoint (e.g. once with use_cbam=false, once with
use_cbam=true) and compare the printed "lung energy fraction" — higher is
better (heatmap concentrating on lung tissue instead of spreading to
shoulders/borders/annotations).

Usage:
    python src/explain/measure_lung_localization.py \
        --checkpoint checkpoints/vision_baseline/best_model.pth \
        --num-examples 30

    python src/explain/measure_lung_localization.py \
        --checkpoint checkpoints/fusion/best_model.pth \
        --train-config configs/fusion.yaml \
        --num-examples 30
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ChestXrayDataset
from src.explain.fusion_wrapper import FusionModelImageWrapper
from src.explain.gradcam import ChestXrayExplainer
from src.explain.lung_segmentation import get_lung_mask
from src.models.fusion import ChestXrayFusionModel
from src.models.vision_encoder import ChestXrayVisionModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def lung_energy_fraction(heatmap: np.ndarray, lung_mask: np.ndarray) -> float:
    """Fraction of the heatmap's total activation that falls inside the
    lung mask. 1.0 = every bit of activation is on lung tissue; lower
    values mean the model is (at least partly) keying off something
    outside the lungs — shoulders, image borders, burned-in markers, etc.
    """
    total = heatmap.sum()
    if total <= 1e-8:
        return float("nan")  # degenerate heatmap, shouldn't normally happen
    inside = (heatmap * lung_mask).sum()
    return float(inside / total)


def is_fusion_checkpoint(checkpoint: dict) -> bool:
    """Fusion checkpoints (src/train_fusion.py) store a "tabular_features"
    key; vision-only ones (src/train.py) don't. Pulled out as its own
    function so this detection logic has one place to change and can be
    unit-tested without constructing a real model or dataset."""
    return "tabular_features" in checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_baseline.yaml")
    parser.add_argument("--num-examples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", default=None, help="Optional path to save per-image results.")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    train_cfg = load_config(args.train_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    use_cbam = checkpoint.get("use_cbam", False)
    is_fusion = is_fusion_checkpoint(checkpoint)

    if is_fusion:
        tabular_features = checkpoint["tabular_features"]
        model = ChestXrayFusionModel(
            num_classes=len(classes),
            num_tabular_features=len(tabular_features),
            pretrained=False,
            use_cbam=use_cbam,
        )
    else:
        tabular_features = data_cfg["tabular_features"]
        model = ChestXrayVisionModel(num_classes=len(classes), pretrained=False, use_cbam=use_cbam)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # For fusion, reuse the checkpoint's fitted tabular_stats rather than
    # re-fitting from the test split — same reasoning as
    # generate_fusion_examples.py: keeps normalization consistent with what
    # the model actually trained on. Doesn't matter for vision-only (its
    # forward pass ignores tabular entirely), but harmless to pass either way.
    test_ds = ChestXrayDataset(
        csv_path=train_cfg["data"].get("test_csv", "data/processed/test.csv"),
        image_dir=train_cfg["data"]["image_dir"],
        classes=classes,
        tabular_features=tabular_features,
        image_size=train_cfg["data"]["image_size"],
        train=False,
        tabular_stats=checkpoint.get("tabular_stats") if is_fusion else None,
    )

    # Vision explainer is stateless w.r.t. any per-image data, so build it
    # once outside the loop. Fusion needs a fresh wrapper (and therefore a
    # fresh explainer, since ChestXrayExplainer registers Grad-CAM hooks at
    # construction time against whatever model it's given) per image, since
    # each image has its own tabular vector that must be fixed correctly —
    # reusing one wrapper across images would silently explain every image
    # using some OTHER patient's vitals.
    explainer = None if is_fusion else ChestXrayExplainer(model, device=str(device))

    rng = np.random.default_rng(args.seed)
    num_examples = min(args.num_examples, len(test_ds))
    indices = rng.choice(len(test_ds), size=num_examples, replace=False)

    fractions = []
    rows = []

    for i in indices:
        image, tabular, _labels = test_ds[int(i)]

        if is_fusion:
            wrapper = FusionModelImageWrapper(model, tabular).to(device)
            image_explainer = ChestXrayExplainer(wrapper, device=str(device))
            with torch.no_grad():
                logits = wrapper(image.unsqueeze(0).to(device))
        else:
            image_explainer = explainer
            with torch.no_grad():
                logits = model(image.unsqueeze(0).to(device))

        probs = torch.sigmoid(logits)[0].cpu()
        pred_idx = int(torch.argmax(probs))

        _overlay, heatmap = image_explainer.explain(image, pred_idx)
        lung_mask = get_lung_mask(image, output_size=heatmap.shape[0])
        frac = lung_energy_fraction(heatmap, lung_mask)

        fractions.append(frac)
        rows.append(
            {
                "index": int(i),
                "predicted_class": classes[pred_idx],
                "predicted_probability": probs[pred_idx].item(),
                "lung_energy_fraction": frac,
            }
        )
        print(f"  [{i}] {classes[pred_idx]:>12s}  p={probs[pred_idx]:.2f}  lung_fraction={frac:.3f}")

    valid_fractions = [f for f in fractions if not np.isnan(f)]
    mean_frac = float(np.mean(valid_fractions))
    std_frac = float(np.std(valid_fractions))

    print(f"\nmodel_type = {'fusion' if is_fusion else 'vision'}")
    print(f"use_cbam = {use_cbam}")
    print(f"n = {len(valid_fractions)} test images")
    print(f"mean lung-energy fraction = {mean_frac:.3f} (std {std_frac:.3f})")

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-image results saved to {out_path}")


if __name__ == "__main__":
    main()
