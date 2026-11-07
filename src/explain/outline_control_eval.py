"""Outline control: does a model need the per-image lung outline, or only the image content?

Evaluates every baseline / lung-aware / fixed-box checkpoint on the same four inputs: the original image,
the dilated thorax only, a fixed centred box only, and everything outside that box. The fixed-box models
were trained to see only the box, whose outline is identical for every image. If they match the
thorax-only models on their own input, the lung outline is not needed for the AUROC. Descriptive: one
dataset, three seeds. Checkpoints missing on disk are listed and skipped.

Usage:
    python src/explain/outline_control_eval.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ChestXrayDataset
from src.data.lung_masking import centered_box
from src.explain.lung_aware_eval import GROUPS as BASE_GROUPS
from src.explain.lung_aware_eval_utils import aggregate_seeds, summary_markdown
from src.explain.outline_control_utils import CONDITIONS, outline_images
from src.explain.shortcut_audit import load_or_compute_masks
from src.explain.shortcut_audit_utils import auroc, dilate_mask
from src.models.vision_encoder import ChestXrayVisionModel

SEEDS = (42, 123, 2024)
GROUPS = dict(BASE_GROUPS)
GROUPS["fixed_box"] = lambda s: f"checkpoints/vision_lung_fixed_box_s{s}"


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_checkpoint(checkpoint_path, dataset, thorax_masks, box, device) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ChestXrayVisionModel(num_classes=1, pretrained=False, use_cbam=checkpoint.get("use_cbam", False))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    labels, logits, areas = [], {name: [] for name in CONDITIONS}, {}
    for i in range(len(dataset)):
        image, _tabular, label = dataset[i]
        images, areas = outline_images(image.numpy(), thorax_masks[i], box)
        batch = np.stack([images[name] for name in CONDITIONS])
        with torch.no_grad():
            out = model(torch.from_numpy(batch).to(device)).reshape(-1).float().cpu().numpy()
        labels.append(float(label[0].item()))
        for name, value in zip(CONDITIONS, out):
            logits[name].append(float(value))
    y = np.array(labels) > 0.5
    return pd.DataFrame(
        [
            {"condition": name, "auroc": auroc(y, np.array(logits[name])), "mean_kept_area": areas[name]}
            for name in CONDITIONS
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Outline control evaluation.")
    parser.add_argument("--test-csv", default="data/processed_patient_disjoint_dedup_rho_v1_0/test.csv")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_patient_disjoint_dedup_rho_v1_0.yaml")
    parser.add_argument(
        "--masks-path", default="docs/shortcut_audit_patient_disjoint_dedup_rho_v1_vision/lung_masks.npz"
    )
    parser.add_argument("--mask-dilation", type=float, default=8.0)
    parser.add_argument("--lung-threshold", type=float, default=0.3)
    parser.add_argument("--box-fraction", type=float, default=0.72)
    parser.add_argument("--output-dir", default="docs/outline_control")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg, train_cfg = load_yaml(args.data_config), load_yaml(args.train_config)
    size = int(train_cfg["data"]["image_size"])
    dataset = ChestXrayDataset(
        csv_path=args.test_csv,
        image_dir=train_cfg["data"]["image_dir"],
        classes=["Pneumonia"],
        tabular_features=data_cfg["tabular_features"],
        image_size=size,
        train=False,
    )
    masks = load_or_compute_masks(
        Path(args.masks_path), dataset, len(dataset), size, args.lung_threshold, False
    )
    thorax_masks = np.stack([dilate_mask(m, args.mask_dilation) for m in masks])
    box = centered_box(size, args.box_fraction)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summaries, missing = {}, []
    for group, directory in GROUPS.items():
        summaries[group] = {}
        for seed in SEEDS:
            checkpoint = Path(directory(seed)) / "best_model.pth"
            if not checkpoint.exists():
                missing.append(str(checkpoint))
                continue
            print(f"=== {group} seed {seed} ===")
            table = evaluate_checkpoint(checkpoint, dataset, thorax_masks, box, device)
            table.to_csv(out_root / f"{group}_s{seed}.csv", index=False)
            summaries[group][seed] = table
    summaries = {group: by_seed for group, by_seed in summaries.items() if by_seed}
    aggregate = aggregate_seeds(summaries)
    aggregate.to_csv(out_root / "outline_control_summary.csv", index=False)
    markdown = summary_markdown(aggregate, CONDITIONS).replace(
        "# Lung-aware training: region probe across seeds", "# Outline control: fixed box versus lung outline"
    )
    (out_root / "outline_control_summary.md").write_text(markdown, encoding="utf-8")
    (out_root / "run_info.json").write_text(
        json.dumps(
            {"box_fraction": args.box_fraction, "mask_dilation_px": args.mask_dilation, "missing": missing},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + markdown)
    if missing:
        print("Skipped (checkpoint not found):", *missing, sep="\n  ")
    print(f"Wrote results to {out_root}")


if __name__ == "__main__":
    main()
