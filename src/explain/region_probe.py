"""Region probe on the frozen vision checkpoint: which parts of the image are enough?

Keeps only one region at a time (dilated-lung thorax, everything outside it, the outer margin, or a
central box of matched area) and measures AUROC. It follows up the mask-dilation run, where the model
still reached AUROC 0.964 with the whole thorax grayed. One dataset, one checkpoint, descriptive only.

Usage (smoke test, then full run):
    python src/explain/region_probe.py \\
        --checkpoint checkpoints/vision_patient_disjoint_dedup_rho_v1_0/best_model.pth --limit 40
    python src/explain/region_probe.py \\
        --checkpoint checkpoints/vision_patient_disjoint_dedup_rho_v1_0/best_model.pth
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
from src.explain.region_probe_utils import CONDITIONS, region_images, summarize_regions, summary_markdown
from src.explain.shortcut_audit import file_sha256, load_or_compute_masks
from src.explain.shortcut_audit_utils import dilate_mask
from src.models.vision_encoder import ChestXrayVisionModel


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Region probe on the frozen vision checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", default="data/processed_patient_disjoint_dedup_rho_v1_0/test.csv")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_baseline.yaml")
    parser.add_argument(
        "--masks-path",
        default="docs/shortcut_audit_patient_disjoint_dedup_rho_v1_vision/lung_masks.npz",
        help="Cached lung masks written by shortcut_audit.py (recomputed if missing or mismatched).",
    )
    parser.add_argument("--mask-dilation", type=float, default=8.0)
    parser.add_argument("--lung-threshold", type=float, default=0.3)
    parser.add_argument("--output-dir", default="docs/region_probe_patient_disjoint_dedup_rho_v1_vision")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test limit; 0 runs the full test set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.limit < 0 or args.mask_dilation < 0 or args.bootstrap_replicates < 100:
        parser.error("--limit >= 0, --mask-dilation >= 0 and --bootstrap-replicates >= 100 are required")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = load_yaml(args.data_config)
    train_cfg = load_yaml(args.train_config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    if len(classes) != 1:
        raise ValueError(f"Expected a one-logit binary Pneumonia model; checkpoint classes={classes!r}")
    model = ChestXrayVisionModel(
        num_classes=1, pretrained=False, use_cbam=checkpoint.get("use_cbam", False)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    size = int(train_cfg["data"]["image_size"])
    dataset = ChestXrayDataset(
        csv_path=args.test_csv,
        image_dir=train_cfg["data"]["image_dir"],
        classes=classes,
        tabular_features=data_cfg["tabular_features"],
        image_size=size,
        train=False,
    )
    n = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))
    print(f"Device: {device}; {n}/{len(dataset)} images x {len(CONDITIONS)} regions")

    masks = load_or_compute_masks(Path(args.masks_path), dataset, n, size, args.lung_threshold, False)
    if args.mask_dilation > 0:
        masks = np.stack([dilate_mask(m, args.mask_dilation) for m in masks])

    rows = []
    for i in range(n):
        image, _tabular, labels = dataset[i]
        images, areas = region_images(image.numpy(), masks[i])
        batch = np.stack([images[name] for name in CONDITIONS])
        with torch.no_grad():
            logits = model(torch.from_numpy(batch).to(device)).reshape(-1).float().cpu().numpy()
        record = {"index": i, "image": dataset.df.iloc[i]["Image Index"], "label": float(labels[0].item())}
        for name, logit in zip(CONDITIONS, logits):
            record[f"logit_{name}"] = float(logit)
            record[f"area_{name}"] = areas[name]
        rows.append(record)
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}")

    measurements = pd.DataFrame(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    measurements.to_csv(out_dir / "region_probe_measurements.csv", index=False)
    summary = summarize_regions(measurements, seed=args.seed, replicates=args.bootstrap_replicates)
    summary.to_csv(out_dir / "region_probe_summary.csv", index=False)
    info = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "test_csv": args.test_csv,
        "n_images": int(len(measurements)),
        "n_pneumonia": int((measurements["label"] > 0.5).sum()),
        "mask_dilation_px": args.mask_dilation,
        "lung_threshold": args.lung_threshold,
        "conditions": CONDITIONS,
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    markdown = summary_markdown(summary, info)
    (out_dir / "region_probe_summary.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
