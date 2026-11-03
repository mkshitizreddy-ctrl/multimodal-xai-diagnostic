"""Shortcut audit: how much does the frozen classifier depend on the lungs?

Evaluates the vision checkpoint on the test set under interventions that keep
or remove the segmented lung fields, with shape- and area-matched displaced-mask
controls. This is a model-sensitivity experiment on one dataset. It does not
establish clinical validity, and a drop under an intervention is not by itself
proof of (or against) shortcut learning.

Usage (smoke test first, then the full run):
    python src/explain/shortcut_audit.py \\
        --checkpoint checkpoints/vision_patient_disjoint_dedup_rho_v1_0/best_model.pth --limit 40
    python src/explain/shortcut_audit.py \\
        --checkpoint checkpoints/vision_patient_disjoint_dedup_rho_v1_0/best_model.pth
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ChestXrayDataset
from src.explain.lung_segmentation import get_lung_mask
from src.explain.shortcut_audit_utils import (
    CONDITIONS,
    audit_image,
    dilate_mask,
    mask_area_fraction,
    summarize,
    summary_markdown,
)
from src.models.vision_encoder import ChestXrayVisionModel

NORM_MEAN, NORM_STD = 0.5, 0.25


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_compute_masks(path: Path, dataset, n: int, size: int, threshold: float, recompute: bool):
    """Lung masks for the first n test images, cached as a packed .npz keyed by image names."""
    names = [str(dataset.df.iloc[i]["Image Index"]) for i in range(n)]
    if path.exists() and not recompute:
        cached = np.load(path, allow_pickle=False)
        same = (
            cached["names"].tolist() == names
            and int(cached["size"]) == size
            and float(cached["threshold"]) == threshold
        )
        if same:
            bits = np.unpackbits(cached["packed"], axis=1, count=size * size)
            print(f"Loaded cached lung masks from {path}")
            return bits.reshape(n, size, size).astype(np.float32)
        print("Cached lung masks do not match this run; recomputing.")

    masks = np.zeros((n, size, size), dtype=bool)
    for i in range(n):
        image, _tabular, _labels = dataset[i]
        masks[i] = get_lung_mask(image, output_size=size, threshold=threshold) > 0.5
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  lung masks: {i + 1}/{n}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        packed=np.packbits(masks.reshape(n, -1), axis=1),
        names=np.array(names),
        size=size,
        threshold=threshold,
    )
    return masks.astype(np.float32)


def save_qc_grid(dataset, masks, indices, path: Path, columns: int = 6) -> None:
    """Tile images with the lung-mask outline in red, for visual inspection of the segmenter."""
    tiles = []
    for i in indices:
        image, _t, _l = dataset[int(i)]
        gray = np.clip(image[0].numpy() * NORM_STD + NORM_MEAN, 0, 1)
        rgb = np.stack([gray, gray, gray], axis=-1)
        m = masks[int(i)] > 0.5
        inner = m.copy()
        inner[1:-1, 1:-1] = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
        edge = m & ~inner
        rgb[edge] = [1.0, 0.1, 0.1]
        tiles.append((rgb * 255).astype(np.uint8))
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r : r + columns], axis=1) for r in range(0, len(tiles), columns)]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main():
    parser = argparse.ArgumentParser(description="Frozen-model lung/background intervention audit.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Vision checkpoint; must match --test-csv's split "
        "(e.g. checkpoints/vision_patient_disjoint_dedup_rho_v1_0/best_model.pth).",
    )
    parser.add_argument("--test-csv", default="data/processed_patient_disjoint_dedup_rho_v1_0/test.csv")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_baseline.yaml")
    parser.add_argument("--output-dir", default="docs/shortcut_audit_patient_disjoint_dedup_rho_v1_vision")
    parser.add_argument(
        "--limit", type=int, default=0, help="Smoke-test limit; 0 evaluates the full test set."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--border-frac", type=float, default=0.10)
    parser.add_argument("--lung-threshold", type=float, default=0.3)
    parser.add_argument("--min-mask-area", type=float, default=0.10)
    parser.add_argument("--max-mask-area", type=float, default=0.70)
    parser.add_argument("--qc-count", type=int, default=24)
    parser.add_argument("--recompute-masks", action="store_true")
    parser.add_argument(
        "--mask-dilation",
        type=float,
        default=0.0,
        help="Grow every lung mask by this many pixels (at model resolution) after loading; "
        "0 keeps the original masks.",
    )
    parser.add_argument(
        "--masks-path",
        default=None,
        help="Cached lung_masks.npz to reuse (default: <output-dir>/lung_masks.npz). Point a dilated "
        "run at the original cache to avoid re-segmenting.",
    )
    args = parser.parse_args()

    if args.limit < 0 or args.control_repeats < 1:
        parser.error("--limit must be >= 0 and --control-repeats must be positive")
    if args.mask_dilation < 0:
        parser.error("--mask-dilation must be >= 0")
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = load_yaml(args.data_config)
    train_cfg = load_yaml(args.train_config)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    if len(classes) != 1:
        raise ValueError(f"Expected a one-logit binary Pneumonia model; checkpoint classes={classes!r}")

    model = ChestXrayVisionModel(
        num_classes=len(classes), pretrained=False, use_cbam=checkpoint.get("use_cbam", False)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dataset = ChestXrayDataset(
        csv_path=args.test_csv,
        image_dir=train_cfg["data"]["image_dir"],
        classes=classes,
        tabular_features=data_cfg["tabular_features"],
        image_size=train_cfg["data"]["image_size"],
        train=False,
    )
    n = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))
    size = int(train_cfg["data"]["image_size"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Evaluating {n}/{len(dataset)} images; {args.control_repeats} displaced-mask repeats")

    masks_path = Path(args.masks_path) if args.masks_path else out_dir / "lung_masks.npz"
    masks = load_or_compute_masks(
        masks_path, dataset, n, size, args.lung_threshold, args.recompute_masks
    )
    if args.mask_dilation > 0:
        masks = np.stack([dilate_mask(m, args.mask_dilation) for m in masks])
        print(f"Dilated every lung mask by {args.mask_dilation:g} px")
    qc_rng = np.random.default_rng(args.seed)
    qc_indices = qc_rng.choice(n, size=min(args.qc_count, n), replace=False)
    save_qc_grid(dataset, masks, qc_indices, out_dir / "qc_lung_masks.png")
    print(f"Saved mask QC image: {out_dir / 'qc_lung_masks.png'} (inspect it before trusting the masks)")

    def infer(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.from_numpy(batch).to(device))
        return logits.reshape(-1).float().cpu().numpy()

    rows = []
    for i in range(n):
        image, _tabular, labels = dataset[i]
        rng = np.random.default_rng([args.seed, i])
        record = audit_image(
            image.numpy(), masks[i], infer, rng, args.control_repeats, args.border_frac
        )
        record.update(
            {
                "index": i,
                "image": dataset.df.iloc[i]["Image Index"],
                "label": float(labels[0].item()),
                "mask_area": mask_area_fraction(masks[i]),
            }
        )
        rows.append(record)
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  audited {i + 1}/{n}")

    measurements = pd.DataFrame(rows)
    lead = ["index", "image", "label", "mask_area"]
    measurements = measurements[lead + [c for c in measurements.columns if c not in lead]]
    measurements.to_csv(out_dir / "shortcut_audit_measurements.csv", index=False)

    summary = summarize(
        measurements,
        seed=args.seed + 1,
        replicates=args.bootstrap_replicates,
        min_area=args.min_mask_area,
        max_area=args.max_mask_area,
    )
    summary.to_csv(out_dir / "shortcut_audit_summary.csv", index=False)

    area = measurements["mask_area"]
    valid = (area >= args.min_mask_area) & (area <= args.max_mask_area)
    info = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "test_csv": args.test_csv,
        "n_images": int(len(measurements)),
        "n_pneumonia": int((measurements["label"] > 0.5).sum()),
        "mask_area_mean": float(measurements["mask_area"].mean()),
        "n_valid_mask": int(valid.sum()),
        "min_mask_area": args.min_mask_area,
        "max_mask_area": args.max_mask_area,
        "control_repeats": args.control_repeats,
        "displaced_iou_mean": float(measurements["displaced_iou_mean"].mean()),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "border_frac": args.border_frac,
        "lung_threshold": args.lung_threshold,
        "mask_dilation_px": args.mask_dilation,
        "conditions": CONDITIONS,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    markdown = summary_markdown(summary, info)
    if args.mask_dilation > 0:
        markdown = f"> Lung masks dilated by {args.mask_dilation:g} px (model resolution).\n\n" + markdown
    (out_dir / "shortcut_audit_summary.md").write_text(markdown, encoding="utf-8")

    print()
    print(markdown)
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
