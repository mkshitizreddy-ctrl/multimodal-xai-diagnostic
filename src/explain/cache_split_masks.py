"""Cache lung masks for the train, val and test splits (needed for lung-aware training).

Uses the same segmenter, threshold and packed format as shortcut_audit.py. Masks are written to
<output-dir>/<split>.npz, which is gitignored; rerun to regenerate.

Usage:
    python src/explain/cache_split_masks.py
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ChestXrayDataset
from src.explain.shortcut_audit import load_or_compute_masks


def main():
    parser = argparse.ArgumentParser(description="Cache lung masks for every split.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_patient_disjoint_dedup_rho_v1_0.yaml")
    parser.add_argument("--output-dir", default="data/lung_masks")
    parser.add_argument("--lung-threshold", type=float, default=0.3)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    with open(args.data_config, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    with open(args.train_config, encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    size = int(train_cfg["data"]["image_size"])
    out_dir = Path(args.output_dir)

    for split in ("train", "val", "test"):
        dataset = ChestXrayDataset(
            csv_path=train_cfg["data"][f"{split}_csv"],
            image_dir=train_cfg["data"]["image_dir"],
            classes=data_cfg["labels"]["classes"],
            tabular_features=data_cfg["tabular_features"],
            image_size=size,
            train=False,
        )
        print(f"{split}: {len(dataset)} images")
        load_or_compute_masks(
            out_dir / f"{split}.npz", dataset, len(dataset), size, args.lung_threshold, args.recompute
        )
    print(f"Masks cached in {out_dir}")


if __name__ == "__main__":
    main()
