"""
Precomputes lung segmentation masks for every image in train/val, at the
same 7x7 resolution as CBAM's spatial attention map (see
src/models/vision_encoder.py's forward_with_attention_map()), and caches
them to a single pickle file.

Why precompute instead of segmenting on the fly during training: the
segmentation model (torchxrayvision's PSPNet, via
src/explain/lung_segmentation.py) runs at 512x512 and is not cheap —
calling it fresh for every image of every epoch would make training
noticeably slower for no benefit, since the mask for a given image never
changes. ~5,800 images total takes a few minutes once, then every
subsequent training run just loads the cached file.

Important tradeoff, stated plainly rather than hidden: masks are computed
from the image WITHOUT random augmentation (no flip, no rotation) — using
ChestXrayDataset(..., train=False) even for the train split. Training
normally applies RandomHorizontalFlip and RandomRotation(degrees=5); if we
cached masks from the deterministic image but training then flips/rotates
the actual image, the mask would face the wrong way roughly half the time
for flips, and be knowably misaligned for rotation. Rather than build a
paired image+mask augmentation pipeline (real, but bigger, engineering
work), this experiment simply disables that augmentation for the training
run that uses these masks — see src/train_attention_consistency.py and
configs/vision_attention_consistency.yaml. This is a deliberate
simplification for a first pass at this idea, not an oversight.

Usage:
    python data/scripts/precompute_lung_masks.py \
        --train-config configs/vision_baseline.yaml \
        --output data/processed/lung_masks.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ChestXrayDataset
from src.explain.lung_segmentation import get_lung_mask

ATTENTION_MAP_RESOLUTION = 7  # must match CBAM's spatial attention map size for a 224x224 input


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def precompute_for_csv(csv_path: str, image_dir: str, image_size: int, classes: list, tabular_features: list) -> dict:
    """Returns {image_filename: mask_array} for every row in csv_path.
    Uses train=False deterministic transform regardless of which split
    this is — see module docstring for why.
    """
    df = pd.read_csv(csv_path)
    ds = ChestXrayDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        classes=classes,
        tabular_features=tabular_features,
        image_size=image_size,
        train=False,  # deliberate — see module docstring
    )

    masks = {}
    for i in tqdm(range(len(ds)), desc=f"segmenting {Path(csv_path).name}"):
        image, _tabular, _labels = ds[i]
        filename = df.iloc[i]["Image Index"]
        masks[filename] = get_lung_mask(image, output_size=ATTENTION_MAP_RESOLUTION)

    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_baseline.yaml")
    parser.add_argument("--output", default="data/processed/lung_masks.pkl")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_csv", "val_csv"],
        help="Which config keys under train_config['data'] to process. "
        "Test split intentionally excluded by default — these masks are "
        "for a training-time loss, not evaluation.",
    )
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    train_cfg = load_config(args.train_config)
    classes = data_cfg["labels"]["classes"]
    tabular_features = data_cfg["tabular_features"]
    image_dir = train_cfg["data"]["image_dir"]
    image_size = train_cfg["data"]["image_size"]

    all_masks = {}
    for split_key in args.splits:
        csv_path = train_cfg["data"][split_key]
        split_masks = precompute_for_csv(csv_path, image_dir, image_size, classes, tabular_features)
        overlap = set(split_masks) & set(all_masks)
        if overlap:
            print(f"WARNING: {len(overlap)} filenames appear in multiple splits — later split's mask wins.")
        all_masks.update(split_masks)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(all_masks, f)

    print(f"\nSaved {len(all_masks)} precomputed lung masks to {output_path}")


if __name__ == "__main__":
    main()
