"""Is the lung-mask silhouette signal just the original image's aspect ratio?

Images are resized to a square for the model, so the original aspect ratio changes how the lungs look in
the mask. This script compares classifiers on aspect ratio / original size with ones on the mask outline,
and reports the outline classifier's AUROC inside aspect-ratio quintiles. Exploratory and descriptive:
a within-quintile AUROC well above 0.5 means the outline carries label information beyond aspect ratio,
without saying what that information is (zoom, positioning, anatomy, or acquisition).

Usage (after cache_split_masks.py and nonanatomical_baseline.py):
    python src/explain/silhouette_aspect_check.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explain.nonanatomical_baseline import evaluate_group
from src.explain.shortcut_audit_utils import auroc
from src.explain.silhouette_baseline import GROUPS, split_table

SETS = {
    "aspect_only": ["aspect"],
    "original_size+aspect": ["orig_w", "orig_h", "aspect"],
    "silhouette_thumbnail": GROUPS["thumbnail_16x16"],
    "silhouette_thumbnail+aspect": GROUPS["thumbnail_16x16"] + ["aspect"],
}


def main():
    parser = argparse.ArgumentParser(description="Silhouette versus aspect-ratio check.")
    parser.add_argument("--split-dir", default="data/processed_patient_disjoint_dedup_rho_v1_0")
    parser.add_argument("--masks-dir", default="data/lung_masks")
    parser.add_argument("--features-dir", default="docs/nonanatomical_baseline_patient_disjoint_dedup_rho_v1")
    parser.add_argument("--output-dir", default="docs/silhouette_baseline_patient_disjoint_dedup_rho_v1")
    parser.add_argument("--dilation", type=float, default=0.0)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--lung-threshold", type=float, default=0.3)
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()

    tables = {}
    for split in ("train", "val", "test"):
        frame = split_table(
            Path(args.split_dir) / f"{split}.csv",
            args.masks_dir,
            split,
            args.size,
            args.lung_threshold,
            args.dilation,
        )
        meta = pd.read_csv(Path(args.features_dir) / f"image_features_{split}.csv")
        meta = meta[["image", "width", "height", "aspect"]]
        meta = meta.rename(columns={"width": "orig_w", "height": "orig_h"})
        frame = frame.merge(meta, on="image", how="left")
        if frame["aspect"].isna().any():
            raise ValueError(f"{split}: some images are missing from the cached image features")
        tables[split] = frame
    train, val, test = tables["train"], tables["val"], tables["test"]
    y = test["label"].to_numpy() > 0.5

    lines = [
        "# Silhouette versus aspect ratio",
        "",
        f"Mask dilation {args.dilation:g} px. Test split, {len(test)} images ({int(y.sum())} Pneumonia).",
        "",
        "| Features | Model (picked on val) | Test AUROC |",
        "|---|---|---|",
    ]
    scores = {}
    for name, columns in SETS.items():
        result = evaluate_group(train, val, test, columns)
        scores[name] = result["test_scores"]
        lines.append(f"| {name} | {result['model']} | {result['test_auroc']:.4f} |")
    median = test.groupby(y)["aspect"].median()
    lines += [
        "",
        f"Median aspect ratio (width/height): Normal {median[False]:.3f}, Pneumonia {median[True]:.3f}.",
        "",
        "Silhouette-thumbnail classifier AUROC within aspect-ratio bins:",
        "",
        "| Aspect bin | n | Pneumonia | Normal | AUROC |",
        "|---|---|---|---|---|",
    ]
    bins = pd.qcut(test["aspect"], args.bins, duplicates="drop")
    score = scores["silhouette_thumbnail"]
    for interval, index in test.groupby(bins, observed=True).indices.items():
        yy = y[index]
        value = auroc(yy, score[index]) if 0 < yy.sum() < len(yy) else float("nan")
        lines.append(f"| {interval} | {len(index)} | {int(yy.sum())} | {int((~yy).sum())} | {value:.3f} |")
    corr = [np.corrcoef(score[y == k], test["aspect"].to_numpy()[y == k])[0, 1] for k in (False, True)]
    lines += [
        "",
        f"Within-class correlation of the silhouette score with aspect ratio: Normal {corr[0]:+.3f}, "
        f"Pneumonia {corr[1]:+.3f}.",
        "Bins at the high-aspect end hold few Normal images, so their AUROC is noisy.",
        "",
    ]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "silhouette_aspect_check.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
