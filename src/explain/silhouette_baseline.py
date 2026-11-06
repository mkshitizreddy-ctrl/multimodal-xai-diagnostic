"""Silhouette baseline: how well does the lung-mask shape alone predict the label?

A thorax-only model sees the lungs through a gray cut-out, so the cut-out's outline (area, position, size,
row and column profile) is visible to it. This script trains simple classifiers on those mask features
only, with no pixel intensities, and reports test AUROC next to the thorax-only CNNs. High AUROC means the
mask outline carries label information (for example through radiograph framing), so a thorax-only CNN could
use it. Low AUROC does not show the CNN ignores the outline. Descriptive, one dataset.

Usage (after cache_split_masks.py):
    python src/explain/silhouette_baseline.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.lung_masking import load_split_masks
from src.explain.nonanatomical_baseline import evaluate_group
from src.explain.shortcut_audit_utils import paired_bootstrap_auroc

PROFILE_BINS = 16
THUMB_SIDE = 16
GEOMETRY = ["area", "x0", "x1", "y0", "y1", "width", "height", "cx", "cy", "left_area", "right_area"]
GROUPS = {
    "area_only": ["area"],
    "geometry": GEOMETRY,
    "profiles": [f"row{i}" for i in range(PROFILE_BINS)] + [f"col{i}" for i in range(PROFILE_BINS)],
    "thumbnail_16x16": [f"m{i}" for i in range(THUMB_SIDE * THUMB_SIDE)],
}
GROUPS["geometry+profiles"] = GROUPS["geometry"] + GROUPS["profiles"]


def _bin_means(values: np.ndarray, bins: int) -> np.ndarray:
    """Mean of `values` [N, L] over `bins` equal slices along axis 1 -> [N, bins]."""
    edges = np.linspace(0, values.shape[1], bins + 1).astype(int)
    return np.stack([values[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)


def mask_features(masks: np.ndarray) -> pd.DataFrame:
    """Shape features of boolean masks [N, H, W]; all coordinates are fractions of the image side."""
    n, h, w = masks.shape
    m = masks.astype(np.float32)
    area = m.mean(axis=(1, 2))
    cols_any, rows_any = masks.any(axis=1), masks.any(axis=2)  # [N, W], [N, H]
    empty = ~rows_any.any(axis=1)
    x0 = np.where(empty, 0, cols_any.argmax(axis=1)) / w
    x1 = np.where(empty, 0, w - cols_any[:, ::-1].argmax(axis=1)) / w
    y0 = np.where(empty, 0, rows_any.argmax(axis=1)) / h
    y1 = np.where(empty, 0, h - rows_any[:, ::-1].argmax(axis=1)) / h
    total = np.maximum(m.sum(axis=(1, 2)), 1.0)
    xs = (np.arange(w) + 0.5) / w
    ys = (np.arange(h) + 0.5) / h
    cx = (m.sum(axis=1) * xs).sum(axis=1) / total
    cy = (m.sum(axis=2) * ys).sum(axis=1) / total
    half = w // 2
    features = {
        "area": area,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "width": x1 - x0,
        "height": y1 - y0,
        "cx": cx,
        "cy": cy,
        "left_area": m[:, :, :half].mean(axis=(1, 2)),
        "right_area": m[:, :, half:].mean(axis=(1, 2)),
    }
    rows = _bin_means(m.mean(axis=2), PROFILE_BINS)
    cols = _bin_means(m.mean(axis=1), PROFILE_BINS)
    for i in range(PROFILE_BINS):
        features[f"row{i}"] = rows[:, i]
        features[f"col{i}"] = cols[:, i]
    side = np.linspace(0, h, THUMB_SIDE + 1).astype(int)
    thumb = np.stack(
        [
            np.stack([m[:, a:b, c:d].mean(axis=(1, 2)) for c, d in zip(side[:-1], side[1:])], axis=1)
            for a, b in zip(side[:-1], side[1:])
        ],
        axis=1,
    ).reshape(n, -1)
    for i in range(thumb.shape[1]):
        features[f"m{i}"] = thumb[:, i]
    return pd.DataFrame(features)


def split_table(csv_path, masks_dir, split: str, size: int, threshold: float, dilation: float):
    table = pd.read_csv(csv_path)
    names = [str(x) for x in table["Image Index"]]
    masks = load_split_masks(masks_dir, split, names, size, threshold, dilation)
    frame = mask_features(masks)
    frame["image"] = names
    frame["label"] = table["Pneumonia"].astype(float).to_numpy()
    return frame


def main():
    parser = argparse.ArgumentParser(description="Lung-mask silhouette baseline.")
    parser.add_argument("--split-dir", default="data/processed_patient_disjoint_dedup_rho_v1_0")
    parser.add_argument("--masks-dir", default="data/lung_masks")
    parser.add_argument("--output-dir", default="docs/silhouette_baseline_patient_disjoint_dedup_rho_v1")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--lung-threshold", type=float, default=0.3)
    parser.add_argument("--dilations", type=float, nargs="+", default=[0.0, 8.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dilation in args.dilations:
        tables = {
            split: split_table(
                Path(args.split_dir) / f"{split}.csv",
                args.masks_dir,
                split,
                args.size,
                args.lung_threshold,
                dilation,
            )
            for split in ("train", "val", "test")
        }
        train, val, test = tables["train"], tables["val"], tables["test"]
        y = test["label"].to_numpy() > 0.5
        print(f"dilation {dilation:g} px: train {len(train)} val {len(val)} test {len(test)}")
        for group, columns in GROUPS.items():
            result = evaluate_group(train, val, test, columns)
            boot = paired_bootstrap_auroc(
                y, result["test_scores"], result["test_scores"], args.seed, args.bootstrap_replicates
            )
            low, high = boot["auroc_ref_ci"]
            rows.append(
                {
                    "dilation_px": dilation,
                    "features": group,
                    "n_features": len(columns),
                    "model": result["model"],
                    "val_auroc": result["val_auroc"],
                    "test_auroc": result["test_auroc"],
                    "test_auroc_ci_low": low,
                    "test_auroc_ci_high": high,
                }
            )
            r = rows[-1]
            print(f"  {group:20s} {r['model']:18s} val {r['val_auroc']:.4f} test {r['test_auroc']:.4f}")
        medians = test.groupby("label")["area"].median()
        normal, pneumonia = medians.get(0.0, float("nan")), medians.get(1.0, float("nan"))
        print(f"  median mask area: Normal {normal:.3f}, Pneumonia {pneumonia:.3f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "silhouette_baseline_summary.csv", index=False)
    lines = [
        "# Silhouette baseline",
        "",
        "Classifiers that see only the shape of the lung mask (no pixel intensities), trained on the",
        "patient-disjoint, deduplicated train split, model picked on validation, scored on the test split.",
        "Intervals are image-level bootstrap and ignore patient clustering.",
        "",
        "| Mask dilation (px) | Features | Model | Val AUROC | Test AUROC [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dilation_px']:g} | {r['features']} | {r['model']} | {r['val_auroc']:.4f} | "
            f"{r['test_auroc']:.4f} [{r['test_auroc_ci_low']:.4f}, {r['test_auroc_ci_high']:.4f}] |"
        )
    lines.append("")
    (out_dir / "silhouette_baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "run_info.json").write_text(
        json.dumps({"seed": args.seed, "dilations": args.dilations, "size": args.size}, indent=2),
        encoding="utf-8",
    )
    print("\n" + "\n".join(lines))
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
