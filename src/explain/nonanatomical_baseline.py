"""Non-anatomical baseline: how well do image-level acquisition cues predict the label?

Trains simple classifiers on features that carry no fine anatomical detail (original image size, file
size, global intensity statistics, a 16x16 thumbnail) using the patient-disjoint, deduplicated split, and
reports test AUROC next to the CNN's. High AUROC from such features would show that the label is
recoverable from acquisition properties, which is a prerequisite for shortcut learning. It does not show
that the CNN uses those cues, and low AUROC would not rule shortcuts out.

Usage:
    python src/explain/nonanatomical_baseline.py --cnn-measurements \\
        docs/shortcut_audit_patient_disjoint_dedup_rho_v1_vision/shortcut_audit_measurements.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explain.shortcut_audit_utils import auroc, paired_bootstrap_auroc

SIZE_FEATURES = ["width", "height", "aspect", "file_kb", "bytes_per_pixel"]
INTENSITY_FEATURES = [
    "mean",
    "std",
    "p01",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
    "p99",
    "frac_dark",
    "frac_bright",
    "border_mean",
    "border_dark",
]
THUMB_SIDE = 16
THUMB_FEATURES = [f"t{i}" for i in range(THUMB_SIDE * THUMB_SIDE)]
GROUPS = {
    "file_size_only": ["file_kb"],
    "dimensions_only": ["width", "height", "aspect"],
    "pixel_mean_std_only": ["mean", "std"],
    "size_meta": SIZE_FEATURES,
    "intensity": INTENSITY_FEATURES,
    "size_meta+intensity": SIZE_FEATURES + INTENSITY_FEATURES,
    "thumbnail_16x16": THUMB_FEATURES,
}


def image_features(path, size: int = 224) -> dict:
    """Acquisition-level features of one image file (no fine anatomical detail)."""
    with Image.open(path) as image:
        width, height = image.size
        gray = image.convert("L")
        resample = Image.Resampling.BILINEAR
        small = np.asarray(gray.resize((size, size), resample), dtype=np.float32) / 255.0
        thumb = np.asarray(gray.resize((THUMB_SIDE, THUMB_SIDE), resample), dtype=np.float32) / 255.0
    file_bytes = os.path.getsize(path)
    margin = max(1, int(round(size * 0.05)))
    border = np.concatenate(
        [
            small[:margin].ravel(),
            small[-margin:].ravel(),
            small[:, :margin].ravel(),
            small[:, -margin:].ravel(),
        ]
    )
    percentiles = np.percentile(small, [1, 5, 25, 50, 75, 95, 99])
    features = {
        "width": float(width),
        "height": float(height),
        "aspect": width / height,
        "file_kb": file_bytes / 1024.0,
        "bytes_per_pixel": file_bytes / (width * height),
        "mean": float(small.mean()),
        "std": float(small.std()),
        "frac_dark": float((small < 0.04).mean()),
        "frac_bright": float((small > 0.96).mean()),
        "border_mean": float(border.mean()),
        "border_dark": float((border < 0.04).mean()),
    }
    for name, value in zip(["p01", "p05", "p25", "p50", "p75", "p95", "p99"], percentiles):
        features[name] = float(value)
    for i, value in enumerate(thumb.ravel()):
        features[f"t{i}"] = float(value)
    return features


def build_feature_table(csv_path, image_dir, split_name: str) -> pd.DataFrame:
    table = pd.read_csv(csv_path)
    rows = []
    for i, row in table.iterrows():
        record = image_features(Path(image_dir) / row["Image Index"])
        record["image"] = row["Image Index"]
        record["label"] = float(row["Pneumonia"])
        record["split"] = split_name
        rows.append(record)
        if (i + 1) % 500 == 0:
            print(f"  {split_name}: {i + 1}/{len(table)}")
    return pd.DataFrame(rows)


def candidate_models() -> dict:
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)),
        "gradient_boosting": HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=0),
    }


def evaluate_group(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> dict:
    """Fit each candidate on train, pick the one with the best validation AUROC, score the test split."""
    best = None
    for name, model in candidate_models().items():
        model.fit(train[columns], train["label"])
        val_auc = auroc(val["label"], model.predict_proba(val[columns])[:, 1])
        if best is None or val_auc > best["val_auroc"]:
            best = {"model": name, "val_auroc": val_auc, "fitted": model}
    test_scores = best["fitted"].predict_proba(test[columns])[:, 1]
    return {
        "model": best["model"],
        "val_auroc": best["val_auroc"],
        "test_auroc": auroc(test["label"], test_scores),
        "test_scores": test_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Non-anatomical baseline classifiers.")
    parser.add_argument("--split-dir", default="data/processed_patient_disjoint_dedup_rho_v1_0")
    parser.add_argument("--image-dir", default="data/raw/images")
    parser.add_argument("--output-dir", default="docs/nonanatomical_baseline_patient_disjoint_dedup_rho_v1")
    parser.add_argument(
        "--cnn-measurements",
        default=None,
        help="shortcut_audit_measurements.csv for a paired comparison with the CNN",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for name in ("train", "val", "test"):
        split_cache = out_dir / f"image_features_{name}.csv"
        if split_cache.exists():
            print(f"Loaded cached {name} features from {split_cache}")
            parts.append(pd.read_csv(split_cache))
        else:
            part = build_feature_table(Path(args.split_dir) / f"{name}.csv", args.image_dir, name)
            part.to_csv(split_cache, index=False)
            parts.append(part)
    features = pd.concat(parts, ignore_index=True)

    train = features[features["split"] == "train"]
    val = features[features["split"] == "val"]
    test = features[features["split"] == "test"].reset_index(drop=True)
    print(f"train {len(train)}  val {len(val)}  test {len(test)} ({int(test['label'].sum())} Pneumonia)")

    cnn = None
    if args.cnn_measurements:
        measurements = pd.read_csv(args.cnn_measurements)
        cnn = test.merge(measurements[["image", "logit_original"]], on="image", how="left")
        if cnn["logit_original"].isna().any():
            raise ValueError("Some test images are missing from the CNN measurements file")

    rows = []
    for group, columns in GROUPS.items():
        result = evaluate_group(train, val, test, columns)
        y = test["label"].to_numpy() > 0.5
        row = {
            "features": group,
            "n_features": len(columns),
            "model": result["model"],
            "val_auroc": result["val_auroc"],
            "test_auroc": result["test_auroc"],
        }
        boot = paired_bootstrap_auroc(
            y, result["test_scores"], result["test_scores"], args.seed, args.bootstrap_replicates
        )
        row["test_auroc_ci_low"], row["test_auroc_ci_high"] = boot["auroc_ref_ci"]
        if cnn is not None:
            paired = paired_bootstrap_auroc(
                y,
                cnn["logit_original"].to_numpy(),
                result["test_scores"],
                args.seed,
                args.bootstrap_replicates,
            )
            row["cnn_auroc"] = paired["auroc_ref"]
            row["delta_vs_cnn"] = paired["delta"]
            row["delta_vs_cnn_ci_low"], row["delta_vs_cnn_ci_high"] = paired["delta_ci"]
        rows.append(row)
        print(f"{group:22s} {result['model']:18s} val {row['val_auroc']:.4f}  test {row['test_auroc']:.4f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "nonanatomical_baseline_summary.csv", index=False)

    # Single-feature test AUROCs (direction-free), to show which cue carries the signal.
    single = []
    for column in SIZE_FEATURES + INTENSITY_FEATURES:
        a = auroc(test["label"], test[column])
        single.append({"feature": column, "test_auroc": a, "direction_free_auroc": max(a, 1 - a)})
    single_df = pd.DataFrame(single).sort_values("direction_free_auroc", ascending=False)
    single_df.to_csv(out_dir / "single_feature_auroc.csv", index=False)

    lines = [
        "# Non-anatomical baseline",
        "",
        "Classifiers that see only acquisition-level features, trained on the patient-disjoint, deduplicated",
        "train split and scored on the test split. This shows how recoverable the label is from such",
        "cues; it does not show that the CNN uses them.",
        "",
        "| Features | Model (picked on val) | Val AUROC | Test AUROC [95% CI] |"
        + (" CNN AUROC | Change vs CNN [95% CI] |" if cnn is not None else ""),
        "|---|---|---|---|" + ("---|---|" if cnn is not None else ""),
    ]
    for r in rows:
        line = (
            f"| {r['features']} | {r['model']} | {r['val_auroc']:.4f} | "
            f"{r['test_auroc']:.4f} [{r['test_auroc_ci_low']:.4f}, {r['test_auroc_ci_high']:.4f}] |"
        )
        if cnn is not None:
            line += (
                f" {r['cnn_auroc']:.4f} | {r['delta_vs_cnn']:+.4f} "
                f"[{r['delta_vs_cnn_ci_low']:+.4f}, {r['delta_vs_cnn_ci_high']:+.4f}] |"
            )
        lines.append(line)
    lines += [
        "",
        "## Strongest single features (test AUROC, direction-free)",
        "",
        "| Feature | AUROC |",
        "|---|---|",
    ]
    for _, r in single_df.head(8).iterrows():
        lines.append(f"| {r['feature']} | {r['direction_free_auroc']:.4f} |")
    medians = test.groupby("label")[["width", "height", "file_kb", "bytes_per_pixel"]].median()
    distinct = {
        int(k): int(g[["width", "height"]].drop_duplicates().shape[0]) for k, g in test.groupby("label")
    }
    lines += [
        "",
        "## Test-split acquisition properties by class (median)",
        "",
        "| Class | Width | Height | File size (KB) | Bytes/pixel | Distinct (width, height) |",
        "|---|---|---|---|---|---|",
    ]
    for label, name in ((0.0, "Normal"), (1.0, "Pneumonia")):
        m = medians.loc[label]
        lines.append(
            f"| {name} | {m['width']:.0f} | {m['height']:.0f} | {m['file_kb']:.1f} | "
            f"{m['bytes_per_pixel']:.3f} | {distinct[int(label)]} |"
        )
    lines.append("")
    (out_dir / "nonanatomical_baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "run_info.json").write_text(
        json.dumps({"seed": args.seed, "split_dir": args.split_dir, "n_test": int(len(test))}, indent=2),
        encoding="utf-8",
    )
    print("\n" + "\n".join(lines))
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
