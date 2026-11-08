"""Does the square-resize step matter? Test AUROC, overall and within aspect-ratio quintiles.

Evaluates a checkpoint with the resize mode it was trained with (``stretch`` for the original models,
``pad`` or ``center_crop`` for the resize controls). Aspect ratio is the original width / height. A
within-quintile AUROC that stays high shows the model separates the classes among images of similar
shape. Descriptive only: one split, image-level bootstrap intervals that ignore patient clustering.

Usage:
    python src/explain/resize_mode_eval.py \\
        --checkpoint checkpoints/vision_resize_pad/best_model.pth \\
        --train-config configs/vision_resize_pad.yaml \\
        --output-csv docs/resize_control/pad_s42.csv
"""

import argparse
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
from src.explain.shortcut_audit import file_sha256
from src.explain.shortcut_audit_utils import auroc
from src.models.vision_encoder import ChestXrayVisionModel


def aspect_quintile_auroc(y_true, scores, aspect, n_bins: int = 5) -> pd.DataFrame:
    """AUROC inside equal-count aspect-ratio bins (rows with one class only give NaN)."""
    y_true, scores, aspect = (np.asarray(v, dtype=float) for v in (y_true, scores, aspect))
    bins = pd.qcut(pd.Series(aspect).rank(method="first"), n_bins, labels=False)
    rows = []
    for b in range(n_bins):
        sel = (bins == b).to_numpy()
        yb = y_true[sel]
        both = len(np.unique(yb)) == 2
        rows.append(
            {
                "bin": b + 1,
                "n": int(sel.sum()),
                "n_pneumonia": int((yb > 0.5).sum()),
                "aspect_min": float(aspect[sel].min()),
                "aspect_max": float(aspect[sel].max()),
                "auroc": auroc(yb, scores[sel]) if both else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(y_true, scores, seed: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    vals = []
    for _ in range(replicates):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) == 2:
            vals.append(auroc(y_true[idx], scores[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    parser = argparse.ArgumentParser(description="Resize-mode evaluation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--test-csv", default="data/processed_patient_disjoint_dedup_rho_v1_0/test.csv")
    parser.add_argument("--output-csv", required=True, help="Per-image predictions with aspect ratio.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = yaml.safe_load(open(args.data_config, encoding="utf-8"))
    train_cfg = yaml.safe_load(open(args.train_config, encoding="utf-8"))
    mode = train_cfg["data"].get("resize_mode", "stretch")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    if len(classes) != 1:
        raise ValueError(f"Expected a one-logit binary model; checkpoint classes={classes!r}")
    model = ChestXrayVisionModel(num_classes=1, pretrained=False, use_cbam=checkpoint.get("use_cbam", False))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dataset = ChestXrayDataset(
        csv_path=args.test_csv,
        image_dir=train_cfg["data"]["image_dir"],
        classes=classes,
        tabular_features=data_cfg["tabular_features"],
        image_size=int(train_cfg["data"]["image_size"]),
        train=False,
        resize_mode=mode,
    )
    n = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))
    rows = []
    for i in range(n):
        image, _tab, labels = dataset[i]
        with torch.no_grad():
            logit = float(model(image.unsqueeze(0).to(device)).reshape(-1).float().cpu()[0])
        name = dataset.df.iloc[i]["Image Index"]
        with Image.open(Path(train_cfg["data"]["image_dir"]) / name) as im:
            w, h = im.size
        rows.append({"image": name, "label": float(labels[0]), "logit": logit, "aspect": w / h})
    df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    overall = auroc(df["label"], df["logit"])
    lo, hi = bootstrap_ci(df["label"], df["logit"], args.seed, args.bootstrap_replicates)
    strata = aspect_quintile_auroc(df["label"], df["logit"], df["aspect"])
    info = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "resize_mode": mode,
        "n_images": int(len(df)),
        "auroc": overall,
        "auroc_ci95": [lo, hi],
    }
    Path(args.output_csv).with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"resize_mode={mode}  n={len(df)}  AUROC {overall:.4f} (95% CI {lo:.4f}-{hi:.4f})")
    print(strata.to_string(index=False))


if __name__ == "__main__":
    main()
