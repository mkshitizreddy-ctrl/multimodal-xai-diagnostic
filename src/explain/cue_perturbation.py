"""Acquisition-cue perturbation test on the frozen vision checkpoint.

Perturbs cues that differ between the two classes' export pipelines (JPEG compression, global contrast,
aspect ratio, fine texture) before the usual resize, and measures how AUROC and errors change, overall and
on the images where a file-size-only classifier is wrong. A large drop on those images would support
reliance on such cues; no change would support robustness. One dataset, one checkpoint; image-level CIs.

Usage (smoke test, then full run):
    python src/explain/cue_perturbation.py \\
        --checkpoint checkpoints/vision_patient_disjoint_dedup_rho_v1_0/best_model.pth --limit 40
    python src/explain/cue_perturbation.py \\
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ChestXrayDataset
from src.explain.baseline_discordance import youden_threshold
from src.explain.cue_perturbation_utils import PERTURBATIONS, summarize_perturbations, summary_markdown
from src.models.vision_encoder import ChestXrayVisionModel


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_size_baseline_errors(features_dir: Path, images: list[str]) -> np.ndarray:
    """True where the file-size-only classifier (validation Youden threshold) is wrong on a test image."""
    train, val, test = (
        pd.read_csv(features_dir / f"image_features_{name}.csv") for name in ("train", "val", "test")
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(train[["file_kb"]], train["label"])
    threshold = youden_threshold(
        val["label"].to_numpy() > 0.5, model.predict_proba(val[["file_kb"]])[:, 1]
    )
    test = test.set_index("image").loc[images]
    predicted = model.predict_proba(test[["file_kb"]])[:, 1] >= threshold
    return predicted != (test["label"].to_numpy() > 0.5)


def main():
    parser = argparse.ArgumentParser(description="Acquisition-cue perturbation test.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", default="data/processed_patient_disjoint_dedup_rho_v1_0/test.csv")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_baseline.yaml")
    parser.add_argument(
        "--features-dir", default="docs/nonanatomical_baseline_patient_disjoint_dedup_rho_v1"
    )
    parser.add_argument("--output-dir", default="docs/cue_perturbation_patient_disjoint_dedup_rho_v1_vision")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test limit; 0 runs the full test set.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1 or args.bootstrap_replicates < 100:
        parser.error("--limit >= 0, --batch-size >= 1 and --bootstrap-replicates >= 100 are required")

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

    dataset = ChestXrayDataset(
        csv_path=args.test_csv,
        image_dir=train_cfg["data"]["image_dir"],
        classes=classes,
        tabular_features=data_cfg["tabular_features"],
        image_size=train_cfg["data"]["image_size"],
        train=False,
    )
    n = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))
    print(f"Device: {device}; evaluating {n}/{len(dataset)} images x {len(PERTURBATIONS)} conditions")

    def logits_for(tensors: list[torch.Tensor]) -> np.ndarray:
        with torch.no_grad():
            out = model(torch.stack(tensors).to(device))
        return out.reshape(-1).float().cpu().numpy()

    records = []
    for start in range(0, n, args.batch_size):
        stop = min(start + args.batch_size, n)
        rows = [dataset.df.iloc[i] for i in range(start, stop)]
        pil = []
        for row in rows:
            with Image.open(dataset.image_dir / row["Image Index"]) as image:
                pil.append(image.convert("L"))
        batch = {"image": [r["Image Index"] for r in rows], "label": [float(r[classes[0]]) for r in rows]}
        for name, perturb in PERTURBATIONS.items():
            tensors = [dataset.transform(perturb(img).convert("RGB")) for img in pil]
            batch[f"logit_{name}"] = logits_for(tensors)
        records.append(pd.DataFrame(batch))
        print(f"  {stop}/{n}")
    measurements = pd.concat(records, ignore_index=True)

    discordant = file_size_baseline_errors(Path(args.features_dir), measurements["image"].tolist())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    measurements["file_size_baseline_wrong"] = discordant
    measurements.to_csv(out_dir / "cue_perturbation_measurements.csv", index=False)
    summary = summarize_perturbations(
        measurements, discordant, seed=args.seed, replicates=args.bootstrap_replicates
    )
    summary.to_csv(out_dir / "cue_perturbation_summary.csv", index=False)
    markdown = summary_markdown(summary)
    (out_dir / "cue_perturbation_summary.md").write_text(markdown, encoding="utf-8")
    info = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "test_csv": args.test_csv,
        "n_images": int(len(measurements)),
        "n_baseline_wrong": int(discordant.sum()),
        "conditions": list(PERTURBATIONS),
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\n" + markdown)
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
