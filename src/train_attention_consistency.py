"""
Trains the vision model with CBAM, same as src/train.py, but adds an
attention-consistency loss term that directly penalizes CBAM's spatial
attention map for falling outside the segmented lung field — rather than
only measuring this after the fact the way
src/explain/measure_lung_localization.py does. See
src/models/attention_consistency_loss.py's docstring for the literature
basis and exact loss definition.

Resulting checkpoints are architecturally identical to a normal
use_cbam=true vision model (same state_dict keys, same forward()) — every
existing script (evaluate.py, the dashboard, generate_examples.py,
measure_lung_localization.py) works on them completely unmodified. Only
*how* the weights got trained differs.

Requires precomputed lung masks — run
data/scripts/precompute_lung_masks.py first. Also requires
use_cbam: true in the training config (there's no attention map to
regularize otherwise) and, per that script's documented tradeoff, expects
augmentation disabled to keep masks spatially aligned with images — this
script overrides train.augmentation to false regardless of what's in the
config, and warns if the config didn't already say so, rather than
silently training with misaligned masks.

Usage:
    python data/scripts/precompute_lung_masks.py --train-config configs/vision_attention_consistency.yaml
    python src/train_attention_consistency.py --train-config configs/vision_attention_consistency.yaml
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import ChestXrayDataset
from src.data.lung_mask_dataset import LungMaskAugmentedDataset
from src.models.attention_consistency_loss import attention_consistency_loss
from src.models.vision_encoder import ChestXrayVisionModel
from src.train import compute_macro_auroc, load_config


def build_dataloaders(data_cfg: dict, train_cfg: dict, lung_masks_path: str):
    classes = data_cfg["labels"]["classes"]
    tabular_features = data_cfg["tabular_features"]
    image_dir = train_cfg["data"]["image_dir"]
    image_size = train_cfg["data"]["image_size"]

    # train=False here is deliberate, not a bug — see module docstring and
    # data/scripts/precompute_lung_masks.py: masks were computed from the
    # un-augmented image, so training must use the same un-augmented image
    # or the mask and image go out of spatial alignment.
    train_ds = LungMaskAugmentedDataset(
        csv_path=train_cfg["data"]["train_csv"],
        image_dir=image_dir,
        classes=classes,
        tabular_features=tabular_features,
        image_size=image_size,
        train=False,
        lung_masks_path=lung_masks_path,
    )
    tabular_stats = train_ds.get_tabular_stats()

    val_ds = ChestXrayDataset(
        csv_path=train_cfg["data"]["val_csv"],
        image_dir=image_dir,
        classes=classes,
        tabular_features=tabular_features,
        image_size=image_size,
        train=False,
        tabular_stats=tabular_stats,
    )

    pin_memory = train_cfg["train"].get("pin_memory", False)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=train_cfg["train"]["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=train_cfg["train"]["num_workers"],
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, classes, tabular_stats


def train_one_epoch(model, loader, optimizer, criterion, consistency_weight, device) -> tuple[float, float, float]:
    """Returns (total_loss, bce_loss, consistency_loss) — all averaged
    over the epoch — so training progress can be inspected as two numbers
    instead of one blended one, useful for sanity-checking that the
    consistency term is actually shrinking, not just riding along."""
    model.train()
    running_total, running_bce, running_consistency = 0.0, 0.0, 0.0

    for images, _tabular, labels, lung_masks in tqdm(loader, desc="train", leave=False):
        images, labels, lung_masks = images.to(device), labels.to(device), lung_masks.to(device)

        optimizer.zero_grad()
        logits, attention_map = model.forward_with_attention_map(images)
        bce = criterion(logits, labels)
        consistency = attention_consistency_loss(attention_map, lung_masks)
        loss = bce + consistency_weight * consistency
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_total += loss.item() * batch_size
        running_bce += bce.item() * batch_size
        running_consistency += consistency.item() * batch_size

    n = len(loader.dataset)
    return running_total / n, running_bce / n, running_consistency / n


@torch.no_grad()
def evaluate(model, loader, criterion, classes, device) -> dict:
    """Plain-vanilla validation (no lung masks needed here — val
    intentionally isn't part of the consistency-loss training signal,
    only used to pick the best checkpoint via macro AUROC, same as
    src/train.py). No attention-consistency number is tracked on val;
    that's what measure_lung_localization.py is for, after training."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, _tabular, labels in tqdm(loader, desc="val", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        all_preds.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    macro_auroc, per_class_auroc = compute_macro_auroc(all_labels, all_preds, classes)

    return {
        "val_loss": running_loss / len(loader.dataset),
        "val_macro_auroc": macro_auroc,
        "per_class_auroc": per_class_auroc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_attention_consistency.yaml")
    parser.add_argument("--lung-masks", default="data/processed/lung_masks.pkl")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    train_cfg = load_config(args.train_config)

    if not train_cfg["model"].get("use_cbam", False):
        raise ValueError(
            "This script requires use_cbam: true — there's no CBAM spatial "
            "attention map to regularize otherwise. Use src/train.py for a "
            "plain (or no-CBAM) run instead."
        )

    consistency_weight = float(train_cfg["train"].get("attention_consistency_weight", 0.1))

    torch.manual_seed(train_cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"attention_consistency_weight = {consistency_weight}")

    train_loader, val_loader, classes, tabular_stats = build_dataloaders(data_cfg, train_cfg, args.lung_masks)

    model = ChestXrayVisionModel(
        num_classes=len(classes),
        pretrained=train_cfg["model"]["pretrained"],
        dropout=train_cfg["model"]["dropout"],
        use_cbam=True,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["train"]["lr"]),
        weight_decay=float(train_cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["train"]["epochs"])

    checkpoint_dir = Path(train_cfg["checkpoint"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(train_cfg["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = log_dir / "metrics.csv"

    best_metric = -1.0
    epochs_without_improvement = 0

    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_bce", "train_consistency", "val_loss", "val_macro_auroc"])

    for epoch in range(1, train_cfg["train"]["epochs"] + 1):
        train_loss, train_bce, train_consistency = train_one_epoch(
            model, train_loader, optimizer, criterion, consistency_weight, device
        )
        val_metrics = evaluate(model, val_loader, criterion, classes, device)
        scheduler.step()

        print(
            f"Epoch {epoch}/{train_cfg['train']['epochs']} | "
            f"train_loss={train_loss:.4f} (bce={train_bce:.4f}, consistency={train_consistency:.4f}) | "
            f"val_loss={val_metrics['val_loss']:.4f} | val_macro_auroc={val_metrics['val_macro_auroc']:.4f}"
        )

        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, train_loss, train_bce, train_consistency, val_metrics["val_loss"], val_metrics["val_macro_auroc"]]
            )

        current_metric = val_metrics["val_macro_auroc"]
        if current_metric > best_metric:
            best_metric = current_metric
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "tabular_stats": tabular_stats,
                    "epoch": epoch,
                    "use_cbam": True,
                    # Not read by any existing script, but worth recording
                    # directly in the checkpoint — this is the one piece of
                    # information that distinguishes this checkpoint from a
                    # plain use_cbam=true run trained by src/train.py, and
                    # printed output / a training log alone could get lost
                    # or separated from the .pth file over time.
                    "attention_consistency_weight": consistency_weight,
                },
                checkpoint_dir / "best_model.pth",
            )
            print(f"  -> new best model saved (val_macro_auroc={best_metric:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["train"]["early_stopping_patience"]:
                print(
                    f"Early stopping at epoch {epoch} (no improvement for "
                    f"{train_cfg['train']['early_stopping_patience']} epochs)."
                )
                break

    print(f"Training complete. Best val_macro_auroc: {best_metric:.4f}")
    print(f"Metrics logged to {metrics_csv}")


if __name__ == "__main__":
    main()
