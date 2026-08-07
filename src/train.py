"""
Train the DenseNet-121 vision-only baseline on NIH Chest X-ray14.

Usage:
    python src/train.py --data-config configs/data.yaml --train-config configs/vision_baseline.yaml
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import ChestXrayDataset
from src.models.vision_encoder import ChestXrayVisionModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(data_cfg: dict, train_cfg: dict):
    classes = data_cfg["labels"]["classes"]
    tabular_features = data_cfg["tabular_features"]
    image_dir = train_cfg["data"]["image_dir"]
    image_size = train_cfg["data"]["image_size"]

    train_ds = ChestXrayDataset(
        csv_path=train_cfg["data"]["train_csv"],
        image_dir=image_dir,
        classes=classes,
        tabular_features=tabular_features,
        image_size=image_size,
        train=True,
    )

    # Fit normalization stats and the categorical vocab ONCE on train, and
    # reuse them as-is for val (and test, in evaluate.py). Each split fitting
    # its own stats independently would let val statistics leak in and could
    # map the same category to a different integer than train used —
    # silently corrupting results. See ChestXrayDataset.get_tabular_stats().
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

    # pin_memory speeds up host->GPU transfer, but is known to cause silent
    # access-violation crashes on some Windows + CUDA driver combinations
    # (no Python traceback, just an immediate process exit). Configurable
    # via train.pin_memory in the config, defaulting to False to be safe —
    # set to true if your setup handles it fine, for a modest speed gain.
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


def compute_macro_auroc(y_true, y_pred, classes) -> tuple[float, dict]:
    """Returns (macro_auroc, per_class_auroc). Skips classes with no positive
    examples in this batch/split, which is common for rare pathologies."""
    per_class = {}
    valid_scores = []
    for i, cls in enumerate(classes):
        if len(set(y_true[:, i].tolist())) < 2:
            continue  # AUROC undefined with only one class present
        score = roc_auc_score(y_true[:, i], y_pred[:, i])
        per_class[cls] = score
        valid_scores.append(score)
    macro = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    return macro, per_class


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for images, _tabular, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, classes, device) -> dict:
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
    parser.add_argument("--train-config", default="configs/vision_baseline.yaml")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    train_cfg = load_config(args.train_config)

    torch.manual_seed(train_cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes, tabular_stats = build_dataloaders(data_cfg, train_cfg)

    model = ChestXrayVisionModel(
        num_classes=len(classes),
        pretrained=train_cfg["model"]["pretrained"],
        dropout=train_cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        # float() guards against a classic PyYAML gotcha: scientific
        # notation like "1e-4" (no decimal point) is parsed as a STRING,
        # not a float, unless written as "1.0e-4". Casting here means a
        # future config edit that reintroduces this can't silently crash
        # training with a confusing TypeError deep inside torch.optim.
        lr=float(train_cfg["train"]["lr"]),
        weight_decay=float(train_cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["train"]["epochs"]
    )

    checkpoint_dir = Path(train_cfg["checkpoint"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(train_cfg["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = log_dir / "metrics.csv"

    best_metric = -1.0
    epochs_without_improvement = 0

    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_macro_auroc"])

    for epoch in range(1, train_cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, classes, device)
        scheduler.step()

        print(
            f"Epoch {epoch}/{train_cfg['train']['epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['val_loss']:.4f} | "
            f"val_macro_auroc={val_metrics['val_macro_auroc']:.4f}"
        )

        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_metrics["val_loss"], val_metrics["val_macro_auroc"]])

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
                },
                checkpoint_dir / "best_model.pth",
            )
            print(f"  -> new best model saved (val_macro_auroc={best_metric:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["train"]["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{train_cfg['train']['early_stopping_patience']} epochs).")
                break

    print(f"Training complete. Best val_macro_auroc: {best_metric:.4f}")
    print(f"Metrics logged to {metrics_csv}")


if __name__ == "__main__":
    main()
