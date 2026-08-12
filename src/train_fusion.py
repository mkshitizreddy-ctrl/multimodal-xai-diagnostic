"""
Train the multimodal fusion model (vision + tabular) on NIH Chest X-ray14.

Structurally mirrors src/train.py, adapted to pass both image and tabular
tensors through the model. Kept as a separate script (rather than branching
inside train.py) so the vision-only baseline stays simple and untouched —
useful since it's the control arm of the fusion ablation.

Usage:
    python src/train_fusion.py --data-config configs/data.yaml --train-config configs/fusion.yaml
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.fusion import ChestXrayFusionModel
from src.train import build_dataloaders, compute_macro_auroc, load_config


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for images, tabular, labels in tqdm(loader, desc="train", leave=False):
        images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images, tabular)
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

    for images, tabular, labels in tqdm(loader, desc="val", leave=False):
        images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
        logits = model(images, tabular)
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
    parser.add_argument("--train-config", default="configs/fusion.yaml")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    train_cfg = load_config(args.train_config)

    torch.manual_seed(train_cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes, tabular_stats = build_dataloaders(data_cfg, train_cfg)
    num_tabular_features = len(data_cfg["tabular_features"])

    model = ChestXrayFusionModel(
        num_classes=len(classes),
        num_tabular_features=num_tabular_features,
        pretrained=train_cfg["model"]["pretrained"],
        tabular_embedding_dim=train_cfg["model"]["tabular_embedding_dim"],
        dropout=train_cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        # See src/train.py — guards against PyYAML parsing "1e-4" as a string.
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
                    "tabular_features": data_cfg["tabular_features"],
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
