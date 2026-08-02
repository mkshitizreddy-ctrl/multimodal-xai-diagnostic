"""
Regression test for a real bug: DataLoader(pin_memory=True) caused a silent
access-violation crash (no Python traceback) on a Windows + CUDA setup
during actual training. build_dataloaders() now reads pin_memory from
config, defaulting to False. This test locks in that default and confirms
the config value is actually respected.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import build_dataloaders

CLASSES = ["Pneumonia"]
TABULAR_FEATURES = ["Patient Age", "Patient Gender"]


@pytest.fixture
def tiny_processed_dataset(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    rows = []
    for i in range(4):
        filename = f"fake_{i}.jpeg"
        Image.fromarray((np.random.rand(64, 64) * 255).astype(np.uint8)).save(image_dir / filename)
        rows.append(
            {
                "Image Index": filename,
                "Patient Age": 3.0 + i * 0.1,
                "Patient Gender": "Male" if i % 2 == 0 else "Female",
                "Pneumonia": i % 2,
            }
        )

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)
    pd.DataFrame(rows).to_csv(val_csv, index=False)

    data_cfg = {"labels": {"classes": CLASSES}, "tabular_features": TABULAR_FEATURES}
    base_train_cfg = {
        "data": {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "image_dir": str(image_dir),
            "image_size": 64,
        },
        "train": {"batch_size": 2, "num_workers": 0},
    }
    return data_cfg, base_train_cfg


def test_pin_memory_defaults_to_false_when_unset(tiny_processed_dataset):
    data_cfg, train_cfg = tiny_processed_dataset
    train_loader, val_loader, classes = build_dataloaders(data_cfg, train_cfg)

    assert train_loader.pin_memory is False
    assert val_loader.pin_memory is False


def test_pin_memory_respects_explicit_config_value(tiny_processed_dataset):
    data_cfg, train_cfg = tiny_processed_dataset
    train_cfg["train"]["pin_memory"] = True

    train_loader, val_loader, classes = build_dataloaders(data_cfg, train_cfg)

    assert train_loader.pin_memory is True
    assert val_loader.pin_memory is True


def test_dataloaders_still_produce_valid_batches(tiny_processed_dataset):
    data_cfg, train_cfg = tiny_processed_dataset
    train_loader, val_loader, classes = build_dataloaders(data_cfg, train_cfg)

    images, tabular, labels = next(iter(train_loader))
    assert images.shape[1:] == (3, 64, 64)
    assert tabular.shape[1] == len(TABULAR_FEATURES)
    assert labels.shape[1] == len(CLASSES)
    assert classes == CLASSES
