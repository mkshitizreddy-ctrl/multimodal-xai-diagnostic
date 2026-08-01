"""
Unit test for ChestXrayDataset. Uses a tiny synthetic dataset (fake images
+ a fake CSV) so it runs fast and doesn't require the real NIH download.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import ChestXrayDataset

CLASSES = ["Atelectasis", "Cardiomegaly"]
TABULAR_FEATURES = ["Patient Age", "Patient Gender"]


@pytest.fixture
def synthetic_dataset(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    rows = []
    for i in range(5):
        filename = f"fake_{i}.png"
        Image.fromarray(
            (np.random.rand(256, 256) * 255).astype(np.uint8)
        ).save(image_dir / filename)

        rows.append(
            {
                "Image Index": filename,
                "Patient Age": 40 + i,
                "Patient Gender": "Male" if i % 2 == 0 else "Female",
                "Atelectasis": i % 2,
                "Cardiomegaly": (i + 1) % 2,
            }
        )

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return ChestXrayDataset(
        csv_path=str(csv_path),
        image_dir=str(image_dir),
        classes=CLASSES,
        tabular_features=TABULAR_FEATURES,
        image_size=64,
        train=False,
    )


def test_dataset_length(synthetic_dataset):
    assert len(synthetic_dataset) == 5


def test_sample_shapes(synthetic_dataset):
    image, tabular, labels = synthetic_dataset[0]
    assert image.shape == (3, 64, 64)
    assert tabular.shape == (len(TABULAR_FEATURES),)
    assert labels.shape == (len(CLASSES),)


def test_labels_are_multi_hot(synthetic_dataset):
    _, _, labels = synthetic_dataset[0]
    assert set(labels.tolist()).issubset({0.0, 1.0})


def test_categorical_tabular_encoding_is_stable(synthetic_dataset):
    _, tabular_a, _ = synthetic_dataset[0]
    _, tabular_b, _ = synthetic_dataset[0]
    assert tabular_a.tolist() == tabular_b.tolist()


def _make_csv(tmp_path, name, rows):
    csv_path = tmp_path / name
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_tabular_stats_can_be_reused_across_splits(tmp_path):
    """Regression test: normalization stats and the categorical vocab must
    be fittable once (e.g. on train) and reused as-is by another split,
    rather than each split silently refitting its own stats — which would
    let val/test statistics leak in and could map the same category to a
    different integer across splits."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(4):
        Image.fromarray((np.random.rand(32, 32) * 255).astype(np.uint8)).save(
            image_dir / f"img_{i}.png"
        )

    # Train split: ages 40-43, both genders present.
    train_csv = _make_csv(
        tmp_path,
        "train.csv",
        [
            {
                "Image Index": f"img_{i}.png",
                "Patient Age": 40 + i,
                "Patient Gender": "Male" if i % 2 == 0 else "Female",
                "Atelectasis": 0,
                "Cardiomegaly": 0,
            }
            for i in range(4)
        ],
    )
    # "Val" split reuses the same 4 images but with a very different age
    # range and, critically, only ONE gender present — if vocab/stats were
    # refit here instead of reused, "Male" could map to a different index
    # than it did on train, and age normalization would shift.
    val_csv = _make_csv(
        tmp_path,
        "val.csv",
        [
            {
                "Image Index": f"img_{i}.png",
                "Patient Age": 90,
                "Patient Gender": "Male",
                "Atelectasis": 0,
                "Cardiomegaly": 0,
            }
            for i in range(4)
        ],
    )

    train_ds = ChestXrayDataset(
        csv_path=str(train_csv),
        image_dir=str(image_dir),
        classes=CLASSES,
        tabular_features=TABULAR_FEATURES,
        image_size=32,
        train=True,
    )
    stats = train_ds.get_tabular_stats()

    val_ds = ChestXrayDataset(
        csv_path=str(val_csv),
        image_dir=str(image_dir),
        classes=CLASSES,
        tabular_features=TABULAR_FEATURES,
        image_size=32,
        train=False,
        tabular_stats=stats,
    )

    # The val dataset must use the train split's fitted means/stds/vocab
    # verbatim, not values fit from its own (very different) data.
    assert val_ds.tabular_means == train_ds.tabular_means
    assert val_ds.tabular_stds == train_ds.tabular_stds
    assert val_ds.tabular_vocab == train_ds.tabular_vocab

    # "Male" must resolve to the same integer code that train fit for it.
    gender_col = "Patient Gender"
    _, train_tabular, _ = train_ds[0]  # Male, age 40
    _, val_tabular, _ = val_ds[0]  # Male, age 90
    gender_feature_idx = TABULAR_FEATURES.index(gender_col)
    assert train_tabular[gender_feature_idx].item() == val_tabular[gender_feature_idx].item()
