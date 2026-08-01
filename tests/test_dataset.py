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
