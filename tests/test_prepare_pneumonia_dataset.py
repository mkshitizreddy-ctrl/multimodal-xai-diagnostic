"""
Tests for data/scripts/prepare_pneumonia_dataset.py. Builds a small fake
Kaggle-style directory (train/val/test x NORMAL/PNEUMONIA) so these run
without needing the real ~2GB download.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.scripts.prepare_pneumonia_dataset import (
    build_split_dataframes,
    find_chest_xray_root,
    synthesize_clinical_features,
)


@pytest.fixture
def fake_kaggle_dataset(tmp_path, monkeypatch):
    """Builds a fake chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/ tree,
    and redirects the module's output paths into tmp_path so the test
    doesn't touch the real data/ directory."""
    root = tmp_path / "chest_xray"
    counts = {
        "train": {"NORMAL": 12, "PNEUMONIA": 18},
        "val": {"NORMAL": 2, "PNEUMONIA": 2},
        "test": {"NORMAL": 6, "PNEUMONIA": 6},
    }

    for split, class_counts in counts.items():
        for cls, n in class_counts.items():
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                if cls == "PNEUMONIA":
                    person_id = i // 3  # groups images into ~3 per synthetic "patient"
                    fname = f"person{person_id}_bacteria_{i}.jpeg"
                else:
                    fname = f"IM-{split}-{i:04d}.jpeg"
                Image.fromarray((np.random.rand(32, 32) * 255).astype(np.uint8)).save(d / fname)

    raw_images_dir = tmp_path / "raw_images"
    processed_dir = tmp_path / "processed"

    import data.scripts.prepare_pneumonia_dataset as module

    monkeypatch.setattr(module, "RAW_IMAGES_DIR", raw_images_dir)
    monkeypatch.setattr(module, "PROCESSED_DIR", processed_dir)

    return tmp_path


def test_find_chest_xray_root_locates_correct_folder(fake_kaggle_dataset):
    root = find_chest_xray_root(fake_kaggle_dataset)
    assert root.name == "chest_xray"
    assert (root / "train").exists()
    assert (root / "test").exists()


def test_build_split_dataframes_row_counts(fake_kaggle_dataset):
    root = find_chest_xray_root(fake_kaggle_dataset)
    train_df, val_df, test_df = build_split_dataframes(root, val_frac=0.2)

    # train pool = train(30) + val(4) = 34, split ~80/20 by patient
    assert len(train_df) + len(val_df) == 34
    assert len(test_df) == 12


def test_no_patient_leakage_between_train_and_val(fake_kaggle_dataset):
    root = find_chest_xray_root(fake_kaggle_dataset)
    train_df, val_df, _ = build_split_dataframes(root, val_frac=0.2)

    overlap = set(train_df["Patient ID"]) & set(val_df["Patient ID"])
    assert len(overlap) == 0


def test_required_columns_present(fake_kaggle_dataset):
    root = find_chest_xray_root(fake_kaggle_dataset)
    train_df, _, _ = build_split_dataframes(root, val_frac=0.2)

    required = {"Image Index", "Pneumonia", "Patient ID", "Patient Age", "Patient Gender", "Temperature", "SpO2"}
    assert required.issubset(set(train_df.columns))


def test_synthetic_vitals_correlate_with_label_direction():
    import pandas as pd

    df = pd.DataFrame({"Pneumonia": [0] * 500 + [1] * 500})
    df = synthesize_clinical_features(df, seed=0)

    mean_temp_positive = df[df["Pneumonia"] == 1]["Temperature"].mean()
    mean_temp_negative = df[df["Pneumonia"] == 0]["Temperature"].mean()
    mean_spo2_positive = df[df["Pneumonia"] == 1]["SpO2"].mean()
    mean_spo2_negative = df[df["Pneumonia"] == 0]["SpO2"].mean()

    # Pneumonia-positive should run a fever and lower oxygen saturation on average
    assert mean_temp_positive > mean_temp_negative
    assert mean_spo2_positive < mean_spo2_negative


def test_synthetic_vitals_within_clinically_plausible_bounds():
    import pandas as pd

    df = pd.DataFrame({"Pneumonia": [0] * 200 + [1] * 200})
    df = synthesize_clinical_features(df, seed=1)

    assert df["Temperature"].between(96.0, 105.0).all()
    assert df["SpO2"].between(80.0, 100.0).all()
    assert df["Patient Age"].between(1.0, 5.0).all()
    assert set(df["Patient Gender"].unique()).issubset({"Male", "Female"})
