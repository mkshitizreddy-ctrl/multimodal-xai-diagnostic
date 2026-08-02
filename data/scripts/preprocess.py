"""
Preprocess NIH Chest X-ray14 metadata into clean, patient-level
train/val/test splits ready for the PyTorch Dataset class.

Reads:
    data/raw/Data_Entry_2017.csv

Writes:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv

Usage:
    python data/scripts/preprocess.py --config configs/data.yaml
"""

import argparse
from pathlib import Path

import pandas as pd

# See src/data/dataset.py for details — avoids a pyarrow arrow.dll crash
# observed on some Windows setups with pandas >= 3.0's default string dtype.
try:
    pd.options.future.infer_string = False
except AttributeError:
    pass
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def multi_hot_encode_labels(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Turn the pipe-separated 'Finding Labels' column into multi-hot columns."""
    for cls in classes:
        df[cls] = df["Finding Labels"].apply(
            lambda labels: int(cls in str(labels).split("|"))
        )
    return df


def clean_metadata(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    # Standardize gender to a fixed vocabulary
    df["Patient Gender"] = df["Patient Gender"].map({"M": "Male", "F": "Female"}).fillna("Unknown")

    # Drop physiologically impossible ages (a few rows in this dataset have age > 120)
    df = df[(df["Patient Age"] > 0) & (df["Patient Age"] < 120)]

    # Multi-hot encode the 14 disease labels
    df = multi_hot_encode_labels(df, cfg["labels"]["classes"])

    return df


def patient_level_split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by Patient ID so no patient's images leak across splits."""
    split_col = cfg["split"]["split_by"]
    seed = cfg["split"]["random_seed"]

    patient_ids = df[split_col].unique()

    train_ids, temp_ids = train_test_split(
        patient_ids, train_size=cfg["split"]["train_frac"], random_state=seed
    )
    relative_val_frac = cfg["split"]["val_frac"] / (cfg["split"]["val_frac"] + cfg["split"]["test_frac"])
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=relative_val_frac, random_state=seed
    )

    train_df = df[df[split_col].isin(train_ids)]
    val_df = df[df[split_col].isin(val_ids)]
    test_df = df[df[split_col].isin(test_ids)]

    return train_df, val_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    metadata_path = Path(cfg["dataset"]["metadata_csv"])
    if not metadata_path.exists():
        raise SystemExit(
            f"{metadata_path} not found. Run data/scripts/download_nih.py first."
        )

    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} rows covering {df['Patient ID'].nunique()} patients.")

    df = clean_metadata(df, cfg)
    train_df, val_df, test_df = patient_level_split(df, cfg)

    processed_dir = Path(cfg["dataset"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    print(
        f"Wrote splits -> train: {len(train_df)} rows, "
        f"val: {len(val_df)} rows, test: {len(test_df)} rows "
        f"(saved to {processed_dir}/)"
    )


if __name__ == "__main__":
    main()
