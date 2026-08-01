"""
Downloads the Kaggle Chest X-ray Pneumonia dataset and prepares it for the
multimodal pipeline, augmented with SYNTHETIC clinical vitals (temperature,
SpO2) alongside age/gender, since the original dataset ships images only —
no EHR metadata.

WHY SYNTHETIC DATA, AND WHY IT'S STILL LEGITIMATE:
This project's point is to demonstrate a multimodal fusion + explainability
architecture, not to publish real clinical findings. The synthetic vitals
are generated with clinically plausible distributions correlated with the
Pneumonia label (fever and lower oxygen saturation more likely when
Pneumonia=1), so the fusion ablation study demonstrates something real
about the architecture rather than fusing against pure noise. This is
disclosed prominently here and in docs/ethics_statement.md — never present
these columns as real patient measurements.

Source dataset: Kelermanov et al. / Guangzhou Women and Children's Medical
Center pediatric chest X-rays (ages 1-5), released on Kaggle by
paultimothymooney: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Usage:
    python data/scripts/prepare_pneumonia_dataset.py
"""

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_IMAGES_DIR = Path("data/raw/images")
PROCESSED_DIR = Path("data/processed")
RANDOM_SEED = 42


def download_dataset() -> Path:
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit(
            "kagglehub is required for this script.\nInstall it with: pip install kagglehub"
        ) from e

    print("Downloading Chest X-ray Pneumonia dataset from Kaggle...")
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"Downloaded to: {path}")
    return Path(path)


def find_chest_xray_root(source_dir: Path) -> Path:
    """Kaggle's download sometimes nests an extra chest_xray/ folder;
    locate the actual root containing train/ and test/ subfolders."""
    for candidate in source_dir.rglob("train"):
        if (candidate.parent / "test").exists():
            return candidate.parent
    raise SystemExit(
        f"Could not locate train/ and test/ folders under {source_dir}. "
        "Kaggle's directory layout may have changed — inspect the download manually."
    )


def collect_split(root: Path, split: str) -> list[dict]:
    """Walks a train/test/val split's NORMAL and PNEUMONIA folders and
    returns one row per image with a best-effort Patient ID (pneumonia
    filenames encode a person ID we can group by; normal images don't
    expose one, so each is treated as its own patient — documented as a
    known limitation in docs/ethics_statement.md)."""
    rows = []
    for is_pneumonia, folder_name in [(1, "PNEUMONIA"), (0, "NORMAL")]:
        folder = root / split / folder_name
        if not folder.exists():
            continue
        image_paths = sorted(
            list(folder.glob("*.jpeg")) + list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        )
        for img_path in image_paths:
            person_match = re.match(r"person(\d+)_", img_path.name)
            patient_id = (
                f"pneumonia_person_{person_match.group(1)}"
                if person_match
                else f"{split}_{folder_name}_{img_path.stem}"
            )
            rows.append(
                {
                    "source_path": img_path,
                    "split": split,
                    "Pneumonia": is_pneumonia,
                    "Patient ID": patient_id,
                }
            )
    return rows


def stage_images(rows: list[dict]) -> None:
    """Copies/symlinks images into a single flat data/raw/images/ folder
    (prefixed to avoid filename collisions across splits/classes), matching
    what ChestXrayDataset expects."""
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for row in rows:
        dest_name = f"{row['split']}_{row['Pneumonia']}_{row['source_path'].name}".replace(" ", "_")
        dest = RAW_IMAGES_DIR / dest_name
        row["Image Index"] = dest_name
        if not dest.exists():
            try:
                dest.symlink_to(row["source_path"].resolve())
            except OSError:
                shutil.copy(row["source_path"], dest)


def synthesize_clinical_features(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Adds SYNTHETIC age, gender, temperature, and SpO2 columns. See module
    docstring and docs/ethics_statement.md — these are not real measurements."""
    rng = np.random.default_rng(seed)
    n = len(df)
    is_pneumonia = df["Pneumonia"].to_numpy().astype(bool)

    # Source cohort is pediatric (ages 1-5) — keep synthetic ages consistent with that.
    df["Patient Age"] = np.clip(rng.normal(3.0, 1.3, n), 1.0, 5.0).round(1)
    df["Patient Gender"] = rng.choice(["Male", "Female"], size=n)

    # Temperature (°F): fever more likely for pneumonia-positive cases
    temp = np.where(is_pneumonia, rng.normal(100.4, 1.3, n), rng.normal(98.3, 0.5, n))
    df["Temperature"] = np.clip(temp, 96.0, 105.0).round(1)

    # SpO2 (%): lower oxygen saturation more likely for pneumonia-positive cases
    spo2 = np.where(is_pneumonia, rng.normal(93.5, 2.5, n), rng.normal(98.0, 1.0, n))
    df["SpO2"] = np.clip(spo2, 80.0, 100.0).round(1)

    return df


def build_split_dataframes(root: Path, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows = collect_split(root, "train")
    # The official val/ folder in this dataset has only 16 images total —
    # too small to be a reliable validation set — so we fold it into the
    # train pool and carve our own stratified validation split below.
    train_rows += collect_split(root, "val")
    test_rows = collect_split(root, "test")

    all_rows = train_rows + test_rows
    stage_images(all_rows)

    train_pool_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    # Patient-level split within the train pool only (test stays exactly as
    # the original curators split it — already patient-disjoint from train).
    labels_per_patient = train_pool_df.groupby("Patient ID")["Pneumonia"].first()
    patient_ids = labels_per_patient.index.to_numpy()

    train_ids, val_ids = train_test_split(
        patient_ids,
        test_size=val_frac,
        random_state=RANDOM_SEED,
        stratify=labels_per_patient.values,
    )

    train_df = train_pool_df[train_pool_df["Patient ID"].isin(train_ids)].copy()
    val_df = train_pool_df[train_pool_df["Patient ID"].isin(val_ids)].copy()

    for split_df in (train_df, val_df, test_df):
        synthesize_clinical_features(split_df)

    return train_df, val_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of the TRAIN split (patient-level) carved out for validation.",
    )
    args = parser.parse_args()

    source_dir = download_dataset()
    root = find_chest_xray_root(source_dir)

    train_df, val_df, test_df = build_split_dataframes(root, args.val_frac)

    keep_cols = [
        "Image Index",
        "Pneumonia",
        "Patient ID",
        "Patient Age",
        "Patient Gender",
        "Temperature",
        "SpO2",
    ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df[keep_cols].to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df[keep_cols].to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df[keep_cols].to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(
        f"train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)} "
        f"(total: {len(train_df) + len(val_df) + len(test_df)})"
    )
    print(f"Wrote CSVs to {PROCESSED_DIR}/")
    print(
        "\nNOTE: Patient Age, Patient Gender, Temperature, and SpO2 are SYNTHETIC "
        "clinical features generated with clinically plausible (but simulated) "
        "distributions correlated with the Pneumonia label. See "
        "docs/ethics_statement.md before presenting results from this data."
    )


if __name__ == "__main__":
    main()
