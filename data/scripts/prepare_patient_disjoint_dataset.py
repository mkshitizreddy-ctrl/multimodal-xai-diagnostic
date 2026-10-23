
"""Create a patient-grouped dataset split without modifying existing data.

Source CSVs: data/processed/{train,val,test}.csv
Output CSVs: data/processed_patient_disjoint/{train,val,test}.csv

Known pneumonia patient IDs are grouped across all existing splits.
NORMAL images receive image-specific groups because reliable patient IDs
are unavailable for those images. Therefore, true patient separation for
NORMAL cases cannot be guaranteed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SOURCE_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed_patient_disjoint")
SEED = 42

KEEP_COLUMNS = [
    "Image Index",
    "Pneumonia",
    "Patient ID",
    "Patient Age",
    "Patient Gender",
    "Temperature",
    "SpO2",
]


def main():
    if OUTPUT_DIR.exists():
        raise SystemExit(
            f"Output directory already exists: {OUTPUT_DIR}\n"
            "Inspect it first; this script will not overwrite it."
        )

    frames = []
    for split in ("train", "val", "test"):
        path = SOURCE_DIR / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)
        missing = set(KEEP_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame[KEEP_COLUMNS].copy())

    df = pd.concat(frames, ignore_index=True)

    if df["Image Index"].duplicated().any():
        raise ValueError("Duplicate Image Index values found.")

    if not df["Pneumonia"].isin([0, 1]).all():
        raise ValueError("Pneumonia labels must be 0 or 1.")

    # All images belonging to a known pneumonia patient share a group.
    # NORMAL images have no reliable patient IDs, so use unique image groups.
    pneumonia = df["Pneumonia"].eq(1)
    df["Group"] = "normal_image:" + df["Image Index"].astype(str)
    df.loc[pneumonia, "Group"] = (
        "pneumonia_patient:"
        + df.loc[pneumonia, "Patient ID"].astype(str)
    )

    # First reserve approximately 20% for test, grouping patients and
    # balancing class labels as closely as the grouped folds allow.
    outer = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )
    development_idx, test_idx = next(
        outer.split(df, df["Pneumonia"], groups=df["Group"])
    )

    development = df.iloc[development_idx].copy().reset_index(drop=True)
    test = df.iloc[test_idx].copy().reset_index(drop=True)

    # Split the remaining ~80% into train and validation. One of five
    # grouped folds gives approximately 16% of development for validation,
    # or about 13% of the full dataset. Exact proportions depend on groups.
    inner = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED + 1
    )
    train_idx, val_idx = next(
        inner.split(
            development,
            development["Pneumonia"],
            groups=development["Group"],
        )
    )

    train = development.iloc[train_idx].copy()
    val = development.iloc[val_idx].copy()

    # Assert that no known pneumonia patient crosses a split.
    split_frames = {"train": train, "val": val, "test": test}
    patient_sets = {
        name: set(
            part.loc[part["Pneumonia"].eq(1), "Patient ID"].astype(str)
        )
        for name, part in split_frames.items()
    }
    for left, right in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        overlap = patient_sets[left] & patient_sets[right]
        if overlap:
            raise RuntimeError(
                f"Patient overlap between {left} and {right}: "
                f"{len(overlap)} IDs"
            )

    # Retain the source synthetic features for this initial split audit.
    # Regenerate rho-specific features in a separate, explicit experiment step.
    OUTPUT_DIR.mkdir(parents=True)

    for name, part in split_frames.items():
        part[KEEP_COLUMNS].to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        print(
            f"{name}: rows={len(part)}, "
            f"normal={(part['Pneumonia'] == 0).sum()}, "
            f"pneumonia={(part['Pneumonia'] == 1).sum()}, "
            f"known pneumonia patients="
            f"{part.loc[part['Pneumonia'] == 1, 'Patient ID'].nunique()}"
        )

    print("\nPairwise known pneumonia patient overlap:")
    print("train vs val:", len(patient_sets["train"] & patient_sets["val"]))
    print("train vs test:", len(patient_sets["train"] & patient_sets["test"]))
    print("val vs test:", len(patient_sets["val"] & patient_sets["test"]))

    print("\nTotal rows:", sum(len(part) for part in split_frames.values()))
    print("Output:", OUTPUT_DIR)
    print(
        "Limitation: NORMAL image patient identities are unavailable; "
        "their true patient-level separation cannot be verified."
    )


if __name__ == "__main__":
    main()
