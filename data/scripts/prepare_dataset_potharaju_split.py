"""
Reproduce the train/val/test split reported in Potharaju et al. 2025
("Enhanced X-ray image classification for pneumonia detection using deep
learning based CBAM and SE mechanisms") as closely as their paper allows,
to test whether their split methodology - rather than their model - explains
their reported accuracy.

Potharaju et al. report 5216 train / 160 val / 480 test (5856 total - the
full Kermany pool) and do not mention patient-level grouping anywhere in
the paper, despite this dataset's known patient-linked filenames
(personXXX_... for pneumonia cases). This script deliberately reproduces
that gap: a stratified-by-class, PATIENT-BLIND random split at the image
level, matching their reported split sizes exactly.

This is a hypothesis test, not an endorsement: if this split alone (with
our own rigorously-evaluated model) produces inflated-looking metrics
relative to our patient-level split, that's evidence their split
methodology - not their model - explains their reported number.

Usage:
    python data/scripts/prepare_dataset_potharaju_split.py
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    pd.options.future.infer_string = False
except AttributeError:
    pass

from prepare_pneumonia_dataset import (
    RAW_IMAGES_DIR,
    RANDOM_SEED,
    collect_split,
    download_dataset,
    find_chest_xray_root,
    stage_images,
    synthesize_clinical_features,
)

PROCESSED_DIR = Path("data/processed_potharaju_split")

# Potharaju et al., Section 5 ("Results and discussion"): "The dataset
# comprised 5816 X-ray images from Kaggle, with 5216 images used for
# training, 160 for validation, and 480 for testing." Note 5216+160+480 =
# 5856, the FULL Kermany pool, not the 5816 stated in their abstract - an
# internal inconsistency in their own paper. We match the breakdown given
# in their Results section, not the abstract's total.
TRAIN_N = 5216
VAL_N = 160
TEST_N = 480


def main() -> None:
    source_dir = download_dataset()
    root = find_chest_xray_root(source_dir)

    # Pool ALL images together (train + val + test folders), matching
    # Potharaju et al.'s apparent approach of using the full dataset before
    # re-splitting, rather than respecting the curators' original test set.
    all_rows = collect_split(root, "train") + collect_split(root, "val") + collect_split(root, "test")
    stage_images(all_rows)
    df = pd.DataFrame(all_rows)

    assert len(df) == TRAIN_N + VAL_N + TEST_N, (
        f"Expected {TRAIN_N + VAL_N + TEST_N} total images to match Potharaju et al.'s "
        f"reported split sizes, got {len(df)}. Their reported total may not exactly match "
        f"this Kaggle mirror's current image count."
    )

    # Deliberately PATIENT-BLIND: plain stratified-by-class split at the
    # image level, matching what their paper describes (no mention of
    # patient grouping). This is the specific methodological choice being
    # tested, not an oversight.
    rest_df, test_df = train_test_split(
        df, test_size=TEST_N, random_state=RANDOM_SEED, stratify=df["Pneumonia"]
    )
    train_df, val_df = train_test_split(
        rest_df, test_size=VAL_N, random_state=RANDOM_SEED, stratify=rest_df["Pneumonia"]
    )

    for split_df in (train_df, val_df, test_df):
        synthesize_clinical_features(split_df)

    keep_cols = [
        "Image Index", "Pneumonia", "Patient ID",
        "Patient Age", "Patient Gender", "Temperature", "SpO2",
    ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df[keep_cols].to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df[keep_cols].to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df[keep_cols].to_csv(PROCESSED_DIR / "test.csv", index=False)

    # Quantify the leakage risk directly: how many test-set pneumonia
    # patients also appear in train? This is the concrete evidence for or
    # against the leakage hypothesis, not just a plausibility argument.
    train_patients = set(train_df["Patient ID"]) | set(val_df["Patient ID"])
    test_patients = set(test_df["Patient ID"])
    overlap = train_patients & test_patients
    # Patient IDs for normal-class images are per-image (no shared patient
    # identifier available in this dataset), so overlap is only meaningful
    # for pneumonia-class patient IDs.
    pneumonia_overlap = {p for p in overlap if p.startswith("pneumonia_person_")}

    print(f"train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
    print(f"Wrote CSVs to {PROCESSED_DIR}/")
    print(
        f"\nPatient-leakage check: {len(pneumonia_overlap)} pneumonia patient IDs appear "
        f"in both (train+val) and test under this patient-blind split."
    )
    if pneumonia_overlap:
        test_leaked_images = test_df[test_df["Patient ID"].isin(pneumonia_overlap)]
        print(
            f"  -> {len(test_leaked_images)} of {len(test_df)} test images belong to a "
            f"patient also seen in training. This is the leakage this experiment is "
            f"designed to detect."
        )


if __name__ == "__main__":
    main()