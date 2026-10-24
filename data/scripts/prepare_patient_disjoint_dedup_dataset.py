
from pathlib import Path
import hashlib

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SOURCE_DIR = Path("data/processed")
IMAGE_DIR = Path("data/raw/images")
OUTPUT_DIR = Path("data/processed_patient_disjoint_dedup_v1")

SPLITS = ("train", "val", "test")
SEED_TEST = 42
SEED_VAL = 43

REQUIRED_COLUMNS = {
    "Image Index",
    "Pneumonia",
    "Patient ID",
}


# ---------------------------------------------------------
# Safety checks
# ---------------------------------------------------------

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT_DIR}\n"
        "Nothing was overwritten. Choose a new output directory "
        "after inspecting the existing files."
    )

if not SOURCE_DIR.is_dir():
    raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")


# ---------------------------------------------------------
# Read all existing processed records
# ---------------------------------------------------------

frames = []

for split in SPLITS:
    csv_path = SOURCE_DIR / f"{split}.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing source CSV: {csv_path}")

    frame = pd.read_csv(csv_path)
    frame["SourceProcessedSplit"] = split

    missing_columns = REQUIRED_COLUMNS - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing columns: {sorted(missing_columns)}"
        )

    frames.append(frame)

df = pd.concat(frames, ignore_index=True)

if df["Image Index"].isna().any():
    raise ValueError("Image Index contains missing values.")

if df["Pneumonia"].isna().any():
    raise ValueError("Pneumonia contains missing values.")

df["Pneumonia"] = df["Pneumonia"].astype(int)

if not set(df["Pneumonia"].unique()).issubset({0, 1}):
    raise ValueError("Pneumonia labels must be 0 or 1.")

if df["Image Index"].duplicated().any():
    duplicated_names = df.loc[
        df["Image Index"].duplicated(keep=False), "Image Index"
    ].tolist()
    raise ValueError(
        f"Duplicate image filenames in source CSVs: {duplicated_names[:10]}"
    )


# ---------------------------------------------------------
# Hash every image; fail if any image is missing
# ---------------------------------------------------------

hashes = []
missing_images = []

for name in df["Image Index"]:
    image_path = IMAGE_DIR / str(name)

    if not image_path.is_file():
        missing_images.append(str(image_path))
        hashes.append(None)
        continue

    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    hashes.append(digest)

if missing_images:
    raise FileNotFoundError(
        f"{len(missing_images)} image files are missing. "
        f"Examples: {missing_images[:10]}"
    )

df["ImageHash"] = hashes

# Identical image contents must have consistent labels.
label_counts = df.groupby("ImageHash")["Pneumonia"].nunique()
conflicting_hashes = label_counts[label_counts > 1]

if not conflicting_hashes.empty:
    raise ValueError(
        f"{len(conflicting_hashes)} identical-image hashes have "
        "conflicting labels. Resolve these before splitting."
    )


# ---------------------------------------------------------
# Create connected groups:
# - known pneumonia patient IDs stay together
# - exact duplicates stay together
# - normal patient IDs are not trusted as real patient IDs
# ---------------------------------------------------------

parent = list(range(len(df)))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


first_by_hash = {}
first_by_pneumonia_patient = {}

for i, row in df.iterrows():
    digest = row["ImageHash"]

    if digest in first_by_hash:
        union(i, first_by_hash[digest])
    else:
        first_by_hash[digest] = i

    # Only pneumonia IDs are treated as known patient groups.
    if int(row["Pneumonia"]) == 1:
        patient_id = str(row["Patient ID"]).strip()

        if not patient_id or patient_id.lower() == "nan":
            raise ValueError(
                f"Missing pneumonia patient ID for {row['Image Index']}"
            )

        if patient_id in first_by_pneumonia_patient:
            union(i, first_by_pneumonia_patient[patient_id])
        else:
            first_by_pneumonia_patient[patient_id] = i


# Resolve all connected components after all unions.
df["GroupID"] = [find(i) for i in range(len(df))]


# ---------------------------------------------------------
# Deduplicate globally, retaining one representative per hash
# ---------------------------------------------------------

original_rows = len(df)
duplicate_rows_removed = int(df["ImageHash"].duplicated().sum())

# The representative row retains its original metadata.
# GroupID links it to all patient/duplicate relationships.
unique_df = df.drop_duplicates(
    subset="ImageHash", keep="first"
).copy()

if unique_df["ImageHash"].duplicated().any():
    raise RuntimeError("Image deduplication failed.")

# A component must not contain contradictory labels.
group_label_counts = unique_df.groupby("GroupID")["Pneumonia"].nunique()

if (group_label_counts > 1).any():
    raise ValueError(
        "A connected patient/duplicate group contains conflicting labels."
    )


# ---------------------------------------------------------
# Split unique images into approximately 64/16/20
# train / validation / test, using grouped stratification
# ---------------------------------------------------------

y = unique_df["Pneumonia"].to_numpy()
groups = unique_df["GroupID"].to_numpy()
indices = pd.RangeIndex(len(unique_df))

outer = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=SEED_TEST
)

development_idx, test_idx = next(
    outer.split(indices, y, groups)
)

development = unique_df.iloc[development_idx].copy()
test = unique_df.iloc[test_idx].copy()

inner = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=SEED_VAL
)

dev_y = development["Pneumonia"].to_numpy()
dev_groups = development["GroupID"].to_numpy()
dev_indices = pd.RangeIndex(len(development))

train_rel_idx, val_rel_idx = next(
    inner.split(dev_indices, dev_y, dev_groups)
)

train = development.iloc[train_rel_idx].copy()
val = development.iloc[val_rel_idx].copy()


# ---------------------------------------------------------
# Verify before writing anything
# ---------------------------------------------------------

split_frames = {
    "train": train,
    "val": val,
    "test": test,
}

for split_name, frame in split_frames.items():
    if frame["ImageHash"].duplicated().any():
        raise RuntimeError(
            f"Duplicate image hashes remain in {split_name}."
        )

for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    hashes_a = set(split_frames[a]["ImageHash"])
    hashes_b = set(split_frames[b]["ImageHash"])

    if hashes_a & hashes_b:
        raise RuntimeError(f"Image-content leakage: {a} vs {b}")

    groups_a = set(split_frames[a]["GroupID"])
    groups_b = set(split_frames[b]["GroupID"])

    if groups_a & groups_b:
        raise RuntimeError(f"Patient/duplicate-group leakage: {a} vs {b}")

total_assigned = sum(len(frame) for frame in split_frames.values())

if total_assigned != len(unique_df):
    raise RuntimeError(
        f"Expected {len(unique_df)} unique images, assigned {total_assigned}."
    )


# ---------------------------------------------------------
# Save to a new directory only after checks pass
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

for split_name, frame in split_frames.items():
    # Internal audit columns are not part of the model input CSVs.
    output_frame = frame.drop(
        columns=["ImageHash", "GroupID", "SourceProcessedSplit"]
    )
    output_frame.to_csv(OUTPUT_DIR / f"{split_name}.csv", index=False)


# Save a report describing deduplication and split composition.
report = {
    "original_rows": original_rows,
    "unique_image_contents": len(unique_df),
    "duplicate_rows_removed": duplicate_rows_removed,
    "train_rows": len(train),
    "val_rows": len(val),
    "test_rows": len(test),
    "known_pneumonia_patients_train": train.loc[
        train["Pneumonia"] == 1, "Patient ID"
    ].nunique(),
    "known_pneumonia_patients_val": val.loc[
        val["Pneumonia"] == 1, "Patient ID"
    ].nunique(),
    "known_pneumonia_patients_test": test.loc[
        test["Pneumonia"] == 1, "Patient ID"
    ].nunique(),
}

pd.Series(report, name="value").to_csv(
    OUTPUT_DIR / "split_report.csv", header=True
)

print("\nNew deduplicated split created successfully.")
print("Output:", OUTPUT_DIR)
print("Original rows:", original_rows)
print("Unique image contents:", len(unique_df))
print("Duplicate records removed:", duplicate_rows_removed)

for split_name, frame in split_frames.items():
    print(
        f"{split_name}: rows={len(frame)}, "
        f"normal={(frame['Pneumonia'] == 0).sum()}, "
        f"pneumonia={(frame['Pneumonia'] == 1).sum()}"
    )

print("\nPairwise group and hash overlap checks: passed")
print("Important: normal-patient identity cannot be verified.")
