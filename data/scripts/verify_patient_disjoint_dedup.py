
from pathlib import Path
import hashlib
import pandas as pd

base = Path("data/processed_patient_disjoint_dedup_v1")
image_dir = Path("data/raw/images")
splits = ("train", "val", "test")

frames = {}
hashes_by_split = {}
patient_ids_by_split = {}
total_rows = 0
errors = []

for split in splits:
    csv_path = base / f"{split}.csv"
    df = pd.read_csv(csv_path)
    frames[split] = df
    total_rows += len(df)

    hashes = set()
    pneumonia_patients = set()

    for _, row in df.iterrows():
        path = image_dir / str(row["Image Index"])

        if not path.is_file():
            errors.append(f"Missing image: {path}")
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        if digest in hashes:
            errors.append(
                f"Duplicate image content within {split}: {row['Image Index']}"
            )

        hashes.add(digest)

        if int(row["Pneumonia"]) == 1:
            pneumonia_patients.add(str(row["Patient ID"]))

    hashes_by_split[split] = hashes
    patient_ids_by_split[split] = pneumonia_patients

    print(
        f"{split}: rows={len(df)}, "
        f"normal={(df['Pneumonia'] == 0).sum()}, "
        f"pneumonia={(df['Pneumonia'] == 1).sum()}, "
        f"unique hashes={len(hashes)}"
    )

print("\nTotal rows:", total_rows)

for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    hash_overlap = hashes_by_split[a] & hashes_by_split[b]
    patient_overlap = (
        patient_ids_by_split[a] & patient_ids_by_split[b]
    )

    print(
        f"{a} vs {b}: "
        f"duplicate hashes={len(hash_overlap)}, "
        f"known pneumonia patient overlap={len(patient_overlap)}"
    )

    if hash_overlap:
        errors.append(f"Hash overlap between {a} and {b}")

    if patient_overlap:
        errors.append(f"Patient overlap between {a} and {b}")

if total_rows != 5824:
    errors.append(f"Expected 5824 rows, found {total_rows}")

if errors:
    print("\nVERIFICATION FAILED:")
    for error in errors:
        print("-", error)
    raise SystemExit(1)

print("\nVERIFICATION PASSED")
print("All image files exist; no exact duplicate content or")
print("known pneumonia patient IDs cross splits.")
print("Normal-patient identity remains unverified.")
