
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Check duplicate image contents across dataset splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed_patient_disjoint_dedup_rho_v1_0"),
        help="Directory containing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/raw/images"),
        help="Directory containing the source images.",
    )
    args = parser.parse_args()

    hash_groups = defaultdict(list)
    missing = []

    for split in ("train", "val", "test"):
        csv_path = args.data_dir / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            name = row["Image Index"]
            path = args.image_dir / name

            if not path.is_file():
                missing.append((split, name))
                continue

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_groups[digest].append(
                {
                    "split": split,
                    "image": name,
                    "label": int(row["Pneumonia"]),
                    "patient_id": row["Patient ID"],
                }
            )

    duplicates = {
        digest: records
        for digest, records in hash_groups.items()
        if len(records) > 1
    }

    print("Dataset directory:", args.data_dir)
    print("Total images checked:", sum(map(len, hash_groups.values())))
    print("Missing images:", len(missing))
    print("Unique image contents:", len(hash_groups))
    print("Duplicate content groups:", len(duplicates))

    for i, records in enumerate(duplicates.values(), start=1):
        print(f"\nDuplicate group {i}:")
        for record in records:
            print(
                f"  split={record['split']}, "
                f"label={record['label']}, "
                f"image={record['image']}"
            )

    if missing:
        print("\nMissing files:")
        for split, name in missing:
            print(f"  split={split}, image={name}")


if __name__ == "__main__":
    main()
