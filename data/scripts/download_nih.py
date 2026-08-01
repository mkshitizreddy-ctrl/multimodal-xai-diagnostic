"""
Download the NIH Chest X-ray14 dataset into data/raw/.

The dataset is ~45GB, so this script uses the Kaggle mirror (much easier
to pull programmatically than NIH's original Box links) via `kagglehub`.

Setup (one-time):
    1. Create a Kaggle account: https://www.kaggle.com
    2. Go to Account -> API -> "Create New Token" to download kaggle.json
    3. Place kaggle.json at ~/.kaggle/kaggle.json (Linux/Mac) or
       C:\\Users\\<you>\\.kaggle\\kaggle.json (Windows)

Usage:
    python data/scripts/download_nih.py
    python data/scripts/download_nih.py --sample 2000   # small subset for dev
"""

import argparse
import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")


def download_full_dataset() -> Path:
    """Download the full NIH Chest X-ray14 dataset via kagglehub."""
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit(
            "kagglehub is required for this script.\n"
            "Install it with: pip install kagglehub"
        ) from e

    print("Downloading NIH Chest X-ray14 from Kaggle (this can take a while)...")
    path = kagglehub.dataset_download("nih-chest-xrays/data")
    print(f"Dataset downloaded to: {path}")
    return Path(path)


def stage_into_raw_dir(source_dir: Path) -> None:
    """Copy/symlink the downloaded dataset into data/raw/ with the expected layout."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    metadata_src = source_dir / "Data_Entry_2017.csv"
    if metadata_src.exists():
        shutil.copy(metadata_src, RAW_DIR / "Data_Entry_2017.csv")
        print(f"Copied metadata CSV to {RAW_DIR / 'Data_Entry_2017.csv'}")
    else:
        print(f"WARNING: could not find Data_Entry_2017.csv under {source_dir}")

    images_dest = RAW_DIR / "images"
    images_dest.mkdir(exist_ok=True)

    # The Kaggle mirror ships images across several images_XXX/images/ folders.
    image_dirs = sorted(source_dir.glob("images_*/images"))
    if not image_dirs:
        print(
            f"WARNING: no images_*/images folders found under {source_dir}. "
            "Check the download layout and adjust this script if Kaggle's "
            "structure has changed."
        )
        return

    for img_dir in image_dirs:
        for img_file in img_dir.glob("*.png"):
            dest = images_dest / img_file.name
            if not dest.exists():
                # Symlink instead of copy to save disk space; falls back to copy
                # on filesystems that don't support symlinks (e.g. some Windows setups).
                try:
                    dest.symlink_to(img_file.resolve())
                except OSError:
                    shutil.copy(img_file, dest)
    print(f"Staged images into {images_dest}")


def make_dev_sample(n: int) -> None:
    """Create a small random subset for fast local development/testing."""
    import pandas as pd

    metadata_path = RAW_DIR / "Data_Entry_2017.csv"
    if not metadata_path.exists():
        raise SystemExit(
            f"{metadata_path} not found. Run the full download first, "
            "then re-run with --sample."
        )

    df = pd.read_csv(metadata_path)
    sample_df = df.sample(n=min(n, len(df)), random_state=42)
    sample_path = RAW_DIR / "Data_Entry_2017_sample.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"Wrote {len(sample_df)}-row dev sample to {sample_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, also write a small metadata sample of N rows for fast local dev.",
    )
    args = parser.parse_args()

    source_dir = download_full_dataset()
    stage_into_raw_dir(source_dir)

    if args.sample:
        make_dev_sample(args.sample)

    print("Done. Raw data is staged under data/raw/.")


if __name__ == "__main__":
    main()
