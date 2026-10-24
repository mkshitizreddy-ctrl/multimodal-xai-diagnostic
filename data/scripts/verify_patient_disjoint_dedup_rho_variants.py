from pathlib import Path

import pandas as pd

RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
SPLITS = ["train", "val", "test"]
COLUMNS = ["Image Index", "Pneumonia", "Patient ID"]

BASE_DIR = Path("data/processed_patient_disjoint_dedup_rho_v1_0")


def main():
    reference = {
        split: pd.read_csv(BASE_DIR / f"{split}.csv")[COLUMNS]
        for split in SPLITS
    }

    print("Reference rho=0 row counts:")
    for split, df in reference.items():
        print(f"  {split}: {len(df)}")

    all_passed = True

    for rho in RHOS[1:]:
        directory = Path(
            f"data/processed_patient_disjoint_dedup_rho_v1_{rho:g}"
        )

        print(f"\nrho={rho:g}")

        for split in SPLITS:
            path = directory / f"{split}.csv"

            if not path.is_file():
                print(f"  {split}: MISSING FILE")
                all_passed = False
                continue

            df = pd.read_csv(path)

            same_rows = (
                len(df) == len(reference[split])
                and all(column in df.columns for column in COLUMNS)
                and df[COLUMNS].equals(reference[split])
            )

            print(
                f"  {split}: rows={len(df)}, "
                f"same image/label/patient rows={same_rows}"
            )

            if not same_rows:
                all_passed = False

    print(
        "\nPARTITION VERIFICATION:",
        "PASSED" if all_passed else "FAILED",
    )

def verify_vital_signs():
    print("\nSynthetic vital-sign verification")
    all_passed = True

    for split in SPLITS:
        frames = {}

        for rho in RHOS:
            path = Path(
                f"data/processed_patient_disjoint_dedup_rho_v1_{rho:g}"
            ) / f"{split}.csv"

            if not path.is_file():
                print(f"FAIL: Missing {path}")
                all_passed = False
                continue

            frames[rho] = pd.read_csv(path)

        if len(frames) != len(RHOS):
            continue

        reference = frames[0.0]

        for rho, df in frames.items():
            unchanged = (
                df["Image Index"].equals(reference["Image Index"])
                and df["Pneumonia"].equals(reference["Pneumonia"])
                and df["Patient ID"].equals(reference["Patient ID"])
                and df["Patient Age"].equals(reference["Patient Age"])
                and df["Patient Gender"].equals(reference["Patient Gender"])
            )

            print(
                f"  {split}, rho={rho:g}: "
                f"non-vital columns unchanged={unchanged}"
            )

            if not unchanged:
                all_passed = False

        for column in ["Temperature", "SpO2"]:
            for rho in RHOS:
                values = frames[rho][column]

                valid = values.notna().all() and values.between(
                    {"Temperature": 96.0, "SpO2": 80.0}[column],
                    {"Temperature": 105.0, "SpO2": 100.0}[column],
                ).all()

                if not valid:
                    print(
                        f"  FAIL: {column} has missing or out-of-range "
                        f"values at rho={rho:g} in {split}"
                    )
                    all_passed = False

    print(
        "\nVITAL-SIGN VERIFICATION:",
        "PASSED" if all_passed else "FAILED",
    )
verify_vital_signs()
if __name__ == "__main__":
    main()