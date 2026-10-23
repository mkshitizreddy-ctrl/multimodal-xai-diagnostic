
"""Generate reproducible synthetic-vitals variants on patient-disjoint splits.

Input:
    data/processed_patient_disjoint/{train,val,test}.csv

Output:
    data/processed_patient_disjoint_rho_v2_{rho}/

All output directories must be new. Existing results are never overwritten.
The clinical variables are synthetic, not real patient measurements.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SOURCE_DIR = Path("data/processed_patient_disjoint")
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
SPLIT_SEEDS = {"train": 101, "val": 202, "test": 303}

TEMP = ((98.3, 0.5), (100.4, 1.3), (96.0, 105.0))
SPO2 = ((98.0, 1.0), (93.5, 2.5), (80.0, 100.0))

KEEP_COLUMNS = [
    "Image Index",
    "Pneumonia",
    "Patient ID",
    "Patient Age",
    "Patient Gender",
    "Temperature",
    "SpO2",
]


def make_vital(y, noise, spec, rho):
    (healthy_mean, shared_sd), (pneumonia_mean, _pneumonia_sd), (lo, hi) = spec
    mean = healthy_mean + rho * (y == 1) * (pneumonia_mean - healthy_mean)
    return np.clip(mean + shared_sd * noise, lo, hi).round(1)


def main():
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"Missing input directory: {SOURCE_DIR}")

    source = {}
    for split in SPLIT_SEEDS:
        path = SOURCE_DIR / f"{split}.csv"
        if not path.is_file():
            raise SystemExit(f"Missing input file: {path}")
        df = pd.read_csv(path)

        missing = set(KEEP_COLUMNS) - set(df.columns)
        if missing:
            raise SystemExit(f"{path} is missing columns: {sorted(missing)}")

        source[split] = df[KEEP_COLUMNS].copy()

    # Generate fixed noise arrays once per split so the same row has the
    # same noise draw across rho levels.
    noises = {}
    for split, seed in SPLIT_SEEDS.items():
        rng = np.random.default_rng(seed)
        n = len(source[split])
        noises[split] = (
            rng.standard_normal(n),
            rng.standard_normal(n),
        )

    # Check all output paths before creating anything.
    output_dirs = {
        rho: Path(f"data/processed_patient_disjoint_rho_v2_{rho:g}")
        for rho in RHOS
    }
    existing = [str(p) for p in output_dirs.values() if p.exists()]
    if existing:
        raise SystemExit(
            "Refusing to overwrite existing output directories:\n"
            + "\n".join(existing)
        )

    for rho, out_dir in output_dirs.items():
        out_dir.mkdir(parents=True)
        generated = {}

        for split, original in source.items():
            df = original.copy()
            y = df["Pneumonia"].to_numpy(dtype=int)
            temp_noise, spo2_noise = noises[split]

            df["Temperature"] = make_vital(y, temp_noise, TEMP, rho)
            df["SpO2"] = make_vital(y, spo2_noise, SPO2, rho)

            df.to_csv(out_dir / f"{split}.csv", index=False)
            generated[split] = df

        # Tabular-only diagnostic: fit on train, evaluate on held-out test.
        cols = ["Temperature", "SpO2"]
        scaler = StandardScaler().fit(generated["train"][cols])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(
            scaler.transform(generated["train"][cols]),
            generated["train"]["Pneumonia"],
        )
        probabilities = clf.predict_proba(
            scaler.transform(generated["test"][cols])
        )[:, 1]

        auc = roc_auc_score(generated["test"]["Pneumonia"], probabilities)
        print(
            f"rho={rho:<5g} | rows={sum(len(generated[s]) for s in generated)} "
            f"| tabular-only test AUROC={auc:.4f} | output={out_dir}"
        )

    print("\nCompleted. All variants use the patient-disjoint partitions.")
    print(
        "Reminder: these vitals are synthetic; rho=0 removes the intended "
        "class-dependent mean shift, not every possible finite-sample association."
    )


if __name__ == "__main__":
    main()
