"""Writes data/processed_rho{X}/{train,val,test}.csv with synthetic vitals whose
class separation is scaled by rho (0 = no signal, 1 = original mean gap).
Noise draws are identical across rho, so only signal strength changes.
Also prints tabular-only test AUROC per rho."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
SPLITS = {"train": 101, "val": 202, "test": 303}  # fixed noise seed per split
# (healthy mean, healthy sd) and (pneumonia mean, pneumonia sd), as in prepare_pneumonia_dataset.py
TEMP = ((98.3, 0.5), (100.4, 1.3), (96.0, 105.0))
SPO2 = ((98.0, 1.0), (93.5, 2.5), (80.0, 100.0))


def make_vital(y, z, spec, rho):
    (m0, s0), (m1, s1), (lo, hi) = spec
    mean = np.where(y == 1, m0 + rho * (m1 - m0), m0)
    sd = np.where(y == 1, s1, s0)
    return np.clip(mean + sd * z, lo, hi).round(1)


def main():
    src = {s: pd.read_csv(f"data/processed/{s}.csv") for s in SPLITS}
    noise = {}
    for s, seed in SPLITS.items():
        rng = np.random.default_rng(seed)
        n = len(src[s])
        noise[s] = (rng.standard_normal(n), rng.standard_normal(n))

    for rho in RHOS:
        out = Path(f"data/processed_rho{rho:g}")
        out.mkdir(parents=True, exist_ok=True)
        frames = {}
        for s, df in src.items():
            df = df.copy()
            y = df["Pneumonia"].to_numpy()
            df["Temperature"] = make_vital(y, noise[s][0], TEMP, rho)
            df["SpO2"] = make_vital(y, noise[s][1], SPO2, rho)
            df.to_csv(out / f"{s}.csv", index=False)
            frames[s] = df
        cols = ["Temperature", "SpO2"]
        sc = StandardScaler().fit(frames["train"][cols])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(frames["train"][cols]), frames["train"]["Pneumonia"])
        p = clf.predict_proba(sc.transform(frames["test"][cols]))[:, 1]
        print(f"rho={rho:<5g} tabular-only test AUROC = {roc_auc_score(frames['test']['Pneumonia'], p):.4f}")


if __name__ == "__main__":
    main()