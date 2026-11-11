"""Patient-clustered bootstrap intervals for the frozen-model audit and the cue-perturbation test.

The intervals elsewhere in this repository resample images, which treats the 849 Pneumonia test images
as independent although they come from 335 patients (mean 2.5 images, maximum 19 per patient). This
script resamples patients instead and reports the cluster interval beside the image-level one.

Pneumonia images are grouped by their patient ID. Every Normal image carries its own unique ID, because
Normal patient identity cannot be recovered in this dataset, so Normal images are treated as independent;
any clustering among them is not accounted for. Descriptive only.

Usage:
    python src/explain/cluster_bootstrap.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explain.shortcut_audit_utils import auroc

AUDIT_CONDITIONS = [
    "original",
    "lung_only",
    "background_only",
    "displaced_lung_only",
    "displaced_background_only",
    "border_blank",
]
CUE_CONDITIONS = ["original", "jpeg_q75", "jpeg_q25", "equalize", "pad_square", "down_up_2x"]


def _interval(values) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_indices(groups, seed: int, replicates: int):
    """Yield index arrays that resample whole groups with replacement."""
    groups = np.asarray(groups)
    _, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        picked = rng.integers(0, len(members), len(members))
        yield np.concatenate([members[g] for g in picked])


def cluster_bootstrap(y_true, scores: dict, contrasts: dict, groups, seed: int = 42, replicates: int = 2000):
    """AUROC per score set and per contrast (name -> (alt, ref), alt minus ref), with percentile intervals.

    ``groups`` assigns each image to a resampling unit; pass distinct values per image for an image-level
    bootstrap. The same resamples are used for every quantity.
    """
    y = np.asarray(y_true).astype(bool)
    scores = {k: np.asarray(v, dtype=np.float64) for k, v in scores.items()}
    boots = {k: [] for k in scores}
    deltas = {k: [] for k in contrasts}
    for idx in bootstrap_indices(groups, seed, replicates):
        yb = y[idx]
        if yb.all() or (~yb).all():
            continue
        values = {k: auroc(yb, s[idx]) for k, s in scores.items()}
        for k, v in values.items():
            boots[k].append(v)
        for k, (alt, ref) in contrasts.items():
            deltas[k].append(values[alt] - values[ref])
    rows = []
    for k, s in scores.items():
        lo, hi = _interval(boots[k])
        rows.append({"quantity": k, "kind": "auroc", "estimate": auroc(y, s), "ci_low": lo, "ci_high": hi})
    for k, (alt, ref) in contrasts.items():
        lo, hi = _interval(deltas[k])
        rows.append(
            {
                "quantity": k,
                "kind": "difference",
                "estimate": auroc(y, scores[alt]) - auroc(y, scores[ref]),
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows)


def compare_levels(y, scores, contrasts, patient_ids, seed, replicates) -> pd.DataFrame:
    """Image-level and patient-cluster intervals side by side."""
    image = cluster_bootstrap(y, scores, contrasts, np.arange(len(y)), seed, replicates)
    patient = cluster_bootstrap(y, scores, contrasts, patient_ids, seed, replicates)
    out = image.rename(columns={"ci_low": "image_ci_low", "ci_high": "image_ci_high"})
    out["patient_ci_low"], out["patient_ci_high"] = patient["ci_low"], patient["ci_high"]
    out["width_ratio"] = (out["patient_ci_high"] - out["patient_ci_low"]) / (
        out["image_ci_high"] - out["image_ci_low"]
    )
    return out


def build_tables(test_csv: str, audit_csv: str, cue_csv: str, seed: int, replicates: int) -> pd.DataFrame:
    test = pd.read_csv(test_csv)
    parts = []
    for name, path, conditions in [("audit", audit_csv, AUDIT_CONDITIONS), ("cue", cue_csv, CUE_CONDITIONS)]:
        df = pd.read_csv(path)
        ids = test[["Image Index", "Patient ID"]].rename(columns={"Image Index": "image"})
        merged = df.merge(ids, on="image", how="left")
        if merged["Patient ID"].isna().any() or len(merged) != len(df):
            raise ValueError(f"{path}: images do not match {test_csv}")
        scores = {c: merged[f"logit_{c}"] for c in conditions}
        contrasts = {f"{c} - original": (c, "original") for c in conditions[1:]}
        if name == "audit":
            contrasts["lung_only - displaced_lung_only"] = ("lung_only", "displaced_lung_only")
            contrasts["background_only - displaced_background_only"] = (
                "background_only",
                "displaced_background_only",
            )
        table = compare_levels(merged["label"], scores, contrasts, merged["Patient ID"], seed, replicates)
        table.insert(0, "experiment", name)
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def markdown(table: pd.DataFrame, n_images: int, n_patients: int, replicates: int) -> str:
    lines = [
        "# Patient-clustered bootstrap intervals",
        "",
        f"Frozen vision checkpoint, test split ({n_images} images, {n_patients} patient or image units).",
        "Pneumonia images are resampled by patient; Normal images are treated as independent because",
        f"their patient identity is unknown. {replicates} resamples; percentile 95% intervals.",
        "`Width ratio` is the patient-cluster interval width divided by the image-level width.",
        "",
        "| Experiment | Quantity | Estimate | Image-level 95% CI | Patient-cluster 95% CI | Width ratio |",
        "|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        f = "{:+.4f}" if r.kind == "difference" else "{:.4f}"
        lines.append(
            f"| {r.experiment} | {r.quantity} | {f.format(r.estimate)} "
            f"| [{f.format(r.image_ci_low)}, {f.format(r.image_ci_high)}] "
            f"| [{f.format(r.patient_ci_low)}, {f.format(r.patient_ci_high)}] | {r.width_ratio:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Patient-clustered bootstrap for the audit results.")
    parser.add_argument("--test-csv", default="data/processed_patient_disjoint_dedup_rho_v1_0/test.csv")
    parser.add_argument(
        "--audit-csv",
        default="docs/shortcut_audit_patient_disjoint_dedup_rho_v1_vision/shortcut_audit_measurements.csv",
    )
    parser.add_argument(
        "--cue-csv",
        default="docs/cue_perturbation_patient_disjoint_dedup_rho_v1_vision/cue_perturbation_measurements.csv",
    )
    parser.add_argument("--output-dir", default="docs/cluster_bootstrap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.replicates < 100:
        parser.error("--replicates must be at least 100")

    table = build_tables(args.test_csv, args.audit_csv, args.cue_csv, args.seed, args.replicates)
    test = pd.read_csv(args.test_csv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "cluster_bootstrap_summary.csv", index=False)
    text = markdown(table, len(test), int(test["Patient ID"].nunique()), args.replicates)
    (out / "cluster_bootstrap_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
