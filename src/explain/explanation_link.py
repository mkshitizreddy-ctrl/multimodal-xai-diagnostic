"""Does the Grad-CAM lung-energy fraction track the model's dependence on lung content?

Within Pneumonia (the Normal class has too few nonzero maps), relates the per-image fraction of Grad-CAM
mass inside the segmented lungs to the frozen model's logits under the audit interventions. The lung
fraction depends strongly on how large the lung mask is, so every correlation is a Spearman partial
correlation controlling for mask area. A matched contrast with the displaced-mask controls shows how much
of the link is specific to the true lung mask. Image-level bootstrap CIs ignore patient clustering, and
the masks have visible defects on dense lungs, so this is descriptive.

Usage:
    python src/explain/explanation_link.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def _residual(y, x) -> np.ndarray:
    ry, rx = rankdata(y), rankdata(x)
    slope, intercept = np.polyfit(rx, ry, 1)
    return ry - (slope * rx + intercept)


def partial_spearman(x, y, control) -> float:
    """Spearman correlation of x and y after removing the rank-linear effect of `control`."""
    x, y, control = (np.asarray(v, dtype=float) for v in (x, y, control))
    return float(spearmanr(_residual(x, control), _residual(y, control))[0])


def bootstrap_partial(x, ys: dict, control, seed: int, replicates: int) -> dict:
    """Point estimate and percentile CI of partial Spearman for each named y, plus its resample draws."""
    x, control = np.asarray(x, dtype=float), np.asarray(control, dtype=float)
    ys = {k: np.asarray(v, dtype=float) for k, v in ys.items()}
    n = len(x)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ys}
    for _ in range(replicates):
        idx = rng.integers(0, n, n)
        for k, v in ys.items():
            draws[k].append(partial_spearman(x[idx], v[idx], control[idx]))
    return {
        k: (partial_spearman(x, v, control), *np.percentile(draws[k], [2.5, 97.5]), np.asarray(draws[k]))
        for k, v in ys.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM lung fraction versus audit logits.")
    parser.add_argument("--localization", default="docs/lung_localization_testset_rho0.csv")
    parser.add_argument(
        "--measurements",
        default="docs/shortcut_audit_patient_disjoint_dedup_rho_v1_vision/shortcut_audit_measurements.csv",
    )
    parser.add_argument("--output-dir", default="docs/explanation_link_patient_disjoint_dedup_rho_v1_vision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    loc = pd.read_csv(args.localization)
    audit = pd.read_csv(args.measurements)
    data = audit.merge(loc[["index", "lung_energy_fraction"]], on="index")
    data = data[data["lung_energy_fraction"].notna() & (data["label"] > 0.5)].copy()
    data["contrast_true"] = data["logit_lung_only"] - data["logit_background_only"]
    data["contrast_displaced"] = data["logit_displaced_lung_only"] - data["logit_displaced_background_only"]
    data["enrichment"] = data["lung_energy_fraction"] / data["mask_area"]

    ys = {
        "original": data["logit_original"],
        "lung_only": data["logit_lung_only"],
        "background_only": data["logit_background_only"],
        "displaced_lung_only": data["logit_displaced_lung_only"],
        "displaced_background_only": data["logit_displaced_background_only"],
        "contrast_true (lung_only - background_only)": data["contrast_true"],
        "contrast_displaced (matched control)": data["contrast_displaced"],
    }
    result = bootstrap_partial(
        data["lung_energy_fraction"], ys, data["mask_area"], args.seed, args.bootstrap_replicates
    )
    true_key, disp_key = list(ys)[-2:]
    diff = result[true_key][3] - result[disp_key][3]
    diff_ci = np.percentile(diff, [2.5, 97.5])

    rows = [
        {"logit": k, "partial_rho": v[0], "ci_low": v[1], "ci_high": v[2]} for k, v in result.items()
    ]
    rows.append(
        {
            "logit": "difference (true contrast - displaced contrast)",
            "partial_rho": result[true_key][0] - result[disp_key][0],
            "ci_low": diff_ci[0],
            "ci_high": diff_ci[1],
        }
    )
    table = pd.DataFrame(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "explanation_link_summary.csv", index=False)

    lines = [
        "# Grad-CAM lung fraction versus audit logits",
        "",
        f"Pneumonia images with a nonzero Grad-CAM map: n={len(data)}. Partial Spearman correlation of the",
        "lung-energy fraction with each logit, controlling for lung-mask area; 95% image-level bootstrap CI.",
        "",
        "| Logit | Partial rho [95% CI] |",
        "|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['logit']} | {r['partial_rho']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] |")
    lines += [
        "",
        f"Enrichment (lung fraction / mask area): median {data['enrichment'].median():.2f}, "
        f"{(data['enrichment'] > 1).mean():.1%} of maps above 1.",
        "",
    ]
    markdown = "\n".join(lines)
    (out_dir / "explanation_link_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
