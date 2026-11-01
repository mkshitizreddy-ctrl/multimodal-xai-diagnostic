"""Does the CNN's behaviour track the acquisition-only baseline?

Uses the cached features from nonanatomical_baseline.py and the CNN logits from shortcut_audit.py.
Two descriptive checks, both on the test split:

1. Within-class Spearman correlation between the CNN logit and the baseline score.
2. CNN AUROC and error rates on images where the file-size baseline is right versus wrong.

This is correlational. A CNN whose errors coincide with the baseline's errors may be using the same cue,
or both may fail on atypical images for an unrelated reason. Image-level bootstrap CIs ignore clustering
of images within a patient.

Usage:
    python src/explain/baseline_discordance.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explain.nonanatomical_baseline import GROUPS, evaluate_group
from src.explain.shortcut_audit_utils import auroc, paired_bootstrap_auroc


def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 (chosen on validation data only)."""
    best_t, best_j = 0.5, -np.inf
    for t in np.linspace(0.05, 0.95, 91):
        pred = score >= t
        j = pred[y].mean() + (~pred[~y]).mean() - 1
        if j > best_j:
            best_t, best_j = float(t), float(j)
    return best_t


def main():
    parser = argparse.ArgumentParser(description="CNN vs acquisition-baseline discordance analysis.")
    parser.add_argument("--features-dir", default="docs/nonanatomical_baseline_patient_disjoint_dedup_rho_v1")
    parser.add_argument(
        "--cnn-measurements",
        default="docs/shortcut_audit_patient_disjoint_dedup_rho_v1_vision/shortcut_audit_measurements.csv",
    )
    parser.add_argument("--output-dir", default=None, help="defaults to --features-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    out_dir = Path(args.output_dir) if args.output_dir else features_dir
    train, val, test = (
        pd.read_csv(features_dir / f"image_features_{name}.csv") for name in ("train", "val", "test")
    )
    cnn = pd.read_csv(args.cnn_measurements)[["image", "logit_original"]]
    test = test.merge(cnn, on="image", how="left")
    if test["logit_original"].isna().any():
        raise ValueError("Some test images are missing from the CNN measurements file")
    y = test["label"].to_numpy() > 0.5
    logit = test["logit_original"].to_numpy()

    lines = ["# CNN versus acquisition-baseline discordance", ""]
    lines += [
        "Within-class Spearman correlation between the CNN logit and each baseline's score (test split).",
        "",
        "| Baseline | Pneumonia (n) | rho | Normal (n) | rho |",
        "|---|---|---|---|---|",
    ]
    for group in ("file_size_only", "size_meta+intensity"):
        scores = evaluate_group(train, val, test, GROUPS[group])["test_scores"]
        cells = []
        for cls in (True, False):
            k = y == cls
            rho = spearmanr(logit[k], scores[k])[0]
            cells.append(f"{int(k.sum())} | {rho:+.3f}")
        lines.append(f"| {group} | {cells[0]} | {cells[1]} |")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(train[["file_kb"]], train["label"])
    threshold = youden_threshold(
        val["label"].to_numpy() > 0.5, model.predict_proba(val[["file_kb"]])[:, 1]
    )
    size_score = model.predict_proba(test[["file_kb"]])[:, 1]
    size_pred = size_score >= threshold
    size_wrong = size_pred != y
    cnn_pred = logit > 0
    cnn_wrong = cnn_pred != y

    lines += [
        "",
        f"File-size baseline at the validation Youden threshold ({threshold:.2f}): "
        f"accuracy {1 - size_wrong.mean():.3f}, {int(size_wrong.sum())} errors "
        f"(FN {int((~size_pred & y).sum())}, FP {int((size_pred & ~y).sum())}).",
        "",
        "| Subset | n | Pneumonia | Normal | CNN AUROC [95% CI] | CNN error rate |",
        "|---|---|---|---|---|---|",
    ]
    for name, k in (("baseline right", ~size_wrong), ("baseline wrong", size_wrong)):
        yy = y[k]
        boot = paired_bootstrap_auroc(yy, logit[k], logit[k], args.seed, args.bootstrap_replicates)
        low, high = boot["auroc_ref_ci"]
        lines.append(
            f"| {name} | {int(k.sum())} | {int(yy.sum())} | {int((~yy).sum())} | "
            f"{auroc(yy, logit[k]):.4f} [{low:.4f}, {high:.4f}] | {cnn_wrong[k].mean():.3f} |"
        )
    both = int((cnn_wrong & size_wrong).sum())
    lines += [
        "",
        f"CNN errors at logit > 0: {int(cnn_wrong.sum())} (FN {int((~cnn_pred & y).sum())}, "
        f"FP {int((cnn_pred & ~y).sum())}); {both} of them are also file-size baseline errors.",
        f"Median file size of CNN false positives: {test['file_kb'][cnn_pred & ~y].median():.1f} KB "
        f"(all Normal: {test['file_kb'][~y].median():.1f} KB).",
        f"Median file size of CNN false negatives: {test['file_kb'][~cnn_pred & y].median():.1f} KB "
        f"(all Pneumonia: {test['file_kb'][y].median():.1f} KB).",
        "",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_discordance.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
