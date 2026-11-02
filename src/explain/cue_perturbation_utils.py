"""Torch-free helpers for the acquisition-cue perturbation test.

Each perturbation maps a grayscale PIL image at its original resolution to another grayscale PIL image.
They are applied before the model's usual resize and normalisation, so only the perturbed cue changes.
"""

import io

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.explain.shortcut_audit_utils import auroc, paired_bootstrap_auroc

JPEG_QUALITY_MILD = 75
JPEG_QUALITY_STRONG = 25


def recompress_jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode as JPEG at a fixed quality, removing differences in the original compression."""
    buffer = io.BytesIO()
    image.convert("L").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    out = Image.open(buffer)
    out.load()
    return out.convert("L")


def equalize_histogram(image: Image.Image) -> Image.Image:
    """Global histogram equalisation, removing brightness/contrast differences between exports."""
    return ImageOps.equalize(image.convert("L"))


def pad_to_square(image: Image.Image) -> Image.Image:
    """Pad to a square with the border median grey, so the later resize does not distort the aspect ratio."""
    gray = image.convert("L")
    width, height = gray.size
    side = max(width, height)
    array = np.asarray(gray)
    m = max(1, int(round(min(width, height) * 0.02)))
    strips = [array[:m], array[-m:], array[:, :m], array[:, -m:]]
    border = np.concatenate([strip.ravel() for strip in strips])
    canvas = Image.new("L", (side, side), int(np.median(border)))
    canvas.paste(gray, ((side - width) // 2, (side - height) // 2))
    return canvas


def down_up(image: Image.Image, factor: int = 2) -> Image.Image:
    """Downsample by `factor` and upsample back, removing fine texture and noise."""
    gray = image.convert("L")
    width, height = gray.size
    small = gray.resize((max(1, width // factor), max(1, height // factor)), Image.Resampling.BILINEAR)
    return small.resize((width, height), Image.Resampling.BILINEAR)


PERTURBATIONS = {
    "original": lambda image: image.convert("L"),
    "jpeg_q75": lambda image: recompress_jpeg(image, JPEG_QUALITY_MILD),
    "jpeg_q25": lambda image: recompress_jpeg(image, JPEG_QUALITY_STRONG),
    "equalize": equalize_histogram,
    "pad_square": pad_to_square,
    "down_up_2x": lambda image: down_up(image, 2),
}


def summarize_perturbations(
    measurements: pd.DataFrame, discordant, seed: int, replicates: int
) -> pd.DataFrame:
    """AUROC, paired change versus `original`, error rates and logit shift per condition and subset.

    measurements: columns `label` and `logit_<condition>` for every condition in PERTURBATIONS.
    discordant: boolean array, True where the file-size baseline is wrong.
    """
    discordant = np.asarray(discordant, dtype=bool)
    y_all = measurements["label"].to_numpy() > 0.5
    subsets = {
        "all": np.ones(len(y_all), dtype=bool),
        "baseline_right": ~discordant,
        "baseline_wrong": discordant,
    }
    rows = []
    for subset, keep in subsets.items():
        if not keep.any():
            continue  # e.g. a smoke test where the baseline makes no errors
        y = y_all[keep]
        ref = measurements["logit_original"].to_numpy()[keep]
        for condition in PERTURBATIONS:
            alt = measurements[f"logit_{condition}"].to_numpy()[keep]
            boot = paired_bootstrap_auroc(y, ref, alt, seed, replicates)
            predicted = alt > 0
            rows.append(
                {
                    "subset": subset,
                    "condition": condition,
                    "n": int(keep.sum()),
                    "n_pneumonia": int(y.sum()),
                    "auroc": auroc(y, alt),
                    "auroc_ci_low": boot["auroc_alt_ci"][0],
                    "auroc_ci_high": boot["auroc_alt_ci"][1],
                    "delta_vs_original": boot["delta"],
                    "delta_ci_low": boot["delta_ci"][0],
                    "delta_ci_high": boot["delta_ci"][1],
                    "error_rate": float((predicted != y).mean()),
                    "false_negative_rate": float((~predicted[y]).mean()) if y.any() else float("nan"),
                    "false_positive_rate": float(predicted[~y].mean()) if (~y).any() else float("nan"),
                    "mean_logit_shift": float((alt - ref).mean()),
                }
            )
    return pd.DataFrame(rows)


def summary_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "# Acquisition-cue perturbation test",
        "",
        "Frozen vision checkpoint on the test split. Each perturbation is applied to the original",
        "image before the usual resize and normalisation. `baseline_wrong` is the set of images where",
        "a file-size-only classifier errs. Intervals are image-level bootstrap and ignore patient",
        "clustering. Error rate uses logit > 0.",
        "",
    ]
    for subset, group in summary.groupby("subset", sort=False):
        first = group.iloc[0]
        lines += [
            f"## {subset} (n={first['n']}, {first['n_pneumonia']} Pneumonia)",
            "",
            "| Condition | AUROC [95% CI] | Change vs original [95% CI] | Error rate | FNR | FPR "
            "| Mean logit shift |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in group.iterrows():
            lines.append(
                f"| {r['condition']} | {r['auroc']:.4f} "
                f"[{r['auroc_ci_low']:.4f}, {r['auroc_ci_high']:.4f}] | "
                f"{r['delta_vs_original']:+.4f} [{r['delta_ci_low']:+.4f}, {r['delta_ci_high']:+.4f}] | "
                f"{r['error_rate']:.3f} | {r['false_negative_rate']:.3f} | {r['false_positive_rate']:.3f} | "
                f"{r['mean_logit_shift']:+.2f} |"
            )
        lines.append("")
    return "\n".join(lines)
