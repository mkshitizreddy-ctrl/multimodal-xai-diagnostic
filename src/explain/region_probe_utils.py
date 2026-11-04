"""Torch-free helpers for the region probe (see region_probe.py).

Asks which parts of the image are enough for the frozen model: the thorax (dilated lung mask), everything
outside it, the outer margin alone, or a central box of matched area. Images are [C, H, W] arrays in the
dataset's normalised space (0.0 is mid-gray).
"""

import numpy as np
import pandas as pd

from src.explain.shortcut_audit_utils import auroc, keep_region, paired_bootstrap_auroc

CONDITIONS = [
    "original",
    "thorax_only",
    "outside_thorax_only",
    "outside_thorax_no_margin",
    "margin_only_10",
    "center_only_60",
    "margin_only_20",
    "center_only_80",
]


def _inner_box(shape: tuple[int, int], frac: float) -> np.ndarray:
    """Boolean mask of the central block left after removing `frac` of each side."""
    if not 0.0 < frac < 0.5:
        raise ValueError("frac must be in (0, 0.5)")
    height, width = shape
    by, bx = int(round(height * frac)), int(round(width * frac))
    box = np.zeros(shape, dtype=bool)
    box[by : height - by, bx : width - bx] = True
    return box


def margin_keep_mask(shape: tuple[int, int], frac: float) -> np.ndarray:
    """The outer frame of width `frac` of the image on every side."""
    return ~_inner_box(shape, frac)


def center_keep_mask(shape: tuple[int, int], side_frac: float) -> np.ndarray:
    """A centred box whose side is `side_frac` of the image side (area side_frac squared)."""
    if not 0.0 < side_frac < 1.0:
        raise ValueError("side_frac must be in (0, 1)")
    return _inner_box(shape, (1.0 - side_frac) / 2.0)


def region_images(image: np.ndarray, thorax_mask: np.ndarray) -> tuple[dict, dict]:
    """Return ({condition: image}, {condition: kept area fraction}) for one image.

    thorax_mask is the (dilated) lung mask [H, W]. `margin_only_10` keeps the outer 10% frame (area 0.36)
    and `center_only_60` a central box of the same area; `margin_only_20` (0.64) pairs with `center_only_80`.
    """
    thorax = np.asarray(thorax_mask) > 0.5
    shape = thorax.shape
    keeps = {
        "original": np.ones(shape, dtype=bool),
        "thorax_only": thorax,
        "outside_thorax_only": ~thorax,
        "outside_thorax_no_margin": ~thorax & _inner_box(shape, 0.10),
        "margin_only_10": margin_keep_mask(shape, 0.10),
        "center_only_60": center_keep_mask(shape, 0.60),
        "margin_only_20": margin_keep_mask(shape, 0.20),
        "center_only_80": center_keep_mask(shape, 0.80),
    }
    images = {name: keep_region(image, keep) for name, keep in keeps.items()}
    return images, {name: float(keep.mean()) for name, keep in keeps.items()}


def summarize_regions(df: pd.DataFrame, seed: int, replicates: int) -> pd.DataFrame:
    """AUROC with CI and the paired change versus the original, per condition, plus threshold-0.5 rates."""
    y = df["label"].to_numpy() > 0.5
    ref = df["logit_original"].to_numpy()
    rows = []
    for condition in CONDITIONS:
        alt = df[f"logit_{condition}"].to_numpy()
        boot = paired_bootstrap_auroc(y, ref, alt, seed, replicates)
        predicted = alt > 0
        rows.append(
            {
                "condition": condition,
                "mean_kept_area": float(df[f"area_{condition}"].mean()),
                "auroc": auroc(y, alt),
                "auroc_ci_low": boot["auroc_alt_ci"][0],
                "auroc_ci_high": boot["auroc_alt_ci"][1],
                "delta_vs_original": boot["delta"],
                "delta_ci_low": boot["delta_ci"][0],
                "delta_ci_high": boot["delta_ci"][1],
                "sensitivity": float(predicted[y].mean()),
                "specificity": float((~predicted[~y]).mean()),
            }
        )
    return pd.DataFrame(rows)


def summary_markdown(summary: pd.DataFrame, info: dict) -> str:
    lines = [
        "# Region probe: which parts of the image are enough?",
        "",
        "Frozen vision checkpoint on the test split. Only the named region is kept; the rest is filled with",
        f"mid-gray. The thorax is the lung mask grown by {info['mask_dilation_px']:g} px.",
        "`margin_only_*` keeps an outer frame; `center_only_*` keeps a centred box of the same area.",
        "Intervals are image-level bootstrap and ignore patient clustering. The gray fill",
        "pushes outputs toward Pneumonia, so sensitivity and specificity are not comparable across rows;",
        "compare AUROC.",
        "",
        "| Condition | Kept area | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['condition']} | {r['mean_kept_area']:.2f} | {r['auroc']:.4f} "
            f"[{r['auroc_ci_low']:.4f}, {r['auroc_ci_high']:.4f}] | {r['delta_vs_original']:+.4f} "
            f"[{r['delta_ci_low']:+.4f}, {r['delta_ci_high']:+.4f}] | "
            f"{r['sensitivity']:.3f} | {r['specificity']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)
