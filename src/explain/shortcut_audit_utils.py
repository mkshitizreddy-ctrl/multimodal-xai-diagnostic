"""Torch-free helpers for the shortcut audit (see shortcut_audit.py).

Everything here works on NumPy arrays so it can be unit-tested without a GPU.
Images are [C, H, W] float arrays in the dataset's normalized space, where 0.0
is mid-gray, matching the fill used by evaluate_gradcam_faithfulness.py.
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import rankdata

GRAY = 0.0


def keep_region(image: np.ndarray, keep_mask: np.ndarray, fill: float = GRAY) -> np.ndarray:
    """Keep pixels where keep_mask is set; replace everything else with `fill`.

    image: [C, H, W]; keep_mask: [H, W] (values > 0.5 count as set).
    """
    image = np.asarray(image, dtype=np.float32)
    keep = (np.asarray(keep_mask) > 0.5).astype(np.float32)[None, :, :]
    return image * keep + np.float32(fill) * (1.0 - keep)


def remove_region(image: np.ndarray, remove_mask: np.ndarray, fill: float = GRAY) -> np.ndarray:
    """Replace pixels where remove_mask is set with `fill`; keep the rest."""
    inverse = 1.0 - (np.asarray(remove_mask) > 0.5).astype(np.float32)
    return keep_region(image, inverse, fill)


def blank_border(image: np.ndarray, frac: float, fill: float = GRAY) -> np.ndarray:
    """Fill the outer `frac` margin on every side, keeping the central block."""
    if not 0.0 <= frac < 0.5:
        raise ValueError("frac must be in [0, 0.5)")
    height, width = image.shape[-2:]
    by, bx = int(round(height * frac)), int(round(width * frac))
    keep = np.zeros((height, width), dtype=np.float32)
    keep[by : height - by, bx : width - bx] = 1.0
    return keep_region(image, keep, fill)


def mask_area_fraction(mask: np.ndarray) -> float:
    return float((np.asarray(mask) > 0.5).mean())


def dilate_mask(mask: np.ndarray, pixels: float) -> np.ndarray:
    """Grow a binary mask by `pixels` (Euclidean distance, in mask pixels).

    Used to cover under-segmented lung edges: any pixel within `pixels` of the mask joins it.
    `pixels` <= 0 returns the mask unchanged.
    """
    base = np.asarray(mask) > 0.5
    if pixels <= 0:
        return base.astype(np.float32)
    distance = ndimage.distance_transform_edt(~base)
    return (distance <= pixels).astype(np.float32)


def displace_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    min_shift: float = 0.10,
    max_shift: float = 0.30,
    max_iou: float = 0.5,
    tries: int = 50,
) -> tuple[np.ndarray, float]:
    """Return a copy of `mask` circularly shifted by a random offset, plus its IoU
    with the original. The shape and area are preserved exactly, so this is an
    area- and shape-matched control for the lung mask. Offsets of at least
    `min_shift` of the image size are drawn until the IoU with the original is
    at most `max_iou` (or the least-overlapping of `tries` draws is returned).
    """
    base = np.asarray(mask) > 0.5
    height, width = base.shape
    best_mask, best_iou = base, 1.0
    for _ in range(tries):
        dy = int(round(rng.uniform(min_shift, max_shift) * height)) * int(rng.choice([-1, 1]))
        dx = int(round(rng.uniform(min_shift, max_shift) * width)) * int(rng.choice([-1, 1]))
        shifted = np.roll(base, (dy, dx), axis=(0, 1))
        union = np.logical_or(base, shifted).sum()
        iou = float(np.logical_and(base, shifted).sum() / union) if union else 0.0
        if iou < best_iou:
            best_mask, best_iou = shifted, iou
        if iou <= max_iou:
            break
    return best_mask.astype(np.float32), best_iou


def auroc(y_true, scores) -> float:
    """Rank-based AUROC with average ranks for ties; NaN if one class is missing."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _interval(values) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def paired_bootstrap_auroc(y_true, ref_scores, alt_scores, seed: int, replicates: int = 2000) -> dict:
    """AUROC of two score sets on the same images, with percentile bootstrap CIs
    for each and for the paired difference (alt - ref). Images are resampled;
    a fresh generator from `seed` is used so every comparison reuses identical
    resamples. Resampling is at image level, so intervals ignore any clustering
    of images within a patient.
    """
    y = np.asarray(y_true).astype(bool)
    ref = np.asarray(ref_scores, dtype=np.float64)
    alt = np.asarray(alt_scores, dtype=np.float64)
    n = len(y)
    rng = np.random.default_rng(seed)

    ref_boot, alt_boot, delta_boot = [], [], []
    for _ in range(replicates):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.all() or (~yb).all():
            continue
        a_ref, a_alt = auroc(yb, ref[idx]), auroc(yb, alt[idx])
        ref_boot.append(a_ref)
        alt_boot.append(a_alt)
        delta_boot.append(a_alt - a_ref)

    point_ref, point_alt = auroc(y, ref), auroc(y, alt)
    return {
        "auroc_ref": point_ref,
        "auroc_ref_ci": _interval(ref_boot),
        "auroc_alt": point_alt,
        "auroc_alt_ci": _interval(alt_boot),
        "delta": point_alt - point_ref,
        "delta_ci": _interval(delta_boot),
    }


def threshold_metrics(y_true, probs, threshold: float = 0.5) -> dict:
    """Accuracy, sensitivity, specificity at `threshold`, plus mean probability per class."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(probs, dtype=np.float64)
    pred = p >= threshold
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return {
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
        "sensitivity": float(pred[y].mean()) if n_pos else float("nan"),
        "specificity": float((~pred[~y]).mean()) if n_neg else float("nan"),
        "mean_prob_normal": float(p[~y].mean()) if n_neg else float("nan"),
        "mean_prob_pneumonia": float(p[y].mean()) if n_pos else float("nan"),
    }


CONDITIONS = [
    "original",
    "lung_only",
    "background_only",
    "displaced_lung_only",
    "displaced_background_only",
    "border_blank",
]


def audit_image(image, mask, infer, rng, repeats: int, border_frac: float) -> dict:
    """Run every audit condition for one image and return per-condition logits.

    infer: callable mapping a float32 batch [B, C, H, W] to B logits.
    The displaced-mask controls average their logits over `repeats` random shifts.
    """
    image = np.asarray(image, dtype=np.float32)
    batch = [
        image,
        keep_region(image, mask),
        remove_region(image, mask),
        blank_border(image, border_frac),
    ]
    keep_controls, remove_controls, ious = [], [], []
    for _ in range(repeats):
        shifted, iou = displace_mask(mask, rng)
        keep_controls.append(keep_region(image, shifted))
        remove_controls.append(remove_region(image, shifted))
        ious.append(iou)

    logits = np.asarray(infer(np.stack(batch + keep_controls + remove_controls)), dtype=np.float64)
    if logits.shape != (len(batch) + 2 * repeats,):
        raise ValueError(f"infer returned shape {logits.shape}; expected ({len(batch) + 2 * repeats},)")
    return {
        "logit_original": float(logits[0]),
        "logit_lung_only": float(logits[1]),
        "logit_background_only": float(logits[2]),
        "logit_border_blank": float(logits[3]),
        "logit_displaced_lung_only": float(logits[4 : 4 + repeats].mean()),
        "logit_displaced_background_only": float(logits[4 + repeats :].mean()),
        "displaced_iou_mean": float(np.mean(ious)) if ious else float("nan"),
    }


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def summarize(df: pd.DataFrame, seed: int, replicates: int, min_area: float, max_area: float) -> pd.DataFrame:
    """One row per (subset, condition): AUROC with CI, paired change vs the original
    images, and threshold-0.5 metrics. Subsets: all images, and images whose lung
    mask covers between min_area and max_area of the frame (a crude mask sanity filter).
    """
    subsets = {
        "all": np.ones(len(df), dtype=bool),
        "valid_mask": ((df["mask_area"] >= min_area) & (df["mask_area"] <= max_area)).to_numpy(),
    }
    rows = []
    for subset_name, keep in subsets.items():
        part = df[keep]
        if len(part) == 0:
            continue
        y = part["label"].to_numpy() > 0.5
        ref = part["logit_original"].to_numpy()
        for condition in CONDITIONS:
            alt = part[f"logit_{condition}"].to_numpy()
            boot = paired_bootstrap_auroc(y, ref, alt, seed=seed, replicates=replicates)
            metrics = threshold_metrics(y, _sigmoid(alt))
            rows.append(
                {
                    "subset": subset_name,
                    "condition": condition,
                    "n": len(part),
                    "n_pneumonia": int(y.sum()),
                    "auroc": boot["auroc_alt"],
                    "auroc_ci_low": boot["auroc_alt_ci"][0],
                    "auroc_ci_high": boot["auroc_alt_ci"][1],
                    "delta_auroc_vs_original": boot["delta"],
                    "delta_ci_low": boot["delta_ci"][0],
                    "delta_ci_high": boot["delta_ci"][1],
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summary_markdown(summary: pd.DataFrame, info: dict) -> str:
    lines = [
        "# Shortcut audit: frozen-model interventions",
        "",
        "Descriptive model-sensitivity experiment on the frozen checkpoint. It does not establish clinical",
        "validity, and a drop under an intervention is not by itself proof of (or against) shortcut "
        "learning.",
        "",
        f"- Checkpoint SHA-256: `{info.get('checkpoint_sha256', 'n/a')}`",
        f"- Test images: {info.get('n_images')} ({info.get('n_pneumonia')} Pneumonia); "
        f"lung-mask area mean {info.get('mask_area_mean', float('nan')):.3f}, "
        f"{info.get('n_valid_mask')} masks within "
        f"[{info.get('min_mask_area')}, {info.get('max_mask_area')}]",
        f"- Displaced-mask controls: {info.get('control_repeats')} random shifts per image, "
        f"mean logit; mean IoU with the true mask {info.get('displaced_iou_mean', float('nan')):.3f}",
        f"- Bootstrap: {info.get('bootstrap_replicates')} image-level resamples "
        "(ignores within-patient clustering)",
        "",
    ]
    for subset_name in summary["subset"].unique():
        part = summary[summary["subset"] == subset_name]
        n_all, n_pos = int(part["n"].iloc[0]), int(part["n_pneumonia"].iloc[0])
        lines += [
            f"## Subset: {subset_name} (n={n_all}, Pneumonia={n_pos})",
            "",
            "| Condition | AUROC [95% CI] | Change vs original [95% CI] | Sens | Spec | Mean P (Normal) |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in part.iterrows():
            auroc_cell = f"{r['auroc']:.4f} [{r['auroc_ci_low']:.4f}, {r['auroc_ci_high']:.4f}]"
            delta_cell = (
                f"{r['delta_auroc_vs_original']:+.4f} [{r['delta_ci_low']:+.4f}, {r['delta_ci_high']:+.4f}]"
            )
            lines.append(
                f"| {r['condition']} | {auroc_cell} | {delta_cell} | "
                f"{r['sensitivity']:.3f} | {r['specificity']:.3f} | {r['mean_prob_normal']:.3f} |"
            )
        lines.append("")
    lines += [
        "## How to read this",
        "",
        "- `lung_only` keeps the segmented lungs and grays everything else; `background_only` grays "
        "the lungs.",
        "- `displaced_*` apply the same operation with a shifted copy of the lung mask (same shape and "
        "area, little overlap with the lungs). They control for the artefact of graying a region of "
        "that size.",
        "- `border_blank` grays the outer margin.",
        "- Lung masks come from an automatic segmenter trained on adult radiographs; inspect "
        "`qc_lung_masks.png` before trusting them on these paediatric images.",
        "- Gray fill is out of distribution for the model, so compare each condition with its displaced "
        "control, not only with the original images.",
        "",
    ]
    return "\n".join(lines)
