"""Aggregate region-probe results across seeds and training variants (baseline vs lung-aware)."""

import numpy as np
import pandas as pd

KEY_CONDITIONS = [
    "original",
    "thorax_only",
    "outside_thorax_only",
    "outside_thorax_no_margin",
    "margin_only_10",
    "center_only_60",
]


def aggregate_seeds(summaries: dict[str, dict[int, pd.DataFrame]]) -> pd.DataFrame:
    """Mean and SD (ddof=1) of AUROC per (group, condition) across seeds.

    `summaries[group][seed]` is the region_probe_summary table of one checkpoint. With a single seed the
    SD is NaN. Groups must share the same set of conditions.
    """
    rows = []
    for group, by_seed in summaries.items():
        if not by_seed:
            raise ValueError(f"no seeds for group {group!r}")
        frames = []
        for seed, table in sorted(by_seed.items()):
            part = table[["condition", "auroc", "mean_kept_area"]].copy()
            part["seed"] = seed
            frames.append(part)
        long = pd.concat(frames, ignore_index=True)
        for condition, sub in long.groupby("condition", sort=False):
            if sub["seed"].nunique() != len(by_seed):
                raise ValueError(f"{group!r}/{condition!r} is missing for some seeds")
            values = sub["auroc"].to_numpy(dtype=float)
            rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "n_seeds": len(values),
                    "auroc_mean": float(values.mean()),
                    "auroc_sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                    "auroc_min": float(values.min()),
                    "auroc_max": float(values.max()),
                    "mean_kept_area": float(sub["mean_kept_area"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summary_markdown(aggregate: pd.DataFrame, conditions: list[str] | None = None) -> str:
    """One row per condition, one column per group, cells 'mean ± SD (min-max)'."""
    groups = list(dict.fromkeys(aggregate["group"]))
    wanted = conditions or list(dict.fromkeys(aggregate["condition"]))
    lines = [
        "# Lung-aware training: region probe across seeds",
        "",
        "Test-split AUROC of each trained variant when only the named region is kept (rest filled with",
        "mid-gray). Cells are mean ± SD over seeds (min to max). With three seeds the SD is a rough guide,",
        "not an interval. Gray fill pushes outputs toward Pneumonia, so compare AUROC only.",
        "",
        "| Condition | " + " | ".join(groups) + " |",
        "|---|" + "---|" * len(groups),
    ]
    for condition in wanted:
        cells = []
        for group in groups:
            hit = aggregate[(aggregate["group"] == group) & (aggregate["condition"] == condition)]
            if hit.empty:
                cells.append("n/a")
                continue
            r = hit.iloc[0]
            sd = "n/a" if np.isnan(r["auroc_sd"]) else f"{r['auroc_sd']:.4f}"
            cells.append(f"{r['auroc_mean']:.4f} ± {sd} ({r['auroc_min']:.3f}-{r['auroc_max']:.3f})")
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)
