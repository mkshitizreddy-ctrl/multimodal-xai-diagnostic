"""Aggregate resize-control predictions across seeds into one table.

Reads ``docs/resize_control/<mode>_s<seed>.csv`` (written by resize_mode_eval.py) and writes
``resize_control_summary.md`` and ``.csv``: test AUROC per mode (mean and SD over seeds, with each
seed) and AUROC inside five aspect-ratio bins (mean over seeds). Descriptive only; three seeds give a
rough spread, not an interval.

Usage:
    python src/explain/resize_mode_summary.py
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explain.resize_mode_eval import aspect_quintile_auroc
from src.explain.shortcut_audit_utils import auroc

FILE_PATTERN = re.compile(r"^(?P<mode>[a-z_]+)_s(?P<seed>\d+)\.csv$")


def summarize(frames: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    """One row per (mode, seed): overall AUROC and the AUROC in each aspect-ratio bin."""
    rows = []
    for (mode, seed), df in sorted(frames.items()):
        record = {"mode": mode, "seed": seed, "n": len(df), "auroc": auroc(df["label"], df["logit"])}
        strata = aspect_quintile_auroc(df["label"], df["logit"], df["aspect"])
        for _, r in strata.iterrows():
            record[f"bin{int(r['bin'])}"] = r["auroc"]
        rows.append(record)
    return pd.DataFrame(rows)


def markdown(per_seed: pd.DataFrame) -> str:
    bins = [c for c in per_seed.columns if c.startswith("bin")]
    lines = [
        "# Resize control: test AUROC by resize mode",
        "",
        "Mean ± SD over seeds (individual seeds in the last column). Bins are equal-count aspect-ratio",
        "quintiles of the original image, bin 1 the squarest. A bin with one class gives no value.",
        "",
        "| Mode | Seeds | AUROC | " + " | ".join(b.replace("bin", "bin ") for b in bins) + " | Per seed |",
        "|---|---|---|" + "---|" * len(bins) + "---|",
    ]
    for mode, g in per_seed.groupby("mode"):
        mean = f"{g['auroc'].mean():.4f}"
        if len(g) > 1:
            mean += f" ± {g['auroc'].std(ddof=1):.4f}"
        cells = [f"{g[b].mean():.4f}" for b in bins]
        seeds = ", ".join(f"{int(s)}: {a:.4f}" for s, a in zip(g["seed"], g["auroc"]))
        lines.append(f"| {mode} | {len(g)} | {mean} | " + " | ".join(cells) + f" | {seeds} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Aggregate resize-control results.")
    parser.add_argument("--input-dir", default="docs/resize_control")
    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    frames = {}
    for path in sorted(in_dir.glob("*_s*.csv")):
        m = FILE_PATTERN.match(path.name)
        if m:
            frames[(m["mode"], int(m["seed"]))] = pd.read_csv(path)
    if not frames:
        raise SystemExit(f"No <mode>_s<seed>.csv files in {in_dir}")
    per_seed = summarize(frames)
    per_seed.to_csv(in_dir / "resize_control_summary.csv", index=False)
    text = markdown(per_seed)
    (in_dir / "resize_control_summary.md").write_text(text, encoding="utf-8")
    print(text)
    assert not np.isnan(per_seed["auroc"]).any()


if __name__ == "__main__":
    main()
