"""Tests for the resize-control aggregation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.resize_mode_summary import markdown, summarize  # noqa: E402


def _frame(seed, n=100):
    rng = np.random.default_rng(seed)
    label = (rng.random(n) > 0.4).astype(float)
    return pd.DataFrame(
        {"label": label, "logit": label * 2 - 1 + rng.normal(0, 0.5, n), "aspect": rng.uniform(0.9, 2.0, n)}
    )


def test_summarize_and_markdown_cover_each_mode_and_seed():
    frames = {("pad", 1): _frame(1), ("pad", 2): _frame(2), ("stretch", 1): _frame(3)}
    table = summarize(frames)
    assert list(table["mode"]) == ["pad", "pad", "stretch"]
    assert table["auroc"].between(0.5, 1.0).all()
    assert {"bin1", "bin5"} <= set(table.columns)
    text = markdown(table)
    assert "| pad | 2 |" in text and "| stretch | 1 |" in text
