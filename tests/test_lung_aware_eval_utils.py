import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.lung_aware_eval_utils import aggregate_seeds, summary_markdown  # noqa: E402


def _table(original, thorax):
    return pd.DataFrame(
        {
            "condition": ["original", "thorax_only"],
            "auroc": [original, thorax],
            "mean_kept_area": [1.0, 0.45],
        }
    )


def test_aggregate_mean_sd_and_markdown():
    summaries = {
        "baseline": {1: _table(0.99, 0.97), 2: _table(0.98, 0.95), 3: _table(0.97, 0.93)},
        "thorax_only": {1: _table(0.96, 0.99)},
    }
    agg = aggregate_seeds(summaries)
    row = agg[(agg.group == "baseline") & (agg.condition == "thorax_only")].iloc[0]
    assert row["auroc_mean"] == pytest.approx(0.95) and row["auroc_sd"] == pytest.approx(0.02)
    assert row["auroc_min"] == 0.93 and row["n_seeds"] == 3
    single = agg[(agg.group == "thorax_only") & (agg.condition == "original")].iloc[0]
    assert np.isnan(single["auroc_sd"])
    text = summary_markdown(agg, ["original", "thorax_only", "margin_only_10"])
    assert "0.9500 ± 0.0200 (0.930-0.970)" in text and "n/a" in text


def test_missing_condition_for_a_seed_raises():
    broken = {"g": {1: _table(0.9, 0.9), 2: _table(0.9, 0.9).iloc[:1]}}
    with pytest.raises(ValueError):
        aggregate_seeds(broken)
