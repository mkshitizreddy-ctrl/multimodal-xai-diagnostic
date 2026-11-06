import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.silhouette_baseline import GROUPS, mask_features  # noqa: E402


def _masks():
    masks = np.zeros((3, 32, 32), dtype=bool)
    masks[0, 8:24, 4:20] = True  # 16 x 16 box, left of centre
    masks[1, 0:32, 0:16] = True  # left half
    return masks  # masks[2] stays empty


def test_geometry_of_known_boxes():
    f = mask_features(_masks())
    assert f["area"].tolist() == pytest.approx([256 / 1024, 0.5, 0.0])
    assert (f.loc[0, "x0"], f.loc[0, "x1"], f.loc[0, "y0"], f.loc[0, "y1"]) == (4 / 32, 20 / 32, 0.25, 0.75)
    assert f.loc[0, "width"] == pytest.approx(0.5) and f.loc[0, "height"] == pytest.approx(0.5)
    assert f.loc[1, "left_area"] == pytest.approx(1.0) and f.loc[1, "right_area"] == 0.0
    assert f.loc[1, "cx"] == pytest.approx(0.25) and f.loc[2, "width"] == 0.0  # empty mask is finite
    assert np.isfinite(f.to_numpy()).all()


def test_every_group_column_exists_and_profiles_sum_sensibly():
    f = mask_features(_masks())
    for group, columns in GROUPS.items():
        assert set(columns) <= set(f.columns), group
    assert f.loc[1, [f"col{i}" for i in range(16)]].tolist() == pytest.approx([1.0] * 8 + [0.0] * 8)
    thumb = f.loc[1, GROUPS["thumbnail_16x16"]].to_numpy().reshape(16, 16)
    assert thumb[:, :8].min() == 1.0 and thumb[:, 8:].max() == 0.0
