import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.outline_control_utils import CONDITIONS, outline_images  # noqa: E402


def test_conditions_keep_the_right_pixels():
    image = np.ones((3, 16, 16), dtype=np.float32)
    thorax = np.zeros((16, 16), dtype=bool)
    thorax[:, :4] = True
    box = np.zeros((16, 16), dtype=bool)
    box[4:12, 4:12] = True
    images, areas = outline_images(image, thorax, box)
    assert list(images) == CONDITIONS
    assert areas == pytest.approx(
        {"original": 1.0, "thorax_only": 0.25, "fixed_box": 0.25, "outside_fixed_box": 0.75}
    )
    assert (images["fixed_box"][:, 4:12, 4:12] == 1).all() and images["fixed_box"][:, 0, 0].max() == 0
    assert images["outside_fixed_box"][:, 8, 8].max() == 0 and images["outside_fixed_box"][:, 0, 0].min() == 1
    assert (images["original"] == image).all()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        outline_images(np.zeros((3, 16, 16)), np.zeros((8, 8)), np.zeros((8, 8)))
