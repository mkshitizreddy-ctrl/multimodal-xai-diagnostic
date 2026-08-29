"""
Tests for src/explain/measure_lung_localization.py. Two tiers:
- Pure arithmetic tests for lung_energy_fraction() — fast, no model/GPU needed.
- Unit tests for is_fusion_checkpoint()'s auto-detection logic.
- One true end-to-end integration test that builds a real (tiny, untrained)
  fusion checkpoint + synthetic dataset and runs the actual script via
  subprocess, exactly as a user would invoke it.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.measure_lung_localization import lung_energy_fraction


def test_all_activation_inside_lungs_gives_fraction_one():
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[3:7, 3:7] = 1.0
    lung_mask = np.ones((10, 10), dtype=np.float32)
    assert lung_energy_fraction(heatmap, lung_mask) == 1.0


def test_all_activation_outside_lungs_gives_fraction_zero():
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[0:2, 0:2] = 1.0  # corner, outside the mask below
    lung_mask = np.zeros((10, 10), dtype=np.float32)
    lung_mask[5:10, 5:10] = 1.0
    assert lung_energy_fraction(heatmap, lung_mask) == 0.0


def test_half_and_half_gives_fraction_half():
    heatmap = np.ones((10, 10), dtype=np.float32)
    lung_mask = np.zeros((10, 10), dtype=np.float32)
    lung_mask[:, :5] = 1.0  # left half is "lung"
    assert abs(lung_energy_fraction(heatmap, lung_mask) - 0.5) < 1e-6


def test_degenerate_zero_heatmap_returns_nan():
    heatmap = np.zeros((10, 10), dtype=np.float32)
    lung_mask = np.ones((10, 10), dtype=np.float32)
    assert np.isnan(lung_energy_fraction(heatmap, lung_mask))


def test_is_fusion_checkpoint_true_when_tabular_features_present():
    from src.explain.measure_lung_localization import is_fusion_checkpoint

    fusion_checkpoint = {"model_state_dict": {}, "classes": ["Pneumonia"], "tabular_features": ["Age"]}
    assert is_fusion_checkpoint(fusion_checkpoint) is True


def test_is_fusion_checkpoint_false_for_vision_only():
    from src.explain.measure_lung_localization import is_fusion_checkpoint

    vision_checkpoint = {"model_state_dict": {}, "classes": ["Pneumonia"], "tabular_stats": {}}
    assert is_fusion_checkpoint(vision_checkpoint) is False


def test_end_to_end_fusion_checkpoint_runs_via_actual_script(tmp_path):
    """Builds a real (tiny, untrained) fusion checkpoint + synthetic
    dataset from scratch and invokes measure_lung_localization.py exactly
    as a user would from the command line — the strongest check available
    that the fusion code path (auto-detection, per-image
    FusionModelImageWrapper construction, tabular_stats reuse) actually
    works end to end, not just that its pieces work in isolation.

    Seed is pinned to one already known to avoid the ~1-in-10
    dead-gradient case that untrained/random-init networks can hit at a
    Grad-CAM target layer (see tests/test_fusion_wrapper.py for the same
    issue, diagnosed there) — this test checks the pipeline runs and
    produces valid structured output, not that any particular heatmap
    looks a certain way.
    """
    import subprocess
    import sys as _sys

    import numpy as _np
    import pandas as pd
    import torch
    import yaml
    from PIL import Image

    repo_root = Path(__file__).resolve().parents[1]

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for i in range(5):
        arr = (_np.random.rand(64, 64) * 255).astype("uint8")
        fname = f"fake_{i}.png"
        Image.fromarray(arr, mode="L").save(image_dir / fname)
        rows.append(
            {
                "Image Index": fname,
                "Pneumonia": float(i % 2),
                "Patient Age": 20 + i * 10,
                "Patient Gender": "M" if i % 2 == 0 else "F",
                "Temperature": 37.0 + i * 0.1,
                "SpO2": 95.0 + i * 0.5,
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "test.csv", index=False)

    data_config = {
        "tabular_features": ["Patient Age", "Patient Gender", "Temperature", "SpO2"]
    }
    with open(tmp_path / "data.yaml", "w") as f:
        yaml.safe_dump(data_config, f)

    train_config = {
        "data": {
            "test_csv": str(tmp_path / "test.csv"),
            "image_dir": str(image_dir),
            "image_size": 224,
        }
    }
    with open(tmp_path / "train.yaml", "w") as f:
        yaml.safe_dump(train_config, f)

    sys.path.insert(0, str(repo_root))
    from src.data.dataset import ChestXrayDataset
    from src.models.fusion import ChestXrayFusionModel

    torch.manual_seed(1)  # known non-degenerate, see docstring
    classes = ["Pneumonia"]
    tabular_features = data_config["tabular_features"]

    fitter_ds = ChestXrayDataset(
        csv_path=str(tmp_path / "test.csv"),
        image_dir=str(image_dir),
        classes=classes,
        tabular_features=tabular_features,
        image_size=224,
        train=True,
    )
    tabular_stats = fitter_ds.get_tabular_stats()

    model = ChestXrayFusionModel(
        num_classes=len(classes), num_tabular_features=len(tabular_features), pretrained=False, use_cbam=True
    )
    checkpoint_path = tmp_path / "fake_fusion.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "tabular_features": tabular_features,
            "tabular_stats": tabular_stats,
            "epoch": 1,
            "use_cbam": True,
        },
        checkpoint_path,
    )

    output_csv = tmp_path / "localization_results.csv"
    result = subprocess.run(
        [
            _sys.executable,
            str(repo_root / "src/explain/measure_lung_localization.py"),
            "--checkpoint", str(checkpoint_path),
            "--data-config", str(tmp_path / "data.yaml"),
            "--train-config", str(tmp_path / "train.yaml"),
            "--num-examples", "3",
            "--seed", "1",
            "--output-csv", str(output_csv),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"script failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "model_type = fusion" in result.stdout
    assert output_csv.exists()

    with open(output_csv) as f:
        written_rows = list(csv.DictReader(f))
    assert len(written_rows) == 3
    assert all("lung_energy_fraction" in row for row in written_rows)
