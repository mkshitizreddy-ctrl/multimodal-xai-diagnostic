"""Shared fixtures."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def segmentation_weights():
    """Skip (rather than fail) tests that need the torchxrayvision lung segmenter when its weights
    cannot be downloaded, for example when the release host returns a gateway timeout."""
    from src.explain.lung_segmentation import _load_segmentation_model

    try:
        _load_segmentation_model()
    except Exception as exc:  # network or corrupt-download problems, not a code defect
        pytest.skip(f"lung segmentation weights unavailable: {type(exc).__name__}: {str(exc)[:100]}")
