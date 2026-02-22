"""
Shared pytest fixtures for maestro-ml.

Loads the PS-S6E2 pipeline.yaml to find data paths. Tests that require
data files skip gracefully if the CSVs are not present (they are
gitignored and must be downloaded separately).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
PS_S6E2_PIPELINE = ROOT_DIR / "competitions" / "ps-s6e2" / "pipeline.yaml"


def _load_pipeline_yaml() -> dict:
    """Load the PS-S6E2 pipeline YAML if it exists."""
    if not PS_S6E2_PIPELINE.exists():
        return {}
    with open(PS_S6E2_PIPELINE) as f:
        return yaml.safe_load(f)


_PIPELINE = _load_pipeline_yaml()


def _data_available() -> bool:
    """Check if the PS-S6E2 CSV files exist."""
    if not _PIPELINE:
        return False
    train_path = ROOT_DIR / _PIPELINE.get("data", {}).get("train_path", "")
    test_path = ROOT_DIR / _PIPELINE.get("data", {}).get("test_path", "")
    return train_path.exists() and test_path.exists()


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_data = pytest.mark.skipif(
    not _data_available(),
    reason="PS-S6E2 data files not found (download from Kaggle first)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_config() -> dict:
    """Return the raw pipeline YAML as a dict."""
    return _PIPELINE


@pytest.fixture
def pipeline_yaml_path() -> Path:
    """Return the path to the PS-S6E2 pipeline.yaml."""
    return PS_S6E2_PIPELINE


@pytest.fixture
def train_path() -> Path:
    """Return the resolved path to train.csv."""
    return ROOT_DIR / _PIPELINE["data"]["train_path"]


@pytest.fixture
def test_path() -> Path:
    """Return the resolved path to test.csv."""
    return ROOT_DIR / _PIPELINE["data"]["test_path"]


@pytest.fixture
def target_column() -> str:
    """Return the target column name."""
    return _PIPELINE["data"]["target_column"]


@pytest.fixture
def id_column() -> str:
    """Return the ID column name."""
    return _PIPELINE["data"]["id_column"]
