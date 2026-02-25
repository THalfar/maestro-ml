"""Tests for src/eda/profiler.py — EDA profiling and report generation."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.profiler import (
    _compute_correlations,
    _detect_column_types,
    _find_feature_clusters,
    _generate_recommendations,
    _identify_weak_features,
    format_eda_for_llm,
    run_eda,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Create a simple binary classification dataset on disk."""
    rng = np.random.default_rng(42)
    n = 200
    train = pd.DataFrame({
        "id": range(n),
        "num_a": rng.normal(0, 1, n),
        "num_b": rng.normal(5, 2, n),
        "cat_x": rng.choice(["low", "mid", "high"], n),
        "binary_flag": rng.choice([0, 1], n),
        "target": rng.choice([0, 1], n),
    })
    # Make num_a correlated with target
    train.loc[train["target"] == 1, "num_a"] += 1.5

    test = pd.DataFrame({
        "id": range(n, n + 50),
        "num_a": rng.normal(0, 1, 50),
        "num_b": rng.normal(5, 2, 50),
        "cat_x": rng.choice(["low", "mid", "high"], 50),
        "binary_flag": rng.choice([0, 1], 50),
    })

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame for unit-testing internal functions."""
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame({
        "numeric_cont": rng.normal(0, 1, n),
        "binary_col": rng.choice([0, 1], n),
        "ordinal_col": rng.choice([1, 2, 3, 4, 5], n),
        "low_card_cat": rng.choice(["A", "B", "C"], n),
        "high_card_cat": [f"val_{i % 50}" for i in range(n)],
    })


# ---------------------------------------------------------------------------
# run_eda (integration)
# ---------------------------------------------------------------------------

class TestRunEda:
    def test_report_keys(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        expected_keys = {
            "dataset_info", "target_analysis", "columns",
            "correlation_matrix", "feature_clusters",
            "weak_features", "recommendations",
        }
        assert set(report.keys()) == expected_keys

    def test_returns_dataframes(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        _, train_df, test_df = run_eda(train_path, test_path, "target")
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert len(train_df) == 200
        assert len(test_df) == 50

    def test_dataset_info(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        info = report["dataset_info"]
        assert info["train_shape"] == [200, 6]
        assert info["test_shape"] == [50, 5]
        assert info["n_features"] == 5  # 6 cols - 1 target

    def test_target_analysis(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        ta = report["target_analysis"]
        assert ta["n_unique"] == 2
        assert sum(ta["distribution"].values()) == 200

    def test_columns_exclude_target(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        assert "target" not in report["columns"]
        assert "num_a" in report["columns"]

    def test_target_correlation_populated(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        # num_a was made correlated with target
        num_a_corr = report["columns"]["num_a"]["target_correlation"]
        assert abs(num_a_corr) > 0.1

    def test_recommendations_not_empty(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        assert len(report["recommendations"]) > 0

    def test_id_col_excluded(self, binary_dataset: tuple[Path, Path]):
        """When id_col is provided, it should be excluded from columns analysis."""
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target", id_col="id")
        assert "id" not in report["columns"]
        assert report["dataset_info"]["n_features"] == 4  # 6 cols - 1 target - 1 id

    def test_id_col_excluded_from_correlations(self, binary_dataset: tuple[Path, Path]):
        """id column should not appear in target_correlations or correlation_matrix."""
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target", id_col="id")
        corr_cols = report["correlation_matrix"]["columns"]
        assert "id" not in corr_cols


# ---------------------------------------------------------------------------
# _detect_column_types
# ---------------------------------------------------------------------------

class TestDetectColumnTypes:
    def test_numeric_continuous(self, sample_df: pd.DataFrame):
        result = _detect_column_types(sample_df)
        assert result["numeric_cont"]["detected_type"] == "numeric_continuous"

    def test_binary(self, sample_df: pd.DataFrame):
        result = _detect_column_types(sample_df)
        assert result["binary_col"]["detected_type"] == "binary"

    def test_ordinal(self, sample_df: pd.DataFrame):
        result = _detect_column_types(sample_df)
        assert result["ordinal_col"]["detected_type"] == "ordinal"

    def test_low_cardinality_categorical(self, sample_df: pd.DataFrame):
        result = _detect_column_types(sample_df)
        assert result["low_card_cat"]["detected_type"] == "low_cardinality_categorical"

    def test_high_cardinality_categorical(self, sample_df: pd.DataFrame):
        result = _detect_column_types(sample_df)
        assert result["high_card_cat"]["detected_type"] == "high_cardinality_categorical"

    def test_str_dtype_correctly_classified(self):
        """String columns (object or StringDtype) should be detected as categorical."""
        df = pd.DataFrame({"col": ["A", "B", "C", "A", "B"]})
        result = _detect_column_types(df)
        assert result["col"]["detected_type"] == "low_cardinality_categorical"

    def test_missing_pct(self):
        df = pd.DataFrame({"col": [1.0, np.nan, 3.0, np.nan, 5.0]})
        result = _detect_column_types(df)
        assert result["col"]["missing_pct"] == 40.0

    def test_stats_for_numeric(self, sample_df: pd.DataFrame):
        result = _detect_column_types(sample_df)
        stats = result["numeric_cont"]["stats"]
        assert stats is not None
        assert set(stats.keys()) == {"mean", "std", "min", "max", "median"}

    def test_stats_none_for_categorical(self):
        """Object dtype categoricals should have stats=None."""
        df = pd.DataFrame({"cat": pd.Series(["A", "B", "C"] * 10, dtype=object)})
        result = _detect_column_types(df)
        assert result["cat"]["stats"] is None

    def test_top_values_limited_to_10(self):
        df = pd.DataFrame({"col": list(range(50))})
        result = _detect_column_types(df)
        assert len(result["col"]["top_values"]) <= 10


# ---------------------------------------------------------------------------
# _compute_correlations
# ---------------------------------------------------------------------------

class TestComputeCorrelations:
    def test_target_correlations_sorted(self):
        rng = np.random.default_rng(0)
        n = 500
        target = rng.choice([0, 1], n)
        df = pd.DataFrame({
            "strong": target + rng.normal(0, 0.3, n),
            "weak": rng.normal(0, 1, n),
            "target": target,
        })
        result = _compute_correlations(df, "target")
        corrs = result["target_correlations"]
        keys = list(corrs.keys())
        # First key should have highest absolute correlation
        assert abs(corrs[keys[0]]) >= abs(corrs[keys[1]])

    def test_target_excluded_from_correlations(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]})
        result = _compute_correlations(df, "target")
        assert "target" not in result["target_correlations"]

    def test_correlation_matrix_structure(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]})
        result = _compute_correlations(df, "target")
        mat = result["correlation_matrix"]
        assert "columns" in mat
        assert "values" in mat
        assert "target" not in mat["columns"]


# ---------------------------------------------------------------------------
# _find_feature_clusters
# ---------------------------------------------------------------------------

class TestFindFeatureClusters:
    def test_correlated_pair_forms_cluster(self):
        corr_matrix = {
            "columns": ["a", "b", "c"],
            "values": [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ],
        }
        clusters = _find_feature_clusters(corr_matrix, threshold=0.5)
        assert len(clusters) >= 1
        # a and b should be in same cluster
        first_cluster = clusters[0]["features"]
        assert "a" in first_cluster and "b" in first_cluster

    def test_no_clusters_when_uncorrelated(self):
        corr_matrix = {
            "columns": ["a", "b", "c"],
            "values": [
                [1.0, 0.1, 0.05],
                [0.1, 1.0, 0.05],
                [0.05, 0.05, 1.0],
            ],
        }
        clusters = _find_feature_clusters(corr_matrix, threshold=0.5)
        assert len(clusters) == 0

    def test_empty_input(self):
        clusters = _find_feature_clusters({"columns": [], "values": []})
        assert clusters == []

    def test_mean_internal_corr_computed(self):
        corr_matrix = {
            "columns": ["a", "b"],
            "values": [[1.0, 0.8], [0.8, 1.0]],
        }
        clusters = _find_feature_clusters(corr_matrix, threshold=0.5)
        assert len(clusters) == 1
        assert abs(clusters[0]["mean_internal_corr"] - 0.8) < 0.01


# ---------------------------------------------------------------------------
# _identify_weak_features
# ---------------------------------------------------------------------------

class TestIdentifyWeakFeatures:
    def test_identifies_weak(self):
        corrs = {"strong": 0.5, "medium": 0.1, "weak": 0.02, "zero": 0.0}
        weak = _identify_weak_features(corrs, threshold=0.05)
        weak_names = [w["column"] for w in weak]
        assert "weak" in weak_names
        assert "zero" in weak_names
        assert "strong" not in weak_names
        assert "medium" not in weak_names

    def test_sorted_by_abs_correlation(self):
        corrs = {"a": 0.04, "b": -0.01, "c": 0.03}
        weak = _identify_weak_features(corrs, threshold=0.05)
        abs_vals = [abs(w["target_correlation"]) for w in weak]
        assert abs_vals == sorted(abs_vals)

    def test_empty_when_all_strong(self):
        corrs = {"a": 0.5, "b": -0.3}
        weak = _identify_weak_features(corrs, threshold=0.05)
        assert weak == []


# ---------------------------------------------------------------------------
# format_eda_for_llm
# ---------------------------------------------------------------------------

class TestFormatEdaForLlm:
    def test_contains_key_sections(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        text = format_eda_for_llm(report)
        assert "MAESTRO-ML EDA REPORT" in text
        assert "TARGET DISTRIBUTION" in text
        assert "FEATURE" in text
        assert "RECOMMENDATIONS" in text

    def test_returns_string(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        text = format_eda_for_llm(report)
        assert isinstance(text, str)
        assert len(text) > 100
