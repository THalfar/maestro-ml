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

    def test_target_mapping(self, tmp_path: Path):
        """target_mapping should convert string labels to numeric in returned train."""
        rng = np.random.default_rng(99)
        n = 50
        train = pd.DataFrame({
            "feat": rng.normal(0, 1, n),
            "target": rng.choice(["Yes", "No"], n),
        })
        test = pd.DataFrame({"feat": rng.normal(0, 1, 20)})
        train.to_csv(tmp_path / "train.csv", index=False)
        test.to_csv(tmp_path / "test.csv", index=False)

        report, train_df, _ = run_eda(
            tmp_path / "train.csv",
            tmp_path / "test.csv",
            "target",
            target_mapping={"Yes": 1, "No": 0},
        )
        # Returned train should have mapped numeric target
        assert set(train_df["target"].unique()) == {0, 1}
        # Target analysis should reflect mapped values
        assert report["target_analysis"]["n_unique"] == 2
        dist_keys = set(report["target_analysis"]["distribution"].keys())
        assert dist_keys == {"0", "1"}

    def test_target_mapping_none_passthrough(self, binary_dataset: tuple[Path, Path]):
        """When target_mapping is None, target values should be unchanged."""
        train_path, test_path = binary_dataset
        _, train_df, _ = run_eda(train_path, test_path, "target", target_mapping=None)
        assert set(train_df["target"].unique()) == {0, 1}

    def test_regression_target(self, tmp_path: Path):
        """Regression targets (continuous float) should produce valid reports."""
        rng = np.random.default_rng(7)
        n = 50
        train = pd.DataFrame({
            "feat": rng.normal(0, 1, n),
            "target": rng.normal(100, 15, n),
        })
        test = pd.DataFrame({"feat": rng.normal(0, 1, 20)})
        train.to_csv(tmp_path / "train.csv", index=False)
        test.to_csv(tmp_path / "test.csv", index=False)

        report, _, _ = run_eda(tmp_path / "train.csv", tmp_path / "test.csv", "target")
        ta = report["target_analysis"]
        assert ta["n_unique"] == n  # All continuous values should be unique
        assert ta["missing_pct"] == 0.0
        assert "feat" in report["columns"]


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

    def test_binary_string_column(self):
        """A string column with exactly 2 unique values should be 'binary'."""
        df = pd.DataFrame({"col": ["yes", "no", "yes", "no", "yes"]})
        result = _detect_column_types(df)
        assert result["col"]["detected_type"] == "binary"

    def test_all_nan_column(self):
        """Column with all NaN values should not crash and should report 100% missing."""
        df = pd.DataFrame({"col": [np.nan, np.nan, np.nan]})
        result = _detect_column_types(df)
        assert result["col"]["missing_pct"] == 100.0
        assert result["col"]["cardinality"] == 0

    def test_numeric_float_low_cardinality_not_ordinal(self):
        """Numeric floats with cardinality <= 20 but non-integer values → numeric_continuous."""
        df = pd.DataFrame({"col": [0.1, 0.2, 0.3, 0.1, 0.2] * 10})
        result = _detect_column_types(df)
        assert result["col"]["detected_type"] == "numeric_continuous"

    def test_categorical_dtype_column(self):
        """Explicit pd.CategoricalDtype should be detected as categorical."""
        df = pd.DataFrame({
            "col": pd.Categorical(["A", "B", "C", "D", "A", "B"] * 5)
        })
        result = _detect_column_types(df)
        assert result["col"]["detected_type"] == "low_cardinality_categorical"
        assert result["col"]["stats"] is None


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

    def test_nan_column_correlation_is_zero(self):
        """Column with all NaN should produce 0 correlation (fillna(0) path)."""
        df = pd.DataFrame({
            "good": [1, 2, 3, 4, 5],
            "all_nan": [np.nan] * 5,
            "target": [0, 1, 0, 1, 0],
        })
        result = _compute_correlations(df, "target")
        assert result["target_correlations"]["all_nan"] == 0.0

    def test_no_numeric_features(self):
        """When all features are non-numeric, correlation matrix should be empty."""
        df = pd.DataFrame({
            "cat_a": ["x", "y", "z"],
            "cat_b": ["a", "b", "c"],
            "target": [0, 1, 0],
        })
        result = _compute_correlations(df, "target")
        assert result["correlation_matrix"]["columns"] == []
        assert result["correlation_matrix"]["values"] == []

    def test_non_numeric_target_fallback(self):
        """Non-numeric target should still produce correlations via category codes."""
        df = pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [10, 20, 30, 40, 50],
            "target": ["low", "low", "high", "high", "high"],
        })
        result = _compute_correlations(df, "target")
        # Should return correlations for both numeric features
        assert "feat_a" in result["target_correlations"]
        assert "feat_b" in result["target_correlations"]
        # Correlations should be non-zero (features monotonically increase with target)
        assert result["target_correlations"]["feat_a"] != 0.0


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

    def test_negative_correlation_forms_cluster(self):
        """Features with strong negative correlation should cluster (abs >= threshold)."""
        corr_matrix = {
            "columns": ["a", "b", "c"],
            "values": [
                [1.0, -0.85, 0.05],
                [-0.85, 1.0, 0.05],
                [0.05, 0.05, 1.0],
            ],
        }
        clusters = _find_feature_clusters(corr_matrix, threshold=0.5)
        assert len(clusters) == 1
        assert set(clusters[0]["features"]) == {"a", "b"}

    def test_clusters_sorted_by_size(self):
        """Larger clusters should appear first."""
        corr_matrix = {
            "columns": ["a", "b", "c", "d", "e"],
            "values": [
                [1.0, 0.9, 0.9, 0.0, 0.0],
                [0.9, 1.0, 0.9, 0.0, 0.0],
                [0.9, 0.9, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.8],
                [0.0, 0.0, 0.0, 0.8, 1.0],
            ],
        }
        clusters = _find_feature_clusters(corr_matrix, threshold=0.5)
        assert len(clusters) == 2
        assert len(clusters[0]["features"]) >= len(clusters[1]["features"])


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

    def test_negative_weak_feature_identified(self):
        """Weak negative correlations should also be identified."""
        corrs = {"pos": 0.5, "neg_weak": -0.02}
        weak = _identify_weak_features(corrs, threshold=0.05)
        assert len(weak) == 1
        assert weak[0]["column"] == "neg_weak"
        assert weak[0]["target_correlation"] == -0.02


# ---------------------------------------------------------------------------
# _generate_recommendations
# ---------------------------------------------------------------------------

class TestGenerateRecommendations:
    def test_top_correlated_recommendation(self):
        columns_analysis = {
            "feat_a": {"detected_type": "numeric_continuous", "missing_pct": 0.0},
            "feat_b": {"detected_type": "numeric_continuous", "missing_pct": 0.0},
        }
        target_corrs = {"feat_a": 0.8, "feat_b": 0.3}
        recs = _generate_recommendations(columns_analysis, target_corrs, [], [])
        assert any("Top correlated" in r for r in recs)
        assert any("feat_a" in r for r in recs)

    def test_weak_features_recommendation(self):
        columns_analysis = {"w": {"detected_type": "numeric_continuous", "missing_pct": 0.0}}
        weak = [{"column": "w", "target_correlation": 0.01, "recommendation": "consider_dropping"}]
        recs = _generate_recommendations(columns_analysis, {"w": 0.01}, [], weak)
        assert any("Weak features" in r for r in recs)

    def test_cluster_recommendation(self):
        columns_analysis = {
            "a": {"detected_type": "numeric_continuous", "missing_pct": 0.0},
            "b": {"detected_type": "numeric_continuous", "missing_pct": 0.0},
        }
        clusters = [{"features": ["a", "b"], "mean_internal_corr": 0.85}]
        recs = _generate_recommendations(columns_analysis, {"a": 0.5, "b": 0.4}, clusters, [])
        assert any("cluster" in r.lower() for r in recs)

    def test_binary_interaction_recommendation(self):
        columns_analysis = {
            "bin": {"detected_type": "binary", "missing_pct": 0.0},
            "cont": {"detected_type": "numeric_continuous", "missing_pct": 0.0},
        }
        recs = _generate_recommendations(columns_analysis, {"bin": 0.3, "cont": 0.5}, [], [])
        assert any("Binary" in r for r in recs)

    def test_categorical_recommendation(self):
        columns_analysis = {
            "cat_col": {"detected_type": "low_cardinality_categorical", "missing_pct": 0.0},
        }
        recs = _generate_recommendations(columns_analysis, {"cat_col": 0.2}, [], [])
        assert any("Categorical" in r or "target encoding" in r for r in recs)

    def test_missing_values_recommendation(self):
        columns_analysis = {
            "miss_col": {"detected_type": "numeric_continuous", "missing_pct": 25.0},
        }
        recs = _generate_recommendations(columns_analysis, {"miss_col": 0.3}, [], [])
        assert any("missing" in r.lower() for r in recs)

    def test_empty_inputs_no_crash(self):
        recs = _generate_recommendations({}, {}, [], [])
        assert isinstance(recs, list)


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

    def test_missing_values_section(self):
        """Report with missing values should show MISSING VALUES section."""
        report = {
            "dataset_info": {"train_shape": [10, 3], "test_shape": [5, 2],
                             "train_memory_mb": 0.1, "test_memory_mb": 0.05, "n_features": 2},
            "target_analysis": {"dtype": "int64", "n_unique": 2,
                                "distribution": {"0": 5, "1": 5},
                                "class_balance_pct": {"0": 50.0, "1": 50.0},
                                "missing_pct": 0.0},
            "columns": {
                "feat_a": {"detected_type": "numeric_continuous", "target_correlation": 0.5,
                           "missing_pct": 15.0, "stats": {"mean": 0, "std": 1, "min": -2, "max": 2, "median": 0}},
            },
            "correlation_matrix": {"columns": ["feat_a"], "values": [[1.0]]},
            "feature_clusters": [],
            "weak_features": [],
            "recommendations": [],
        }
        text = format_eda_for_llm(report)
        assert "MISSING VALUES" in text
        assert "feat_a" in text
        assert "15.00%" in text

    def test_weak_features_section(self):
        """Report with weak features should display them."""
        report = {
            "dataset_info": {"train_shape": [10, 2], "test_shape": [5, 1],
                             "train_memory_mb": 0.1, "test_memory_mb": 0.05, "n_features": 1},
            "target_analysis": {"dtype": "int64", "n_unique": 2,
                                "distribution": {"0": 5, "1": 5},
                                "class_balance_pct": {"0": 50.0, "1": 50.0},
                                "missing_pct": 0.0},
            "columns": {"weak_f": {"detected_type": "numeric_continuous",
                                   "target_correlation": 0.01, "missing_pct": 0.0}},
            "correlation_matrix": {"columns": [], "values": []},
            "feature_clusters": [],
            "weak_features": [{"column": "weak_f", "target_correlation": 0.01,
                               "recommendation": "consider_dropping"}],
            "recommendations": [],
        }
        text = format_eda_for_llm(report)
        assert "WEAK FEATURES" in text
        assert "weak_f" in text
        assert "consider_dropping" in text

    def test_column_type_breakdown(self, binary_dataset: tuple[Path, Path]):
        """Report should include column type breakdown section."""
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        text = format_eda_for_llm(report)
        assert "COLUMN TYPE BREAKDOWN" in text

    def test_empty_report(self):
        """Minimal/empty report should not crash."""
        report = {
            "dataset_info": {},
            "target_analysis": {"dtype": "", "n_unique": 0,
                                "distribution": {},
                                "class_balance_pct": {},
                                "missing_pct": 0.0},
            "columns": {},
            "correlation_matrix": {"columns": [], "values": []},
            "feature_clusters": [],
            "weak_features": [],
            "recommendations": [],
        }
        text = format_eda_for_llm(report)
        assert isinstance(text, str)
        assert "MAESTRO-ML EDA REPORT" in text
        assert "(none)" in text or "(no " in text

    def test_feature_clusters_section(self):
        """Report with clusters should display them."""
        report = {
            "dataset_info": {"train_shape": [10, 3], "test_shape": [5, 2],
                             "train_memory_mb": 0.1, "test_memory_mb": 0.05, "n_features": 2},
            "target_analysis": {"dtype": "int64", "n_unique": 2,
                                "distribution": {"0": 5, "1": 5},
                                "class_balance_pct": {"0": 50.0, "1": 50.0},
                                "missing_pct": 0.0},
            "columns": {
                "a": {"detected_type": "numeric_continuous", "target_correlation": 0.5, "missing_pct": 0.0},
                "b": {"detected_type": "numeric_continuous", "target_correlation": 0.4, "missing_pct": 0.0},
            },
            "correlation_matrix": {"columns": ["a", "b"], "values": [[1.0, 0.9], [0.9, 1.0]]},
            "feature_clusters": [{"features": ["a", "b"], "mean_internal_corr": 0.9}],
            "weak_features": [],
            "recommendations": [],
        }
        text = format_eda_for_llm(report)
        assert "FEATURE CLUSTERS" in text
        assert "Cluster 1" in text
        assert "a" in text and "b" in text
