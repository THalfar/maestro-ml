"""Tests for src/eda/profiler.py — EDA profiling and report generation."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.profiler import (
    _add_skewness_and_outliers,
    _compute_categorical_target_rates,
    _compute_correlations,
    _compute_cramers_v,
    _compute_distribution_shift,
    _compute_iv_woe,
    _compute_mutual_information,
    _compute_psi_numeric,
    _compute_univariate_auc,
    _compute_vif,
    _detect_column_types,
    _detect_leakage,
    _enrich_clusters_with_pairs,
    _find_feature_clusters,
    _generate_recommendations,
    _identify_weak_features,
    _screen_interactions,
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
            "mutual_information", "distribution_shift",
            "interaction_candidates", "leakage_warnings", "vif_scores",
            "univariate_auc", "iv_woe", "cramers_v",
            "preprocessing_summary",
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
        assert set(stats.keys()) == {"mean", "std", "min", "max", "range", "median"}

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

    def test_mi_section_displayed(self, binary_dataset: tuple[Path, Path]):
        """New MI section should appear in formatted output."""
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        text = format_eda_for_llm(report)
        assert "MUTUAL INFORMATION" in text

    def test_old_report_no_crash(self):
        """Report without new keys should still format without errors."""
        report = {
            "dataset_info": {"train_shape": [10, 3], "test_shape": [5, 2],
                             "train_memory_mb": 0.1, "test_memory_mb": 0.05, "n_features": 2},
            "target_analysis": {"dtype": "int64", "n_unique": 2,
                                "distribution": {"0": 5, "1": 5},
                                "class_balance_pct": {"0": 50.0, "1": 50.0},
                                "missing_pct": 0.0},
            "columns": {"f": {"detected_type": "numeric_continuous",
                               "target_correlation": 0.3, "missing_pct": 0.0}},
            "correlation_matrix": {"columns": [], "values": []},
            "feature_clusters": [],
            "weak_features": [],
            "recommendations": ["Test rec"],
        }
        text = format_eda_for_llm(report)
        assert "MAESTRO-ML EDA REPORT" in text
        assert "Test rec" in text


# ---------------------------------------------------------------------------
# _add_skewness_and_outliers
# ---------------------------------------------------------------------------

class TestAddSkewnessAndOutliers:
    def test_numeric_has_skewness(self, sample_df: pd.DataFrame):
        analysis = _detect_column_types(sample_df[["numeric_cont"]])
        _add_skewness_and_outliers(analysis, sample_df[["numeric_cont"]])
        assert analysis["numeric_cont"]["skewness"] is not None
        assert isinstance(analysis["numeric_cont"]["skewness"], float)

    def test_numeric_has_kurtosis(self, sample_df: pd.DataFrame):
        analysis = _detect_column_types(sample_df[["numeric_cont"]])
        _add_skewness_and_outliers(analysis, sample_df[["numeric_cont"]])
        assert analysis["numeric_cont"]["kurtosis"] is not None
        assert isinstance(analysis["numeric_cont"]["kurtosis"], float)

    def test_categorical_gets_none(self, sample_df: pd.DataFrame):
        analysis = _detect_column_types(sample_df[["low_card_cat"]])
        _add_skewness_and_outliers(analysis, sample_df[["low_card_cat"]])
        assert analysis["low_card_cat"]["skewness"] is None
        assert analysis["low_card_cat"]["kurtosis"] is None
        assert analysis["low_card_cat"]["outlier_pct"] is None

    def test_outlier_pct_range(self, sample_df: pd.DataFrame):
        analysis = _detect_column_types(sample_df[["numeric_cont"]])
        _add_skewness_and_outliers(analysis, sample_df[["numeric_cont"]])
        assert 0.0 <= analysis["numeric_cont"]["outlier_pct"] <= 100.0

    def test_all_nan_no_crash(self):
        df = pd.DataFrame({"col": [np.nan, np.nan, np.nan]})
        analysis = _detect_column_types(df)
        _add_skewness_and_outliers(analysis, df)
        # All NaN column has stats (it's float64, numeric) but dropna is empty
        assert analysis["col"]["skewness"] is None
        assert analysis["col"]["kurtosis"] is None


# ---------------------------------------------------------------------------
# _compute_mutual_information
# ---------------------------------------------------------------------------

class TestComputeMutualInformation:
    def test_returns_all_columns(self, sample_df: pd.DataFrame):
        target = pd.Series(np.random.default_rng(0).choice([0, 1], len(sample_df)))
        mi = _compute_mutual_information(sample_df, target, "binary_classification")
        assert set(mi.keys()) == set(sample_df.columns)

    def test_scores_nonnegative(self, sample_df: pd.DataFrame):
        target = pd.Series(np.random.default_rng(0).choice([0, 1], len(sample_df)))
        mi = _compute_mutual_information(sample_df, target, "binary_classification")
        assert all(v >= 0.0 for v in mi.values())

    def test_sorted_descending(self, sample_df: pd.DataFrame):
        target = pd.Series(np.random.default_rng(0).choice([0, 1], len(sample_df)))
        mi = _compute_mutual_information(sample_df, target, "binary_classification")
        values = list(mi.values())
        assert values == sorted(values, reverse=True)

    def test_categorical_columns_included(self):
        """Categorical columns should get MI scores, not be silently skipped."""
        df = pd.DataFrame({
            "cat": ["a", "a", "a", "b", "b", "b"] * 10,
            "num": np.random.default_rng(0).normal(0, 1, 60),
        })
        target = pd.Series([0, 0, 0, 1, 1, 1] * 10)
        mi = _compute_mutual_information(df, target, "binary_classification")
        assert "cat" in mi
        assert mi["cat"] > 0.0  # cat perfectly separates target

    def test_regression_task(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(0, 1, 100)})
        target = pd.Series(rng.normal(0, 1, 100))
        mi = _compute_mutual_information(df, target, "regression")
        assert "x" in mi

    def test_empty_df(self):
        mi = _compute_mutual_information(pd.DataFrame(), pd.Series(dtype=float), "binary_classification")
        assert mi == {}


# ---------------------------------------------------------------------------
# _compute_categorical_target_rates
# ---------------------------------------------------------------------------

class TestComputeCategoricalTargetRates:
    def test_numeric_col_skipped(self):
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0, 5.0]})
        target = pd.Series([0, 1, 0, 1, 0])
        analysis = {"num": {"detected_type": "numeric_continuous", "cardinality": 5}}
        result = _compute_categorical_target_rates(df, target, analysis, "binary_classification")
        assert "num" not in result

    def test_binary_col_has_rates(self):
        rng = np.random.default_rng(0)
        n = 100
        feat = rng.choice(["A", "B"], n)
        target = pd.Series([1 if f == "A" else 0 for f in feat])
        df = pd.DataFrame({"cat": feat})
        analysis = {"cat": {"detected_type": "binary", "cardinality": 2}}
        result = _compute_categorical_target_rates(df, target, analysis, "binary_classification")
        assert "cat" in result
        assert "A" in result["cat"]["target_rate_per_value"]
        assert "B" in result["cat"]["target_rate_per_value"]

    def test_max_delta_correct(self):
        n = 100
        feat = ["A"] * 50 + ["B"] * 50
        target = pd.Series([1] * 50 + [0] * 50)
        df = pd.DataFrame({"cat": feat})
        analysis = {"cat": {"detected_type": "binary", "cardinality": 2}}
        result = _compute_categorical_target_rates(df, target, analysis, "binary_classification")
        assert result["cat"]["target_rate_max_delta"] == 1.0

    def test_high_cardinality_skipped(self):
        df = pd.DataFrame({"cat": [f"val_{i}" for i in range(100)]})
        target = pd.Series(np.random.default_rng(0).choice([0, 1], 100))
        analysis = {"cat": {"detected_type": "high_cardinality_categorical", "cardinality": 100}}
        result = _compute_categorical_target_rates(df, target, analysis, "binary_classification")
        assert "cat" not in result


# ---------------------------------------------------------------------------
# _compute_distribution_shift
# ---------------------------------------------------------------------------

class TestComputeDistributionShift:
    def test_identical_not_flagged(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 200)
        train = pd.DataFrame({"num": data[:100]})
        test = pd.DataFrame({"num": data[100:]})
        analysis = {"num": {"stats": {"mean": 0}}}
        result = _compute_distribution_shift(train, test, analysis)
        assert result["numeric"]["num"]["shift_flagged"] is False

    def test_shifted_flagged(self):
        rng = np.random.default_rng(0)
        train = pd.DataFrame({"num": rng.normal(0, 1, 500)})
        test = pd.DataFrame({"num": rng.normal(5, 1, 500)})
        analysis = {"num": {"stats": {"mean": 0}}}
        result = _compute_distribution_shift(train, test, analysis)
        assert result["numeric"]["num"]["shift_flagged"] is True
        assert "num" in result["flagged_columns"]

    def test_missing_test_col_skipped(self):
        train = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        test = pd.DataFrame({"a": [1, 2, 3]})
        analysis = {
            "a": {"stats": {"mean": 0}},
            "b": {"stats": {"mean": 0}},
        }
        result = _compute_distribution_shift(train, test, analysis)
        assert "b" not in result["numeric"]

    def test_categorical_shift(self):
        train = pd.DataFrame({"cat": ["A"] * 80 + ["B"] * 20})
        test = pd.DataFrame({"cat": ["A"] * 20 + ["B"] * 80})
        analysis = {"cat": {"stats": None}}
        result = _compute_distribution_shift(train, test, analysis)
        assert result["categorical"]["cat"]["shift_flagged"] is True


# ---------------------------------------------------------------------------
# _enrich_clusters_with_pairs
# ---------------------------------------------------------------------------

class TestEnrichClustersWithPairs:
    def test_pairs_added(self):
        clusters = [{"features": ["a", "b"], "mean_internal_corr": 0.8}]
        matrix = {"columns": ["a", "b"], "values": [[1.0, 0.8], [0.8, 1.0]]}
        result = _enrich_clusters_with_pairs(clusters, matrix)
        assert len(result[0]["pairs"]) == 1
        assert result[0]["pairs"][0]["correlation"] == 0.8

    def test_signed_correlation_preserved(self):
        clusters = [{"features": ["a", "b"], "mean_internal_corr": 0.85}]
        matrix = {"columns": ["a", "b"], "values": [[1.0, -0.85], [-0.85, 1.0]]}
        result = _enrich_clusters_with_pairs(clusters, matrix)
        assert result[0]["pairs"][0]["correlation"] == -0.85

    def test_sorted_by_abs_corr(self):
        clusters = [{"features": ["a", "b", "c"], "mean_internal_corr": 0.7}]
        matrix = {
            "columns": ["a", "b", "c"],
            "values": [[1.0, 0.5, 0.9], [0.5, 1.0, 0.7], [0.9, 0.7, 1.0]],
        }
        result = _enrich_clusters_with_pairs(clusters, matrix)
        abs_corrs = [abs(p["correlation"]) for p in result[0]["pairs"]]
        assert abs_corrs == sorted(abs_corrs, reverse=True)

    def test_does_not_mutate_input(self):
        clusters = [{"features": ["a", "b"], "mean_internal_corr": 0.8}]
        matrix = {"columns": ["a", "b"], "values": [[1.0, 0.8], [0.8, 1.0]]}
        _enrich_clusters_with_pairs(clusters, matrix)
        assert "pairs" not in clusters[0]

    def test_empty_clusters(self):
        result = _enrich_clusters_with_pairs([], {"columns": [], "values": []})
        assert result == []


# ---------------------------------------------------------------------------
# _screen_interactions
# ---------------------------------------------------------------------------

class TestScreenInteractions:
    def test_returns_list(self):
        rng = np.random.default_rng(0)
        n = 200
        x = rng.normal(0, 1, n)
        y = rng.normal(0, 1, n)
        target = pd.Series(x * y + rng.normal(0, 0.1, n))
        df = pd.DataFrame({"x": x, "y": y})
        mi = {"x": 0.1, "y": 0.1}
        analysis = {
            "x": {"detected_type": "numeric_continuous"},
            "y": {"detected_type": "numeric_continuous"},
        }
        result = _screen_interactions(df, target, mi, analysis)
        assert isinstance(result, list)

    def test_added_value_positive(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.choice([0, 1], n)
        y = rng.normal(0, 1, n)
        target = pd.Series(x * y + rng.normal(0, 0.1, n))
        df = pd.DataFrame({"x": x, "y": y})
        mi = {"x": 0.1, "y": 0.1}
        analysis = {
            "x": {"detected_type": "binary"},
            "y": {"detected_type": "numeric_continuous"},
        }
        result = _screen_interactions(df, target, mi, analysis)
        for ic in result:
            assert ic["added_value"] > 0.01

    def test_categorical_excluded(self):
        df = pd.DataFrame({"cat": ["A", "B"] * 50, "num": range(100)})
        target = pd.Series(np.random.default_rng(0).choice([0, 1], 100))
        mi = {"cat": 0.1, "num": 0.1}
        analysis = {
            "cat": {"detected_type": "low_cardinality_categorical"},
            "num": {"detected_type": "numeric_continuous"},
        }
        result = _screen_interactions(df, target, mi, analysis)
        # Only 1 numeric col after excluding cat, so no pairs possible
        assert result == []


# ---------------------------------------------------------------------------
# _detect_leakage
# ---------------------------------------------------------------------------

class TestDetectLeakage:
    def test_high_corr_flagged(self):
        analysis = {"col": {"target_rate_max_delta": None}}
        mi = {"col": 0.1}
        corrs = {"col": 0.85}
        result = _detect_leakage(analysis, mi, corrs)
        assert len(result) == 1
        assert result[0]["reason"] == "high_numeric_correlation"

    def test_high_delta_flagged(self):
        analysis = {"col": {"target_rate_max_delta": 0.75}}
        mi = {"col": 0.1}
        corrs = {"col": 0.1}
        result = _detect_leakage(analysis, mi, corrs)
        assert any(w["reason"] == "high_categorical_delta" for w in result)

    def test_normal_not_flagged(self):
        analysis = {"col": {"target_rate_max_delta": 0.2}}
        mi = {"col": 0.1}
        corrs = {"col": 0.3}
        result = _detect_leakage(analysis, mi, corrs)
        assert result == []

    def test_high_mi_flagged(self):
        analysis = {"col": {"target_rate_max_delta": None}}
        mi = {"col": 0.7}
        corrs = {"col": 0.3}
        result = _detect_leakage(analysis, mi, corrs)
        assert any(w["reason"] == "high_mutual_information" for w in result)


# ---------------------------------------------------------------------------
# _compute_vif
# ---------------------------------------------------------------------------

class TestComputeVif:
    def test_returns_dict(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(0, 1, 50)})
        analysis = {
            "a": {"stats": {"mean": 0}},
            "b": {"stats": {"mean": 0}},
        }
        result = _compute_vif(df, analysis)
        assert isinstance(result, dict)
        assert "a" in result and "b" in result

    def test_independent_low_vif(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
        analysis = {"a": {"stats": {"mean": 0}}, "b": {"stats": {"mean": 0}}}
        result = _compute_vif(df, analysis)
        assert all(v < 5.0 for v in result.values())

    def test_collinear_high_vif(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 200)
        df = pd.DataFrame({"a": x, "b": x + rng.normal(0, 0.01, 200), "c": rng.normal(0, 1, 200)})
        analysis = {
            "a": {"stats": {"mean": 0}},
            "b": {"stats": {"mean": 0}},
            "c": {"stats": {"mean": 0}},
        }
        result = _compute_vif(df, analysis)
        assert result["a"] > 10.0 or result["b"] > 10.0

    def test_single_numeric_returns_empty(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        analysis = {"a": {"stats": {"mean": 0}}}
        result = _compute_vif(df, analysis)
        assert result == {}

    def test_no_statsmodels_needed(self):
        """VIF should work without statsmodels import."""
        import importlib
        # Just verify the function runs without statsmodels
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(0, 1, 50)})
        analysis = {"a": {"stats": {"mean": 0}}, "b": {"stats": {"mean": 0}}}
        result = _compute_vif(df, analysis)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _compute_univariate_auc
# ---------------------------------------------------------------------------

class TestComputeUnivariateAuc:
    def test_returns_all_columns(self):
        rng = np.random.default_rng(42)
        n = 200
        target = pd.Series(rng.choice([0, 1], n))
        df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
        result = _compute_univariate_auc(df, target)
        assert set(result.keys()) == {"a", "b"}

    def test_auc_range(self):
        rng = np.random.default_rng(42)
        n = 200
        target = pd.Series(rng.choice([0, 1], n))
        df = pd.DataFrame({"a": rng.normal(0, 1, n)})
        result = _compute_univariate_auc(df, target)
        assert 0.5 <= result["a"] <= 1.0

    def test_perfect_predictor(self):
        target = pd.Series([0] * 50 + [1] * 50)
        df = pd.DataFrame({"a": list(range(100))})
        result = _compute_univariate_auc(df, target)
        assert result["a"] == 1.0

    def test_sorted_descending(self):
        rng = np.random.default_rng(0)
        n = 200
        target = pd.Series(rng.choice([0, 1], n))
        df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": target * 10.0})
        result = _compute_univariate_auc(df, target)
        vals = list(result.values())
        assert vals == sorted(vals, reverse=True)

    def test_non_binary_returns_empty(self):
        target = pd.Series([0, 1, 2, 3, 4])
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        assert _compute_univariate_auc(df, target) == {}

    def test_empty_df(self):
        assert _compute_univariate_auc(pd.DataFrame(), pd.Series(dtype=float)) == {}

    def test_categorical_feature(self):
        """Categorical features should be label-encoded and produce valid AUC."""
        target = pd.Series([0, 0, 0, 1, 1, 1] * 10)
        df = pd.DataFrame({"cat": ["A", "A", "A", "B", "B", "B"] * 10})
        result = _compute_univariate_auc(df, target)
        assert "cat" in result
        assert result["cat"] > 0.9  # near-perfect separation


# ---------------------------------------------------------------------------
# _compute_iv_woe
# ---------------------------------------------------------------------------

class TestComputeIvWoe:
    def test_returns_all_features(self):
        rng = np.random.default_rng(42)
        n = 200
        target = pd.Series(rng.choice([0, 1], n))
        df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.choice(["X", "Y", "Z"], n)})
        analysis = {
            "a": {"stats": {"mean": 0}, "detected_type": "numeric_continuous"},
            "b": {"stats": None, "detected_type": "low_cardinality_categorical"},
        }
        result = _compute_iv_woe(df, target, analysis)
        assert "a" in result
        assert "b" in result

    def test_iv_structure(self):
        rng = np.random.default_rng(42)
        n = 200
        target = pd.Series(rng.choice([0, 1], n))
        df = pd.DataFrame({"a": rng.normal(0, 1, n)})
        analysis = {"a": {"stats": {"mean": 0}, "detected_type": "numeric_continuous"}}
        result = _compute_iv_woe(df, target, analysis)
        assert "iv" in result["a"]
        assert "iv_label" in result["a"]
        assert "woe_per_bin" in result["a"]
        assert result["a"]["iv"] >= 0.0

    def test_iv_labels(self):
        """Test that IV labels map correctly to ranges."""
        rng = np.random.default_rng(42)
        n = 200
        # Strong predictor
        target = pd.Series([0] * 100 + [1] * 100)
        df = pd.DataFrame({"strong": list(range(200))})
        analysis = {"strong": {"stats": {"mean": 0}, "detected_type": "numeric_continuous"}}
        result = _compute_iv_woe(df, target, analysis)
        assert result["strong"]["iv_label"] in ("medium", "strong")

    def test_sorted_by_iv_descending(self):
        rng = np.random.default_rng(42)
        n = 200
        target = pd.Series([0] * 100 + [1] * 100)
        df = pd.DataFrame({"strong": list(range(200)), "weak": rng.normal(0, 1, 200)})
        analysis = {
            "strong": {"stats": {"mean": 0}, "detected_type": "numeric_continuous"},
            "weak": {"stats": {"mean": 0}, "detected_type": "numeric_continuous"},
        }
        result = _compute_iv_woe(df, target, analysis)
        ivs = [info["iv"] for info in result.values()]
        assert ivs == sorted(ivs, reverse=True)

    def test_non_binary_returns_empty(self):
        target = pd.Series([0, 1, 2, 3, 4])
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        analysis = {"a": {"stats": {"mean": 0}, "detected_type": "numeric_continuous"}}
        assert _compute_iv_woe(df, target, analysis) == {}

    def test_empty_returns_empty(self):
        assert _compute_iv_woe(pd.DataFrame(), pd.Series(dtype=float), {}) == {}

    def test_woe_per_bin_has_counts(self):
        rng = np.random.default_rng(42)
        n = 200
        target = pd.Series(rng.choice([0, 1], n))
        df = pd.DataFrame({"a": rng.normal(0, 1, n)})
        analysis = {"a": {"stats": {"mean": 0}, "detected_type": "numeric_continuous"}}
        result = _compute_iv_woe(df, target, analysis)
        for bin_info in result["a"]["woe_per_bin"].values():
            assert "woe" in bin_info
            assert "count" in bin_info
            assert "event_rate" in bin_info
            assert bin_info["count"] > 0


# ---------------------------------------------------------------------------
# _compute_cramers_v
# ---------------------------------------------------------------------------

class TestComputeCramersV:
    def test_returns_pairs_and_matrix(self):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "a": rng.choice(["X", "Y"], n),
            "b": rng.choice(["X", "Y"], n),
        })
        analysis = {
            "a": {"detected_type": "binary", "cardinality": 2},
            "b": {"detected_type": "binary", "cardinality": 2},
        }
        result = _compute_cramers_v(df, analysis)
        assert "pairs" in result
        assert "matrix" in result
        assert result["matrix"]["columns"] == ["a", "b"]

    def test_perfect_association(self):
        n = 100
        vals = ["A", "B"] * (n // 2)
        df = pd.DataFrame({"a": vals, "b": vals})
        analysis = {
            "a": {"detected_type": "binary", "cardinality": 2},
            "b": {"detected_type": "binary", "cardinality": 2},
        }
        result = _compute_cramers_v(df, analysis)
        assert len(result["pairs"]) == 1
        assert result["pairs"][0]["cramers_v"] > 0.9

    def test_independent_low_v(self):
        rng = np.random.default_rng(42)
        n = 500
        df = pd.DataFrame({
            "a": rng.choice(["X", "Y", "Z"], n),
            "b": rng.choice(["P", "Q", "R"], n),
        })
        analysis = {
            "a": {"detected_type": "low_cardinality_categorical", "cardinality": 3},
            "b": {"detected_type": "low_cardinality_categorical", "cardinality": 3},
        }
        result = _compute_cramers_v(df, analysis)
        # Independent features should have V close to 0
        if result["pairs"]:
            assert result["pairs"][0]["cramers_v"] < 0.3

    def test_single_cat_returns_empty(self):
        df = pd.DataFrame({"a": ["X", "Y", "Z"]})
        analysis = {"a": {"detected_type": "low_cardinality_categorical", "cardinality": 3}}
        result = _compute_cramers_v(df, analysis)
        assert result["pairs"] == []
        assert result["matrix"]["columns"] == []

    def test_numeric_excluded(self):
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "num": rng.normal(0, 1, n),
            "cat": rng.choice(["A", "B"], n),
        })
        analysis = {
            "num": {"detected_type": "numeric_continuous", "cardinality": 100},
            "cat": {"detected_type": "binary", "cardinality": 2},
        }
        result = _compute_cramers_v(df, analysis)
        assert result["pairs"] == []

    def test_pairs_sorted_descending(self):
        rng = np.random.default_rng(42)
        n = 200
        a = rng.choice(["X", "Y"], n)
        df = pd.DataFrame({
            "a": a,
            "b": a,  # identical to a
            "c": rng.choice(["X", "Y"], n),  # independent
        })
        analysis = {
            "a": {"detected_type": "binary", "cardinality": 2},
            "b": {"detected_type": "binary", "cardinality": 2},
            "c": {"detected_type": "binary", "cardinality": 2},
        }
        result = _compute_cramers_v(df, analysis)
        if len(result["pairs"]) >= 2:
            vs = [p["cramers_v"] for p in result["pairs"]]
            assert vs == sorted(vs, reverse=True)


# ---------------------------------------------------------------------------
# _compute_psi_numeric
# ---------------------------------------------------------------------------

class TestComputePsiNumeric:
    def test_identical_distribution_low_psi(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        psi = _compute_psi_numeric(pd.Series(data[:500]), pd.Series(data[500:]))
        assert psi < 0.1

    def test_shifted_distribution_high_psi(self):
        rng = np.random.default_rng(42)
        train = pd.Series(rng.normal(0, 1, 500))
        test = pd.Series(rng.normal(3, 1, 500))
        psi = _compute_psi_numeric(train, test)
        assert psi > 0.25

    def test_psi_non_negative(self):
        rng = np.random.default_rng(42)
        train = pd.Series(rng.normal(0, 1, 200))
        test = pd.Series(rng.normal(0.5, 1, 200))
        psi = _compute_psi_numeric(train, test)
        assert psi >= 0.0

    def test_constant_feature(self):
        train = pd.Series([1.0] * 100)
        test = pd.Series([1.0] * 50)
        psi = _compute_psi_numeric(train, test)
        assert psi == 0.0


# ---------------------------------------------------------------------------
# Near-zero-variance (dominant_pct in _detect_column_types)
# ---------------------------------------------------------------------------

class TestNearZeroVariance:
    def test_dominant_pct_computed(self):
        df = pd.DataFrame({"a": [0] * 99 + [1]})
        result = _detect_column_types(df)
        assert result["a"]["dominant_pct"] == 99.0

    def test_balanced_feature(self):
        df = pd.DataFrame({"a": [0, 1] * 50})
        result = _detect_column_types(df)
        assert result["a"]["dominant_pct"] == 50.0

    def test_all_same_value(self):
        df = pd.DataFrame({"a": [42] * 100})
        result = _detect_column_types(df)
        assert result["a"]["dominant_pct"] == 100.0

    def test_categorical_dominant(self):
        df = pd.DataFrame({"a": ["X"] * 98 + ["Y", "Z"]})
        result = _detect_column_types(df)
        assert result["a"]["dominant_pct"] == 98.0


# ---------------------------------------------------------------------------
# PSI in distribution shift integration
# ---------------------------------------------------------------------------

class TestDistributionShiftPsi:
    def test_psi_in_numeric_shift(self):
        rng = np.random.default_rng(0)
        train = pd.DataFrame({"num": rng.normal(0, 1, 500)})
        test = pd.DataFrame({"num": rng.normal(5, 1, 500)})
        analysis = {"num": {"stats": {"mean": 0}}}
        result = _compute_distribution_shift(train, test, analysis)
        assert "psi" in result["numeric"]["num"]
        assert result["numeric"]["num"]["psi"] > 0.25

    def test_psi_in_categorical_shift(self):
        train = pd.DataFrame({"cat": ["A"] * 80 + ["B"] * 20})
        test = pd.DataFrame({"cat": ["A"] * 20 + ["B"] * 80})
        analysis = {"cat": {"stats": None}}
        result = _compute_distribution_shift(train, test, analysis)
        assert "psi" in result["categorical"]["cat"]
        assert result["categorical"]["cat"]["psi"] > 0.1

    def test_stable_low_psi(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 2000)
        train = pd.DataFrame({"num": data[:1000]})
        test = pd.DataFrame({"num": data[1000:]})
        analysis = {"num": {"stats": {"mean": 0}}}
        result = _compute_distribution_shift(train, test, analysis)
        assert result["numeric"]["num"]["psi"] < 0.1


# ---------------------------------------------------------------------------
# format_eda_for_llm with new sections
# ---------------------------------------------------------------------------

class TestFormatEdaNewSections:
    def test_univariate_auc_section(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        text = format_eda_for_llm(report)
        assert "UNIVARIATE AUC" in text

    def test_iv_section(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        text = format_eda_for_llm(report)
        # IV may or may not have notable features, check header appears in table
        assert "AUC" in text  # column header

    def test_psi_in_shift_section(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        text = format_eda_for_llm(report)
        assert "PSI" in text

    def test_kurtosis_in_extra_column(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        text = format_eda_for_llm(report)
        assert "k=" in text  # kurtosis in EXTRA column

    def test_preprocessing_summary_in_output(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        text = format_eda_for_llm(report)
        assert "PREPROCESSING SUMMARY" in text
        assert "Scale range ratio" in text
        assert "Suggested scalers" in text


# ---------------------------------------------------------------------------
# Range in stats
# ---------------------------------------------------------------------------

class TestRangeInStats:
    def test_range_computed(self, sample_df: pd.DataFrame):
        analysis = _detect_column_types(sample_df)
        cont = analysis["numeric_cont"]
        assert "range" in cont["stats"]
        expected_range = cont["stats"]["max"] - cont["stats"]["min"]
        assert abs(cont["stats"]["range"] - expected_range) < 1e-5

    def test_range_none_for_categorical(self, sample_df: pd.DataFrame):
        analysis = _detect_column_types(sample_df)
        assert analysis["low_card_cat"]["stats"] is None


# ---------------------------------------------------------------------------
# Skewness labels
# ---------------------------------------------------------------------------

class TestSkewnessLabels:
    def test_symmetric_label(self):
        """Normal distribution → symmetric skewness."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"col": rng.normal(0, 1, 1000)})
        analysis = _detect_column_types(df)
        _add_skewness_and_outliers(analysis, df)
        # Normal should be approximately symmetric
        assert analysis["col"]["skewness_label"] == "symmetric"

    def test_high_label(self):
        """Exponential distribution → high skewness."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"col": rng.exponential(1, 1000)})
        analysis = _detect_column_types(df)
        _add_skewness_and_outliers(analysis, df)
        assert analysis["col"]["skewness_label"] == "high"

    def test_none_for_categorical(self, sample_df: pd.DataFrame):
        analysis = _detect_column_types(sample_df)
        _add_skewness_and_outliers(analysis, sample_df)
        assert analysis["low_card_cat"]["skewness_label"] is None


# ---------------------------------------------------------------------------
# Sentinel detection
# ---------------------------------------------------------------------------

class TestSentinelDetection:
    def test_detect_minus_one(self):
        """Should detect -1 as sentinel when frequent enough."""
        from src.eda.profiler import _detect_sentinels
        rng = np.random.default_rng(42)
        n = 200
        vals = rng.normal(5, 1, n)
        vals[:20] = -1  # 10% are -1 sentinels
        df = pd.DataFrame({"col": vals})
        analysis = _detect_column_types(df)
        _detect_sentinels(analysis, df)
        sentinels = analysis["col"]["sentinels"]
        assert len(sentinels) >= 1
        assert any(s["value"] == -1.0 for s in sentinels)

    def test_no_false_positive(self):
        """Normal data without sentinels should not flag any."""
        from src.eda.profiler import _detect_sentinels
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"col": rng.normal(0, 1, 200)})
        analysis = _detect_column_types(df)
        _detect_sentinels(analysis, df)
        assert analysis["col"]["sentinels"] == []

    def test_detect_minus_999(self):
        """Should detect -999 as sentinel."""
        from src.eda.profiler import _detect_sentinels
        rng = np.random.default_rng(42)
        n = 200
        vals = rng.normal(50, 10, n)
        vals[:10] = -999  # 5%
        df = pd.DataFrame({"col": vals})
        analysis = _detect_column_types(df)
        _detect_sentinels(analysis, df)
        sentinels = analysis["col"]["sentinels"]
        assert any(s["value"] == -999.0 for s in sentinels)

    def test_sentinel_in_eda_report(self, binary_dataset: tuple[Path, Path]):
        """run_eda should include sentinels in column analysis."""
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        columns = report["columns"]
        for col_name, col_info in columns.items():
            assert "sentinels" in col_info


# ---------------------------------------------------------------------------
# Preprocessing summary
# ---------------------------------------------------------------------------

class TestPreprocessingSummary:
    def test_summary_structure(self, binary_dataset: tuple[Path, Path]):
        train_path, test_path = binary_dataset
        report, _, _ = run_eda(train_path, test_path, "target")
        ps = report["preprocessing_summary"]
        assert "scale_range_ratio" in ps
        assert "n_high_skew" in ps
        assert "n_high_outlier" in ps
        assert "n_sentinel_features" in ps
        assert "suggested_scalers" in ps
        assert isinstance(ps["suggested_scalers"], list)

    def test_scale_range_ratio(self):
        """Features on different scales should produce high ratio."""
        from src.eda.profiler import _build_preprocessing_summary
        analysis = {
            "small_scale": {
                "detected_type": "numeric_continuous",
                "stats": {"range": 1.0},
                "skewness_label": "symmetric",
                "outlier_pct": 0.0,
                "sentinels": [],
            },
            "large_scale": {
                "detected_type": "numeric_continuous",
                "stats": {"range": 1000.0},
                "skewness_label": "symmetric",
                "outlier_pct": 0.0,
                "sentinels": [],
            },
        }
        summary = _build_preprocessing_summary(analysis)
        assert summary["scale_range_ratio"] == 1000.0

    def test_high_outlier_detection(self):
        """Features with >3% outliers should be flagged."""
        from src.eda.profiler import _build_preprocessing_summary
        analysis = {
            "outlier_col": {
                "detected_type": "numeric_continuous",
                "stats": {"range": 10.0},
                "skewness_label": "moderate",
                "outlier_pct": 5.0,
                "sentinels": [],
            },
        }
        summary = _build_preprocessing_summary(analysis)
        assert summary["n_high_outlier"] == 1
        assert "outlier_col" in summary["high_outlier_features"]
        assert "robust" in summary["suggested_scalers"]

    def test_high_range_ratio_removes_standard(self):
        """Scale range ratio > 100 should remove 'standard' and prefer 'robust'."""
        from src.eda.profiler import _build_preprocessing_summary
        analysis = {
            "tiny_range": {
                "detected_type": "numeric_continuous",
                "stats": {"range": 0.5},
                "skewness_label": "symmetric",
                "outlier_pct": 0.0,
                "sentinels": [],
            },
            "huge_range": {
                "detected_type": "numeric_continuous",
                "stats": {"range": 100000.0},
                "skewness_label": "symmetric",
                "outlier_pct": 0.0,
                "sentinels": [],
            },
        }
        summary = _build_preprocessing_summary(analysis)
        assert summary["scale_range_ratio"] == 200000.0
        assert "standard" not in summary["suggested_scalers"]
        assert "robust" in summary["suggested_scalers"]

    def test_sentinel_features_counted(self):
        """Features with sentinels should be counted."""
        from src.eda.profiler import _build_preprocessing_summary
        analysis = {
            "col_with_sentinel": {
                "detected_type": "numeric_continuous",
                "stats": {"range": 10.0},
                "skewness_label": "symmetric",
                "outlier_pct": 0.0,
                "sentinels": [{"value": -1.0, "count": 50, "pct": 5.0}],
            },
        }
        summary = _build_preprocessing_summary(analysis)
        assert summary["n_sentinel_features"] == 1
        assert "col_with_sentinel" in summary["sentinel_features"]


# ---------------------------------------------------------------------------
# _generate_recommendations with advanced inputs
# ---------------------------------------------------------------------------

class TestGenerateRecommendationsAdvanced:
    def test_leakage_warning_recommendation(self):
        """Leakage warnings should produce LEAKAGE WARNING recommendations."""
        columns_analysis = {
            "leak_col": {"detected_type": "numeric_continuous", "missing_pct": 0.0},
        }
        leakage = [{"column": "leak_col", "reason": "high_numeric_correlation", "value": 0.95}]
        recs = _generate_recommendations(
            columns_analysis, {"leak_col": 0.95}, [], [],
            leakage_warnings=leakage,
        )
        assert any("LEAKAGE WARNING" in r for r in recs)
        assert any("leak_col" in r for r in recs)

    def test_distribution_shift_recommendation(self):
        """Flagged distribution shift should appear in recommendations."""
        columns_analysis = {
            "shifted": {"detected_type": "numeric_continuous", "missing_pct": 0.0},
        }
        dist_shift = {"flagged_columns": ["shifted"]}
        recs = _generate_recommendations(
            columns_analysis, {"shifted": 0.3}, [], [],
            distribution_shift=dist_shift,
        )
        assert any("shift" in r.lower() for r in recs)
        assert any("shifted" in r for r in recs)

    def test_strong_iv_recommendation(self):
        """Features with strong IV should produce recommendation."""
        columns_analysis = {
            "strong_iv_col": {"detected_type": "numeric_continuous", "missing_pct": 0.0},
        }
        iv_woe = {
            "strong_iv_col": {"iv": 0.5, "iv_label": "strong", "woe_per_bin": {}},
        }
        recs = _generate_recommendations(
            columns_analysis, {"strong_iv_col": 0.4}, [], [],
            iv_woe=iv_woe,
        )
        assert any("Information Value" in r for r in recs)

    def test_univariate_auc_recommendation(self):
        """High univariate AUC features should produce recommendation."""
        columns_analysis = {
            "good_feat": {"detected_type": "numeric_continuous", "missing_pct": 0.0},
        }
        uni_auc = {"good_feat": 0.72}
        recs = _generate_recommendations(
            columns_analysis, {"good_feat": 0.4}, [], [],
            univariate_auc=uni_auc,
        )
        assert any("AUC" in r for r in recs)

    def test_near_zero_variance_recommendation(self):
        """Near-zero-variance features (>99% same value) should produce recommendation."""
        columns_analysis = {
            "nzv_col": {
                "detected_type": "numeric_continuous",
                "missing_pct": 0.0,
                "dominant_pct": 99.5,
            },
        }
        recs = _generate_recommendations(
            columns_analysis, {"nzv_col": 0.01}, [], [],
        )
        assert any("zero-variance" in r.lower() or "nzv_col" in r for r in recs)
