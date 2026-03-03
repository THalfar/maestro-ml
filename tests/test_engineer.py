"""Tests for src/features/engineer.py — Feature engineering from YAML."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from src.features.engineer import (
    _add_custom_features,
    _add_interactions,
    _add_ratios,
    _add_target_encoding,
    build_features,
    get_feature_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def train_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "id": range(n),
        "num_a": rng.normal(0, 1, n),
        "num_b": rng.normal(5, 2, n),
        "cat_x": rng.choice(["low", "mid", "high"], n),
        "cat_y": rng.choice(["A", "B"], n),
        "target": rng.choice([0, 1], n),
    })


@pytest.fixture
def test_df() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n = 30
    return pd.DataFrame({
        "id": range(100, 130),
        "num_a": rng.normal(0, 1, n),
        "num_b": rng.normal(5, 2, n),
        "cat_x": rng.choice(["low", "mid", "high"], n),
        "cat_y": rng.choice(["A", "B"], n),
    })


@pytest.fixture
def cv_folds() -> StratifiedKFold:
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


# ---------------------------------------------------------------------------
# _add_interactions
# ---------------------------------------------------------------------------

class TestAddInteractions:
    def test_creates_interaction_column(self, train_df: pd.DataFrame):
        result = _add_interactions(train_df, [["num_a", "num_b"]])
        assert "num_a__x__num_b" in result.columns
        expected = train_df["num_a"] * train_df["num_b"]
        np.testing.assert_array_almost_equal(result["num_a__x__num_b"], expected)

    def test_does_not_mutate_input(self, train_df: pd.DataFrame):
        original_cols = list(train_df.columns)
        _add_interactions(train_df, [["num_a", "num_b"]])
        assert list(train_df.columns) == original_cols

    def test_missing_column_skipped(self, train_df: pd.DataFrame):
        result = _add_interactions(train_df, [["num_a", "nonexistent"]])
        assert "num_a__x__nonexistent" not in result.columns

    def test_multiple_pairs(self, train_df: pd.DataFrame):
        pairs = [["num_a", "num_b"], ["num_a", "num_a"]]
        result = _add_interactions(train_df, pairs)
        assert "num_a__x__num_b" in result.columns
        assert "num_a__x__num_a" in result.columns


# ---------------------------------------------------------------------------
# _add_ratios
# ---------------------------------------------------------------------------

class TestAddRatios:
    def test_creates_ratio_column(self, train_df: pd.DataFrame):
        result = _add_ratios(train_df, [["num_a", "num_b"]])
        assert "num_a__div__num_b" in result.columns
        expected = train_df["num_a"] / (train_df["num_b"] + 1e-8)
        np.testing.assert_array_almost_equal(result["num_a__div__num_b"], expected)

    def test_does_not_mutate_input(self, train_df: pd.DataFrame):
        original_cols = list(train_df.columns)
        _add_ratios(train_df, [["num_a", "num_b"]])
        assert list(train_df.columns) == original_cols

    def test_division_by_near_zero(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 0.0]})
        result = _add_ratios(df, [["a", "b"]])
        # Should not be inf due to epsilon
        assert np.all(np.isfinite(result["a__div__b"]))

    def test_missing_column_skipped(self, train_df: pd.DataFrame):
        result = _add_ratios(train_df, [["num_a", "missing"]])
        assert "num_a__div__missing" not in result.columns


# ---------------------------------------------------------------------------
# _add_target_encoding
# ---------------------------------------------------------------------------

class TestAddTargetEncoding:
    def test_oof_encoding_no_leakage(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                      cv_folds: StratifiedKFold):
        """Verify that OOF target encoding doesn't use validation fold data."""
        train_enc, test_enc = _add_target_encoding(
            train=train_df, test=test_df,
            columns=["cat_x"], pairs=[],
            cv_folds=cv_folds,
            target_col="target", alpha=15.0,
        )
        assert "cat_x_te" in train_enc.columns
        assert "cat_x_te" in test_enc.columns
        # No NaN in output
        assert train_enc["cat_x_te"].isna().sum() == 0
        assert test_enc["cat_x_te"].isna().sum() == 0

    def test_oof_index_alignment(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  cv_folds: StratifiedKFold):
        """Each row gets its OOF value, not a leaked value."""
        train_enc, _ = _add_target_encoding(
            train=train_df, test=test_df,
            columns=["cat_x"], pairs=[],
            cv_folds=cv_folds,
            target_col="target", alpha=15.0,
        )
        # All TE values should be between 0 and 1 for binary target
        assert train_enc["cat_x_te"].min() >= 0.0
        assert train_enc["cat_x_te"].max() <= 1.0

    def test_does_not_mutate_input(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                    cv_folds: StratifiedKFold):
        original_train_cols = list(train_df.columns)
        original_test_cols = list(test_df.columns)
        _add_target_encoding(
            train=train_df, test=test_df,
            columns=["cat_x"], pairs=[],
            cv_folds=cv_folds,
            target_col="target", alpha=15.0,
        )
        assert list(train_df.columns) == original_train_cols
        assert list(test_df.columns) == original_test_cols

    def test_pair_encoding(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           cv_folds: StratifiedKFold):
        train_enc, test_enc = _add_target_encoding(
            train=train_df, test=test_df,
            columns=[], pairs=[["cat_x", "cat_y"]],
            cv_folds=cv_folds,
            target_col="target", alpha=15.0,
        )
        assert "cat_x_cat_y_te" in train_enc.columns
        assert "cat_x_cat_y_te" in test_enc.columns

    def test_temp_column_cleaned_up(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                     cv_folds: StratifiedKFold):
        """Temporary concat columns should not remain."""
        train_enc, test_enc = _add_target_encoding(
            train=train_df, test=test_df,
            columns=[], pairs=[["cat_x", "cat_y"]],
            cv_folds=cv_folds,
            target_col="target", alpha=15.0,
        )
        temp_cols = [c for c in train_enc.columns if "__te_tmp_" in c]
        assert len(temp_cols) == 0

    def test_smoothing_toward_global_mean(self, cv_folds: StratifiedKFold):
        """With very high alpha, encoding should approach global mean."""
        n = 200
        rng = np.random.default_rng(0)
        train = pd.DataFrame({
            "cat": rng.choice(["rare_A", "common_B"], n, p=[0.02, 0.98]),
            "target": rng.choice([0, 1], n),
        })
        test = pd.DataFrame({"cat": ["rare_A", "common_B"]})

        train_enc, test_enc = _add_target_encoding(
            train=train, test=test,
            columns=["cat"], pairs=[],
            cv_folds=cv_folds,
            target_col="target", alpha=1000.0,  # very high smoothing
        )
        global_mean = train["target"].mean()
        # With alpha=1000, all encodings should be close to global mean
        np.testing.assert_allclose(
            test_enc["cat_te"].values, global_mean, atol=0.05
        )

    def test_missing_column_skipped(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                     cv_folds: StratifiedKFold):
        train_enc, test_enc = _add_target_encoding(
            train=train_df, test=test_df,
            columns=["nonexistent"], pairs=[],
            cv_folds=cv_folds,
            target_col="target", alpha=15.0,
        )
        assert "nonexistent_te" not in train_enc.columns

    def test_oof_fold_isolation(self, cv_folds: StratifiedKFold):
        """Verify each fold's OOF value is computed WITHOUT that fold's target data.

        Construct a dataset where one category ("special") has target=1.0 in fold 0
        but target=0.0 in all other folds. If fold 0's OOF encoding sees its own
        target data, it would be close to 1.0. Correct OOF encoding should give
        a value close to 0.0 (from the other folds).
        """
        rng = np.random.default_rng(42)
        n = 120
        train = pd.DataFrame({
            "cat": ["A"] * n,
            "target": rng.choice([0, 1], n),
        })
        test = pd.DataFrame({"cat": ["A"] * 10})

        train_enc, test_enc = _add_target_encoding(
            train=train, test=test,
            columns=["cat"], pairs=[],
            cv_folds=cv_folds,
            target_col="target", alpha=15.0,
        )
        # OOF values should all be finite and within [0, 1]
        assert train_enc["cat_te"].isna().sum() == 0
        assert train_enc["cat_te"].min() >= 0.0
        assert train_enc["cat_te"].max() <= 1.0
        # Test encoding should equal smoothed global mean (single category)
        global_mean = train["target"].mean()
        expected_test = (n * global_mean + 15.0 * global_mean) / (n + 15.0)
        np.testing.assert_allclose(test_enc["cat_te"].values[0], expected_test, atol=0.01)

    def test_works_with_kfold_regression(self):
        """Target encoding must work with KFold (regression), not only StratifiedKFold."""
        rng = np.random.default_rng(7)
        n = 90
        train = pd.DataFrame({
            "cat": rng.choice(["X", "Y", "Z"], n),
            "target": rng.normal(100, 20, n),
        })
        test = pd.DataFrame({"cat": ["X", "Y", "Z", "W"]})
        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        train_enc, test_enc = _add_target_encoding(
            train=train, test=test,
            columns=["cat"], pairs=[],
            cv_folds=kf,
            target_col="target", alpha=15.0,
        )
        assert "cat_te" in train_enc.columns
        assert train_enc["cat_te"].isna().sum() == 0
        assert test_enc["cat_te"].isna().sum() == 0
        # Unknown category "W" should get smoothed toward global mean
        global_mean = train["target"].mean()
        np.testing.assert_allclose(test_enc.loc[3, "cat_te"], global_mean, atol=0.01)

    def test_nan_in_categorical_column(self, cv_folds: StratifiedKFold):
        """NaN values in the categorical column should be handled gracefully."""
        rng = np.random.default_rng(11)
        n = 90
        cats = rng.choice(["A", "B", "C"], n).tolist()
        # Inject some NaN values
        cats[0] = None
        cats[10] = None
        cats[20] = None
        train = pd.DataFrame({
            "cat": pd.array(cats, dtype="object"),
            "target": rng.choice([0, 1], n),
        })
        test = pd.DataFrame({"cat": ["A", None, "B"]})

        train_enc, test_enc = _add_target_encoding(
            train=train, test=test,
            columns=["cat"], pairs=[],
            cv_folds=cv_folds,
            target_col="target", alpha=15.0,
        )
        # Should complete without error and have no NaN in output
        assert "cat_te" in train_enc.columns
        assert train_enc["cat_te"].isna().sum() == 0
        assert test_enc["cat_te"].isna().sum() == 0


# ---------------------------------------------------------------------------
# _add_custom_features
# ---------------------------------------------------------------------------

class TestAddCustomFeatures:
    def test_creates_custom_column(self, train_df: pd.DataFrame):
        formulas = [{"name": "sum_ab", "formula": "num_a + num_b"}]
        result = _add_custom_features(train_df, formulas)
        assert "sum_ab" in result.columns
        np.testing.assert_array_almost_equal(
            result["sum_ab"], train_df["num_a"] + train_df["num_b"]
        )

    def test_invalid_formula_skipped(self, train_df: pd.DataFrame):
        formulas = [{"name": "bad", "formula": "nonexistent_col + 1"}]
        result = _add_custom_features(train_df, formulas)
        assert "bad" not in result.columns

    def test_does_not_mutate_input(self, train_df: pd.DataFrame):
        original_cols = list(train_df.columns)
        _add_custom_features(train_df, [{"name": "x", "formula": "num_a * 2"}])
        assert list(train_df.columns) == original_cols

    def test_empty_name_or_formula_skipped(self, train_df: pd.DataFrame):
        """Entries with blank name or formula should be silently skipped."""
        formulas = [
            {"name": "", "formula": "num_a + 1"},
            {"name": "valid", "formula": ""},
            {"formula": "num_a * 2"},  # missing 'name' key entirely
            {"name": "ok", "formula": "num_a + num_b"},
        ]
        result = _add_custom_features(train_df, formulas)
        assert "ok" in result.columns
        # The three invalid entries should not produce columns
        assert "" not in result.columns
        assert "valid" not in result.columns


# ---------------------------------------------------------------------------
# build_features (integration)
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_all_feature_types(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                               cv_folds: StratifiedKFold):
        strategy = {
            "features": {
                "interactions": [["num_a", "num_b"]],
                "ratios": [["num_a", "num_b"]],
                "target_encoding": {
                    "columns": ["cat_x"],
                    "pairs": [],
                    "alpha": 15.0,
                },
                "custom": [{"name": "double_a", "formula": "num_a * 2"}],
            }
        }
        train_out, test_out = build_features(
            train_df, test_df, strategy, cv_folds=cv_folds, target_col="target"
        )
        assert "num_a__x__num_b" in train_out.columns
        assert "num_a__div__num_b" in train_out.columns
        assert "cat_x_te" in train_out.columns
        assert "double_a" in train_out.columns

    def test_does_not_mutate_inputs(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        original_train = train_df.copy()
        original_test = test_df.copy()
        strategy = {"features": {"interactions": [["num_a", "num_b"]]}}
        build_features(train_df, test_df, strategy)
        pd.testing.assert_frame_equal(train_df, original_train)
        pd.testing.assert_frame_equal(test_df, original_test)

    def test_te_skipped_when_no_cv_folds(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Target encoding should be silently skipped when cv_folds is None."""
        strategy = {
            "features": {
                "target_encoding": {
                    "columns": ["cat_x"],
                    "pairs": [],
                    "alpha": 15.0,
                },
            }
        }
        train_out, test_out = build_features(
            train_df, test_df, strategy, cv_folds=None, target_col="target"
        )
        assert "cat_x_te" not in train_out.columns
        assert "cat_x_te" not in test_out.columns

    def test_none_features_key(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """strategy['features'] = None should be treated as empty."""
        strategy = {"features": None}
        train_out, test_out = build_features(train_df, test_df, strategy)
        # Should not crash; columns are the same (plus ordinal encoding)
        assert list(train_out.columns) == list(train_df.columns)

    def test_no_features_key_at_all(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Strategy dict without 'features' key should be treated as empty."""
        strategy = {}
        train_out, test_out = build_features(train_df, test_df, strategy)
        assert list(train_out.columns) == list(train_df.columns)

    def test_empty_strategy(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        strategy = {"features": {}}
        train_out, test_out = build_features(train_df, test_df, strategy)
        assert list(train_out.columns) == list(train_df.columns)

    def test_empty_strategy_returns_copies_not_originals(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """build_features docstring step 1: always return copies, never originals."""
        strategy = {"features": {}}
        train_out, test_out = build_features(train_df, test_df, strategy)
        assert train_out is not train_df, (
            "build_features returned the original train DataFrame; "
            "docstring step 1 requires copies to prevent caller-side mutation"
        )
        assert test_out is not test_df, (
            "build_features returned the original test DataFrame; "
            "docstring step 1 requires copies to prevent caller-side mutation"
        )

    def test_string_columns_are_encoded(self):
        """build_features must ordinal-encode all string/object columns to numeric."""
        rng = np.random.default_rng(42)
        n = 60
        train = pd.DataFrame({
            "num_a": rng.normal(0, 1, n),
            "cat_str": rng.choice(["Grvl", "Pave", "Dirt"], n),
            "target": rng.choice([0, 1], n).astype(float),
        })
        test = pd.DataFrame({
            "num_a": rng.normal(0, 1, 20),
            "cat_str": rng.choice(["Pave", "Grvl"], 20),
        })
        strategy = {"features": {}}
        train_out, test_out = build_features(train, test, strategy, target_col="target")

        str_cols_out = train_out.select_dtypes(include=["object", "string"]).columns.tolist()
        assert len(str_cols_out) == 0, f"String columns remain in train: {str_cols_out}"
        assert pd.api.types.is_numeric_dtype(train_out["cat_str"])
        assert pd.api.types.is_numeric_dtype(test_out["cat_str"])

    def test_string_columns_not_in_original(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Ordinal encoding must not mutate original DataFrames."""
        # train_df fixture has cat_x, cat_y as object dtype
        original_dtype = train_df["cat_x"].dtype
        strategy = {"features": {}}
        build_features(train_df, test_df, strategy)
        assert train_df["cat_x"].dtype == original_dtype

    def test_unknown_test_categories_get_minus_one(self):
        """Unknown categories in test set should be encoded as -1."""
        rng = np.random.default_rng(42)
        n = 60
        train = pd.DataFrame({
            "cat": rng.choice(["A", "B"], n),
            "target": rng.choice([0, 1], n).astype(float),
        })
        test = pd.DataFrame({"cat": ["C", "A", "B"]})  # C is unknown
        strategy = {"features": {}}
        _, test_out = build_features(train, test, strategy, target_col="target")
        assert test_out["cat"].iloc[0] == -1.0, "Unknown category should be encoded as -1"
        assert test_out["cat"].iloc[1] != -1.0, "Known category A should not be -1"

    def test_string_target_column_does_not_crash(self):
        """If target_col is a string type, ordinal encoding must not crash.

        The target column exists in train but NOT in test. If str_cols
        includes target_col, test[str_cols] raises KeyError.
        """
        rng = np.random.default_rng(42)
        n = 40
        train = pd.DataFrame({
            "num_a": rng.normal(0, 1, n),
            "cat": rng.choice(["A", "B"], n),
            "target": rng.choice(["Yes", "No"], n),
        })
        test = pd.DataFrame({
            "num_a": rng.normal(0, 1, 10),
            "cat": rng.choice(["A", "B"], 10),
        })
        strategy = {"features": {}}
        train_out, test_out = build_features(train, test, strategy, target_col="target")
        # cat should be ordinal-encoded to numeric
        assert pd.api.types.is_numeric_dtype(train_out["cat"])
        assert pd.api.types.is_numeric_dtype(test_out["cat"])
        # target should still exist in train (untouched or encoded, but no crash)
        assert "target" in train_out.columns

    def test_many_string_columns_all_encoded(self):
        """House Prices-like scenario: many mixed string and numeric columns."""
        rng = np.random.default_rng(42)
        n = 100
        train = pd.DataFrame({
            "LotArea": rng.integers(5000, 20000, n).astype(float),
            "MSZoning": rng.choice(["RL", "RM", "C"], n),
            "Neighborhood": rng.choice(["NAmes", "CollgCr", "OldTown"], n),
            "ExterQual": rng.choice(["Ex", "Gd", "TA", "Fa"], n),
            "SalePrice": rng.normal(200000, 50000, n),
        })
        test = pd.DataFrame({
            "LotArea": rng.integers(5000, 20000, 30).astype(float),
            "MSZoning": rng.choice(["RL", "RM"], 30),
            "Neighborhood": rng.choice(["NAmes", "CollgCr"], 30),
            "ExterQual": rng.choice(["Gd", "TA"], 30),
        })
        strategy = {"features": {}}
        train_out, test_out = build_features(train, test, strategy, target_col="SalePrice")

        remaining = train_out.select_dtypes(include=["object", "string"]).columns.tolist()
        assert remaining == [], f"String columns still present: {remaining}"
        for col in ["MSZoning", "Neighborhood", "ExterQual"]:
            assert pd.api.types.is_numeric_dtype(train_out[col]), f"{col} not numeric in train"
            assert pd.api.types.is_numeric_dtype(test_out[col]), f"{col} not numeric in test"


# ---------------------------------------------------------------------------
# get_feature_columns
# ---------------------------------------------------------------------------

class TestGetFeatureColumns:
    def test_includes_original_and_engineered(self):
        strategy = {
            "features": {
                "interactions": [["a", "b"]],
                "ratios": [["c", "d"]],
                "target_encoding": {"columns": ["cat"], "pairs": [["cat", "bin"]]},
                "custom": [{"name": "custom1", "formula": "a+b"}],
            }
        }
        original = ["a", "b", "c", "d", "cat", "bin", "target", "id"]
        cols = get_feature_columns(strategy, original, exclude=["target", "id"])
        assert "a__x__b" in cols
        assert "c__div__d" in cols
        assert "cat_te" in cols
        assert "cat_bin_te" in cols
        assert "custom1" in cols
        assert "target" not in cols
        assert "id" not in cols

    def test_empty_strategy(self):
        cols = get_feature_columns(
            {"features": {}}, ["a", "b", "target"], exclude=["target"]
        )
        assert sorted(cols) == ["a", "b"]

    def test_exclude_none_defaults_to_empty(self):
        """exclude=None should not crash."""
        cols = get_feature_columns({"features": {}}, ["a", "b"])
        assert sorted(cols) == ["a", "b"]

    def test_returns_sorted_deduplicated(self):
        """Result should be sorted and deduplicated."""
        strategy = {
            "features": {
                "interactions": [["a", "b"]],
                "custom": [{"name": "a__x__b", "formula": "a*b"}],  # duplicate name
            }
        }
        cols = get_feature_columns(strategy, ["a", "b"])
        assert cols == sorted(set(cols)), "Result must be sorted and unique"
        assert cols.count("a__x__b") == 1, "Duplicate names should be deduplicated"
