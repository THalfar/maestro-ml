"""
Feature Engineering — Part of Layer 3 of the Maestro pipeline.

Dynamically creates features based on the strategy YAML from Layer 2.
All feature operations are defined in YAML, not hardcoded. The strategy
specifies which interactions, ratios, target encodings, and custom
features to create.

IMPORTANT: Functions in this module return new DataFrames — they do NOT
modify the input DataFrames in-place.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    strategy: dict,
    cv_folds: StratifiedKFold | KFold | None = None,
    target_col: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all features specified in the strategy dictionary.

    This is the main entry point for feature engineering. It applies
    all feature transformations in order: interactions, ratios, target
    encoding, and custom features.

    Args:
        train: Training DataFrame (must include target column).
        test: Test DataFrame.
        strategy: Strategy dictionary with a 'features' key containing:
                  - interactions: list of [col_a, col_b] pairs
                  - ratios: list of [numerator, denominator] pairs
                  - target_encoding: {columns, pairs, alpha}
                  - custom: list of {name, formula} dicts
        cv_folds: Sklearn CV splitter for target encoding. Must be the
                  same splitter used for model training to prevent leakage.
        target_col: Name of the target column in train.

    Returns:
        Tuple of (train_featured, test_featured) DataFrames with new
        columns appended. Original columns are preserved.

    Steps:
        1. Make copies of train and test to avoid in-place modification.
        2. Extract features config from strategy['features'].
        3. If interactions exist, call _add_interactions on both DataFrames.
        4. If ratios exist, call _add_ratios on both DataFrames.
        5. If target_encoding config exists, call _add_target_encoding
           (this uses cv_folds to prevent leakage).
        6. If custom features exist, call _add_custom_features on both.
        7. Return the modified (train, test) tuple.
    """
    raise NotImplementedError


def _add_interactions(
    df: pd.DataFrame,
    pairs: list[list[str]],
) -> pd.DataFrame:
    """Add interaction features (column products) to the DataFrame.

    Args:
        df: Input DataFrame.
        pairs: List of [col_a, col_b] pairs. Each pair produces a new
               column named '{col_a}__x__{col_b}' = col_a * col_b.

    Returns:
        New DataFrame with interaction columns appended.

    Steps:
        1. Copy the input DataFrame.
        2. For each [col_a, col_b] pair:
           a. Verify both columns exist in df.
           b. Create new column: df[f'{col_a}__x__{col_b}'] = df[col_a] * df[col_b]
        3. Return the modified DataFrame.
    """
    raise NotImplementedError


def _add_ratios(
    df: pd.DataFrame,
    ratios: list[list[str]],
) -> pd.DataFrame:
    """Add ratio features (column divisions) to the DataFrame.

    Args:
        df: Input DataFrame.
        ratios: List of [numerator, denominator] pairs. Each produces a
                new column named '{num}__div__{den}' = num / (den + 1e-8).
                The epsilon prevents division by zero.

    Returns:
        New DataFrame with ratio columns appended.

    Steps:
        1. Copy the input DataFrame.
        2. For each [numerator, denominator] pair:
           a. Verify both columns exist in df.
           b. Create new column with epsilon for numerical stability:
              df[f'{num}__div__{den}'] = df[num] / (df[den] + 1e-8)
        3. Return the modified DataFrame.
    """
    raise NotImplementedError


def _add_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    pairs: list[list[str]],
    cv_folds: StratifiedKFold | KFold,
    target_col: str,
    alpha: float = 15.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add target-encoded features using cross-validation to prevent leakage.

    Target encoding replaces categorical values with smoothed target means.
    The smoothing formula is:
        encoded = (count * mean_target + alpha * global_mean) / (count + alpha)

    For training data, uses OOF (out-of-fold) encoding: each fold's values
    are computed from the other folds only. Test data uses the full
    training set statistics.

    Args:
        train: Training DataFrame (includes target column).
        test: Test DataFrame.
        columns: List of single column names to target-encode.
        pairs: List of [col_a, col_b] pairs to encode jointly
               (concatenated into a temporary column before encoding).
        cv_folds: Sklearn CV splitter — MUST be the same one used for
                  model training to prevent data leakage.
        target_col: Name of the target column.
        alpha: Smoothing parameter. Higher values regularize toward the
               global mean more strongly (default: 15.0).

    Returns:
        Tuple of (train_encoded, test_encoded) DataFrames with new
        target-encoded columns (named '{col}_te' or '{col_a}_{col_b}_te').

    Steps:
        1. Copy train and test DataFrames.
        2. Compute global_mean = train[target_col].mean().
        3. For each column in columns:
           a. Initialize train OOF target-encoded column with NaN.
           b. For each fold in cv_folds.split(train, train[target_col]):
              - Compute per-value target stats from training fold.
              - Apply smoothed encoding to validation fold.
           c. For test: compute stats from full train, apply encoding.
           d. Fill any remaining NaN with global_mean.
        4. For each pair in pairs:
           a. Create temporary concatenated column in both DataFrames.
           b. Apply the same OOF encoding logic as single columns.
           c. Drop the temporary concatenated column.
        5. Return (train_encoded, test_encoded).
    """
    raise NotImplementedError


def _add_custom_features(
    df: pd.DataFrame,
    formulas: list[dict[str, str]],
) -> pd.DataFrame:
    """Add custom features defined by pandas eval expressions.

    Args:
        df: Input DataFrame.
        formulas: List of dicts with 'name' and 'formula' keys.
                  Example: {'name': 'bmi', 'formula': 'weight / height**2'}
                  Formulas are evaluated using pd.eval with the DataFrame
                  columns available as variables.

    Returns:
        New DataFrame with custom feature columns appended.

    Steps:
        1. Copy the input DataFrame.
        2. For each formula dict:
           a. Extract 'name' and 'formula'.
           b. Evaluate: df[name] = df.eval(formula).
           c. Handle any evaluation errors gracefully (log warning,
              skip the feature).
        3. Return the modified DataFrame.
    """
    raise NotImplementedError


def get_feature_columns(
    strategy: dict,
    original_columns: list[str],
    exclude: list[str] | None = None,
) -> list[str]:
    """Get the complete list of feature column names after engineering.

    Useful for knowing which columns to pass to model training after
    feature engineering is complete.

    Args:
        strategy: Strategy dictionary with 'features' key.
        original_columns: List of original DataFrame column names
                          (before feature engineering).
        exclude: Column names to exclude (e.g., target_col, id_col).

    Returns:
        Sorted list of all feature column names (original + engineered),
        excluding any columns in the exclude list.

    Steps:
        1. Start with original_columns.
        2. Remove any columns in the exclude list.
        3. Add interaction column names: '{col_a}__x__{col_b}'.
        4. Add ratio column names: '{num}__div__{den}'.
        5. Add target encoding column names: '{col}_te', '{a}_{b}_te'.
        6. Add custom feature names from strategy['features']['custom'].
        7. Return the sorted list.
    """
    raise NotImplementedError
