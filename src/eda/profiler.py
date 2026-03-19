"""
EDA Profiler — Layer 1 of the Maestro pipeline.

Performs automatic dataset profiling on raw CSV data. Produces a structured
report (matching eda_schema.yaml) that the LLM strategist consumes to make
informed decisions about feature engineering and model selection.

This module is pure pandas/numpy/scipy/sklearn. No randomness beyond MI
estimation (seeded). Fully deterministic given the same seed.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections import Counter, deque
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


def _add_skewness_and_outliers(
    columns_analysis: dict[str, dict],
    feature_df: pd.DataFrame,
) -> None:
    """Add skewness, kurtosis and outlier_pct to numeric column entries in-place.

    Args:
        columns_analysis: Per-column analysis dict (from _detect_column_types).
            Modified in-place — adds 'skewness', 'kurtosis' and 'outlier_pct'
            to each numeric column entry. Non-numeric columns get None for all.
        feature_df: Feature DataFrame (same as passed to _detect_column_types).
    """
    for col_name, col_info in columns_analysis.items():
        if col_info["stats"] is not None and col_name in feature_df.columns:
            series = feature_df[col_name].dropna()
            if len(series) > 0:
                skew_val = float(scipy_stats.skew(series, bias=False))
                kurt_val = float(scipy_stats.kurtosis(series, bias=False))
                q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
                    outlier_pct = round(float(outliers / len(series) * 100), 3)
                else:
                    outlier_pct = 0.0
                col_info["skewness"] = round(skew_val, 4)
                col_info["kurtosis"] = round(kurt_val, 4)
                col_info["outlier_pct"] = outlier_pct
                # Skewness label for LLM decision-making
                abs_skew = abs(skew_val)
                if abs_skew < 0.5:
                    col_info["skewness_label"] = "symmetric"
                elif abs_skew < 1.0:
                    col_info["skewness_label"] = "moderate"
                else:
                    col_info["skewness_label"] = "high"
            else:
                col_info["skewness"] = None
                col_info["kurtosis"] = None
                col_info["outlier_pct"] = None
                col_info["skewness_label"] = None
        else:
            col_info["skewness"] = None
            col_info["kurtosis"] = None
            col_info["outlier_pct"] = None
            col_info["skewness_label"] = None


def _detect_sentinels(
    columns_analysis: dict[str, dict],
    feature_df: pd.DataFrame,
) -> None:
    """Detect likely sentinel values (e.g. -1, -999) in numeric columns in-place.

    A sentinel is a value that is far from the rest of the distribution and
    appears frequently — likely a masked missing value. Detection criteria:
    - Value is at least 3 IQR below Q1 (or is exactly -1, -999, -9999, or >= 9999)
    - Value accounts for >= 1% of the column

    Args:
        columns_analysis: Per-column analysis dict. Modified in-place — adds
            'sentinels' (list of dicts with 'value', 'count', 'pct') to each
            numeric column. Non-numeric columns get an empty list.
        feature_df: Feature DataFrame.
    """
    common_sentinels = {-1, -999, -9999, 9999}

    for col_name, col_info in columns_analysis.items():
        col_info["sentinels"] = []
        if col_info["stats"] is None or col_name not in feature_df.columns:
            continue
        series = feature_df[col_name].dropna()
        if len(series) < 10:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        n = len(series)
        vc = series.value_counts()

        detected = []
        for val, count in vc.items():
            pct = count / n * 100
            if pct < 1.0:
                continue
            val_f = float(val)
            is_common = val_f in common_sentinels
            is_extreme = iqr > 0 and (val_f < q1 - 3 * iqr or val_f > q3 + 3 * iqr)
            if is_common or is_extreme:
                detected.append({
                    "value": val_f,
                    "count": int(count),
                    "pct": round(pct, 2),
                })
        if detected:
            detected.sort(key=lambda x: x["count"], reverse=True)
            col_info["sentinels"] = detected


def _compute_univariate_auc(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
) -> dict[str, float]:
    """Compute univariate AUC for each feature against a binary target.

    For each feature, computes AUC using the raw feature values as scores.
    Non-numeric features are label-encoded first. AUC is flipped to always
    be >= 0.5 (i.e. ``max(auc, 1 - auc)``).

    Args:
        feature_df: Feature columns (no target, no id).
        target_series: Binary target (0/1).

    Returns:
        Dict {column_name: auc_score} sorted descending. Empty dict if
        target is not binary or inputs are empty.
    """
    if feature_df.empty or len(target_series) == 0:
        return {}
    unique_vals = target_series.dropna().unique()
    if len(unique_vals) != 2:
        return {}

    y = target_series.values
    result: dict[str, float] = {}

    for col in feature_df.columns:
        series = feature_df[col]
        if not pd.api.types.is_numeric_dtype(series):
            codes, _ = pd.factorize(series, sort=True)
            arr = codes.astype(float)
        else:
            arr = series.fillna(series.median()).values.astype(float)
        # Handle all-NaN → all zeros
        arr = np.nan_to_num(arr, nan=0.0)

        mask = ~np.isnan(y)
        y_clean, arr_clean = y[mask], arr[mask]
        if len(np.unique(y_clean)) < 2:
            result[col] = 0.5
            continue

        try:
            auc = roc_auc_score(y_clean, arr_clean)
            auc = max(auc, 1.0 - auc)  # flip if inverted
        except ValueError:
            auc = 0.5
        result[col] = round(float(auc), 6)

    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def _compute_iv_woe(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    columns_analysis: dict[str, dict],
    n_bins: int = 10,
) -> dict[str, dict]:
    """Compute Information Value and Weight of Evidence for each feature.

    Numeric features are binned into ``n_bins`` equal-frequency bins.
    Categorical features use their natural categories. Only computed for
    binary targets.

    IV interpretation: <0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, >0.3 strong.

    Args:
        feature_df: Feature columns (no target, no id).
        target_series: Binary target (0/1).
        columns_analysis: Per-column type info from _detect_column_types.
        n_bins: Number of bins for numeric features.

    Returns:
        Dict keyed by column name, each value has:
        - 'iv': float — total Information Value
        - 'iv_label': str — 'useless'/'weak'/'medium'/'strong'
        - 'woe_per_bin': dict {bin_label: {'woe': float, 'count': int, 'event_rate': float}}
        Empty dict if target is not binary.
    """
    if feature_df.empty or len(target_series) == 0:
        return {}
    unique_vals = target_series.dropna().unique()
    if len(unique_vals) != 2:
        return {}

    y = target_series.values
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {}

    result: dict[str, dict] = {}

    for col in feature_df.columns:
        series = feature_df[col]
        col_info = columns_analysis.get(col, {})
        is_numeric = col_info.get("stats") is not None and pd.api.types.is_numeric_dtype(series)

        # Bin the feature
        if is_numeric:
            try:
                binned = pd.qcut(series.fillna(series.median()), q=n_bins, duplicates="drop")
            except (ValueError, TypeError):
                # All same value, very low variance, or non-numeric dtype
                continue
        else:
            binned = series.fillna("__MISSING__")

        # Group and compute
        df_temp = pd.DataFrame({"bin": binned, "target": y})
        grouped = df_temp.groupby("bin", observed=True)["target"]
        stats = grouped.agg(["sum", "count"])
        stats.columns = ["events", "total"]
        stats["non_events"] = stats["total"] - stats["events"]

        # Skip single-bin results (no discriminative power)
        if len(stats) < 2:
            continue

        # Use Laplace smoothing: add 0.5 to avoid log(0)
        pct_pos = (stats["events"] + 0.5) / (n_pos + 0.5 * len(stats))
        pct_neg = (stats["non_events"] + 0.5) / (n_neg + 0.5 * len(stats))

        woe = np.log(pct_pos / pct_neg)
        iv_per_bin = (pct_pos - pct_neg) * woe
        total_iv = float(iv_per_bin.sum())

        if total_iv < 0.02:
            iv_label = "useless"
        elif total_iv < 0.1:
            iv_label = "weak"
        elif total_iv < 0.3:
            iv_label = "medium"
        else:
            iv_label = "strong"

        woe_detail: dict[str, dict] = {}
        for idx, row in stats.iterrows():
            bin_label = str(idx)
            event_rate = float(row["events"] / row["total"]) if row["total"] > 0 else 0.0
            woe_detail[bin_label] = {
                "woe": round(float(woe.loc[idx]), 4),
                "count": int(row["total"]),
                "event_rate": round(event_rate, 4),
            }

        result[col] = {
            "iv": round(total_iv, 4),
            "iv_label": iv_label,
            "woe_per_bin": woe_detail,
        }

    return dict(sorted(result.items(), key=lambda x: x[1]["iv"], reverse=True))


def _compute_cramers_v(
    feature_df: pd.DataFrame,
    columns_analysis: dict[str, dict],
) -> dict[str, dict]:
    """Compute Cramér's V for all pairs of categorical features.

    Cramér's V is based on chi-square and measures association between
    two categorical variables, scaled to [0, 1].

    Args:
        feature_df: Feature columns (no target, no id).
        columns_analysis: Per-column type info from _detect_column_types.

    Returns:
        Dict with:
        - 'pairs': list of {'features': [a, b], 'cramers_v': float}
          sorted by cramers_v descending, only pairs with V > 0.1.
        - 'matrix': {'columns': [...], 'values': [[...]]}
          Full pairwise Cramér's V matrix for categorical features.
    """
    cat_types = {"binary", "low_cardinality_categorical", "ordinal"}
    cat_cols = [
        col for col, info in columns_analysis.items()
        if info["detected_type"] in cat_types
        and col in feature_df.columns
        and info.get("cardinality", 0) >= 2
    ]

    if len(cat_cols) < 2:
        return {"pairs": [], "matrix": {"columns": [], "values": []}}

    n = len(cat_cols)
    v_matrix = np.zeros((n, n))

    for i in range(n):
        v_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            contingency = pd.crosstab(feature_df[cat_cols[i]], feature_df[cat_cols[j]])
            chi2 = scipy_stats.chi2_contingency(contingency)[0]
            n_obs = contingency.values.sum()
            min_dim = min(contingency.shape) - 1
            if min_dim > 0 and n_obs > 0:
                v = float(np.sqrt(chi2 / (n_obs * min_dim)))
            else:
                v = 0.0
            v_matrix[i, j] = v
            v_matrix[j, i] = v

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            v_val = v_matrix[i, j]
            if v_val > 0.1:
                pairs.append({
                    "features": [cat_cols[i], cat_cols[j]],
                    "cramers_v": round(float(v_val), 4),
                })
    pairs.sort(key=lambda x: x["cramers_v"], reverse=True)

    matrix = {
        "columns": cat_cols,
        "values": [[round(float(v_matrix[i, j]), 4) for j in range(n)] for i in range(n)],
    }

    return {"pairs": pairs, "matrix": matrix}


def _compute_mutual_information(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    task_type: str,
) -> dict[str, float]:
    """Compute mutual information between each feature and target.

    Uses sklearn mutual_info_classif or mutual_info_regression depending
    on task_type. Categorical columns are label-encoded before computation.

    Args:
        feature_df: Feature columns (no target, no id).
        target_series: Numeric target series.
        task_type: 'binary_classification'/'multiclass' → classif;
                   'regression' → regression variant.

    Returns:
        Dict {column_name: mi_score} sorted by mi_score descending.
    """
    if feature_df.empty or len(target_series) == 0:
        return {}

    cols = list(feature_df.columns)
    discrete_mask = np.zeros(len(cols), dtype=bool)
    arrays: list[np.ndarray] = []

    for i, col in enumerate(cols):
        series = feature_df[col]
        if not pd.api.types.is_numeric_dtype(series):
            codes, _ = pd.factorize(series, sort=False)  # NaN → -1
            arrays.append(codes.astype(float))
            discrete_mask[i] = True
        else:
            arr = series.fillna(series.median()).values.astype(float)
            arrays.append(arr)

    X_arr = np.column_stack(arrays) if arrays else np.empty((len(feature_df), 0))
    # Handle all-NaN columns whose median was NaN
    X_arr = np.nan_to_num(X_arr, nan=0.0)

    y = target_series.values
    # Auto-detect: if task_type says classification but target is continuous, fall back
    mi_func = mutual_info_classif if task_type != "regression" else mutual_info_regression
    try:
        mi_values = mi_func(X_arr, y, discrete_features=discrete_mask, random_state=42)
    except ValueError:
        # Classification MI on continuous target — fall back to regression MI
        mi_values = mutual_info_regression(
            X_arr, y, discrete_features=discrete_mask, random_state=42
        )

    result = {col: round(float(mi), 6) for col, mi in zip(cols, mi_values)}
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def _compute_categorical_target_rates(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    columns_analysis: dict[str, dict],
    task_type: str,
) -> dict[str, dict]:
    """Compute per-category target rate for each categorical/binary column.

    Args:
        feature_df: Feature columns (no target, no id).
        target_series: Numeric target series.
        columns_analysis: Per-column type info (to check detected_type).
        task_type: Task type string (unused for now — mean works for both).

    Returns:
        Dict keyed by column name. Each value has:
        - 'target_rate_per_value': {category: {'rate': float, 'count': int}}
        - 'target_rate_max_delta': float (max - min rate across categories)
    """
    cat_types = {"binary", "low_cardinality_categorical", "ordinal"}
    result: dict[str, dict] = {}

    for col_name, col_info in columns_analysis.items():
        if col_info["detected_type"] not in cat_types:
            continue
        if col_info["cardinality"] > 50 or col_info["cardinality"] < 2:
            continue
        if col_name not in feature_df.columns:
            continue

        grouped = pd.DataFrame({"feature": feature_df[col_name], "target": target_series})
        stats = grouped.groupby("feature")["target"].agg(["mean", "count"])
        # Filter groups with very few samples
        stats = stats[stats["count"] >= 5]

        if len(stats) < 2:
            continue

        rates = {
            str(idx): {"rate": round(float(row["mean"]), 4), "count": int(row["count"])}
            for idx, row in stats.iterrows()
        }
        max_delta = round(float(stats["mean"].max() - stats["mean"].min()), 4)

        result[col_name] = {
            "target_rate_per_value": rates,
            "target_rate_max_delta": max_delta,
        }

    return result


def _compute_psi_numeric(
    train_vals: pd.Series,
    test_vals: pd.Series,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index for a single numeric feature.

    Bins are defined by train quantiles, then proportions in each bin are
    compared between train and test.

    PSI interpretation: <0.1 stable, 0.1–0.25 moderate shift, >0.25 significant.

    Args:
        train_vals: Non-null numeric values from train.
        test_vals: Non-null numeric values from test.
        n_bins: Number of quantile bins.

    Returns:
        PSI value (float >= 0).
    """
    # Create bin edges from train quantiles
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(train_vals, quantiles)
    bin_edges = np.unique(bin_edges)  # deduplicate for low-variance features
    if len(bin_edges) < 2:
        return 0.0

    train_counts = np.histogram(train_vals, bins=bin_edges)[0].astype(float)
    test_counts = np.histogram(test_vals, bins=bin_edges)[0].astype(float)

    train_sum = train_counts.sum()
    test_sum = test_counts.sum()
    if train_sum == 0 or test_sum == 0:
        return 0.0

    # Proportions with small epsilon to avoid division by zero
    train_pct = train_counts / train_sum
    test_pct = test_counts / test_sum
    train_pct = np.maximum(train_pct, 1e-4)
    test_pct = np.maximum(test_pct, 1e-4)

    psi = float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))
    return max(psi, 0.0)


def _compute_distribution_shift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns_analysis: dict[str, dict],
) -> dict[str, Any]:
    """Compare train and test feature distributions to detect covariate shift.

    Args:
        train_df: Feature columns from train.
        test_df: Feature columns from test.
        columns_analysis: Per-column type info.

    Returns:
        Dict with 'numeric', 'categorical', and 'flagged_columns' keys.
    """
    numeric_shifts: dict[str, dict] = {}
    categorical_shifts: dict[str, dict] = {}
    flagged: list[str] = []

    for col_name, col_info in columns_analysis.items():
        if col_name not in test_df.columns:
            continue

        if col_info["stats"] is not None:
            # Numeric: KS test + PSI
            train_vals = train_df[col_name].dropna()
            test_vals = test_df[col_name].dropna()
            if len(train_vals) > 0 and len(test_vals) > 0:
                ks_stat, ks_pval = scipy_stats.ks_2samp(train_vals, test_vals)
                # PSI: bin using train quantiles, compare proportions
                psi_val = _compute_psi_numeric(train_vals, test_vals)
                shift_flagged = bool(ks_stat > 0.1 and ks_pval < 0.05) or psi_val > 0.25
                numeric_shifts[col_name] = {
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_pvalue": round(float(ks_pval), 6),
                    "psi": round(float(psi_val), 4),
                    "shift_flagged": shift_flagged,
                }
                if shift_flagged:
                    flagged.append(col_name)
        else:
            # Categorical: proportion delta + PSI
            train_props = train_df[col_name].value_counts(normalize=True)
            test_props = test_df[col_name].value_counts(normalize=True)
            all_cats = set(train_props.index) | set(test_props.index)
            max_delta = 0.0
            psi_val = 0.0
            for cat in all_cats:
                p_train = max(train_props.get(cat, 0.0), 1e-4)
                p_test = max(test_props.get(cat, 0.0), 1e-4)
                delta = abs(float(p_train) - float(p_test))
                max_delta = max(max_delta, delta)
                psi_val += float((p_test - p_train) * np.log(p_test / p_train))
            psi_val = max(psi_val, 0.0)
            shift_flagged = bool(max_delta > 0.1) or psi_val > 0.25
            categorical_shifts[col_name] = {
                "max_proportion_delta": round(float(max_delta), 4),
                "psi": round(float(psi_val), 4),
                "shift_flagged": shift_flagged,
            }
            if shift_flagged:
                flagged.append(col_name)

    return {
        "numeric": numeric_shifts,
        "categorical": categorical_shifts,
        "flagged_columns": flagged,
    }


def _enrich_clusters_with_pairs(
    feature_clusters: list[dict[str, Any]],
    correlation_matrix: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add pairwise correlation details to each feature cluster.

    Args:
        feature_clusters: Output from _find_feature_clusters.
        correlation_matrix: Full pairwise matrix from _compute_correlations.

    Returns:
        New list of cluster dicts with added 'pairs' key.
    """
    columns = correlation_matrix.get("columns", [])
    values = correlation_matrix.get("values", [])
    if not columns or not values:
        return feature_clusters

    col_idx = {col: i for i, col in enumerate(columns)}
    mat = np.array(values)

    enriched = []
    for cluster in feature_clusters:
        new_cluster = dict(cluster)
        pairs = []
        for a, b in itertools.combinations(cluster["features"], 2):
            if a in col_idx and b in col_idx:
                corr_val = float(mat[col_idx[a], col_idx[b]])
                pairs.append({
                    "features": [a, b],
                    "correlation": round(corr_val, 4),
                })
        pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
        new_cluster["pairs"] = pairs
        enriched.append(new_cluster)

    return enriched


def _screen_interactions(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    mi_scores: dict[str, float],
    columns_analysis: dict[str, dict],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Screen pairwise feature interactions for target predictive value.

    Takes the top-N features by MI score and tests whether their pairwise
    products correlate with target better than either feature alone.

    Args:
        feature_df: Feature columns (no target, no id).
        target_series: Numeric target series.
        mi_scores: Dict {col: mi_score} from _compute_mutual_information.
        columns_analysis: Per-column type info.
        top_n: Maximum features to consider.

    Returns:
        List of interaction candidates sorted by added_value descending.
    """
    # Select top_n numeric/binary features by MI
    numeric_types = {"numeric_continuous", "binary", "ordinal"}
    candidates = [
        col for col in mi_scores
        if col in columns_analysis
        and columns_analysis[col]["detected_type"] in numeric_types
        and col in feature_df.columns
    ][:top_n]

    if len(candidates) < 2:
        return []

    # Pre-compute individual |corr| with target
    target_arr = target_series.values.astype(float)
    ind_corrs: dict[str, float] = {}
    col_arrays: dict[str, np.ndarray] = {}
    for col in candidates:
        if not pd.api.types.is_numeric_dtype(feature_df[col]):
            continue
        arr = feature_df[col].fillna(feature_df[col].median()).values.astype(float)
        col_arrays[col] = arr
        corr_mat = np.corrcoef(arr, target_arr)
        ind_corrs[col] = abs(float(corr_mat[0, 1])) if not np.isnan(corr_mat[0, 1]) else 0.0

    results = []
    numeric_candidates = list(col_arrays.keys())
    for a, b in itertools.combinations(numeric_candidates, 2):
        product = col_arrays[a] * col_arrays[b]
        corr_mat = np.corrcoef(product, target_arr)
        interaction_corr = abs(float(corr_mat[0, 1])) if not np.isnan(corr_mat[0, 1]) else 0.0
        individual_max = max(ind_corrs[a], ind_corrs[b])
        added_value = interaction_corr - individual_max

        if added_value > 0.01:
            results.append({
                "features": [a, b],
                "interaction_corr": round(interaction_corr, 4),
                "individual_max_corr": round(individual_max, 4),
                "added_value": round(added_value, 4),
            })

    results.sort(key=lambda x: x["added_value"], reverse=True)
    return results[:20]


def _detect_leakage(
    columns_analysis: dict[str, dict],
    mi_scores: dict[str, float],
    target_correlations: dict[str, float],
) -> list[dict[str, Any]]:
    """Flag potential target leakage based on suspiciously high predictive power.

    Args:
        columns_analysis: Per-column analysis (with target_rate_max_delta).
        mi_scores: Dict {col: mi_score}.
        target_correlations: Dict {col: pearson_corr}.

    Returns:
        List of warning dicts with 'column', 'reason', and 'value' keys.
    """
    warnings: list[dict[str, Any]] = []

    for col, corr in target_correlations.items():
        if abs(corr) > 0.7:
            warnings.append({
                "column": col,
                "reason": "high_numeric_correlation",
                "value": round(abs(corr), 4),
            })

    for col, col_info in columns_analysis.items():
        delta = col_info.get("target_rate_max_delta")
        if delta is not None and delta > 0.6:
            warnings.append({
                "column": col,
                "reason": "high_categorical_delta",
                "value": round(delta, 4),
            })

    for col, mi in mi_scores.items():
        if mi > 0.5:
            # Avoid duplicate if already flagged by correlation
            already_flagged = any(w["column"] == col for w in warnings)
            if not already_flagged:
                warnings.append({
                    "column": col,
                    "reason": "high_mutual_information",
                    "value": round(mi, 4),
                })

    return warnings


def _compute_vif(
    feature_df: pd.DataFrame,
    columns_analysis: dict[str, dict],
) -> dict[str, float]:
    """Compute Variance Inflation Factor for numeric features.

    Uses manual computation (no statsmodels): sklearn LinearRegression → R²
    → VIF = 1 / (1 - R²).

    Args:
        feature_df: Feature DataFrame.
        columns_analysis: Per-column type info.

    Returns:
        Dict {column_name: vif_score} sorted by VIF descending.
    """
    numeric_cols = [
        col for col, info in columns_analysis.items()
        if info["stats"] is not None and col in feature_df.columns
    ]

    if len(numeric_cols) < 2:
        return {}

    # Drop rows with any NaN in numeric columns
    numeric_data = feature_df[numeric_cols].dropna()
    if len(numeric_data) < 10:
        return {}

    X = numeric_data.values
    vif_scores: dict[str, float] = {}

    # Fast path: invert the correlation matrix once — O(n_features³) vs
    # O(n_features × n_samples × n_features) for the iterative OLS approach.
    # Identity: VIF_i = (C⁻¹)_{ii} where C is the Pearson correlation matrix.
    corr = np.corrcoef(X.T)
    try:
        inv_corr = np.linalg.inv(corr)
        if not np.isfinite(inv_corr).all():
            raise np.linalg.LinAlgError("Non-finite values in inverse")
        for i, col in enumerate(numeric_cols):
            vif = min(float(inv_corr[i, i]), 1000.0)
            vif_scores[col] = round(vif, 2)
    except np.linalg.LinAlgError:
        # Singular correlation matrix (perfect multicollinearity): fall back to OLS
        for i, col in enumerate(numeric_cols):
            y_i = X[:, i]
            X_others = np.delete(X, i, axis=1)
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X_others, y_i)
            r_squared = reg.score(X_others, y_i)
            vif = min(1.0 / (1.0 - r_squared + 1e-10), 1000.0)
            vif_scores[col] = round(vif, 2)

    return dict(sorted(vif_scores.items(), key=lambda x: x[1], reverse=True))


def _count_duplicates(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
) -> dict[str, Any]:
    """Count duplicate and conflicting rows for a given feature set.

    Helper used by _detect_duplicates for both full and signal-only passes.

    Args:
        feature_df: Feature columns to check.
        target_series: Target series (same length).

    Returns:
        Dict with n_duplicate_rows, duplicate_pct, n_conflicting_rows,
        conflicting_pct, n_duplicate_groups, n_conflicting_groups.
    """
    n_total = len(feature_df)
    if n_total == 0 or feature_df.shape[1] == 0:
        return {
            "n_duplicate_rows": 0, "duplicate_pct": 0.0,
            "n_conflicting_rows": 0, "conflicting_pct": 0.0,
            "n_duplicate_groups": 0, "n_conflicting_groups": 0,
        }

    row_hash = pd.util.hash_pandas_object(feature_df, index=False)
    hash_counts = row_hash.value_counts()
    dup_hashes = hash_counts[hash_counts > 1]

    n_dup_groups = len(dup_hashes)
    dup_mask = row_hash.isin(dup_hashes.index)
    n_dup_rows = int(dup_mask.sum())

    n_conflict_rows = 0
    n_conflict_groups = 0
    if n_dup_rows > 0:
        combined = pd.DataFrame({"hash": row_hash, "target": target_series.values})
        dup_combined = combined[dup_mask]
        target_per_hash = dup_combined.groupby("hash")["target"].nunique()
        conflict_hashes = target_per_hash[target_per_hash > 1]
        n_conflict_groups = len(conflict_hashes)
        if n_conflict_groups > 0:
            conflict_mask = row_hash.isin(conflict_hashes.index)
            n_conflict_rows = int(conflict_mask.sum())

    return {
        "n_duplicate_rows": n_dup_rows,
        "duplicate_pct": round(n_dup_rows / n_total * 100, 2),
        "n_conflicting_rows": n_conflict_rows,
        "conflicting_pct": round(n_conflict_rows / n_total * 100, 2),
        "n_duplicate_groups": n_dup_groups,
        "n_conflicting_groups": n_conflict_groups,
    }


def _detect_duplicates(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    columns_analysis: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Detect duplicate and conflicting rows in training data.

    Performs two passes:
    1. **Full pass**: all feature columns — exact duplicates.
    2. **Signal-only pass**: excludes noise columns (IV=0 AND AUC~0.5,
       or near-zero-variance >99%). This catches hidden duplicates masked by
       noise features (e.g., Porto Seguro's ps_calc_* columns).

    Uses hash-based approach for speed (avoids groupby on many columns).

    Args:
        feature_df: Feature columns (no target, no id).
        target_series: Target series.
        columns_analysis: Per-column analysis dict from run_eda. If provided,
            enables the signal-only pass. Keys used: ``iv``, ``dominant_pct``,
            ``mutual_information``, ``univariate_auc``.

    Returns:
        Dict with full-pass keys (n_duplicate_rows, duplicate_pct, etc.)
        and optional signal_only sub-dict with same keys + dropped_columns list.
    """
    result = _count_duplicates(feature_df, target_series)

    # Signal-only pass: drop noise columns
    if columns_analysis:
        noise_cols: list[str] = []
        for col, info in columns_analysis.items():
            if col not in feature_df.columns:
                continue
            dom_pct = info.get("dominant_pct", 0) or 0
            iv = info.get("iv")
            mi = info.get("mutual_information", 0) or 0
            auc = info.get("univariate_auc")

            # Near-zero-variance: >99% same value
            if dom_pct > 99.0:
                noise_cols.append(col)
                continue

            # Near-zero IV + AUC ~0.5 → pure noise
            # IV has small estimation noise on large N (ps_calc_* get IV≈0.0001–0.001).
            # Threshold IV < 0.001 catches noise but preserves weak-signal features.
            if iv is not None and iv < 0.001:
                if auc is not None and abs(auc - 0.5) < 0.01:
                    noise_cols.append(col)
                    continue

        if noise_cols:
            signal_cols = [c for c in feature_df.columns if c not in noise_cols]
            if signal_cols and len(signal_cols) < len(feature_df.columns):
                signal_df = feature_df[signal_cols]
                signal_result = _count_duplicates(signal_df, target_series)
                signal_result["dropped_columns"] = sorted(noise_cols)
                signal_result["n_signal_features"] = len(signal_cols)
                result["signal_only"] = signal_result

    return result


def _compute_unseen_categories(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns_analysis: dict[str, dict],
) -> dict[str, dict[str, Any]]:
    """Find categorical values in test that don't appear in train.

    Unseen categories get only the global prior in target encoding, so high
    unseen counts signal the need for higher smoothing alpha or frequency-based
    encoding fallback.

    Args:
        train_df: Feature columns from train.
        test_df: Feature columns from test.
        columns_analysis: Per-column type info.

    Returns:
        Dict keyed by column name (only columns with unseen values) with:
        - n_unseen: count of unique values in test not in train
        - n_test_unique: total unique values in test
        - unseen_pct: n_unseen / n_test_unique * 100
        - unseen_row_pct: percentage of test rows containing unseen values
        - unseen_values: list of unseen values (up to 20)
    """
    cat_types = {"low_cardinality_categorical", "high_cardinality_categorical",
                 "ordinal", "binary"}
    result: dict[str, dict[str, Any]] = {}

    for col, info in columns_analysis.items():
        if info["detected_type"] not in cat_types:
            continue
        if col not in train_df.columns or col not in test_df.columns:
            continue

        train_vals = set(train_df[col].dropna().unique())
        test_vals = set(test_df[col].dropna().unique())
        unseen = test_vals - train_vals

        if not unseen:
            continue

        n_test_unique = len(test_vals)
        test_series = test_df[col].dropna()
        unseen_row_count = int(test_series.isin(unseen).sum())
        unseen_row_pct = (
            round(float(unseen_row_count) / len(test_series) * 100, 2)
            if len(test_series) > 0 else 0.0
        )

        result[col] = {
            "n_unseen": len(unseen),
            "n_test_unique": n_test_unique,
            "unseen_pct": round(len(unseen) / n_test_unique * 100, 1) if n_test_unique > 0 else 0.0,
            "unseen_row_pct": unseen_row_pct,
            "unseen_values": sorted([str(v) for v in unseen])[:20],
        }

    return result


def _detect_monotonicity(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    columns_analysis: dict[str, dict],
    n_bins: int = 10,
) -> dict[str, dict[str, Any]]:
    """Detect monotonic relationships between features and target.

    Bins each numeric feature and computes mean target per bin. Features with
    strong monotonic trends (|Spearman rho| > 0.7 on binned means) can benefit
    from monotone_constraints in gradient boosting models.

    Args:
        feature_df: Feature columns (no target, no id).
        target_series: Target series.
        columns_analysis: Per-column type info.
        n_bins: Number of bins for numeric features.

    Returns:
        Dict keyed by column name with:
        - spearman_rho: Spearman correlation on binned target rates
        - direction: "increasing", "decreasing", or "non_monotonic"
        - is_monotonic: True if |rho| > 0.7
    """
    result: dict[str, dict[str, Any]] = {}
    y = target_series.values.astype(float)

    for col, info in columns_analysis.items():
        if info["stats"] is None or col not in feature_df.columns:
            continue
        if info["cardinality"] < 3:
            continue

        series = feature_df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        try:
            binned = pd.qcut(
                series.fillna(series.median()), q=n_bins,
                duplicates="drop", labels=False,
            )
        except (ValueError, TypeError):
            continue

        if binned.nunique() < 3:
            continue

        df_temp = pd.DataFrame({"bin": binned, "target": y})
        bin_means = df_temp.groupby("bin")["target"].mean()
        if len(bin_means) < 3:
            continue

        rho, _ = scipy_stats.spearmanr(bin_means.index, bin_means.values)
        if np.isnan(rho):
            continue

        is_mono = abs(rho) > 0.7
        if rho > 0.7:
            direction = "increasing"
        elif rho < -0.7:
            direction = "decreasing"
        else:
            direction = "non_monotonic"

        result[col] = {
            "spearman_rho": round(float(rho), 4),
            "direction": direction,
            "is_monotonic": is_mono,
        }

    return result


def _compute_cardinality_profile(
    feature_df: pd.DataFrame,
    columns_analysis: dict[str, dict],
) -> dict[str, dict[str, Any]]:
    """Profile the distribution shape of categorical features.

    Reports concentration (top-K share) and entropy to distinguish uniform
    distributions from long-tail ones. Long-tail categoricals need different
    encoding strategies (frequency encoding, rare-category binning).

    Args:
        feature_df: Feature columns.
        columns_analysis: Per-column type info.

    Returns:
        Dict keyed by column name with:
        - top5_share: percentage of rows in top-5 categories
        - top10_share: percentage of rows in top-10 categories
        - entropy: Shannon entropy (higher = more uniform)
        - normalized_entropy: entropy / max_entropy (1.0 = perfectly uniform)
        - shape: "uniform", "moderate", or "long_tail"
    """
    cat_types = {"low_cardinality_categorical", "high_cardinality_categorical", "ordinal"}
    result: dict[str, dict[str, Any]] = {}

    for col, info in columns_analysis.items():
        if info["detected_type"] not in cat_types:
            continue
        if info["cardinality"] < 3 or col not in feature_df.columns:
            continue

        vc = feature_df[col].value_counts(normalize=True)
        n_cats = len(vc)
        top5_share = float(vc.head(5).sum()) * 100
        top10_share = float(vc.head(10).sum()) * 100

        probs = vc.values
        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        max_entropy = float(np.log2(n_cats)) if n_cats > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

        if normalized_entropy > 0.8:
            shape = "uniform"
        elif normalized_entropy > 0.5:
            shape = "moderate"
        else:
            shape = "long_tail"

        result[col] = {
            "top5_share": round(top5_share, 1),
            "top10_share": round(top10_share, 1),
            "entropy": round(entropy, 3),
            "normalized_entropy": round(normalized_entropy, 3),
            "shape": shape,
        }

    return result


def _compute_target_encoding_preview(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    columns_analysis: dict[str, dict],
    n_folds: int = 5,
    alpha: float = 10.0,
) -> dict[str, dict[str, Any]]:
    """Preview target encoding effectiveness per categorical feature.

    Simulates OOF target encoding with smoothing and reports the resulting
    correlation with target. Gives the LLM a concrete number for whether
    target encoding is worthwhile for each feature.

    Args:
        feature_df: Feature columns.
        target_series: Target series.
        columns_analysis: Per-column type info.
        n_folds: Number of CV folds for OOF encoding.
        alpha: Smoothing parameter (higher = more regularization).

    Returns:
        Dict keyed by column name, sorted by |encoded_corr| descending, with:
        - encoded_corr: Pearson correlation of OOF-encoded column with target
        - encoded_auc: AUC of OOF-encoded column (binary only, else None)
    """
    cat_types = {"low_cardinality_categorical", "high_cardinality_categorical",
                 "ordinal"}
    result: dict[str, dict[str, Any]] = {}
    y = target_series.values.astype(float)
    is_binary = len(np.unique(y[~np.isnan(y)])) == 2

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for col, info in columns_analysis.items():
        if info["detected_type"] not in cat_types:
            continue
        if info["cardinality"] < 2 or col not in feature_df.columns:
            continue

        series = feature_df[col].values
        encoded = np.full(len(y), np.nan)

        for train_idx, val_idx in kf.split(series):
            # Fold-specific mean (train fold only) — prevents val leakage into
            # unseen-category fallback and smoothing prior.
            global_mean = float(np.nanmean(y[train_idx]))
            # Smoothed target encoding on train fold
            agg = (
                pd.DataFrame({"cat": series[train_idx], "y": y[train_idx]})
                .dropna(subset=["cat"])
                .groupby("cat")["y"]
                .agg(["sum", "count"])
            )
            cat_stats: dict[Any, tuple[float, int]] = {
                cat: (float(row["sum"]), int(row["count"]))
                for cat, row in agg.iterrows()
            }

            for idx in val_idx:
                cat = series[idx]
                if pd.isna(cat) or cat not in cat_stats:
                    encoded[idx] = global_mean
                else:
                    s, c = cat_stats[cat]
                    encoded[idx] = (s + alpha * global_mean) / (c + alpha)

        mask = ~np.isnan(encoded)
        if mask.sum() < 10:
            continue

        corr_mat = np.corrcoef(encoded[mask], y[mask])
        corr_val = float(corr_mat[0, 1]) if not np.isnan(corr_mat[0, 1]) else 0.0

        auc_val = None
        if is_binary:
            try:
                auc_val = float(roc_auc_score(y[mask], encoded[mask]))
                auc_val = max(auc_val, 1.0 - auc_val)
            except ValueError:
                pass

        result[col] = {
            "encoded_corr": round(corr_val, 4),
            "encoded_auc": round(auc_val, 4) if auc_val is not None else None,
        }

    return dict(sorted(result.items(), key=lambda x: abs(x[1]["encoded_corr"]), reverse=True))


def _prepare_eda_features(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    task_type: str,
    max_samples: int = 50000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], bool]:
    """Shared data prep for EDA baseline functions.

    Label-encodes categoricals, fills NaN with median, subsamples large
    datasets, and auto-detects classification vs regression.

    Args:
        feature_df: Feature columns (raw).
        target_series: Target series.
        task_type: ``'binary_classification'`` or ``'regression'``.
        max_samples: Subsample to this many rows for speed.
        seed: Random seed for deterministic subsampling.

    Returns:
        Tuple of ``(X, y, column_names, is_classification)``.
        Returns ``(empty_array, empty_array, [], False)`` if input is empty.
    """
    if feature_df.empty or len(target_series) == 0:
        return np.empty((0, 0)), np.array([]), [], False

    y = target_series.values.astype(float)
    cols = list(feature_df.columns)

    # Label-encode categoricals, fill NaN with median
    arrays: list[np.ndarray] = []
    for col in cols:
        series = feature_df[col]
        if not pd.api.types.is_numeric_dtype(series):
            codes, _ = pd.factorize(series, sort=False)
            arrays.append(codes.astype(float))
        else:
            arrays.append(series.fillna(series.median()).values.astype(float))

    X = np.column_stack(arrays) if arrays else np.empty((len(feature_df), 0))
    X = np.nan_to_num(X, nan=0.0)

    # Subsample for speed on large datasets
    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        idx = rng.choice(len(X), max_samples, replace=False)
        idx.sort()
        X = X[idx]
        y = y[idx]

    # Auto-detect: classification if task_type says so AND target has few unique values
    n_unique = len(np.unique(y[~np.isnan(y)]))
    is_classification = task_type != "regression" and n_unique <= 30

    return X, y, cols, is_classification


def _compute_quick_importance_and_baseline(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    task_type: str,
    n_folds: int = 3,
    max_samples: int = 50000,
) -> dict[str, Any]:
    """Train a quick random forest to get feature importances and baseline score.

    Combines two analyses: (1) feature importance from a quick model (sees
    interactions and non-linear effects unlike univariate metrics), and
    (2) baseline OOF score as a performance estimate without feature engineering.

    Subsamples large datasets for speed. Fully deterministic (seeded).

    Args:
        feature_df: Feature columns. Categoricals are label-encoded.
        target_series: Target series.
        task_type: 'binary_classification' or 'regression'.
        n_folds: Number of CV folds.
        max_samples: Subsample to this many rows for speed.

    Returns:
        Dict with:
        - feature_importances: dict {col: importance} sorted descending
        - baseline_score: float (AUC for classification, RMSE for regression)
        - baseline_metric: str describing the metric used
    """
    X, y, cols, is_classification = _prepare_eda_features(
        feature_df, target_series, task_type, max_samples=max_samples
    )
    if len(y) == 0:
        return {"feature_importances": {}, "baseline_score": None, "baseline_metric": "N/A"}

    if is_classification:
        model_cls = RandomForestClassifier
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_target = y
    else:
        model_cls = RandomForestRegressor
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_target = None

    model_kwargs = dict(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    oof = np.zeros(len(y))
    importances = np.zeros(len(cols))

    for train_idx, val_idx in cv.split(X, split_target):
        model = model_cls(**model_kwargs)
        model.fit(X[train_idx], y[train_idx])
        if is_classification:
            oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        else:
            oof[val_idx] = model.predict(X[val_idx])
        importances += model.feature_importances_ / n_folds

    if is_classification:
        try:
            score = float(roc_auc_score(y, oof))
        except ValueError:
            score = 0.5
        metric_name = "AUC"
    else:
        score = float(np.sqrt(np.mean((y - oof) ** 2)))
        metric_name = "RMSE"

    imp_dict = {col: round(float(imp), 6) for col, imp in zip(cols, importances)}
    imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))

    return {
        "feature_importances": imp_dict,
        "baseline_score": round(score, 6),
        "baseline_metric": metric_name,
    }


def _compute_prediction_diversity_probe(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    task_type: str,
    seeds: tuple[int, ...] = (42, 123, 456),
    n_folds: int = 3,
    max_samples: int = 50000,
) -> dict[str, Any]:
    """Train quick RF models with different seeds and measure prediction stability.

    Measures how much RF predictions change when only the random seed differs.
    The primary metric is **signal-noise ratio** (SNR):

    - ``within_seed_std``: how much predictions vary across *samples* within
      one seed.  This reflects how much signal the model finds — strong
      signal → predictions spread out from the class prior.
    - ``prediction_std``: how much predictions vary across *seeds* for the
      same sample.  This reflects seed-dependent noise — noise features
      and bootstrap randomness inflate this.
    - ``signal_noise_ratio = within_seed_std / prediction_std``: how much
      of the model's output is stable signal vs random noise.

    **Why not Pearson correlation?**  RF is inherently high-variance
    (bootstrap + random feature selection).  On low-signal data with noise
    features (e.g., Porto Seguro with 20 ps_calc_* noise columns), Pearson
    correlation between seeds can be as low as 0.79 — *not* because models
    find diverse solutions, but because noise features inject random
    variation.  SNR is immune to this because noise features inflate both
    within- and across-seed std proportionally.

    Classification (based on SNR):

    - SNR > 15: ``high`` — strong signal, models find robust patterns.
      Standard per-fold settings are sufficient.
    - SNR 8–15: ``moderate`` — decent signal.  Tiered tracker optional.
    - SNR 3–8: ``low`` — weak signal relative to seed noise.  Tiered
      tracker and diversity pruning recommended for neural nets.
    - SNR < 3: ``very_low`` — seed noise dominates.  All models will
      converge to near-identical predictions.  Aggressive diversity
      management needed.

    Also reports Pearson correlations and Fisher z-transform CI for
    reference.

    Args:
        feature_df: Feature columns (categoricals will be label-encoded).
        target_series: Target series.
        task_type: ``'binary_classification'`` or ``'regression'``.
        seeds: Random seeds for the RF models.
        n_folds: Number of CV folds per seed.
        max_samples: Subsample to this many rows for speed.

    Returns:
        Dict with:
        - ``pairwise_correlations``: list of ``(seed_a, seed_b, corr)`` tuples
        - ``mean_corr``: mean pairwise Pearson correlation
        - ``min_corr``: minimum pairwise correlation
        - ``prediction_std``: mean pointwise std across seeds
        - ``within_seed_std``: mean std of predictions within each seed
        - ``signal_noise_ratio``: within_seed_std / prediction_std (primary metric)
        - ``diversity_class``: ``"very_low"`` / ``"low"`` / ``"moderate"`` / ``"high"``
        - ``fisher_z_ci``: 95% CI for mean correlation via Fisher z-transform
        - ``n_samples_used``: how many samples were used (after subsampling)
    """
    X, y, _cols, is_classification = _prepare_eda_features(
        feature_df, target_series, task_type, max_samples=max_samples
    )
    if len(y) == 0:
        return {}

    n = len(y)

    # Collect OOF predictions per seed
    oof_per_seed: list[np.ndarray] = []

    for seed in seeds:
        if is_classification:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split_target = y
            model_cls = RandomForestClassifier
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split_target = None
            model_cls = RandomForestRegressor

        model_kwargs = dict(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1)
        oof = np.zeros(n)

        for train_idx, val_idx in cv.split(X, split_target):
            model = model_cls(**model_kwargs)
            model.fit(X[train_idx], y[train_idx])
            if is_classification:
                oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
            else:
                oof[val_idx] = model.predict(X[val_idx])

        oof_per_seed.append(oof)

    # Pairwise Pearson correlations
    pairwise: list[tuple[int, int, float]] = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            corr = float(np.corrcoef(oof_per_seed[i], oof_per_seed[j])[0, 1])
            if np.isnan(corr):
                corr = 1.0  # constant predictions
            pairwise.append((seeds[i], seeds[j], round(corr, 6)))

    corr_values = [c for _, _, c in pairwise]
    mean_corr = float(np.mean(corr_values))
    min_corr = float(np.min(corr_values))

    # Pointwise prediction std across seeds (seed noise)
    stacked = np.column_stack(oof_per_seed)  # (n, n_seeds)
    prediction_std = float(np.mean(np.std(stacked, axis=1)))

    # Within-seed std: how much predictions vary across samples (signal strength)
    within_seed_std = float(np.mean([np.std(oof) for oof in oof_per_seed]))

    # Signal-noise ratio: primary classification metric
    snr = within_seed_std / max(prediction_std, 1e-10)

    # Fisher z-transform 95% CI for mean correlation (reference)
    z_values = [0.5 * np.log((1 + r) / (1 - r + 1e-15)) for r in corr_values]
    z_mean = float(np.mean(z_values))
    se_z = 1.0 / np.sqrt(max(n - 3, 1))
    z_lo = z_mean - 1.96 * se_z
    z_hi = z_mean + 1.96 * se_z
    ci_lo = float(np.tanh(z_lo))
    ci_hi = float(np.tanh(z_hi))

    # Classification based on SNR
    if snr < 3:
        diversity_class = "very_low"
    elif snr < 8:
        diversity_class = "low"
    elif snr < 15:
        diversity_class = "moderate"
    else:
        diversity_class = "high"

    return {
        "pairwise_correlations": pairwise,
        "mean_corr": round(mean_corr, 6),
        "min_corr": round(min_corr, 6),
        "prediction_std": round(prediction_std, 6),
        "within_seed_std": round(within_seed_std, 6),
        "signal_noise_ratio": round(snr, 2),
        "diversity_class": diversity_class,
        "fisher_z_ci": [round(ci_lo, 6), round(ci_hi, 6)],
        "n_samples_used": n,
    }


def _run_single_baseline(
    name: str,
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    is_classification: bool,
    n_folds: int,
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Run CV for one baseline model. Returns OOF predictions, score, std, time.

    Args:
        name: Model name for logging.
        model: Unfitted sklearn-compatible estimator (will be cloned per fold).
        X: Feature matrix.
        y: Target array.
        is_classification: Whether to use predict_proba.
        n_folds: Number of CV folds.
        scaler: If not None, applied per fold (fit on train, transform train+val).

    Returns:
        Dict with ``oof``, ``score``, ``fold_scores``, ``std``, ``time_sec``,
        ``feature_importances`` (if available), ``metric``.
    """
    from sklearn.base import clone

    if is_classification:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_target = y
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_target = None

    oof = np.zeros(len(y))
    importances = np.zeros(X.shape[1]) if X.shape[1] > 0 else np.array([])
    has_importances = False
    fold_scores: list[float] = []

    t0 = time.perf_counter()
    for train_idx, val_idx in cv.split(X, split_target):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if scaler is not None:
            from sklearn.base import clone as clone_scaler

            sc = clone_scaler(scaler)
            X_tr = sc.fit_transform(X_tr)
            X_val = sc.transform(X_val)

        m = clone(model)
        m.fit(X_tr, y_tr)

        if is_classification:
            preds = m.predict_proba(X_val)[:, 1]
        else:
            preds = m.predict(X_val)
        oof[val_idx] = preds

        # Per-fold score
        if is_classification:
            try:
                fold_scores.append(float(roc_auc_score(y_val, preds)))
            except ValueError:
                fold_scores.append(0.5)
        else:
            fold_scores.append(float(np.sqrt(np.mean((y_val - preds) ** 2))))

        if hasattr(m, "feature_importances_"):
            importances += m.feature_importances_ / n_folds
            has_importances = True

    elapsed = time.perf_counter() - t0

    # Overall OOF score
    if is_classification:
        try:
            score = float(roc_auc_score(y, oof))
        except ValueError:
            score = 0.5
        metric = "AUC"
    else:
        score = float(np.sqrt(np.mean((y - oof) ** 2)))
        metric = "RMSE"

    result: dict[str, Any] = {
        "oof": oof,
        "score": round(score, 6),
        "fold_scores": [round(s, 6) for s in fold_scores],
        "std": round(float(np.std(fold_scores)), 6),
        "time_sec": round(elapsed, 2),
        "metric": metric,
    }
    if has_importances:
        result["feature_importances"] = importances
    return result


def _classify_data_personality(
    scores: dict[str, float],
    metric: str,
) -> tuple[str, str]:
    """Classify data personality from multi-baseline scores.

    Args:
        scores: ``{model_name: score}`` dict.
        metric: ``"AUC"`` (higher-is-better) or ``"RMSE"`` (lower-is-better).

    Returns:
        Tuple of ``(personality, detail_string)``.
        Personality is one of: ``nn_goldmine``, ``tree_dominant``,
        ``linear_friendly``, ``all_similar``, ``mixed``.
    """
    tree_models = {"random_forest", "lightgbm", "catboost"}
    nn_models = {"realmlp", "tabm"}
    linear_models = {"ridge"}

    tree_scores = [scores[k] for k in tree_models if k in scores]
    nn_scores = [scores[k] for k in nn_models if k in scores]
    linear_score = next((scores[k] for k in linear_models if k in scores), None)

    if not tree_scores:
        return "unknown", "No tree models available."

    higher_is_better = metric != "RMSE"

    best_tree = max(tree_scores) if higher_is_better else min(tree_scores)
    best_nn = (max(nn_scores) if higher_is_better else min(nn_scores)) if nn_scores else None

    # For comparisons: always work in "higher is better" space
    def gap(a: float, b: float) -> float:
        return (a - b) if higher_is_better else (b - a)

    parts: list[str] = []
    personality = "mixed"

    nn_gap = gap(best_nn, best_tree) if best_nn is not None else None
    linear_gap = gap(best_tree, linear_score) if linear_score is not None else None

    # Check all similar first (takes priority over linear_friendly)
    all_scores_list = list(scores.values())
    spread = max(all_scores_list) - min(all_scores_list)

    if nn_gap is not None and nn_gap > 0.005:
        personality = "nn_goldmine"
        parts.append(f"Neural nets lead trees by {abs(nn_gap):.4f}.")
    elif linear_gap is not None and linear_gap > 0.02 and (
        nn_gap is None or nn_gap < -0.005
    ):
        personality = "tree_dominant"
        parts.append("Tree boosters dominate.")
    elif spread < 0.005:
        personality = "all_similar"
        parts.append("All models perform similarly.")
    elif linear_gap is not None and linear_gap < 0.005:
        personality = "linear_friendly"
        parts.append("Linear model is competitive with trees.")

    if linear_gap is not None:
        parts.append(f"Linear gap: {linear_gap:.4f}.")
    if nn_gap is not None:
        parts.append(f"NN gap vs trees: {nn_gap:+.4f}.")
    elif not nn_scores:
        parts.append("NNs not tested (pytabkit not available or no GPU).")

    return personality, " ".join(parts)


def _compute_multi_baseline(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    task_type: str,
    n_folds: int = 3,
    max_samples: int = 50000,
    nn_max_samples: int = 20000,
    nn_timeout: float = 120.0,
) -> dict[str, Any]:
    """Train multiple baseline models and classify data personality.

    Trains up to 6 models (RF, LightGBM, CatBoost, Ridge + RealMLP, TabM)
    with default hyperparameters on a 3-fold CV. Returns per-model scores,
    cross-model diversity, and a data personality classification.

    CPU models (RF, LightGBM, CatBoost, Ridge) always run on the full
    subsample. Neural nets (RealMLP, TabM) run on a smaller subsample
    with a per-model timeout, and only when pytabkit + torch are available.

    Args:
        feature_df: Raw feature columns.
        target_series: Target series.
        task_type: ``'binary_classification'`` or ``'regression'``.
        n_folds: Number of CV folds (default 3).
        max_samples: Subsample for CPU models.
        nn_max_samples: Smaller subsample for neural nets.
        nn_timeout: Max seconds per NN model (skip if exceeded).

    Returns:
        Dict with ``scores``, ``stds``, ``best_model``, ``best_score``,
        ``metric``, ``feature_importances``, ``linear_gap``,
        ``cross_model_correlations``, ``training_times_sec``, ``personality``,
        ``personality_detail``, ``n_samples_used``, ``nn_samples_used``,
        ``all_importances`` (per-model importances for ghost detector),
        ``lgbm_models`` (fitted LightGBM models for interaction orchestra).
    """
    logger = logging.getLogger(__name__)

    X, y, cols, is_classification = _prepare_eda_features(
        feature_df, target_series, task_type, max_samples=max_samples
    )
    if len(y) == 0:
        return {
            "scores": {}, "stds": {}, "best_model": None, "best_score": None,
            "metric": "N/A", "feature_importances": {},
            "linear_gap": None, "cross_model_correlations": [],
            "training_times_sec": {}, "personality": "unknown",
            "personality_detail": "No data.", "n_samples_used": 0,
            "nn_samples_used": 0, "all_importances": {}, "lgbm_models": [],
        }

    # ─── Define CPU baseline models ───────────────────────────────────────
    cpu_baselines: list[tuple[str, Any, Any]] = []  # (name, model, scaler|None)

    # 1. RandomForest
    if is_classification:
        cpu_baselines.append((
            "random_forest",
            RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
            None,
        ))
    else:
        cpu_baselines.append((
            "random_forest",
            RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
            None,
        ))

    # 2. LightGBM
    try:
        import lightgbm as lgb

        if is_classification:
            cpu_baselines.append((
                "lightgbm",
                lgb.LGBMClassifier(n_estimators=200, verbosity=-1, random_state=42, n_jobs=-1),
                None,
            ))
        else:
            cpu_baselines.append((
                "lightgbm",
                lgb.LGBMRegressor(n_estimators=200, verbosity=-1, random_state=42, n_jobs=-1),
                None,
            ))
    except ImportError:
        logger.warning("LightGBM not installed — skipping baseline.")

    # 3. CatBoost
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        if is_classification:
            cpu_baselines.append((
                "catboost",
                CatBoostClassifier(
                    iterations=200, verbose=0, random_seed=42,
                    allow_writing_files=False,
                ),
                None,
            ))
        else:
            cpu_baselines.append((
                "catboost",
                CatBoostRegressor(
                    iterations=200, verbose=0, random_seed=42,
                    allow_writing_files=False,
                ),
                None,
            ))
    except ImportError:
        logger.warning("CatBoost not installed — skipping baseline.")

    # 4. Ridge / LogisticRegression (with scaling)
    from sklearn.preprocessing import StandardScaler

    if is_classification:
        from sklearn.linear_model import LogisticRegression

        cpu_baselines.append((
            "ridge",
            LogisticRegression(max_iter=10000, solver="lbfgs", random_state=42),
            StandardScaler(),
        ))
    else:
        from sklearn.linear_model import Ridge

        cpu_baselines.append((
            "ridge",
            Ridge(fit_intercept=True, random_state=42),
            StandardScaler(),
        ))

    # ─── Run CPU baselines ────────────────────────────────────────────────
    results: dict[str, dict[str, Any]] = {}
    lgbm_models: list[Any] = []

    for name, model, scaler in cpu_baselines:
        try:
            res = _run_single_baseline(
                name, model, X, y, is_classification, n_folds, scaler=scaler
            )
            results[name] = res
            logger.info(
                "Baseline %s: %s=%.4f (±%.4f) in %.1fs",
                name, res["metric"], res["score"], res["std"], res["time_sec"],
            )
        except Exception as exc:
            logger.warning("Baseline %s failed: %s", name, exc)
            logger.debug("Baseline %s traceback:", name, exc_info=True)

    # Collect LightGBM fitted models for interaction orchestra (refit one for tree dump)
    if "lightgbm" in results:
        try:
            import lightgbm as lgb

            # Fit a single LightGBM on full data for tree structure extraction
            if is_classification:
                lgbm_full = lgb.LGBMClassifier(
                    n_estimators=200, verbosity=-1, random_state=42, n_jobs=-1
                )
            else:
                lgbm_full = lgb.LGBMRegressor(
                    n_estimators=200, verbosity=-1, random_state=42, n_jobs=-1
                )
            lgbm_full.fit(X, y)
            lgbm_models.append(lgbm_full)
        except Exception:
            pass  # interaction orchestra will skip

    # ─── Run NN baselines (optional) ──────────────────────────────────────
    nn_samples_used = 0
    try:
        import pytabkit  # noqa: F401
        import torch  # noqa: F401

        has_gpu = torch.cuda.is_available()
        device = "cuda" if has_gpu else "cpu"

        # Prepare smaller subsample for NNs
        X_nn, y_nn, cols_nn, _ = _prepare_eda_features(
            feature_df, target_series, task_type,
            max_samples=nn_max_samples, seed=42,
        )
        nn_samples_used = len(y_nn)

        nn_baselines: list[tuple[str, Any]] = []

        # RealMLP
        try:
            if is_classification:
                nn_baselines.append((
                    "realmlp",
                    pytabkit.RealMLP_TD_Classifier(
                        hidden_sizes="rectangular", n_hidden_layers=3, hidden_width=128,
                        n_epochs=50, n_cv=1, n_refit=0, use_early_stopping=True,
                        verbosity=0, random_state=42, n_ens=1, device=device,
                    ),
                ))
            else:
                nn_baselines.append((
                    "realmlp",
                    pytabkit.RealMLP_TD_Regressor(
                        hidden_sizes="rectangular", n_hidden_layers=3, hidden_width=128,
                        n_epochs=50, n_cv=1, n_refit=0, use_early_stopping=True,
                        verbosity=0, random_state=42, n_ens=1, device=device,
                    ),
                ))
        except Exception as exc:
            logger.warning("RealMLP init failed: %s", exc)

        # TabM
        try:
            if is_classification:
                nn_baselines.append((
                    "tabm",
                    pytabkit.TabM_D_Classifier(
                        arch_type="tabm", tabm_k=8, n_blocks=2, d_block=128,
                        dropout=0.1, num_emb_type="pwl", d_embedding=16,
                        num_emb_n_bins=32, lr=0.001, weight_decay=0.001,
                        batch_size=2048, n_epochs=100, patience=16,
                        n_cv=1, n_refit=0, tfms=["quantile_tabr"],
                        val_metric_name="1-auc_ovr", verbosity=0,
                        random_state=42, device=device,
                    ),
                ))
            else:
                nn_baselines.append((
                    "tabm",
                    pytabkit.TabM_D_Regressor(
                        arch_type="tabm", tabm_k=8, n_blocks=2, d_block=128,
                        dropout=0.1, num_emb_type="pwl", d_embedding=16,
                        num_emb_n_bins=32, lr=0.001, weight_decay=0.001,
                        batch_size=2048, n_epochs=100, patience=16,
                        n_cv=1, n_refit=0, tfms=["quantile_tabr"],
                        val_metric_name="rmse", verbosity=0,
                        random_state=42, device=device,
                    ),
                ))
        except Exception as exc:
            logger.warning("TabM init failed: %s", exc)

        for name, model in nn_baselines:
            try:
                t0 = time.perf_counter()
                res = _run_single_baseline(
                    name, model, X_nn, y_nn, is_classification, n_folds
                )
                elapsed = time.perf_counter() - t0
                if elapsed > nn_timeout:
                    logger.warning("NN baseline %s took %.0fs (timeout=%.0fs) — included but slow.", name, elapsed, nn_timeout)
                results[name] = res
                logger.info(
                    "Baseline %s: %s=%.4f (±%.4f) in %.1fs [%s, %d samples]",
                    name, res["metric"], res["score"], res["std"],
                    res["time_sec"], device, nn_samples_used,
                )
            except Exception as exc:
                logger.warning("NN baseline %s failed: %s", name, exc)

    except (ImportError, OSError):
        logger.info("pytabkit/torch not available — skipping NN baselines.")

    if not results:
        return {
            "scores": {}, "stds": {}, "best_model": None, "best_score": None,
            "metric": "N/A", "feature_importances": {},
            "linear_gap": None, "cross_model_correlations": [],
            "training_times_sec": {}, "personality": "unknown",
            "personality_detail": "All baselines failed.", "n_samples_used": len(y),
            "nn_samples_used": nn_samples_used, "all_importances": {},
            "lgbm_models": [],
        }

    # ─── Aggregate results ────────────────────────────────────────────────
    scores = {k: v["score"] for k, v in results.items()}
    stds = {k: v["std"] for k, v in results.items()}
    times = {k: v["time_sec"] for k, v in results.items()}
    metric = next(iter(results.values()))["metric"]

    # Best model
    if metric == "RMSE":
        best_model = min(scores, key=scores.get)  # type: ignore[arg-type]
    else:
        best_model = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_model]

    # RF feature importances
    rf_importances: dict[str, float] = {}
    if "random_forest" in results and "feature_importances" in results["random_forest"]:
        imp_arr = results["random_forest"]["feature_importances"]
        rf_importances = {col: round(float(imp), 6) for col, imp in zip(cols, imp_arr)}
        rf_importances = dict(sorted(rf_importances.items(), key=lambda x: x[1], reverse=True))

    # All importances (for ghost detector)
    all_importances: dict[str, dict[str, float]] = {}
    if rf_importances:
        all_importances["random_forest"] = rf_importances

    # LightGBM importances (gain-based)
    if "lightgbm" in results and lgbm_models:
        try:
            lgbm_imp = lgbm_models[0].feature_importances_
            lgbm_imp_dict = {col: round(float(v), 6) for col, v in zip(cols, lgbm_imp)}
            all_importances["lightgbm"] = lgbm_imp_dict
        except Exception:
            pass

    # CatBoost importances
    if "catboost" in results:
        try:
            # Fit a quick CatBoost on full data for importances
            from catboost import CatBoostClassifier, CatBoostRegressor

            if is_classification:
                cb_full = CatBoostClassifier(
                    iterations=200, verbose=0, random_seed=42,
                    allow_writing_files=False,
                )
            else:
                cb_full = CatBoostRegressor(
                    iterations=200, verbose=0, random_seed=42,
                    allow_writing_files=False,
                )
            cb_full.fit(X, y)
            cb_imp = cb_full.get_feature_importance()
            cb_imp_dict = {col: round(float(v), 6) for col, v in zip(cols, cb_imp)}
            all_importances["catboost"] = cb_imp_dict
        except Exception:
            pass

    # Ridge coefficients as importances
    if "ridge" in results:
        try:
            from sklearn.preprocessing import StandardScaler

            sc = StandardScaler()
            X_scaled = sc.fit_transform(X)
            if is_classification:
                from sklearn.linear_model import LogisticRegression

                ridge_full = LogisticRegression(max_iter=10000, solver="lbfgs", random_state=42)
            else:
                from sklearn.linear_model import Ridge as RidgeModel

                ridge_full = RidgeModel(fit_intercept=True, random_state=42)
            ridge_full.fit(X_scaled, y)
            coefs = np.abs(ridge_full.coef_.ravel())
            ridge_imp_dict = {col: round(float(v), 6) for col, v in zip(cols, coefs)}
            all_importances["ridge"] = ridge_imp_dict
        except Exception:
            pass

    # Linear gap
    tree_models = {"random_forest", "lightgbm", "catboost"}
    tree_scores = [scores[k] for k in tree_models if k in scores]
    ridge_score = scores.get("ridge")
    if tree_scores and ridge_score is not None:
        if metric == "RMSE":
            linear_gap = round(ridge_score - min(tree_scores), 6)  # positive = trees better
        else:
            linear_gap = round(max(tree_scores) - ridge_score, 6)  # positive = trees better
    else:
        linear_gap = None

    # Cross-model OOF correlations (CPU models only — same sample size)
    cpu_oof = {k: v["oof"] for k, v in results.items() if k in tree_models | {"ridge"}}
    cross_corr: list[tuple[str, str, float]] = []
    for (na, oa), (nb, ob) in combinations(cpu_oof.items(), 2):
        corr = float(np.corrcoef(oa, ob)[0, 1])
        if np.isnan(corr):
            corr = 1.0
        cross_corr.append((na, nb, round(corr, 6)))

    # Data personality
    personality, personality_detail = _classify_data_personality(scores, metric)

    return {
        "scores": scores,
        "stds": stds,
        "best_model": best_model,
        "best_score": best_score,
        "metric": metric,
        "feature_importances": rf_importances,
        "linear_gap": linear_gap,
        "cross_model_correlations": cross_corr,
        "training_times_sec": times,
        "personality": personality,
        "personality_detail": personality_detail,
        "n_samples_used": len(y),
        "nn_samples_used": nn_samples_used,
        "all_importances": all_importances,
        "lgbm_models": lgbm_models,
    }


def _extract_lgbm_interaction_pairs(
    lgbm_model: Any,
    cols: list[str],
    top_n: int = 15,
) -> list[dict[str, Any]]:
    """Extract feature interaction pairs from LightGBM tree structure.

    Walks the tree JSON and records parent→child split feature pairs
    weighted by the parent node's split gain.

    Args:
        lgbm_model: Fitted LightGBM model with ``booster_`` attribute.
        cols: Feature column names.
        top_n: Number of top interactions to return.

    Returns:
        List of dicts with ``feature_a``, ``feature_b``, ``interaction_strength``.
    """
    try:
        dump = lgbm_model.booster_.dump_model()
    except Exception:
        return []

    feature_names = dump.get("feature_names", [str(i) for i in range(len(cols))])
    # Map internal names (Column_0, ...) back to user names
    name_map = {}
    for i, fn in enumerate(feature_names):
        if i < len(cols):
            name_map[fn] = cols[i]
        else:
            name_map[fn] = fn

    pair_gains: dict[tuple[str, str], float] = Counter()

    def _walk_tree(node: dict) -> str | None:
        """Walk tree recursively, return split feature name of this node."""
        split_feature = node.get("split_feature")
        if split_feature is None:
            return None  # leaf node

        gain = node.get("split_gain", 0.0)
        parent_name = name_map.get(split_feature, split_feature)

        # Visit children
        for child_key in ("left_child", "right_child"):
            child = node.get(child_key)
            if child and child.get("split_feature") is not None:
                child_name = name_map.get(child["split_feature"], child["split_feature"])
                if parent_name != child_name:
                    pair = tuple(sorted([parent_name, child_name]))
                    pair_gains[pair] += gain
                _walk_tree(child)

        return parent_name

    for tree_info in dump.get("tree_info", []):
        tree = tree_info.get("tree_structure")
        if tree:
            _walk_tree(tree)

    # Sort by cumulative gain
    top_pairs = pair_gains.most_common(top_n)
    total_gain = sum(pair_gains.values()) or 1.0

    return [
        {
            "feature_a": pair[0],
            "feature_b": pair[1],
            "interaction_strength": round(gain / total_gain, 6),
        }
        for pair, gain in top_pairs
    ]


def _compute_interaction_orchestra(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    task_type: str,
    lgbm_models: list[Any] | None = None,
    n_folds: int = 3,
    max_samples: int = 50000,
    top_n: int = 15,
) -> dict[str, Any]:
    """Extract top feature interaction pairs from LightGBM tree structure.

    Uses consecutive parent-child split pairs weighted by split gain
    to identify features that the model frequently uses together.

    Args:
        feature_df: Raw feature columns.
        target_series: Target series.
        task_type: ``'binary_classification'`` or ``'regression'``.
        lgbm_models: Pre-fitted LightGBM models from multi_baseline.
            If provided, skips re-training.
        n_folds: Number of CV folds (only used if lgbm_models is None).
        max_samples: Subsample size (only used if lgbm_models is None).
        top_n: Number of top interactions to return.

    Returns:
        Dict with ``top_interactions`` (list of interaction dicts) and
        ``method`` describing the extraction approach.
    """
    logger = logging.getLogger(__name__)
    cols = list(feature_df.columns)

    # Use pre-fitted model if available
    if lgbm_models:
        interactions = _extract_lgbm_interaction_pairs(lgbm_models[0], cols, top_n)
        return {"top_interactions": interactions, "method": "lgbm_split_gain_pairs"}

    # Otherwise, train a quick LightGBM
    try:
        import lightgbm as lgb
    except ImportError:
        logger.info("LightGBM not available — skipping interaction orchestra.")
        return {"top_interactions": [], "method": "skipped_no_lgbm"}

    X, y, cols, is_classification = _prepare_eda_features(
        feature_df, target_series, task_type, max_samples=max_samples
    )
    if len(y) == 0:
        return {"top_interactions": [], "method": "skipped_empty_data"}

    try:
        if is_classification:
            model = lgb.LGBMClassifier(n_estimators=200, verbosity=-1, random_state=42, n_jobs=-1)
        else:
            model = lgb.LGBMRegressor(n_estimators=200, verbosity=-1, random_state=42, n_jobs=-1)
        model.fit(X, y)
        interactions = _extract_lgbm_interaction_pairs(model, cols, top_n)
        return {"top_interactions": interactions, "method": "lgbm_split_gain_pairs"}
    except Exception as exc:
        logger.warning("Interaction orchestra failed: %s", exc)
        return {"top_interactions": [], "method": "failed"}


def _compute_ghost_features(
    all_importances: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Detect ghost features by comparing importance rankings across models.

    A ghost feature is important for one model family but irrelevant for
    others — potentially indicating leakage, MDI bias, or data artifacts.

    Args:
        all_importances: ``{model_name: {feature: importance}}`` dict.
            At least 2 models needed for meaningful comparison.

    Returns:
        Dict with ``ghost_features`` (list of ghost dicts with feature,
        severity, ranks, explanation) and ``rank_correlations`` (pairwise
        Spearman rank correlations between model importances).
    """
    if len(all_importances) < 2:
        return {"ghost_features": [], "rank_correlations": {}}

    # Get union of all features
    all_features = sorted(
        set().union(*(imp.keys() for imp in all_importances.values()))
    )
    n_features = len(all_features)
    if n_features == 0:
        return {"ghost_features": [], "rank_correlations": {}}

    # Build rank arrays per model (rank 1 = most important)
    model_ranks: dict[str, dict[str, int]] = {}
    model_arrays: dict[str, np.ndarray] = {}

    for model_name, imp_dict in all_importances.items():
        # Get importance values (0 for missing features)
        values = np.array([imp_dict.get(f, 0.0) for f in all_features])
        # Rank: higher importance → lower rank number (1 = best)
        ranks = n_features + 1 - scipy_stats.rankdata(values, method="average")
        model_ranks[model_name] = {f: int(r) for f, r in zip(all_features, ranks)}
        model_arrays[model_name] = values

    # Detect ghost features
    ghost_features: list[dict[str, Any]] = []
    rank_gap_threshold_high = n_features * 0.7
    rank_gap_threshold_medium = n_features * 0.5

    for feature in all_features:
        ranks_for_feature = {m: model_ranks[m][feature] for m in model_ranks}
        min_rank = min(ranks_for_feature.values())
        max_rank = max(ranks_for_feature.values())
        gap = max_rank - min_rank

        # Must be important in at least one model (top 10%)
        if min_rank > max(n_features * 0.1, 1):
            continue

        if gap >= rank_gap_threshold_medium:
            best_model = min(ranks_for_feature, key=ranks_for_feature.get)  # type: ignore[arg-type]
            worst_model = max(ranks_for_feature, key=ranks_for_feature.get)  # type: ignore[arg-type]
            severity = "high" if gap >= rank_gap_threshold_high else "medium"
            ghost_features.append({
                "feature": feature,
                "severity": severity,
                "ranks": ranks_for_feature,
                "explanation": (
                    f"Rank {min_rank} in {best_model} but rank {max_rank} in "
                    f"{worst_model} (gap={int(gap)}). "
                    f"Possible {'MDI bias' if best_model == 'random_forest' else 'model-specific artifact'}."
                ),
            })

    # Sort by severity (high first), then by min rank
    ghost_features.sort(key=lambda x: (0 if x["severity"] == "high" else 1, min(x["ranks"].values())))

    # Pairwise Spearman rank correlations
    rank_correlations: dict[str, float] = {}
    model_names = list(all_importances.keys())
    for i, ma in enumerate(model_names):
        for mb in model_names[i + 1:]:
            arr_a = model_arrays[ma]
            arr_b = model_arrays[mb]
            corr, _ = scipy_stats.spearmanr(arr_a, arr_b)
            if np.isnan(corr):
                corr = 0.0
            rank_correlations[f"{ma}_vs_{mb}"] = round(float(corr), 4)

    return {
        "ghost_features": ghost_features,
        "rank_correlations": rank_correlations,
    }


def run_eda(
    train_path: str | Path,
    test_path: str | Path,
    target_col: str,
    id_col: str | None = None,
    target_mapping: dict[str, int] | None = None,
    task_type: str = "binary_classification",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Run the complete EDA pipeline and produce a structured report.

    This is the main entry point for Layer 1. It reads the raw CSVs,
    analyzes every column, computes correlations, clusters features,
    identifies weak features, and generates LLM-readable recommendations.

    Args:
        train_path: Path to the training CSV file.
        test_path: Path to the test CSV file.
        target_col: Name of the target variable column.
        id_col: Optional name of the ID column to exclude from analysis.
        target_mapping: Optional dict mapping string labels to numeric values
            (e.g. ``{"Yes": 1, "No": 0}``). Applied to the target column
            before analysis so that correlation and distribution stats are
            computed on the mapped numeric values.
        task_type: Task type — ``'binary_classification'`` or ``'multiclass'``
            uses ``mutual_info_classif``; ``'regression'`` uses
            ``mutual_info_regression``. Defaults to
            ``'binary_classification'``.

    Returns:
        Tuple of (report, train_df, test_df) where report is a dictionary
        matching the eda_schema.yaml structure with keys:
        - dataset_info: shapes, memory usage
        - target_analysis: distribution, class balance
        - columns: per-column analysis dict
        - correlation_matrix: full pairwise correlations
        - feature_clusters: groups of correlated features
        - weak_features: features with low target correlation
        - recommendations: list of LLM-readable suggestion strings
        train_df and test_df are the loaded DataFrames (returned so callers
        can reuse them without re-reading from disk).

    Steps:
        1. Read train and test CSVs with pd.read_csv.
        2. Compute dataset_info: shapes, memory in MB.
        3. Analyze the target column: distribution (value_counts),
           class balance (percentage per class).
        4. For each non-target, non-id column, call _detect_column_types
           to classify columns and compute per-column stats (missing_pct,
           cardinality, top_values, numeric stats).
        5. Call _compute_correlations to get target correlations and the
           full correlation matrix for numeric columns.
        6. Merge target_correlation into the per-column analysis.
        7. Call _find_feature_clusters on the correlation matrix.
        8. Call _identify_weak_features with threshold=0.05.
        9. Generate the recommendations list based on:
           - Strongest correlated features (suggest target-encoded pairs)
           - Weak features (suggest dropping or interactions)
           - Feature clusters (suggest keeping one or creating ratios)
           - Binary features (suggest interactions with top continuous)
           - Missing value patterns (suggest imputation strategy)
        10. Assemble and return the complete report dict.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Apply target mapping (e.g., {"Presence": 1, "Absence": 0})
    if target_mapping and target_col in train.columns:
        train[target_col] = train[target_col].map(target_mapping)

    # Auto-detect binary string targets (e.g., "Yes"/"No") and map to 0/1
    if (
        target_col in train.columns
        and not pd.api.types.is_numeric_dtype(train[target_col])
        and task_type == "binary_classification"
    ):
        unique_vals = sorted(train[target_col].dropna().unique().tolist())
        if len(unique_vals) == 2:
            auto_map = {unique_vals[0]: 0, unique_vals[1]: 1}
            train[target_col] = train[target_col].map(auto_map).astype(int)
            logging.getLogger(__name__).info(
                "Auto-mapped string target %s: %s", target_col, auto_map,
            )

    # Dataset info
    train_mem = train.memory_usage(deep=True).sum() / 1024 / 1024
    test_mem = test.memory_usage(deep=True).sum() / 1024 / 1024
    dataset_info = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_memory_mb": round(float(train_mem), 3),
        "test_memory_mb": round(float(test_mem), 3),
        "n_features": train.shape[1] - 1 - (1 if id_col and id_col in train.columns else 0),
    }

    # Target analysis
    target_series = train[target_col]
    vc = target_series.value_counts()
    vc_pct = target_series.value_counts(normalize=True) * 100
    target_analysis = {
        "dtype": str(target_series.dtype),
        "n_unique": int(target_series.nunique()),
        "distribution": {str(k): int(v) for k, v in vc.items()},
        "class_balance_pct": {str(k): round(float(v), 2) for k, v in vc_pct.items()},
        "missing_pct": round(float(target_series.isna().mean() * 100), 3),
    }

    # Feature columns (exclude target and id)
    feature_cols = [
        c for c in train.columns
        if c != target_col and (id_col is None or c != id_col)
    ]
    feature_df = train[feature_cols]

    # Per-column analysis
    columns_analysis = _detect_column_types(feature_df)

    # Skewness and outliers (Feature 6)
    _add_skewness_and_outliers(columns_analysis, feature_df)

    # Sentinel detection (masked missing values like -1, -999)
    _detect_sentinels(columns_analysis, feature_df)

    # Correlations (pass feature_df + target to exclude id_col)
    corr_result = _compute_correlations(
        pd.concat([feature_df, train[[target_col]]], axis=1),
        target_col,
    )
    target_correlations = corr_result["target_correlations"]
    correlation_matrix = corr_result["correlation_matrix"]

    # Merge target_correlation into per-column analysis
    for col_name, col_info in columns_analysis.items():
        col_info["target_correlation"] = round(float(target_correlations.get(col_name, 0.0)), 6)

    # Mutual information (Feature 2)
    mi_scores = _compute_mutual_information(feature_df, target_series, task_type)
    for col_name in columns_analysis:
        columns_analysis[col_name]["mutual_information"] = round(
            float(mi_scores.get(col_name, 0.0)), 6
        )

    # Categorical target rates (Feature 1)
    cat_target_rates = _compute_categorical_target_rates(
        feature_df, target_series, columns_analysis, task_type
    )
    for col_name in columns_analysis:
        if col_name in cat_target_rates:
            columns_analysis[col_name]["target_rate_per_value"] = cat_target_rates[col_name]["target_rate_per_value"]
            columns_analysis[col_name]["target_rate_max_delta"] = cat_target_rates[col_name]["target_rate_max_delta"]
        else:
            columns_analysis[col_name]["target_rate_per_value"] = None
            columns_analysis[col_name]["target_rate_max_delta"] = None

    # Feature clusters + pairwise enrichment (Feature 4)
    feature_clusters = _find_feature_clusters(correlation_matrix, threshold=0.5)
    feature_clusters = _enrich_clusters_with_pairs(feature_clusters, correlation_matrix)

    # Distribution shift (Feature 3)
    test_feature_df = test[[c for c in feature_cols if c in test.columns]]
    distribution_shift = _compute_distribution_shift(
        feature_df, test_feature_df, columns_analysis
    )

    # Interaction screening (Feature 5)
    interaction_candidates = _screen_interactions(
        feature_df, target_series, mi_scores, columns_analysis, top_n=10
    )

    # Leakage detection (Feature 7)
    leakage_warnings = _detect_leakage(columns_analysis, mi_scores, target_correlations)

    # VIF (Feature 8)
    vif_scores = _compute_vif(feature_df, columns_analysis)

    # Univariate AUC (binary classification only)
    univariate_auc = _compute_univariate_auc(feature_df, target_series)
    for col_name in columns_analysis:
        columns_analysis[col_name]["univariate_auc"] = univariate_auc.get(col_name)

    # IV/WoE (binary classification only)
    iv_woe = _compute_iv_woe(feature_df, target_series, columns_analysis)
    for col_name in columns_analysis:
        if col_name in iv_woe:
            columns_analysis[col_name]["iv"] = iv_woe[col_name]["iv"]
            columns_analysis[col_name]["iv_label"] = iv_woe[col_name]["iv_label"]
        else:
            columns_analysis[col_name]["iv"] = None
            columns_analysis[col_name]["iv_label"] = None

    # Cramér's V (categorical-categorical associations)
    cramers_v = _compute_cramers_v(feature_df, columns_analysis)

    # Duplicate and conflicting row detection
    duplicates = _detect_duplicates(feature_df, target_series, columns_analysis)

    # Unseen categories in test
    unseen_categories = _compute_unseen_categories(
        feature_df, test_feature_df, columns_analysis
    )

    # Monotonicity detection
    monotonicity = _detect_monotonicity(feature_df, target_series, columns_analysis)

    # Categorical cardinality profiles
    cardinality_profiles = _compute_cardinality_profile(feature_df, columns_analysis)

    # Target encoding preview
    te_preview = _compute_target_encoding_preview(
        feature_df, target_series, columns_analysis
    )

    # Multi-model baseline (replaces single RF quick_model)
    multi_baseline = _compute_multi_baseline(feature_df, target_series, task_type)

    # Backward-compat alias for quick_model
    quick_model = {
        "feature_importances": multi_baseline.get("feature_importances", {}),
        "baseline_score": multi_baseline.get("best_score"),
        "baseline_metric": multi_baseline.get("metric", "N/A"),
    }

    # Interaction orchestra (LightGBM split-gain pairs)
    interaction_orchestra = _compute_interaction_orchestra(
        feature_df, target_series, task_type,
        lgbm_models=multi_baseline.get("lgbm_models"),
    )

    # Ghost feature detector (cross-model importance comparison)
    ghost_features = _compute_ghost_features(
        multi_baseline.get("all_importances", {}),
    )

    # Prediction diversity probe (multi-seed RF correlation)
    prediction_diversity = _compute_prediction_diversity_probe(
        feature_df, target_series, task_type
    )

    # Weak features
    weak_features = _identify_weak_features(target_correlations, threshold=0.05)

    # Generate recommendations
    recommendations = _generate_recommendations(
        columns_analysis, target_correlations, feature_clusters, weak_features,
        mi_scores=mi_scores,
        leakage_warnings=leakage_warnings,
        distribution_shift=distribution_shift,
        iv_woe=iv_woe,
        univariate_auc=univariate_auc,
        duplicates=duplicates,
        unseen_categories=unseen_categories,
        monotonicity=monotonicity,
        te_preview=te_preview,
        quick_model=quick_model,
        prediction_diversity=prediction_diversity,
        multi_baseline=multi_baseline,
        interaction_orchestra=interaction_orchestra,
        ghost_features=ghost_features,
    )

    # Preprocessing summary: aggregate scaling/transform signals for LLM
    preprocessing_summary = _build_preprocessing_summary(columns_analysis)

    report = {
        "dataset_info": dataset_info,
        "target_analysis": target_analysis,
        "columns": columns_analysis,
        "correlation_matrix": correlation_matrix,
        "feature_clusters": feature_clusters,
        "weak_features": weak_features,
        "recommendations": recommendations,
        "mutual_information": mi_scores,
        "distribution_shift": distribution_shift,
        "interaction_candidates": interaction_candidates,
        "leakage_warnings": leakage_warnings,
        "vif_scores": vif_scores,
        "univariate_auc": univariate_auc,
        "iv_woe": iv_woe,
        "cramers_v": cramers_v,
        "preprocessing_summary": preprocessing_summary,
        "duplicates": duplicates,
        "unseen_categories": unseen_categories,
        "monotonicity": monotonicity,
        "cardinality_profiles": cardinality_profiles,
        "te_preview": te_preview,
        "quick_model": quick_model,
        "prediction_diversity": prediction_diversity,
        "multi_baseline": {
            k: v for k, v in multi_baseline.items()
            if k not in ("lgbm_models", "all_importances")
        },
        "interaction_orchestra": interaction_orchestra,
        "ghost_features": ghost_features,
    }
    return report, train, test


def _build_preprocessing_summary(columns_analysis: dict[str, dict]) -> dict[str, Any]:
    """Build a preprocessing summary from column analysis for LLM decision-making.

    Aggregates scaling-relevant signals across all numeric features:
    - Scale range ratio (max range / min range) — indicates if features
      are on vastly different scales
    - Count of high-skewness features (candidates for log transform)
    - Count of features with significant outliers (candidates for clipping/RobustScaler)
    - Count of features with sentinel values (candidates for NaN replacement)
    - Suggested scaler based on data characteristics

    Args:
        columns_analysis: Per-column analysis dict from _detect_column_types
            enriched with skewness/outlier/sentinel data.

    Returns:
        Dict with keys: scale_range_ratio, n_high_skew, n_high_outlier,
        n_sentinel_features, high_skew_features, high_outlier_features,
        sentinel_features, suggested_scalers.
    """
    numeric_ranges: list[tuple[str, float]] = []
    high_skew: list[str] = []
    high_outlier: list[str] = []
    sentinel_feats: list[str] = []

    for col, info in columns_analysis.items():
        if info.get("detected_type") not in ("numeric_continuous",):
            continue
        stats = info.get("stats")
        if stats and stats.get("range") is not None and stats["range"] > 0:
            numeric_ranges.append((col, stats["range"]))
        skew_label = info.get("skewness_label")
        if skew_label == "high":
            high_skew.append(col)
        outlier_pct = info.get("outlier_pct", 0.0)
        if outlier_pct is not None and outlier_pct > 3.0:
            high_outlier.append(col)
        if info.get("sentinels"):
            sentinel_feats.append(col)

    # Scale range ratio
    if len(numeric_ranges) >= 2:
        ranges = [r for _, r in numeric_ranges]
        min_range = min(r for r in ranges if r > 0)
        max_range = max(ranges)
        scale_range_ratio = round(max_range / min_range, 2) if min_range > 0 else 0.0
    else:
        scale_range_ratio = 1.0

    # Suggest scalers based on data characteristics
    suggested_scalers: list[str] = ["standard"]
    if high_outlier:
        suggested_scalers = ["robust", "quantile"]
    if scale_range_ratio > 100:
        if "standard" in suggested_scalers:
            suggested_scalers.remove("standard")
        if "robust" not in suggested_scalers:
            suggested_scalers.insert(0, "robust")

    return {
        "scale_range_ratio": scale_range_ratio,
        "n_high_skew": len(high_skew),
        "n_high_outlier": len(high_outlier),
        "n_sentinel_features": len(sentinel_feats),
        "high_skew_features": high_skew,
        "high_outlier_features": high_outlier,
        "sentinel_features": sentinel_feats,
        "suggested_scalers": suggested_scalers,
    }


def _generate_recommendations(
    columns_analysis: dict,
    target_correlations: dict,
    feature_clusters: list,
    weak_features: list,
    mi_scores: dict[str, float] | None = None,
    leakage_warnings: list[dict] | None = None,
    distribution_shift: dict | None = None,
    iv_woe: dict[str, dict] | None = None,
    univariate_auc: dict[str, float] | None = None,
    duplicates: dict[str, Any] | None = None,
    unseen_categories: dict[str, dict] | None = None,
    monotonicity: dict[str, dict] | None = None,
    te_preview: dict[str, dict] | None = None,
    quick_model: dict[str, Any] | None = None,
    prediction_diversity: dict[str, Any] | None = None,
    multi_baseline: dict[str, Any] | None = None,
    interaction_orchestra: dict[str, Any] | None = None,
    ghost_features: dict[str, Any] | None = None,
) -> list[str]:
    """Generate LLM-readable recommendation strings from EDA results."""
    recs = []

    # Leakage warnings (highest priority)
    if leakage_warnings:
        for w in leakage_warnings:
            recs.append(
                f"LEAKAGE WARNING: {w['column']} has suspiciously high predictive "
                f"power ({w['reason']}={w['value']:.3f}). Verify this is not data leakage."
            )

    # Duplicate/conflict warnings
    if duplicates and duplicates.get("n_conflicting_rows", 0) > 0:
        recs.append(
            f"CONFLICTING DUPLICATES: {duplicates['n_conflicting_rows']} rows "
            f"({duplicates['conflicting_pct']:.1f}%) have identical features but "
            f"different targets. This sets a hard ceiling on model performance — "
            f"no feature engineering can resolve these."
        )
    elif duplicates and duplicates.get("duplicate_pct", 0) > 5:
        recs.append(
            f"High duplicate rate: {duplicates['duplicate_pct']:.1f}% of rows are "
            f"exact duplicates (but same target). Consider deduplication or "
            f"sample weighting."
        )

    # Signal-only duplicates (noise features mask real duplicates)
    signal_dupes = duplicates.get("signal_only", {}) if duplicates else {}
    if signal_dupes and signal_dupes.get("n_conflicting_rows", 0) > 0:
        n_dropped = len(signal_dupes.get("dropped_columns", []))
        recs.append(
            f"HIDDEN CONFLICTS: After dropping {n_dropped} noise columns, "
            f"{signal_dupes['n_conflicting_rows']} rows "
            f"({signal_dupes['conflicting_pct']:.1f}%) have identical signal features "
            f"but different targets. Noise features mask real conflicts — drop them "
            f"to reveal the true performance ceiling."
        )
    elif signal_dupes and signal_dupes.get("n_duplicate_rows", 0) > 0 and duplicates.get("n_duplicate_rows", 0) == 0:
        n_dropped = len(signal_dupes.get("dropped_columns", []))
        recs.append(
            f"HIDDEN DUPLICATES: {signal_dupes['n_duplicate_rows']} rows "
            f"({signal_dupes['duplicate_pct']:.1f}%) are duplicates in signal features "
            f"but differ only in {n_dropped} noise columns. Dropping noise features "
            f"will expose this redundancy."
        )

    # Quick model baseline
    if quick_model and quick_model.get("baseline_score") is not None:
        metric = quick_model["baseline_metric"]
        score = quick_model["baseline_score"]
        recs.append(
            f"Quick RF baseline (no feature engineering): {metric}={score:.4f}. "
            f"This is the performance floor — feature engineering should improve on this."
        )

    # Prediction diversity probe
    if prediction_diversity and prediction_diversity.get("diversity_class"):
        dc = prediction_diversity["diversity_class"]
        snr = prediction_diversity.get("signal_noise_ratio", 0)
        wss = prediction_diversity.get("within_seed_std", 0)
        pstd = prediction_diversity.get("prediction_std", 0)
        if dc == "very_low":
            recs.append(
                f"VERY LOW SIGNAL DIVERSITY: SNR={snr:.1f} "
                f"(within-seed std={wss:.4f}, across-seed std={pstd:.4f}). "
                f"Seed noise dominates — all models will converge to similar "
                f"predictions. Enable tiered tracker (diversity_mode: tiered) and "
                f"diversity pruning for neural nets. Use aggressive "
                f"tier2_corr_threshold (0.99) and corr_threshold (0.995)."
            )
        elif dc == "low":
            recs.append(
                f"LOW SIGNAL DIVERSITY: SNR={snr:.1f} "
                f"(within-seed std={wss:.4f}, across-seed std={pstd:.4f}). "
                f"Weak signal relative to seed noise. Consider tiered tracker "
                f"and diversity pruning for neural nets to avoid wasting compute "
                f"on redundant trials."
            )
        elif dc == "moderate":
            recs.append(
                f"Moderate signal diversity: SNR={snr:.1f}. Some diversity "
                f"benefit from tiered tracker; diversity pruning optional."
            )
        # "high" diversity — no special recommendation needed

    # Monotonicity signals
    if monotonicity:
        mono_feats = [
            (col, info["direction"]) for col, info in monotonicity.items()
            if info["is_monotonic"]
        ]
        if mono_feats:
            desc = ", ".join(f"{c}({d})" for c, d in mono_feats[:8])
            recs.append(
                f"Monotonic features: {desc}. "
                "Consider monotone_constraints in gradient boosting for regularization."
            )

    # Target encoding preview
    if te_preview:
        strong_te = [
            (col, info["encoded_corr"]) for col, info in te_preview.items()
            if abs(info["encoded_corr"]) > 0.05
        ]
        if strong_te:
            desc = ", ".join(f"{c}(corr={corr:+.3f})" for c, corr in strong_te[:5])
            recs.append(
                f"Target encoding effective for: {desc}. "
                "Prioritize these for OOF target encoding."
            )

    # Unseen categories
    if unseen_categories:
        high_unseen = [
            (col, info) for col, info in unseen_categories.items()
            if info["unseen_row_pct"] > 1.0
        ]
        if high_unseen:
            desc = ", ".join(
                f"{c}({info['n_unseen']} unseen, {info['unseen_row_pct']:.1f}% rows)"
                for c, info in high_unseen[:5]
            )
            recs.append(
                f"Unseen test categories: {desc}. "
                "Increase target encoding smoothing alpha or use frequency encoding fallback."
            )

    # Strong IV features
    if iv_woe:
        strong_iv = [
            (col, info["iv"]) for col, info in iv_woe.items()
            if info["iv_label"] in ("medium", "strong")
        ]
        if strong_iv:
            strong_iv.sort(key=lambda x: x[1], reverse=True)
            desc = ", ".join(f"{c}(IV={iv:.3f})" for c, iv in strong_iv[:5])
            recs.append(
                f"Strong Information Value features: {desc}. "
                "These have high discriminative power for the target."
            )

    # Near-zero-variance features
    nzv_cols = [
        (c, info["dominant_pct"]) for c, info in columns_analysis.items()
        if info.get("dominant_pct", 0) > 99.0
    ]
    if nzv_cols:
        desc = ", ".join(f"{c}({pct:.1f}%)" for c, pct in nzv_cols[:5])
        recs.append(
            f"Near-zero-variance features (>99% same value): {desc}. "
            "Consider dropping — too little information for most models."
        )

    # Univariate AUC highlights
    if univariate_auc:
        high_auc = [(c, a) for c, a in univariate_auc.items() if a > 0.55]
        if high_auc:
            desc = ", ".join(f"{c}(AUC={a:.3f})" for c, a in high_auc[:5])
            recs.append(f"Top univariate AUC features: {desc}.")

    # MI vs Pearson discrepancy for categoricals
    if mi_scores:
        for col, col_info in columns_analysis.items():
            if col_info["detected_type"] in ("low_cardinality_categorical", "high_cardinality_categorical"):
                mi = mi_scores.get(col, 0.0)
                corr = abs(col_info.get("target_correlation", 0.0))
                if mi > 0.02 and corr < 0.01:
                    recs.append(
                        f"MI reveals {col} has MI={mi:.3f} but Pearson corr≈0 — "
                        "categorical feature with real predictive power. Prioritize target encoding."
                    )

    # Distribution shift warnings
    if distribution_shift:
        flagged = distribution_shift.get("flagged_columns", [])
        if flagged:
            recs.append(
                f"Distribution shift detected in: {', '.join(flagged)}. "
                "Consider robust features or binning for these columns."
            )

    # Top correlated features → suggest target encoding or interactions
    sorted_corrs = sorted(
        target_correlations.items(), key=lambda x: abs(x[1]), reverse=True
    )
    top_features = [f for f, _ in sorted_corrs[:5]]
    if top_features:
        recs.append(
            f"Top correlated features: {', '.join(top_features)}. "
            "Consider creating interaction features between these."
        )

    # Weak features → suggest dropping or interactions
    weak_cols = [w["column"] for w in weak_features]
    if weak_cols:
        recs.append(
            f"Weak features (|corr| < 0.05): {', '.join(weak_cols[:10])}. "
            "Consider dropping or using only in interactions with stronger features."
        )

    # Feature clusters → suggest ratios
    for cluster in feature_clusters[:3]:
        members = cluster["features"]
        mean_corr = cluster["mean_internal_corr"]
        recs.append(
            f"Feature cluster (mean corr={mean_corr:.2f}): {', '.join(members)}. "
            "These are highly correlated — consider keeping only the strongest, "
            "or create ratio features between them."
        )

    # Binary features → suggest interactions with top continuous
    binary_cols = [
        c for c, info in columns_analysis.items()
        if info["detected_type"] == "binary"
    ]
    continuous_cols = [
        c for c, info in columns_analysis.items()
        if info["detected_type"] == "numeric_continuous"
    ]
    if binary_cols and continuous_cols:
        b_sample = binary_cols[:3]
        c_sample = continuous_cols[:3]
        recs.append(
            f"Binary features ({', '.join(b_sample)}) can be multiplied with "
            f"top continuous features ({', '.join(c_sample)}) to create "
            "conditional interaction features."
        )

    # Categorical features → suggest target encoding
    cat_cols = [
        c for c, info in columns_analysis.items()
        if info["detected_type"] in ("low_cardinality_categorical", "high_cardinality_categorical")
    ]
    if cat_cols:
        recs.append(
            f"Categorical features: {', '.join(cat_cols[:5])}. "
            "Apply OOF target encoding with smoothing (alpha=15) to avoid leakage."
        )

    # Missing values
    missing_cols = [
        (c, info["missing_pct"])
        for c, info in columns_analysis.items()
        if info["missing_pct"] > 0
    ]
    if missing_cols:
        missing_desc = ", ".join(f"{c}({pct:.1f}%)" for c, pct in missing_cols[:5])
        recs.append(
            f"Columns with missing values: {missing_desc}. "
            "Tree-based models handle these natively; linear/KNN models need imputation."
        )

    # ─── Multi-baseline personality ──────────────────────────────────
    if multi_baseline and multi_baseline.get("personality"):
        p = multi_baseline["personality"]
        detail = multi_baseline.get("personality_detail", "")
        lg = multi_baseline.get("linear_gap")

        recs.append(f"DATA PERSONALITY: {p.upper()}. {detail}")

        if lg is not None and lg > 0.05:
            recs.append(
                f"Linear gap = {lg:.4f} — significant non-linear patterns. "
                "Use interaction features, target encoding, and tree models."
            )
        elif lg is not None and lg < 0.005:
            recs.append(
                f"Linear gap = {lg:.4f} — linear model is competitive. "
                "Data may be linearly separable; simpler models may suffice."
            )

    # ─── Interaction orchestra ──────────────────────────────────
    if interaction_orchestra:
        top_int = interaction_orchestra.get("top_interactions", [])
        if top_int:
            top5 = top_int[:5]
            desc = ", ".join(
                f"{i['feature_a']}*{i['feature_b']}({i['interaction_strength']:.3f})"
                for i in top5
            )
            recs.append(
                f"Top LightGBM interaction pairs: {desc}. "
                "Prioritize these for explicit interaction features."
            )

    # ─── Ghost feature warnings ──────────────────────────────────
    if ghost_features:
        ghosts = ghost_features.get("ghost_features", [])
        high_ghosts = [g for g in ghosts if g["severity"] == "high"]
        if high_ghosts:
            names = ", ".join(g["feature"] for g in high_ghosts[:5])
            recs.append(
                f"GHOST FEATURES (high severity): {names}. "
                "Important in one model but irrelevant in others. "
                "Investigate for leakage, MDI bias, or data artifacts."
            )

    return recs


def _compute_correlations(
    df: pd.DataFrame,
    target_col: str,
) -> dict[str, Any]:
    """Compute target correlations and the full correlation matrix.

    Args:
        df: Training DataFrame (numeric columns only will be used).
        target_col: Name of the target column.

    Returns:
        Dictionary with:
        - target_correlations: dict of {column_name: correlation_value},
          sorted by absolute value descending.
        - correlation_matrix: dict with 'columns' (list of names) and
          'values' (nested list of floats).

    Steps:
        1. Select only numeric columns from df.
        2. Compute Pearson correlation of each numeric column with target.
        3. Sort by absolute correlation value, descending.
        4. Compute the full pairwise correlation matrix.
        5. Convert the matrix to the schema format: {columns: [...],
           values: [[...], ...]}.
        6. Return both target_correlations and correlation_matrix.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    # Pearson correlation of each column with target
    if target_col in numeric_df.columns:
        target_corrs = numeric_df.corrwith(numeric_df[target_col])
        target_corrs = target_corrs.drop(target_col, errors="ignore")
    else:
        # Target is non-numeric (not in numeric_df); encode as category codes for correlation.
        # Unconditional: covers object, StringDtype (pandas 2.x), CategoricalDtype, etc.
        target_series = pd.Categorical(df[target_col]).codes
        feature_numeric = numeric_df.drop(columns=[target_col], errors="ignore")
        target_corrs = feature_numeric.corrwith(pd.Series(target_series, name=target_col))

    target_corrs = target_corrs.fillna(0.0)
    # Sort by absolute value descending
    target_corrs_sorted = target_corrs.reindex(
        target_corrs.abs().sort_values(ascending=False).index
    )
    target_correlations = {col: float(val) for col, val in target_corrs_sorted.items()}

    # Full pairwise correlation matrix (features only, no target)
    feature_cols_num = [c for c in numeric_df.columns if c != target_col]
    if feature_cols_num:
        corr_matrix_df = numeric_df[feature_cols_num].corr().fillna(0.0)
        corr_columns = list(corr_matrix_df.columns)
        corr_values = corr_matrix_df.values.tolist()
    else:
        corr_columns = []
        corr_values = []

    correlation_matrix = {
        "columns": corr_columns,
        "values": [[round(v, 6) for v in row] for row in corr_values],
    }

    return {
        "target_correlations": target_correlations,
        "correlation_matrix": correlation_matrix,
    }


def _detect_column_types(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Classify each column and compute per-column statistics.

    Args:
        df: DataFrame to analyze (all columns except target and id).

    Returns:
        Dictionary keyed by column name, each value is a dict with:
        - dtype: string representation of pandas dtype
        - detected_type: one of 'numeric_continuous', 'binary',
          'low_cardinality_categorical', 'ordinal',
          'high_cardinality_categorical'
        - missing_pct: percentage of missing values (0.0-100.0)
        - cardinality: number of unique values
        - top_values: dict of top 10 values by frequency
        - stats: {mean, std, min, max, median} for numeric, else None

    Steps:
        1. For each column in df:
           a. Compute dtype, missing_pct, cardinality.
           b. Compute top_values (value_counts().head(10) as dict).
           c. If numeric: compute stats dict (mean, std, min, max, median).
           d. Classify detected_type:
              - If cardinality == 2 → 'binary'
              - If dtype is object/category → if cardinality <= 20:
                'low_cardinality_categorical', else
                'high_cardinality_categorical'
              - If numeric and cardinality <= 20 and all values are
                integers → 'ordinal'
              - Else → 'numeric_continuous'
        2. Return the assembled dictionary.
    """
    result = {}

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        missing_pct = round(float(series.isna().mean() * 100), 3)
        cardinality = int(series.nunique(dropna=True))
        top_values = {
            str(k): int(v) for k, v in series.value_counts().head(10).items()
        }

        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_object_or_cat = (
            pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )

        if is_numeric:
            non_null = series.dropna()
            min_val = round(float(non_null.min()), 6) if len(non_null) > 0 else None
            max_val = round(float(non_null.max()), 6) if len(non_null) > 0 else None
            stats = {
                "mean": round(float(non_null.mean()), 6) if len(non_null) > 0 else None,
                "std": round(float(non_null.std()), 6) if len(non_null) > 0 else None,
                "min": min_val,
                "max": max_val,
                "range": round(max_val - min_val, 6) if min_val is not None and max_val is not None else None,
                "median": round(float(non_null.median()), 6) if len(non_null) > 0 else None,
            }
        else:
            stats = None

        # Classify detected_type
        if cardinality == 2:
            detected_type = "binary"
        elif is_object_or_cat:
            detected_type = (
                "low_cardinality_categorical" if cardinality <= 20
                else "high_cardinality_categorical"
            )
        elif is_numeric and cardinality <= 20:
            # non_null already computed above in the stats block (is_numeric is True here)
            if len(non_null) > 0 and (non_null == non_null.round()).all():
                detected_type = "ordinal"
            else:
                detected_type = "numeric_continuous"
        else:
            detected_type = "numeric_continuous"

        # Near-zero-variance: percentage of most frequent value
        n_total = len(series)
        if n_total > 0:
            dominant_count = series.value_counts(dropna=True).iloc[0] if series.notna().any() else 0
            dominant_pct = round(float(dominant_count / n_total * 100), 2)
        else:
            dominant_pct = 0.0

        result[col] = {
            "dtype": dtype_str,
            "detected_type": detected_type,
            "missing_pct": missing_pct,
            "cardinality": cardinality,
            "top_values": top_values,
            "stats": stats,
            "dominant_pct": dominant_pct,
            "target_correlation": 0.0,  # filled in by run_eda
        }

    return result


def _find_feature_clusters(
    corr_matrix: dict[str, Any],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Find groups of features that are highly correlated with each other.

    Args:
        corr_matrix: Correlation matrix dict with 'columns' and 'values'
                     keys (as produced by _compute_correlations).
        threshold: Minimum absolute correlation to consider features
                   as belonging to the same cluster (default: 0.5).

    Returns:
        List of cluster dicts, each containing:
        - features: list of column names in this cluster
        - mean_internal_corr: average pairwise |correlation| within cluster

    Steps:
        1. Convert the correlation matrix values to a numpy array.
        2. Build an adjacency graph: two features are connected if their
           |correlation| >= threshold (excluding self-correlations).
        3. Find connected components using a simple BFS/DFS or union-find.
        4. Filter out singleton clusters (only keep groups of 2+).
        5. For each cluster, compute mean_internal_corr as the average
           of all pairwise |correlations| within the group.
        6. Sort clusters by size descending.
        7. Return the list of cluster dicts.
    """
    columns = corr_matrix.get("columns", [])
    values = corr_matrix.get("values", [])

    if not columns or not values:
        return []

    mat = np.array(values)
    n = len(columns)

    # Build adjacency list
    adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if abs(mat[i, j]) >= threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)

    # BFS to find connected components
    visited = [False] * n
    components = []
    for start in range(n):
        if visited[start]:
            continue
        component = []
        queue = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(component)

    # Filter singletons and compute stats
    clusters = []
    for component in components:
        if len(component) < 2:
            continue
        feature_names = [columns[i] for i in component]
        # Compute mean internal correlation
        pairs = []
        for i in range(len(component)):
            for j in range(i + 1, len(component)):
                pairs.append(abs(mat[component[i], component[j]]))
        mean_internal_corr = float(np.mean(pairs)) if pairs else 0.0
        clusters.append({
            "features": feature_names,
            "mean_internal_corr": round(mean_internal_corr, 4),
        })

    # Sort by cluster size descending
    clusters.sort(key=lambda c: len(c["features"]), reverse=True)
    return clusters


def _identify_weak_features(
    target_correlations: dict[str, float],
    threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Identify features with very low target correlation.

    Args:
        target_correlations: Dict of {column_name: correlation_value}
                             (from _compute_correlations).
        threshold: Maximum |correlation| to consider a feature weak
                   (default: 0.05).

    Returns:
        List of dicts, each containing:
        - column: feature name
        - target_correlation: actual correlation value
        - recommendation: 'consider_dropping' if not correlated with
          any strong features, else 'try_interactions'

    Steps:
        1. Filter target_correlations to features where
           |correlation| < threshold.
        2. For each weak feature, set recommendation to
           'consider_dropping' (the strategist may upgrade this to
           'try_interactions' if the feature is correlated with strong
           predictors — but this simple version defaults conservatively).
        3. Sort by |correlation| ascending.
        4. Return the list of weak feature dicts.
    """
    weak = [
        {
            "column": col,
            "target_correlation": float(corr),
            "recommendation": "consider_dropping",
        }
        for col, corr in target_correlations.items()
        if abs(corr) < threshold
    ]
    weak.sort(key=lambda x: abs(x["target_correlation"]))
    return weak


def format_eda_for_llm(eda_report: dict) -> str:
    """Format the EDA report as a human/LLM-readable text summary.

    Used in manual mode: the output is printed to console so the user
    can copy-paste it into a chat with Claude or ChatGPT.

    Args:
        eda_report: Complete EDA report dict (as returned by run_eda).

    Returns:
        Formatted string containing:
        - Dataset overview (shapes, memory)
        - Target distribution
        - Top features by target correlation (table format)
        - Weak features list
        - Feature clusters
        - Missing value summary
        - Column type breakdown
        - Recommendations (numbered list)

    Steps:
        1. Build a header with dataset shape and memory info.
        2. Format target distribution as a simple table.
        3. Sort columns by |target_correlation| and format as a table
           with columns: Feature, Correlation, Type, Missing%.
        4. List weak features with their correlations.
        5. List feature clusters with member features and mean correlation.
        6. Summarize missing values (only columns with > 0% missing).
        7. Count columns by detected_type.
        8. Number and list all recommendations.
        9. Join all sections with clear headers and separators.
        10. Return the complete formatted string.
    """
    sep = "=" * 70
    lines = []

    # Header / dataset overview
    info = eda_report.get("dataset_info", {})
    lines.append(sep)
    lines.append("MAESTRO-ML EDA REPORT")
    lines.append(sep)
    lines.append(
        f"Train shape : {info.get('train_shape', 'N/A')}  "
        f"({info.get('train_memory_mb', 0):.1f} MB)"
    )
    lines.append(
        f"Test shape  : {info.get('test_shape', 'N/A')}  "
        f"({info.get('test_memory_mb', 0):.1f} MB)"
    )
    lines.append(f"Features    : {info.get('n_features', 'N/A')}")
    train_shape = info.get("train_shape", [0, 0])
    if isinstance(train_shape, list) and len(train_shape) >= 1 and train_shape[0] > 0:
        n_rows = train_shape[0]
        for n_folds in (5, 10):
            fold_train = n_rows * (n_folds - 1) // n_folds
            fold_val = n_rows // n_folds
            lines.append(
                f"  {n_folds}-fold CV  : ~{fold_train:,} train / ~{fold_val:,} val per fold"
            )
    lines.append("")

    # Target distribution
    target = eda_report.get("target_analysis", {})
    lines.append("─" * 40)
    lines.append("TARGET DISTRIBUTION")
    lines.append("─" * 40)
    dist = target.get("distribution", {})
    pct = target.get("class_balance_pct", {})
    for k in dist:
        lines.append(f"  {k:>10s} : {dist[k]:>8d}  ({pct.get(k, 0):.2f}%)")
    lines.append(f"  dtype={target.get('dtype', '')}, unique={target.get('n_unique', '')}")
    lines.append("")

    # Feature correlation table (with MI, AUC, IV and optional extra)
    columns = eda_report.get("columns", {})
    sorted_cols = sorted(
        columns.items(),
        key=lambda kv: abs(kv[1].get("target_correlation", 0)),
        reverse=True,
    )
    has_auc = any(ci.get("univariate_auc") is not None for _, ci in sorted_cols)
    has_iv = any(ci.get("iv") is not None for _, ci in sorted_cols)
    header = f"{'FEATURE':<28} {'CORR':>7} {'MI':>6}"
    if has_auc:
        header += f" {'AUC':>5}"
    if has_iv:
        header += f" {'IV':>6} {'IVlbl':>6}"
    header += f" {'TYPE':<25} {'MISS%':>5} {'EXTRA':>12}"
    lines.append("─" * len(header))
    lines.append(header)
    lines.append("─" * len(header))
    for col_name, col_info in sorted_cols:
        corr = col_info.get("target_correlation", 0.0)
        mi = col_info.get("mutual_information", 0.0)
        dtype = col_info.get("detected_type", "")
        miss = col_info.get("missing_pct", 0.0)
        row = f"{col_name:<28} {corr:>+7.4f} {mi:>6.4f}"
        if has_auc:
            auc = col_info.get("univariate_auc")
            row += f" {auc:>5.3f}" if auc is not None else "      "
        if has_iv:
            iv = col_info.get("iv")
            iv_lbl = col_info.get("iv_label", "")
            row += f" {iv:>6.3f} {iv_lbl:>6}" if iv is not None else "              "
        # Extra: kurtosis+skewness+outlier for numeric, Δrate for categorical
        extra = ""
        skew = col_info.get("skewness")
        skew_lbl = col_info.get("skewness_label", "")
        outlier = col_info.get("outlier_pct")
        delta = col_info.get("target_rate_max_delta")
        sentinels = col_info.get("sentinels", [])
        if delta is not None:
            extra = f"Δ={delta:.2f}"
        elif skew is not None:
            extra = f"sk={skew:+.1f}({skew_lbl[:3]}) o={outlier:.1f}%" if outlier else f"sk={skew:+.1f}({skew_lbl[:3]})"
        if sentinels:
            sentinel_str = ",".join(str(int(s["value"])) if s["value"] == int(s["value"]) else str(s["value"]) for s in sentinels[:2])
            extra += f" S[{sentinel_str}]"
        row += f" {dtype:<25} {miss:>5.1f} {extra:>12}"
        lines.append(row)
    lines.append("")

    # Weak features
    weak = eda_report.get("weak_features", [])
    lines.append("─" * 40)
    lines.append("WEAK FEATURES (|corr| < 0.05)")
    lines.append("─" * 40)
    if weak:
        for w in weak:
            lines.append(
                f"  {w['column']:<30} corr={w['target_correlation']:+.4f}  → {w['recommendation']}"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    # Feature clusters (with pairwise details)
    clusters = eda_report.get("feature_clusters", [])
    lines.append("─" * 40)
    lines.append("FEATURE CLUSTERS (|corr| >= 0.5)")
    lines.append("─" * 40)
    if clusters:
        for i, c in enumerate(clusters, 1):
            members = ", ".join(c["features"])
            lines.append(f"  Cluster {i} (mean_corr={c['mean_internal_corr']:.2f}): {members}")
            pairs = c.get("pairs", [])
            if pairs:
                pair_strs = [
                    f"{p['features'][0]}/{p['features'][1]}={p['correlation']:+.2f}"
                    for p in pairs
                ]
                lines.append(f"    Pairs: {', '.join(pair_strs)}")
    else:
        lines.append("  (no clusters found)")
    lines.append("")

    # Missing values
    missing_cols = [
        (col, info["missing_pct"])
        for col, info in columns.items()
        if info.get("missing_pct", 0) > 0
    ]
    lines.append("─" * 40)
    lines.append("MISSING VALUES")
    lines.append("─" * 40)
    if missing_cols:
        for col, pct_val in sorted(missing_cols, key=lambda x: x[1], reverse=True):
            lines.append(f"  {col:<35} {pct_val:.2f}%")
    else:
        lines.append("  (no missing values)")
    lines.append("")

    # Column type breakdown
    type_counts: dict[str, int] = {}
    for col_info in columns.values():
        t = col_info.get("detected_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    lines.append("─" * 40)
    lines.append("COLUMN TYPE BREAKDOWN")
    lines.append("─" * 40)
    for t, cnt in sorted(type_counts.items()):
        lines.append(f"  {t:<35} {cnt}")
    lines.append("")

    # Mutual information (top 10)
    mi_scores = eda_report.get("mutual_information", {})
    if mi_scores:
        lines.append("─" * 40)
        lines.append("MUTUAL INFORMATION (top 10)")
        lines.append("─" * 40)
        for col, mi in list(mi_scores.items())[:10]:
            lines.append(f"  {col:<35} {mi:>8.4f}")
        lines.append("")

    # Categorical target rates (top features by max_delta)
    cat_rate_cols = [
        (col, info)
        for col, info in columns.items()
        if info.get("target_rate_max_delta") is not None
    ]
    if cat_rate_cols:
        cat_rate_cols.sort(key=lambda x: x[1].get("target_rate_max_delta", 0), reverse=True)
        lines.append("─" * 50)
        lines.append("CATEGORICAL TARGET RATES (by max_delta)")
        lines.append("─" * 50)
        for col_name, col_info in cat_rate_cols[:10]:
            delta = col_info["target_rate_max_delta"]
            strength = "STRONG" if delta > 0.3 else "moderate" if delta > 0.15 else "weak"
            lines.append(f"  {col_name}:  max_delta={delta:.3f}  ({strength})")
            rates = col_info.get("target_rate_per_value", {})
            if rates:
                for val, info_dict in rates.items():
                    if isinstance(info_dict, dict):
                        rate = info_dict.get("rate", 0.0)
                        count = info_dict.get("count", 0)
                        lines.append(f"    {val:<25} rate={rate:.3f}  (n={count})")
        lines.append("")

    # Distribution shift (with PSI)
    dist_shift = eda_report.get("distribution_shift", {})
    flagged_cols = dist_shift.get("flagged_columns", [])
    if dist_shift.get("numeric") or dist_shift.get("categorical"):
        lines.append("─" * 60)
        lines.append("DISTRIBUTION SHIFT (train vs test) — PSI: <0.1 stable, 0.1–0.25 moderate, >0.25 significant")
        lines.append("─" * 60)
        # Show all features with PSI, flagged ones first
        all_shift_items = []
        for cn, info in dist_shift.get("numeric", {}).items():
            all_shift_items.append((cn, info.get("psi", 0.0), info.get("shift_flagged", False), "numeric", info))
        for cn, info in dist_shift.get("categorical", {}).items():
            all_shift_items.append((cn, info.get("psi", 0.0), info.get("shift_flagged", False), "categorical", info))
        all_shift_items.sort(key=lambda x: x[1], reverse=True)
        for cn, psi_val, flagged_flag, shift_type, info in all_shift_items[:15]:
            flag_str = " [FLAGGED]" if flagged_flag else ""
            if shift_type == "numeric":
                lines.append(
                    f"  {cn:<30} PSI={psi_val:.3f}  KS={info['ks_statistic']:.3f}  "
                    f"p={info['ks_pvalue']:.4f}{flag_str}"
                )
            else:
                lines.append(
                    f"  {cn:<30} PSI={psi_val:.3f}  max_delta={info['max_proportion_delta']:.3f}{flag_str}"
                )
        if not all_shift_items:
            lines.append("  (no significant shift detected)")
        lines.append("")

    # Leakage warnings
    leakage = eda_report.get("leakage_warnings", [])
    if leakage:
        lines.append("─" * 50)
        lines.append("LEAKAGE WARNINGS")
        lines.append("─" * 50)
        for w in leakage:
            lines.append(f"  {w['column']:<30} {w['reason']}={w['value']:.3f}")
        lines.append("")

    # VIF scores (only VIF > 5)
    vif = eda_report.get("vif_scores", {})
    high_vif = {k: v for k, v in vif.items() if v > 5.0}
    if high_vif:
        lines.append("─" * 40)
        lines.append("VIF SCORES (VIF > 5)")
        lines.append("─" * 40)
        for col_name, vif_val in high_vif.items():
            lines.append(f"  {col_name:<35} VIF={vif_val:.1f}")
        lines.append("")

    # Interaction candidates (top 5)
    interactions = eda_report.get("interaction_candidates", [])
    if interactions:
        lines.append("─" * 50)
        lines.append("INTERACTION CANDIDATES (top 5)")
        lines.append("─" * 50)
        for ic in interactions[:5]:
            a, b = ic["features"]
            lines.append(
                f"  {a} x {b}: corr={ic['interaction_corr']:.3f}  "
                f"added_value=+{ic['added_value']:.3f}"
            )
        lines.append("")

    # Univariate AUC (top 10)
    uni_auc = eda_report.get("univariate_auc", {})
    if uni_auc:
        lines.append("─" * 40)
        lines.append("UNIVARIATE AUC (top 10)")
        lines.append("─" * 40)
        for col, auc in list(uni_auc.items())[:10]:
            strength = "STRONG" if auc > 0.6 else "moderate" if auc > 0.55 else "weak"
            lines.append(f"  {col:<35} AUC={auc:.4f}  ({strength})")
        lines.append("")

    # Information Value (top features with IV >= weak)
    iv_woe = eda_report.get("iv_woe", {})
    if iv_woe:
        notable_iv = [(c, info) for c, info in iv_woe.items() if info["iv"] >= 0.02]
        if notable_iv:
            lines.append("─" * 50)
            lines.append("INFORMATION VALUE (IV >= 0.02)")
            lines.append("─" * 50)
            for col, info in notable_iv[:15]:
                lines.append(
                    f"  {col:<35} IV={info['iv']:.4f}  ({info['iv_label']})"
                )
            lines.append("")

    # Cramér's V (categorical associations)
    cramers = eda_report.get("cramers_v", {})
    cramers_pairs = cramers.get("pairs", []) if isinstance(cramers, dict) else []
    if cramers_pairs:
        lines.append("─" * 50)
        lines.append("CRAMÉR'S V (categorical associations, V > 0.1)")
        lines.append("─" * 50)
        for p in cramers_pairs[:10]:
            a, b = p["features"]
            lines.append(f"  {a} × {b}: V={p['cramers_v']:.3f}")
        lines.append("")

    # Near-zero-variance
    nzv = [
        (col, info.get("dominant_pct", 0))
        for col, info in columns.items()
        if info.get("dominant_pct", 0) > 95.0
    ]
    if nzv:
        nzv.sort(key=lambda x: x[1], reverse=True)
        lines.append("─" * 40)
        lines.append("NEAR-ZERO-VARIANCE (>95% same value)")
        lines.append("─" * 40)
        for col, dom_pct in nzv:
            lines.append(f"  {col:<35} dominant={dom_pct:.1f}%")
        lines.append("")

    # Duplicate analysis
    dupes = eda_report.get("duplicates", {})
    signal_dupes = dupes.get("signal_only", {}) if dupes else {}
    has_exact = dupes and (dupes.get("n_duplicate_rows", 0) > 0 or dupes.get("n_conflicting_rows", 0) > 0)
    has_signal = signal_dupes and (signal_dupes.get("n_duplicate_rows", 0) > 0 or signal_dupes.get("n_conflicting_rows", 0) > 0)
    if has_exact or has_signal:
        lines.append("─" * 50)
        lines.append("DUPLICATE ANALYSIS")
        lines.append("─" * 50)
        lines.append(f"  Exact duplicate rows : {dupes.get('n_duplicate_rows', 0)} ({dupes.get('duplicate_pct', 0):.1f}%)")
        lines.append(f"  Exact dup groups     : {dupes.get('n_duplicate_groups', 0)}")
        lines.append(f"  Conflicting rows     : {dupes.get('n_conflicting_rows', 0)} ({dupes.get('conflicting_pct', 0):.1f}%)")
        lines.append(f"  Conflicting groups   : {dupes.get('n_conflicting_groups', 0)}")
        if dupes.get("n_conflicting_rows", 0) > 0:
            lines.append("  ⚠ Conflicting duplicates set a hard ceiling on achievable performance.")
        if has_signal:
            n_dropped = len(signal_dupes.get("dropped_columns", []))
            n_signal = signal_dupes.get("n_signal_features", 0)
            lines.append(f"  --- Signal-only ({n_signal} features, {n_dropped} noise columns dropped) ---")
            dropped_list = ", ".join(signal_dupes.get("dropped_columns", []))
            lines.append(f"  Dropped: {dropped_list}")
            lines.append(f"  Signal duplicate rows : {signal_dupes['n_duplicate_rows']} ({signal_dupes['duplicate_pct']:.1f}%)")
            lines.append(f"  Signal dup groups     : {signal_dupes['n_duplicate_groups']}")
            lines.append(f"  Signal conflicts      : {signal_dupes['n_conflicting_rows']} ({signal_dupes['conflicting_pct']:.1f}%)")
            lines.append(f"  Signal conflict groups: {signal_dupes['n_conflicting_groups']}")
            if signal_dupes.get("n_conflicting_rows", 0) > 0:
                lines.append("  ⚠ Noise features mask real conflicts — these set a hard AUC ceiling.")
        lines.append("")

    # Unseen categories (train-test overlap)
    unseen = eda_report.get("unseen_categories", {})
    if unseen:
        lines.append("─" * 50)
        lines.append("TRAIN-TEST OVERLAP (unseen categories in test)")
        lines.append("─" * 50)
        for col, info in sorted(unseen.items(), key=lambda x: x[1]["unseen_row_pct"], reverse=True):
            lines.append(
                f"  {col:<30} {info['n_unseen']} unseen / {info['n_test_unique']} unique  "
                f"({info['unseen_row_pct']:.1f}% test rows)"
            )
            if info.get("unseen_values"):
                vals_str = ", ".join(info["unseen_values"][:10])
                lines.append(f"    values: {vals_str}")
        lines.append("")

    # Monotonicity detection
    mono = eda_report.get("monotonicity", {})
    mono_feats = {col: info for col, info in mono.items() if info.get("is_monotonic")}
    if mono_feats:
        lines.append("─" * 50)
        lines.append("MONOTONICITY DETECTION (|Spearman rho| > 0.7 on binned target rates)")
        lines.append("─" * 50)
        for col, info in sorted(mono_feats.items(), key=lambda x: abs(x[1]["spearman_rho"]), reverse=True):
            lines.append(
                f"  {col:<35} rho={info['spearman_rho']:+.3f}  ({info['direction']})"
            )
        lines.append("  → Use monotone_constraints in gradient boosting for regularization.")
        lines.append("")

    # Cardinality profiles
    card = eda_report.get("cardinality_profiles", {})
    if card:
        lines.append("─" * 50)
        lines.append("CATEGORICAL CARDINALITY PROFILES")
        lines.append("─" * 50)
        for col, info in sorted(card.items(), key=lambda x: x[1]["normalized_entropy"]):
            lines.append(
                f"  {col:<30} shape={info['shape']:<10} "
                f"top5={info['top5_share']:.0f}%  entropy={info['entropy']:.2f} "
                f"(norm={info['normalized_entropy']:.2f})"
            )
        lines.append("")

    # Target encoding preview
    te = eda_report.get("te_preview", {})
    if te:
        lines.append("─" * 50)
        lines.append("TARGET ENCODING PREVIEW (OOF, alpha=10)")
        lines.append("─" * 50)
        for col, info in te.items():
            auc_str = f"  AUC={info['encoded_auc']:.4f}" if info.get("encoded_auc") is not None else ""
            lines.append(
                f"  {col:<35} corr={info['encoded_corr']:+.4f}{auc_str}"
            )
        lines.append("")

    # Multi-model baseline (personality fingerprint)
    mb = eda_report.get("multi_baseline", {})
    if mb and mb.get("scores"):
        lines.append("─" * 50)
        lines.append("MULTI-MODEL BASELINE (3-fold CV, no feature engineering)")
        lines.append("─" * 50)
        personality = mb.get("personality", "unknown").upper()
        lines.append(f"  Data Personality: {personality}")
        if mb.get("personality_detail"):
            lines.append(f"  {mb['personality_detail']}")
        lines.append("")

        metric = mb.get("metric", "Score")
        lines.append(f"  {'Model':<20} {metric:<10} {'Std':<10} {'Time'}")
        lines.append(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")
        scores = mb.get("scores", {})
        stds = mb.get("stds", {})
        times = mb.get("training_times_sec", {})
        nn_models = {"realmlp", "tabm"}
        nn_samples = mb.get("nn_samples_used", 0)
        for model_name in scores:
            score_val = scores[model_name]
            std_val = stds.get(model_name, 0)
            time_val = times.get(model_name, 0)
            suffix = ""
            if model_name in nn_models and nn_samples > 0:
                suffix = f"  ({nn_samples:,} subsample)"
            lines.append(
                f"  {model_name:<20} {score_val:<10.4f} {std_val:<10.4f} {time_val:.1f}s{suffix}"
            )
        lines.append("")

        if mb.get("best_model"):
            lines.append(f"  Best: {mb['best_model']} ({metric}={mb['best_score']:.4f})")
        if mb.get("linear_gap") is not None:
            lg = mb["linear_gap"]
            desc = "trees significantly better" if lg > 0.02 else (
                "trees slightly better" if lg > 0.005 else "linear competitive"
            )
            lines.append(f"  Linear gap: {lg:.4f} ({desc})")
        lines.append("")

        cross_corr = mb.get("cross_model_correlations", [])
        if cross_corr:
            lines.append("  Cross-model diversity:")
            for na, nb, corr in cross_corr:
                lines.append(f"    {na} vs {nb}: {corr:.4f}")
            lines.append("")

        imp = mb.get("feature_importances", {})
        if imp:
            lines.append("  Feature importances (RF, top 15):")
            for col, importance in list(imp.items())[:15]:
                bar = "█" * int(importance * 200)
                lines.append(f"    {col:<35} {importance:.4f}  {bar}")
            lines.append("")
    else:
        # Backward compat: show quick_model if multi_baseline not available
        qm = eda_report.get("quick_model", {})
        if qm and qm.get("baseline_score") is not None:
            lines.append("─" * 50)
            lines.append("QUICK MODEL BASELINE (RandomForest, 3-fold CV, no feature engineering)")
            lines.append("─" * 50)
            lines.append(f"  Baseline {qm['baseline_metric']}: {qm['baseline_score']:.4f}")
            lines.append("")
            imp = qm.get("feature_importances", {})
            if imp:
                lines.append("  Feature importances (top 15):")
                for col, importance in list(imp.items())[:15]:
                    bar = "█" * int(importance * 200)
                    lines.append(f"    {col:<35} {importance:.4f}  {bar}")
                lines.append("")

    # Interaction orchestra (LightGBM split-gain pairs)
    io_data = eda_report.get("interaction_orchestra", {})
    if io_data and io_data.get("top_interactions"):
        lines.append("─" * 50)
        lines.append("INTERACTION ORCHESTRA (LightGBM split-gain pairs)")
        lines.append("─" * 50)
        lines.append("  Top interaction candidates:")
        for i, pair in enumerate(io_data["top_interactions"][:15], 1):
            lines.append(
                f"    {i:>2}. {pair['feature_a']} * {pair['feature_b']}"
                f"   strength={pair['interaction_strength']:.4f}"
            )
        lines.append(f"  Method: {io_data.get('method', 'N/A')}")
        lines.append("")

    # Ghost feature detector (cross-model importance comparison)
    gf_data = eda_report.get("ghost_features", {})
    if gf_data:
        ghosts = gf_data.get("ghost_features", [])
        rank_corrs = gf_data.get("rank_correlations", {})
        if ghosts or rank_corrs:
            lines.append("─" * 50)
            lines.append("GHOST FEATURE DETECTOR (cross-model importance comparison)")
            lines.append("─" * 50)
            if ghosts:
                for g in ghosts[:10]:
                    ranks_str = ", ".join(f"{m}={r}" for m, r in g["ranks"].items())
                    lines.append(
                        f"  ⚠ {g['feature']} [{g['severity'].upper()}]: {ranks_str}"
                    )
                    lines.append(f"    {g['explanation']}")
                lines.append("")
            else:
                lines.append("  No ghost features detected — model importances are consistent.")
                lines.append("")
            if rank_corrs:
                lines.append("  Model importance rank correlations (Spearman):")
                corr_parts = [f"{k}: {v:.3f}" for k, v in rank_corrs.items()]
                lines.append(f"    {' | '.join(corr_parts)}")
                lines.append("")

    # Prediction diversity probe (multi-seed RF signal-noise analysis)
    pd_probe = eda_report.get("prediction_diversity", {})
    if pd_probe and pd_probe.get("signal_noise_ratio") is not None:
        lines.append("─" * 50)
        lines.append("PREDICTION DIVERSITY PROBE (3 RF seeds, signal-noise analysis)")
        lines.append("─" * 50)
        lines.append(f"  Within-seed std (signal)  : {pd_probe['within_seed_std']:.6f}")
        lines.append(f"  Across-seed std (noise)   : {pd_probe['prediction_std']:.6f}")
        lines.append(f"  Signal-noise ratio (SNR)  : {pd_probe['signal_noise_ratio']:.1f}")
        lines.append(f"  Diversity classification  : {pd_probe['diversity_class'].upper()}")
        lines.append(f"  Mean pairwise correlation : {pd_probe['mean_corr']:.6f}")
        ci = pd_probe.get("fisher_z_ci", [0, 0])
        lines.append(f"  95% CI (Fisher z)         : [{ci[0]:.6f}, {ci[1]:.6f}]")
        lines.append(f"  Samples used              : {pd_probe.get('n_samples_used', 'N/A')}")
        lines.append("")
        pairs = pd_probe.get("pairwise_correlations", [])
        if pairs:
            lines.append("  Pairwise correlations:")
            for sa, sb, corr in pairs:
                lines.append(f"    seed {sa} vs {sb}: {corr:.6f}")
            lines.append("")
        lines.append("  SNR thresholds: <3 very_low | 3-8 low | 8-15 moderate | >15 high")
        dc = pd_probe["diversity_class"]
        if dc in ("very_low", "low"):
            lines.append(f"  ⚠ {dc.upper()} signal diversity: Models find weak signal relative to")
            lines.append(f"    seed noise. Neural nets with different hyperparameters will converge")
            lines.append(f"    to similar predictions. Enable tiered tracker (diversity_mode: tiered)")
            lines.append(f"    and diversity pruning to avoid redundant compute.")
        elif dc == "moderate":
            lines.append("  Moderate signal diversity — tiered tracker may help, diversity pruning optional.")
        else:
            lines.append("  High signal diversity — standard per-fold settings are sufficient.")
        lines.append("")

    # Preprocessing summary (scaling/transform signals for LLM)
    preproc = eda_report.get("preprocessing_summary", {})
    if preproc:
        lines.append("─" * 50)
        lines.append("PREPROCESSING SUMMARY (for scaling/transform decisions)")
        lines.append("─" * 50)
        lines.append(f"  Scale range ratio    : {preproc.get('scale_range_ratio', 'N/A')}×")
        lines.append(f"  High-skewness feats  : {preproc.get('n_high_skew', 0)} (|skew| > 1.0, candidates for log transform)")
        if preproc.get("high_skew_features"):
            lines.append(f"    → {', '.join(preproc['high_skew_features'][:10])}")
        lines.append(f"  High-outlier feats   : {preproc.get('n_high_outlier', 0)} (outlier% > 3%, candidates for clipping/RobustScaler)")
        if preproc.get("high_outlier_features"):
            lines.append(f"    → {', '.join(preproc['high_outlier_features'][:10])}")
        lines.append(f"  Sentinel feats       : {preproc.get('n_sentinel_features', 0)} (likely masked NaN values)")
        if preproc.get("sentinel_features"):
            lines.append(f"    → {', '.join(preproc['sentinel_features'][:10])}")
        lines.append(f"  Suggested scalers    : {', '.join(preproc.get('suggested_scalers', ['standard']))}")
        lines.append("")

    # Recommendations
    recs = eda_report.get("recommendations", [])
    lines.append("─" * 40)
    lines.append("RECOMMENDATIONS")
    lines.append("─" * 40)
    if recs:
        for i, rec in enumerate(recs, 1):
            lines.append(f"  {i}. {rec}")
    else:
        lines.append("  (no recommendations)")
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)
