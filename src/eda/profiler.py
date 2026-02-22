"""
EDA Profiler — Layer 1 of the Maestro pipeline.

Performs automatic dataset profiling on raw CSV data. Produces a structured
report (matching eda_schema.yaml) that the LLM strategist consumes to make
informed decisions about feature engineering and model selection.

This module is pure pandas/numpy. No ML, no randomness, fully deterministic.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def run_eda(
    train_path: str | Path,
    test_path: str | Path,
    target_col: str,
) -> dict:
    """Run the complete EDA pipeline and produce a structured report.

    This is the main entry point for Layer 1. It reads the raw CSVs,
    analyzes every column, computes correlations, clusters features,
    identifies weak features, and generates LLM-readable recommendations.

    Args:
        train_path: Path to the training CSV file.
        test_path: Path to the test CSV file.
        target_col: Name of the target variable column.

    Returns:
        Dictionary matching the eda_schema.yaml structure with keys:
        - dataset_info: shapes, memory usage
        - target_analysis: distribution, class balance
        - columns: per-column analysis dict
        - correlation_matrix: full pairwise correlations
        - feature_clusters: groups of correlated features
        - weak_features: features with low target correlation
        - recommendations: list of LLM-readable suggestion strings

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

    # Dataset info
    train_mem = train.memory_usage(deep=True).sum() / 1024 / 1024
    test_mem = test.memory_usage(deep=True).sum() / 1024 / 1024
    dataset_info = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_memory_mb": round(float(train_mem), 3),
        "test_memory_mb": round(float(test_mem), 3),
        "n_features": train.shape[1] - 1,  # excluding target
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

    # Feature columns (exclude target)
    feature_cols = [c for c in train.columns if c != target_col]
    feature_df = train[feature_cols].copy()

    # Per-column analysis
    columns_analysis = _detect_column_types(feature_df)

    # Correlations
    corr_result = _compute_correlations(train, target_col)
    target_correlations = corr_result["target_correlations"]
    correlation_matrix = corr_result["correlation_matrix"]

    # Merge target_correlation into per-column analysis
    for col_name, col_info in columns_analysis.items():
        col_info["target_correlation"] = round(float(target_correlations.get(col_name, 0.0)), 6)

    # Feature clusters
    feature_clusters = _find_feature_clusters(correlation_matrix, threshold=0.5)

    # Weak features
    weak_features = _identify_weak_features(target_correlations, threshold=0.05)

    # Generate recommendations
    recommendations = _generate_recommendations(
        columns_analysis, target_correlations, feature_clusters, weak_features, train
    )

    return {
        "dataset_info": dataset_info,
        "target_analysis": target_analysis,
        "columns": columns_analysis,
        "correlation_matrix": correlation_matrix,
        "feature_clusters": feature_clusters,
        "weak_features": weak_features,
        "recommendations": recommendations,
    }


def _generate_recommendations(
    columns_analysis: dict,
    target_correlations: dict,
    feature_clusters: list,
    weak_features: list,
    train: pd.DataFrame,
) -> list[str]:
    """Generate LLM-readable recommendation strings from EDA results."""
    recs = []

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
        # Target might be non-numeric; convert for correlation
        target_series = df[target_col]
        if target_series.dtype == object:
            target_series = pd.Categorical(target_series).codes
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
        is_object_or_cat = series.dtype == object or hasattr(series, "cat")

        if is_numeric:
            non_null = series.dropna()
            stats = {
                "mean": round(float(non_null.mean()), 6) if len(non_null) > 0 else None,
                "std": round(float(non_null.std()), 6) if len(non_null) > 0 else None,
                "min": round(float(non_null.min()), 6) if len(non_null) > 0 else None,
                "max": round(float(non_null.max()), 6) if len(non_null) > 0 else None,
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
            # Check if all values are integers
            non_null = series.dropna()
            if len(non_null) > 0 and (non_null == non_null.round()).all():
                detected_type = "ordinal"
            else:
                detected_type = "numeric_continuous"
        else:
            detected_type = "numeric_continuous"

        result[col] = {
            "dtype": dtype_str,
            "detected_type": detected_type,
            "missing_pct": missing_pct,
            "cardinality": cardinality,
            "top_values": top_values,
            "stats": stats,
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

    # Feature correlation table
    columns = eda_report.get("columns", {})
    sorted_cols = sorted(
        columns.items(),
        key=lambda kv: abs(kv[1].get("target_correlation", 0)),
        reverse=True,
    )
    lines.append("─" * 70)
    lines.append(f"{'FEATURE':<35} {'CORR':>8} {'TYPE':<30} {'MISS%':>6}")
    lines.append("─" * 70)
    for col_name, col_info in sorted_cols:
        corr = col_info.get("target_correlation", 0.0)
        dtype = col_info.get("detected_type", "")
        miss = col_info.get("missing_pct", 0.0)
        lines.append(f"{col_name:<35} {corr:>+8.4f} {dtype:<30} {miss:>6.1f}")
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

    # Feature clusters
    clusters = eda_report.get("feature_clusters", [])
    lines.append("─" * 40)
    lines.append("FEATURE CLUSTERS (|corr| >= 0.5)")
    lines.append("─" * 40)
    if clusters:
        for i, c in enumerate(clusters, 1):
            members = ", ".join(c["features"])
            lines.append(f"  Cluster {i} (mean_corr={c['mean_internal_corr']:.2f}): {members}")
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
