"""
Ensemble Diversity — Diversity-aware model selection using NSGA-II.

Measures diversity between model predictions using correlation matrix
eigenvalues (effective ensemble size). Optimizes two objectives
simultaneously:
1. Primary metric (e.g., AUC-ROC)
2. Ensemble diversity (effective ensemble size)

This dual-objective approach prevents the ensemble from collapsing into
near-identical models, which is a common failure mode.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score, mean_squared_error

logger = logging.getLogger("maestro")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _score_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute metric (higher is better: returns AUC or negative RMSE)."""
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_pred))
    elif metric == "neg_rmse":
        return -float(np.sqrt(mean_squared_error(y_true, y_pred)))
    else:
        try:
            return float(roc_auc_score(y_true, y_pred))
        except Exception:
            return -float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_correlation_matrix(oof_list: list[np.ndarray]) -> np.ndarray:
    """Compute the pairwise Pearson correlation matrix between model OOF predictions.

    Args:
        oof_list: List of OOF prediction arrays, one per model.
                  Each array has shape (n_samples,).

    Returns:
        Correlation matrix as a numpy array of shape (n_models, n_models).

    Steps:
        1. Stack the OOF predictions into a matrix (n_samples, n_models).
        2. Compute Pearson correlation using np.corrcoef (transposed so
           each row is a model).
        3. Return the correlation matrix.
    """
    mat = np.column_stack(oof_list)  # (n_samples, n_models)
    corr = np.corrcoef(mat.T)       # (n_models, n_models)
    # Zero-variance OOF arrays (degenerate model) produce NaN in corrcoef.
    # Treat undefined correlation as uncorrelated (0.0) and restore diagonal to 1.0.
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def effective_ensemble_size(corr_matrix: np.ndarray) -> float:
    """Compute the effective ensemble size from the correlation matrix.

    Uses the eigenvalue-based formula:
        N_eff = (sum of eigenvalues)^2 / (sum of eigenvalues^2)

    This measures how many "independent" models are in the ensemble.
    If all models are identical, N_eff = 1. If all models are perfectly
    uncorrelated, N_eff = N.

    Args:
        corr_matrix: Correlation matrix of shape (n_models, n_models).

    Returns:
        Effective ensemble size (float >= 1.0).

    Steps:
        1. Compute eigenvalues of the correlation matrix using
           np.linalg.eigvalsh.
        2. Clip eigenvalues to be >= 0 (numerical stability).
        3. Compute N_eff = (sum(eigenvalues))^2 / sum(eigenvalues^2).
        4. Return N_eff.
    """
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.clip(eigenvalues, 0, None)

    sum_eig = eigenvalues.sum()
    sum_eig_sq = (eigenvalues ** 2).sum()

    if sum_eig_sq == 0:
        return 1.0

    n_eff = (sum_eig ** 2) / sum_eig_sq
    return float(n_eff)


def greedy_diverse_select(
    oof_list: list[np.ndarray],
    scores: list[float],
    n_select: int,
    min_score_ratio: float = 0.95,
) -> list[int]:
    """Greedily select a diverse subset of models.

    Iteratively adds the model that maximizes effective ensemble size
    while maintaining a minimum quality threshold.

    Args:
        oof_list: List of OOF prediction arrays.
        scores: List of CV scores for each model (higher = better).
        n_select: Number of models to select.
        min_score_ratio: Minimum score as fraction of best score
                         (e.g., 0.95 = at least 95% of best model's score).

    Returns:
        List of selected model indices (0-based).

    Steps:
        1. Compute the minimum acceptable score:
           min_score = max(scores) * min_score_ratio.
        2. Filter to models meeting the quality threshold.
        3. Start with the model that has the highest score.
        4. Iteratively add the model that maximizes effective_ensemble_size
           of the current selection:
           a. For each remaining candidate, compute the effective ensemble
              size if it were added.
           b. Select the candidate with the highest effective ensemble size.
        5. Repeat until n_select models are selected or no candidates remain.
        6. Return the list of selected indices.
    """
    max_score = max(scores)
    min_score = max_score * min_score_ratio

    # Filter candidates meeting quality threshold
    candidates = [i for i, s in enumerate(scores) if s >= min_score]
    if not candidates:
        candidates = list(range(len(scores)))

    # Start with best-scoring model
    best_start = max(candidates, key=lambda i: scores[i])
    selected = [best_start]
    remaining = [i for i in candidates if i != best_start]

    while len(selected) < n_select and remaining:
        best_candidate = None
        best_n_eff = -np.inf

        for candidate in remaining:
            current_selection = selected + [candidate]
            candidate_oofs = [oof_list[i] for i in current_selection]

            if len(candidate_oofs) == 1:
                n_eff = 1.0
            else:
                corr = compute_correlation_matrix(candidate_oofs)
                n_eff = effective_ensemble_size(corr)

            if n_eff > best_n_eff:
                best_n_eff = n_eff
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


def select_from_pareto(
    pareto_trials: list,
    oof_list: list[np.ndarray],
    test_list: list[np.ndarray],
    y: np.ndarray,
    n_models: int,
    diversity_weight: float,
    metric: str,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the best ensemble from a NSGA-II Pareto front.

    Given completed Pareto front trials, selects the best solution using
    weighted scoring: (1-dw)*normalized_metric + dw*normalized_diversity.

    This allows producing multiple submissions from a single NSGA-II run
    by calling this function with different diversity_weight values.

    Args:
        pareto_trials: List of Optuna FrozenTrial from study.best_trials.
        oof_list: All OOF prediction arrays.
        test_list: All test prediction arrays.
        y: True target values.
        n_models: Total number of prediction arrays.
        diversity_weight: Weight for diversity (0=pure metric, 1=pure diversity).
        metric: Metric name.
        labels: Model labels for logging.

    Returns:
        Tuple of (test_preds, info_dict).
    """
    _labels = labels or [f"model[{i}]" for i in range(n_models)]

    # Normalize Pareto front values for selection
    metric_values = np.array([t.values[0] for t in pareto_trials])
    diversity_values = np.array([t.values[1] for t in pareto_trials])

    def _normalize(arr: np.ndarray) -> np.ndarray:
        rng = arr.max() - arr.min()
        if rng == 0:
            return np.ones_like(arr)
        return (arr - arr.min()) / rng

    norm_metric = _normalize(metric_values)
    norm_diversity = _normalize(diversity_values)

    combined = (1 - diversity_weight) * norm_metric + diversity_weight * norm_diversity
    best_idx = int(np.argmax(combined))
    best_trial = pareto_trials[best_idx]

    # Reconstruct from best trial
    best_included = []
    best_weights = []
    for i in range(n_models):
        include = best_trial.params.get(f"include_{i}", False)
        weight = best_trial.params.get(f"weight_{i}", 0.0)
        if include:
            best_included.append(i)
            best_weights.append(weight)

    if not best_included:
        best_included = list(range(n_models))
        best_weights = [1.0] * n_models

    total_w = sum(best_weights)
    norm_w = [w / total_w for w in best_weights] if total_w > 0 else [1.0 / len(best_included)] * len(best_included)

    # Test predictions
    selected_test = [test_list[i] for i in best_included]
    test_preds = np.array(sum(w * p for w, p in zip(norm_w, selected_test)))

    # Final metrics
    selected_oofs = [oof_list[i] for i in best_included]
    blended_oof = sum(w * p for w, p in zip(norm_w, selected_oofs))
    final_metric = _score_metric(y, blended_oof, metric)
    final_eff_size = (
        effective_ensemble_size(compute_correlation_matrix(selected_oofs))
        if len(best_included) > 1
        else 1.0
    )

    pareto_front = [
        (t.values[0], t.values[1]) for t in pareto_trials
    ]

    info_dict = {
        "selected_models": best_included,
        "weights": norm_w,
        "metric_score": final_metric,
        "effective_size": final_eff_size,
        "pareto_front": pareto_front,
    }

    # --- Detailed report ---
    logger.info("=" * 60)
    logger.info(f"NSGA-II ENSEMBLE RESULT (dw={diversity_weight:.2f})")
    logger.info("-" * 60)
    logger.info(
        f"  Selected {len(best_included)}/{n_models} models | "
        f"{metric}={final_metric:.6f} | N_eff={final_eff_size:.2f}"
    )

    # Per-model weights (sorted by weight descending)
    model_weight_pairs = sorted(
        zip(best_included, norm_w), key=lambda x: x[1], reverse=True
    )
    for idx, w in model_weight_pairs:
        try:
            solo_score = _score_metric(y, oof_list[idx], metric)
        except Exception:
            solo_score = float("nan")
        logger.info(f"    {_labels[idx]:<25s}  weight={w:.4f}  solo_{metric}={solo_score:.6f}")

    # Pareto front summary
    if pareto_trials:
        pf_metrics = [t.values[0] for t in pareto_trials]
        pf_diversity = [t.values[1] for t in pareto_trials]
        logger.info("-" * 60)
        logger.info(
            f"  Pareto front: {len(pareto_trials)} solutions | "
            f"{metric}=[{min(pf_metrics):.6f}, {max(pf_metrics):.6f}] | "
            f"N_eff=[{min(pf_diversity):.2f}, {max(pf_diversity):.2f}]"
        )

        best_metric_trial = max(pareto_trials, key=lambda t: t.values[0])
        best_div_trial = max(pareto_trials, key=lambda t: t.values[1])
        logger.info(
            f"  Best {metric} on Pareto: {best_metric_trial.values[0]:.6f} "
            f"(N_eff={best_metric_trial.values[1]:.2f})"
        )
        logger.info(
            f"  Best diversity on Pareto: N_eff={best_div_trial.values[1]:.2f} "
            f"({metric}={best_div_trial.values[0]:.6f})"
        )

    # Selected ensemble correlation matrix
    if len(best_included) > 1:
        sel_corr = compute_correlation_matrix(selected_oofs)
        sel_labels = [_labels[i] for i in best_included]
        logger.info("-" * 60)
        logger.info(f"  Selected ensemble pairwise correlations:")
        sel_pairs = []
        for i in range(len(best_included)):
            for j in range(i + 1, len(best_included)):
                sel_pairs.append((sel_labels[i], sel_labels[j], sel_corr[i, j]))
        sel_pairs.sort(key=lambda x: x[2], reverse=True)
        for a, b, r in sel_pairs:
            logger.info(f"    {a} <-> {b}: {r:.4f}")
        avg_corr = np.mean([r for _, _, r in sel_pairs])
        logger.info(f"  Avg pairwise correlation: {avg_corr:.4f}")
    logger.info("=" * 60)

    return test_preds, info_dict


def run_nsga2_ensemble(
    oof_list: list[np.ndarray],
    test_list: list[np.ndarray],
    y: np.ndarray,
    n_trials: int = 300,
    metric: str = "roc_auc",
    diversity_weight: float = 0.3,
    seed: int = 42,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run NSGA-II multi-objective optimization for ensemble selection.

    Simultaneously optimizes:
    - Objective 1: Primary metric (AUC-ROC, RMSE, etc.)
    - Objective 2: Effective ensemble size (diversity)

    Each trial selects a subset of models and their blend weights.
    The diversity_weight is only used for Pareto front selection, not
    during optimization. Use select_from_pareto() to re-select from
    the same Pareto front with different weights.

    Args:
        oof_list: List of OOF prediction arrays from all base models.
        test_list: List of test prediction arrays from all base models.
        y: True target values.
        n_trials: Number of NSGA-II trials.
        metric: Primary metric name.
        diversity_weight: Weight for diversity in final selection
                          (0.0 = pure metric, 1.0 = pure diversity).
        seed: Random seed.
        labels: Model labels for logging.

    Returns:
        Tuple of (best_test_preds, info_dict):
        - best_test_preds: Test predictions from the best ensemble.
        - info_dict: Dictionary containing:
          - selected_models: indices of selected models
          - weights: blend weights
          - metric_score: primary metric value
          - effective_size: effective ensemble size
          - pareto_front: list of (metric, diversity) tuples
          - pareto_trials: Optuna FrozenTrial list (for reuse with
            select_from_pareto)

    Steps:
        1. Create an Optuna study with NSGAIISampler and two directions:
           - 'maximize' for the primary metric
           - 'maximize' for effective ensemble size
        2. Define the objective:
           a. For each model, suggest a boolean (include/exclude) and
              a weight (0.0-1.0).
           b. Filter to included models. If none included, return
              worst possible values.
           c. Normalize weights of included models to sum to 1.0.
           d. Compute blended OOF predictions.
           e. Compute the primary metric.
           f. Compute the correlation matrix of included models' OOFs.
           g. Compute effective ensemble size.
           h. Return (metric_score, effective_size).
        3. Run study.optimize for n_trials.
        4. Call select_from_pareto to pick the best solution.
        5. Return (test_preds, info_dict) with pareto_trials included.
    """
    n_models = len(oof_list)
    _labels = labels or [f"model[{i}]" for i in range(n_models)]
    worst_metric = -np.inf

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        # For each model: include flag + weight
        included = []
        weights = []
        for i in range(n_models):
            include = trial.suggest_categorical(f"include_{i}", [True, False])
            weight = trial.suggest_float(f"weight_{i}", 0.0, 1.0)
            if include:
                included.append(i)
                weights.append(weight)

        if not included:
            return (worst_metric, 1.0)

        # Normalize weights
        total_w = sum(weights)
        if total_w == 0:
            norm_weights = [1.0 / len(included)] * len(included)
        else:
            norm_weights = [w / total_w for w in weights]

        # Blend OOF
        selected_oofs = [oof_list[i] for i in included]
        blended_oof = sum(w * p for w, p in zip(norm_weights, selected_oofs))

        # Metric
        try:
            metric_score = _score_metric(y, blended_oof, metric)
        except Exception:
            return (worst_metric, 1.0)

        # Diversity
        if len(included) == 1:
            n_eff = 1.0
        else:
            corr = compute_correlation_matrix(selected_oofs)
            n_eff = effective_ensemble_size(corr)

        return (metric_score, n_eff)

    import time as _time
    nsga2_start = _time.time()

    # --- Input diagnostics ---
    logger.info("=" * 60)
    logger.info("NSGA-II ENSEMBLE INPUT")
    logger.info("-" * 60)
    logger.info(f"  {n_models} prediction arrays, {len(oof_list[0])} samples each")
    logger.info(f"  diversity_weight={diversity_weight}, n_trials={n_trials}")

    # Per-model solo scores
    solo_scores = []
    for i in range(n_models):
        try:
            s = _score_metric(y, oof_list[i], metric)
        except Exception:
            s = float("nan")
        solo_scores.append(s)
    sorted_by_score = sorted(
        enumerate(solo_scores), key=lambda x: x[1], reverse=True
    )
    logger.info(f"  Solo {metric} scores (best → worst):")
    for idx, score in sorted_by_score:
        logger.info(f"    {_labels[idx]:<25s}  {score:.6f}")

    # Full correlation matrix
    full_corr = compute_correlation_matrix(oof_list)
    full_neff = effective_ensemble_size(full_corr)
    logger.info(f"  Full matrix N_eff={full_neff:.2f} (all {n_models} arrays)")

    # Correlation heatmap (top-left triangle, compact)
    logger.info(f"  Pairwise correlations (high=redundant):")
    # Show pairs with highest and lowest correlation
    corr_pairs = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            corr_pairs.append((_labels[i], _labels[j], full_corr[i, j]))
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    n_show = min(5, len(corr_pairs))
    logger.info(f"    Most correlated (top {n_show}):")
    for a, b, r in corr_pairs[:n_show]:
        logger.info(f"      {a} <-> {b}: {r:.4f}")
    logger.info(f"    Least correlated (top {n_show}):")
    for a, b, r in corr_pairs[-n_show:]:
        logger.info(f"      {a} <-> {b}: {r:.4f}")
    logger.info("=" * 60)

    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    nsga2_elapsed = _time.time() - nsga2_start

    # Extract Pareto front trials
    pareto_trials = study.best_trials
    n_completed = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    logger.info(
        f"NSGA-II: {n_completed}/{n_trials} trials completed in {nsga2_elapsed:.1f}s "
        f"({nsga2_elapsed / max(n_trials, 1) * 1000:.1f}ms/trial), "
        f"Pareto front: {len(pareto_trials)} solutions"
    )
    if not pareto_trials:
        # Fallback: use all models with equal weights
        n_models_use = n_models
        selected_indices = list(range(n_models_use))
        equal_weights = [1.0 / n_models_use] * n_models_use
        test_preds = sum(w * p for w, p in zip(equal_weights, test_list))
        return np.array(test_preds), {
            "selected_models": selected_indices,
            "weights": equal_weights,
            "metric_score": 0.0,
            "effective_size": 1.0,
            "pareto_front": [],
            "pareto_trials": [],
        }

    # Select from Pareto front
    test_preds, info_dict = select_from_pareto(
        pareto_trials, oof_list, test_list, y,
        n_models=n_models,
        diversity_weight=diversity_weight,
        metric=metric,
        labels=_labels,
    )

    # Include pareto_trials for reuse with different diversity_weights
    info_dict["pareto_trials"] = pareto_trials

    return test_preds, info_dict


def print_diversity_report(
    corr_matrix: np.ndarray,
    labels: list[str],
) -> None:
    """Print a formatted diversity report to the console.

    Args:
        corr_matrix: Correlation matrix of shape (n_models, n_models).
        labels: List of model names corresponding to matrix rows/columns.

    Steps:
        1. Print header: "Ensemble Diversity Report".
        2. Print the correlation matrix as a formatted table with labels.
        3. Compute and print effective ensemble size.
        4. Highlight the most correlated pair (highest off-diagonal value).
        5. Highlight the least correlated pair (lowest off-diagonal value).
        6. Print a summary recommendation:
           - If effective_size < 2: "Warning: ensemble is highly redundant"
           - If effective_size >= n/2: "Good diversity in ensemble"
    """
    n = len(labels)
    print("\n" + "=" * 60)
    print("ENSEMBLE DIVERSITY REPORT")
    print("=" * 60)

    # Correlation matrix table
    label_width = max(len(lb) for lb in labels) + 2
    header = " " * label_width + "".join(f"{lb:>12s}" for lb in labels)
    print(header)
    print("-" * len(header))
    for i, row_label in enumerate(labels):
        row_str = f"{row_label:<{label_width}}"
        for j in range(n):
            row_str += f"{corr_matrix[i, j]:>12.3f}"
        print(row_str)

    # Effective ensemble size
    n_eff = effective_ensemble_size(corr_matrix)
    print(f"\nEffective ensemble size (N_eff): {n_eff:.3f} / {n}")

    # Most and least correlated pairs
    max_corr = -np.inf
    min_corr = np.inf
    max_pair = ("", "")
    min_pair = ("", "")

    for i in range(n):
        for j in range(i + 1, n):
            val = corr_matrix[i, j]
            if val > max_corr:
                max_corr = val
                max_pair = (labels[i], labels[j])
            if val < min_corr:
                min_corr = val
                min_pair = (labels[i], labels[j])

    if n > 1:
        print(f"Most correlated pair : {max_pair[0]} & {max_pair[1]}  (r={max_corr:.3f})")
        print(f"Least correlated pair: {min_pair[0]} & {min_pair[1]}  (r={min_corr:.3f})")

    # Summary recommendation
    if n_eff < 2:
        print("\n⚠  Warning: ensemble is highly redundant (N_eff < 2). Consider more diverse models.")
    elif n_eff >= n / 2:
        print(f"\n✓  Good diversity in ensemble (N_eff={n_eff:.2f} >= {n/2:.1f}).")
    else:
        print(f"\n~  Moderate diversity (N_eff={n_eff:.2f}). Adding diverse models may help.")

    print("=" * 60 + "\n")
