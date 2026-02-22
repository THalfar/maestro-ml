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


def run_nsga2_ensemble(
    oof_list: list[np.ndarray],
    test_list: list[np.ndarray],
    y: np.ndarray,
    n_trials: int = 300,
    metric: str = "roc_auc",
    diversity_weight: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run NSGA-II multi-objective optimization for ensemble selection.

    Simultaneously optimizes:
    - Objective 1: Primary metric (AUC-ROC, RMSE, etc.)
    - Objective 2: Effective ensemble size (diversity)

    Each trial selects a subset of models and their blend weights.

    Args:
        oof_list: List of OOF prediction arrays from all base models.
        test_list: List of test prediction arrays from all base models.
        y: True target values.
        n_trials: Number of NSGA-II trials.
        metric: Primary metric name.
        diversity_weight: Weight for diversity in final selection
                          (0.0 = pure metric, 1.0 = pure diversity).
        seed: Random seed.

    Returns:
        Tuple of (best_test_preds, info_dict):
        - best_test_preds: Test predictions from the best ensemble.
        - info_dict: Dictionary containing:
          - selected_models: indices of selected models
          - weights: blend weights
          - metric_score: primary metric value
          - effective_size: effective ensemble size
          - pareto_front: list of (metric, diversity) tuples

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
        4. Select the best trial from the Pareto front using a weighted
           score: diversity_weight * normalized_diversity +
                  (1 - diversity_weight) * normalized_metric.
        5. Reconstruct the test predictions using the best trial's
           model selection and weights.
        6. Return (test_preds, info_dict).
    """
    n_models = len(oof_list)
    worst_metric = -np.inf if metric != "neg_rmse" else np.inf

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

    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Extract Pareto front trials
    pareto_trials = study.best_trials
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
        }

    # Normalize Pareto front values for selection
    metric_values = np.array([t.values[0] for t in pareto_trials])
    diversity_values = np.array([t.values[1] for t in pareto_trials])

    # Normalize to [0, 1]
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

    logger.info(
        f"NSGA-II ensemble: {len(best_included)} models selected, "
        f"{metric}={final_metric:.6f}, N_eff={final_eff_size:.2f}"
    )

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
