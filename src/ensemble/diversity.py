"""
Ensemble Diversity — Diversity-aware model selection using NSGA-II.

Measures diversity between model predictions using configurable metrics:
- **pearson_neff**: Effective ensemble size from Pearson error correlation
  eigenvalues (default). Classic approach.
- **spearman_neff**: Same eigenvalue formula but on Spearman rank
  correlation of errors. Theoretically better for AUC (a ranking metric)
  because it captures monotonic — not just linear — relationships.
- **ambiguity**: Ambiguity decomposition (weighted variance of predictions
  across models). Directly measures "how much does this subset benefit
  from ensembling?" instead of proxy correlation.

NSGA-II optimizes two objectives simultaneously:
1. Primary metric (e.g., AUC-ROC)
2. Ensemble diversity (one of the metrics above)

Uses pymoo's NSGA-II with SBX crossover and polynomial mutation for
proper evolutionary multi-objective optimization, and HighTradeoffPoints
for principled knee-point selection on the Pareto front.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, mean_squared_error

logger = logging.getLogger("maestro")

_WEIGHT_THRESHOLD = 0.01

# Valid diversity_metric values
DIVERSITY_METRICS = ("pearson_neff", "spearman_neff", "ambiguity")


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


def compute_error_correlation_matrix(
    oof_list: list[np.ndarray], y_true: np.ndarray
) -> np.ndarray:
    """Compute pairwise Pearson correlation matrix of model errors (residuals).

    Errors are defined as ``oof - y_true`` (soft residuals). For binary
    classification with probability outputs this strips out the common signal
    learned from y_true, isolating whether models are *wrong on the same
    samples*. For regression it is the standard residual.

    Prediction correlation is inflated when all models learn the same signal:
    if both models predict 0.9 on positive samples and 0.1 on negatives they
    appear highly correlated even if their *mistakes* occur on completely
    different rows. Error correlation reveals the true diversity.

    Args:
        oof_list: List of OOF prediction arrays (probabilities or values).
        y_true: True target values, same length as each OOF array.

    Returns:
        Correlation matrix of shape (n_models, n_models).
    """
    errors = [oof - y_true for oof in oof_list]
    return compute_correlation_matrix(errors)


def compute_spearman_correlation_matrix(oof_list: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise Spearman rank correlation matrix.

    Spearman correlation captures monotonic (not just linear) relationships
    between predictions. Theoretically better suited for AUC-ROC because
    AUC is a ranking metric — it only cares about the ordering of
    predictions, not their exact values.

    Args:
        oof_list: List of OOF prediction arrays, one per model.

    Returns:
        Spearman correlation matrix of shape (n_models, n_models).
    """
    mat = np.column_stack(oof_list)  # (n_samples, n_models)
    corr, _ = spearmanr(mat)         # (n_models, n_models)
    # spearmanr returns a scalar for 2 variables — force to 2D
    if mat.shape[1] == 1:
        corr = np.array([[1.0]])
    elif mat.shape[1] == 2:
        corr = np.array([[1.0, corr], [corr, 1.0]])
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def compute_spearman_error_correlation_matrix(
    oof_list: list[np.ndarray], y_true: np.ndarray
) -> np.ndarray:
    """Compute Spearman rank correlation matrix of model errors.

    Combines the benefits of error-based diversity (strips shared signal)
    with Spearman's rank-based correlation (matches AUC's ranking nature).

    Args:
        oof_list: List of OOF prediction arrays.
        y_true: True target values.

    Returns:
        Spearman correlation matrix of errors, shape (n_models, n_models).
    """
    errors = [oof - y_true for oof in oof_list]
    return compute_spearman_correlation_matrix(errors)


def compute_ambiguity(
    oof_list: list[np.ndarray],
    weights: np.ndarray,
) -> float:
    """Compute weighted ambiguity (ensemble disagreement).

    From the bias-variance-diversity decomposition:
        ensemble_error = weighted_avg_error - ambiguity

    Ambiguity = sum_i(w_i * mean((f_i - f_bar)^2)) where f_bar is the
    weighted ensemble prediction. Higher ambiguity means more disagreement
    among models, which translates to larger ensemble benefit.

    Unlike N_eff (which measures correlation structure), ambiguity directly
    measures "how much does averaging these models reduce error?" It is
    the quantity you want to maximize in an ensemble.

    Args:
        oof_list: List of OOF prediction arrays.
        weights: Normalized weight vector (same length as oof_list, sum=1).

    Returns:
        Ambiguity value (float >= 0). Higher = more diverse = better.
    """
    # f_bar = weighted ensemble prediction
    f_bar = sum(w * oof for w, oof in zip(weights, oof_list))
    # ambiguity = sum_i(w_i * mean((f_i - f_bar)^2))
    ambiguity = sum(
        w * float(np.mean((oof - f_bar) ** 2))
        for w, oof in zip(weights, oof_list)
    )
    return float(ambiguity)


def _compute_diversity(
    oof_list: list[np.ndarray],
    y_true: np.ndarray,
    weights: np.ndarray,
    diversity_metric: str = "pearson_neff",
) -> float:
    """Compute diversity score using the specified metric.

    Dispatcher for all diversity metrics. Higher is always better.

    Args:
        oof_list: List of OOF prediction arrays for selected models.
        y_true: True target values.
        weights: Normalized weight vector.
        diversity_metric: One of 'pearson_neff', 'spearman_neff', 'ambiguity'.

    Returns:
        Diversity score (higher = more diverse).
    """
    if len(oof_list) == 1:
        return 1.0 if "neff" in diversity_metric else 0.0

    if diversity_metric == "pearson_neff":
        corr = compute_error_correlation_matrix(oof_list, y_true)
        return effective_ensemble_size(corr)
    elif diversity_metric == "spearman_neff":
        corr = compute_spearman_error_correlation_matrix(oof_list, y_true)
        return effective_ensemble_size(corr)
    elif diversity_metric == "ambiguity":
        return compute_ambiguity(oof_list, weights)
    else:
        raise ValueError(
            f"Unknown diversity_metric '{diversity_metric}'. "
            f"Valid options: {DIVERSITY_METRICS}"
        )


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
    y_true: np.ndarray | None = None,
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
        y_true: True target values. When provided, diversity is measured on
                *errors* (oof - y_true) instead of raw predictions. Error
                diversity is more meaningful: two models can have correlated
                predictions but uncorrelated mistakes.

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
                corr = (
                    compute_error_correlation_matrix(candidate_oofs, y_true)
                    if y_true is not None
                    else compute_correlation_matrix(candidate_oofs)
                )
                n_eff = effective_ensemble_size(corr)

            if n_eff > best_n_eff:
                best_n_eff = n_eff
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


# ---------------------------------------------------------------------------
# pymoo NSGA-II problem definition
# ---------------------------------------------------------------------------

class _EnsembleProblem(ElementwiseProblem):
    """Multi-objective ensemble weight optimization problem for pymoo.

    Decision variables: one continuous weight w_i in [0, 1] per model.
    Models with w_i < _WEIGHT_THRESHOLD are excluded from the blend.
    Two objectives (both minimized — negate to maximize):
        F[0] = -metric_score   (e.g., -AUC)
        F[1] = -diversity       (N_eff or ambiguity, configurable)
    """

    def __init__(
        self,
        oof_list: list[np.ndarray],
        y: np.ndarray,
        metric: str,
        diversity_metric: str = "pearson_neff",
    ) -> None:
        n_models = len(oof_list)
        super().__init__(
            n_var=n_models,
            n_obj=2,
            xl=np.zeros(n_models),
            xu=np.ones(n_models),
        )
        self._oof_list = oof_list
        self._y = y
        self._metric = metric
        self._diversity_metric = diversity_metric

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        included = [i for i in range(self.n_var) if x[i] >= _WEIGHT_THRESHOLD]

        if not included:
            out["F"] = np.array([1e6, 1e6])
            return

        raw_w = np.array([x[i] for i in included])
        norm_w = raw_w / raw_w.sum()

        selected_oofs = [self._oof_list[i] for i in included]
        blended = sum(w * p for w, p in zip(norm_w, selected_oofs))

        try:
            metric_val = _score_metric(self._y, np.asarray(blended), self._metric)
        except Exception:
            out["F"] = np.array([1e6, 1e6])
            return

        diversity_val = _compute_diversity(
            selected_oofs, self._y, norm_w, self._diversity_metric
        )

        # Negate both — pymoo minimises
        out["F"] = np.array([-metric_val, -diversity_val])


# ---------------------------------------------------------------------------
# Pareto front utilities
# ---------------------------------------------------------------------------

def _weights_to_selection(
    x: np.ndarray,
) -> tuple[list[int], list[float]]:
    """Convert a raw weight vector to (included_indices, normalised_weights).

    Models with weight < _WEIGHT_THRESHOLD are excluded.
    """
    included = [i for i in range(len(x)) if x[i] >= _WEIGHT_THRESHOLD]
    if not included:
        included = list(range(len(x)))
    raw = np.array([x[i] for i in included])
    total = raw.sum()
    norm = (raw / total).tolist() if total > 0 else [1.0 / len(included)] * len(included)
    return included, norm


def select_from_pareto(
    pareto_F: np.ndarray,
    pareto_X: np.ndarray,
    oof_list: list[np.ndarray],
    test_list: list[np.ndarray],
    y: np.ndarray,
    n_models: int,
    diversity_weight: float,
    metric: str,
    labels: list[str] | None = None,
    use_knee: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the best ensemble from a NSGA-II Pareto front.

    Given the Pareto front objectives (pareto_F) and decision variables
    (pareto_X), selects a single solution using either:
    - **Knee-point**: ``HighTradeoffPoints`` from pymoo (``use_knee=True``)
    - **Weighted scoring**: ``(1-dw)*norm_metric + dw*norm_diversity``

    This allows producing multiple submissions from a single NSGA-II run
    by calling this function with different diversity_weight values.

    Args:
        pareto_F: Objective values, shape (n_solutions, 2). Stored as
                  negated values (pymoo minimises), i.e. [-metric, -n_eff].
        pareto_X: Decision variables, shape (n_solutions, n_models).
                  Each row is a weight vector.
        oof_list: All OOF prediction arrays.
        test_list: All test prediction arrays.
        y: True target values.
        n_models: Total number of prediction arrays.
        diversity_weight: Weight for diversity (0=pure metric, 1=pure diversity).
        metric: Metric name.
        labels: Model labels for logging.
        use_knee: If True, use HighTradeoffPoints knee-point selection
                  instead of linear weighting.

    Returns:
        Tuple of (test_preds, info_dict).
    """
    _labels = labels or [f"model[{i}]" for i in range(n_models)]

    # Negate back to "higher is better"
    metric_values = -pareto_F[:, 0]
    diversity_values = -pareto_F[:, 1]

    if use_knee and len(pareto_F) >= 3:
        try:
            dm = HighTradeoffPoints()
            knee_indices = dm(pareto_F)
            # Pick the knee point with best combined score
            if len(knee_indices) == 1:
                best_idx = int(knee_indices[0])
            else:
                # Multiple knee points — use diversity_weight to pick
                sub_metric = metric_values[knee_indices]
                sub_div = diversity_values[knee_indices]
                sub_combined = (
                    (1 - diversity_weight) * _normalize(sub_metric)
                    + diversity_weight * _normalize(sub_div)
                )
                best_idx = int(knee_indices[np.argmax(sub_combined)])
            knee_metric = metric_values[best_idx]
            knee_div = diversity_values[best_idx]
            best_m = metric_values.max()
            best_d = diversity_values.max()
            logger.info(
                f"  Knee-point selection: picked index {best_idx} "
                f"from {len(knee_indices)} candidate(s)"
            )
            logger.info(
                f"    Knee:  {metric}={knee_metric:.6f}  N_eff={knee_div:.2f}"
            )
            logger.info(
                f"    vs best {metric}={best_m:.6f} (delta={knee_metric - best_m:+.6f})"
                f"  |  vs best N_eff={best_d:.2f} (delta={knee_div - best_d:+.2f})"
            )
        except Exception as exc:
            logger.warning(f"  HighTradeoffPoints failed ({exc}), falling back to linear weighting")
            use_knee = False

    if not use_knee or len(pareto_F) < 3:
        norm_metric = _normalize(metric_values)
        norm_diversity = _normalize(diversity_values)
        combined = (1 - diversity_weight) * norm_metric + diversity_weight * norm_diversity
        best_idx = int(np.argmax(combined))

    best_x = pareto_X[best_idx]
    best_included, norm_w = _weights_to_selection(best_x)

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
    final_err_eff_size = (
        effective_ensemble_size(compute_error_correlation_matrix(selected_oofs, y))
        if len(best_included) > 1
        else 1.0
    )

    pareto_front = list(zip(metric_values.tolist(), diversity_values.tolist()))

    info_dict = {
        "selected_models": best_included,
        "weights": norm_w,
        "metric_score": final_metric,
        "effective_size": final_eff_size,
        "error_effective_size": final_err_eff_size,
        "pareto_front": pareto_front,
    }

    # --- Detailed report ---
    logger.info("=" * 60)
    sel_method = "knee" if use_knee else f"dw={diversity_weight:.2f}"
    logger.info(f"NSGA-II ENSEMBLE RESULT ({sel_method})")
    logger.info("-" * 60)
    logger.info(
        f"  Selected {len(best_included)}/{n_models} models | "
        f"{metric}={final_metric:.6f} | "
        f"Pred N_eff={final_eff_size:.2f} | Error N_eff={final_err_eff_size:.2f}"
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
    n_pareto = len(pareto_F)
    if n_pareto > 0:
        logger.info("-" * 60)
        logger.info(
            f"  Pareto front: {n_pareto} solutions | "
            f"{metric}=[{metric_values.min():.6f}, {metric_values.max():.6f}] | "
            f"N_eff=[{diversity_values.min():.2f}, {diversity_values.max():.2f}]"
        )

        best_m_idx = int(np.argmax(metric_values))
        best_d_idx = int(np.argmax(diversity_values))
        logger.info(
            f"  Best {metric} on Pareto: {metric_values[best_m_idx]:.6f} "
            f"(N_eff={diversity_values[best_m_idx]:.2f})"
        )
        logger.info(
            f"  Best diversity on Pareto: N_eff={diversity_values[best_d_idx]:.2f} "
            f"({metric}={metric_values[best_d_idx]:.6f})"
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


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise an array to [0, 1]."""
    rng = arr.max() - arr.min()
    if rng == 0:
        return np.ones_like(arr)
    return (arr - arr.min()) / rng


def log_fold_diversity(
    selected_oofs: list[np.ndarray],
    y: np.ndarray,
    fold_indices: list[np.ndarray],
    weights: list[float],
    metric: str = "roc_auc",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Log per-fold error diversity and metric scores for the selected ensemble.

    Computes error N_eff and per-model/blend metric in each CV fold
    separately. Combining diversity with metric context lets you
    distinguish "hard fold, models collapse" (low AUC + low N_eff =
    red flag) from "easy fold, nothing to diversify" (high AUC + low
    N_eff = harmless).

    This is purely diagnostic: it does not alter the ensemble.

    Args:
        selected_oofs: OOF arrays for the selected models.
        y: True target values.
        fold_indices: List of validation index arrays, one per fold.
        weights: Blend weights for the selected models (sum to 1).
        metric: Metric name ('roc_auc' or 'neg_rmse').
        labels: Model labels for logging.

    Returns:
        Dict with fold_neffs, fold_blend_scores, mean, std, worst_fold.
    """
    n_models = len(selected_oofs)
    if n_models < 2:
        return {"fold_neffs": [], "mean": 0.0, "std": 0.0}

    _labels = labels or [f"model[{i}]" for i in range(n_models)]
    n_folds = len(fold_indices)

    logger.info("-" * 60)
    logger.info(f"FOLD-LEVEL ERROR DIVERSITY ({n_folds} folds, {n_models} models)")

    fold_neffs: list[float] = []
    fold_blend_scores: list[float] = []
    fold_corr_matrices: list[np.ndarray] = []

    for fold_idx, val_idx in enumerate(fold_indices):
        fold_oofs = [oof[val_idx] for oof in selected_oofs]
        fold_y = y[val_idx]

        # Error diversity
        fold_err_corr = compute_error_correlation_matrix(fold_oofs, fold_y)
        fold_neff = effective_ensemble_size(fold_err_corr)
        fold_neffs.append(fold_neff)
        fold_corr_matrices.append(fold_err_corr)

        # Blend score for this fold
        fold_blend = sum(w * p for w, p in zip(weights, fold_oofs))
        try:
            blend_score = _score_metric(fold_y, np.array(fold_blend), metric)
        except Exception:
            blend_score = float("nan")
        fold_blend_scores.append(blend_score)

        # Per-model scores for this fold
        model_scores = []
        for i in range(n_models):
            try:
                ms = _score_metric(fold_y, fold_oofs[i], metric)
            except Exception:
                ms = float("nan")
            model_scores.append(ms)

        # Main fold line
        metric_short = "AUC" if metric == "roc_auc" else metric
        logger.info(
            f"  Fold {fold_idx}: Error N_eff={fold_neff:.2f}  "
            f"{metric_short}(blend)={blend_score:.4f}  "
            f"({len(val_idx)} samples)"
        )
        # Per-model scores (compact, one line)
        parts = [f"{_labels[i]}={model_scores[i]:.4f}" for i in range(n_models)]
        logger.info(f"    {'  '.join(parts)}")

    mean_neff = float(np.mean(fold_neffs))
    std_neff = float(np.std(fold_neffs))
    min_neff = float(min(fold_neffs))
    max_neff = float(max(fold_neffs))
    mean_blend = float(np.mean(fold_blend_scores))

    logger.info(
        f"  N_eff: mean={mean_neff:.2f}  std={std_neff:.3f}  "
        f"range=[{min_neff:.2f}, {max_neff:.2f}]"
    )

    # Flag worst fold by blend metric (lowest AUC = hardest fold)
    worst_blend_idx = int(np.argmin(fold_blend_scores))
    worst_blend_delta = fold_blend_scores[worst_blend_idx] - mean_blend
    metric_short = "AUC" if metric == "roc_auc" else metric
    if worst_blend_delta < -0.005:
        logger.info(
            f"  Worst fold by blend {metric_short}: fold {worst_blend_idx} "
            f"(delta={worst_blend_delta:+.4f} vs mean)"
        )

    # Flag instability
    if std_neff > 0.5:
        logger.warning(
            "  HIGH fold variance — diversity is data-split dependent. "
            "Consider fold-specific model selection."
        )
        # Show pairwise error correlations of worst-diversity fold
        worst_div_fold = int(np.argmin(fold_neffs))
        worst_corr = fold_corr_matrices[worst_div_fold]
        logger.info(
            f"  Worst diversity fold {worst_div_fold} "
            f"(N_eff={fold_neffs[worst_div_fold]:.2f}) "
            f"pairwise error correlations:"
        )
        for i in range(n_models):
            for j in range(i + 1, n_models):
                logger.info(
                    f"    {_labels[i]} <-> {_labels[j]}: "
                    f"{worst_corr[i, j]:.4f}"
                )
    elif std_neff > 0.3:
        logger.info("  ~ Moderate fold variance in diversity")
    else:
        logger.info("  Stable diversity across folds")
    logger.info("=" * 60)

    return {
        "fold_neffs": fold_neffs,
        "fold_blend_scores": fold_blend_scores,
        "mean": mean_neff,
        "std": std_neff,
        "min": min_neff,
        "max": max_neff,
        "worst_fold": int(np.argmin(fold_neffs)),
    }


def run_nsga2_ensemble(
    oof_list: list[np.ndarray],
    test_list: list[np.ndarray],
    y: np.ndarray,
    n_trials: int = 300,
    metric: str = "roc_auc",
    diversity_weight: float = 0.3,
    seed: int = 42,
    labels: list[str] | None = None,
    pop_size: int = 40,
    diversity_metric: str = "pearson_neff",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run NSGA-II multi-objective optimization for ensemble selection.

    Uses pymoo's NSGA-II with SBX crossover and polynomial mutation.
    Simultaneously optimizes:
    - Objective 1: Primary metric (AUC-ROC, RMSE, etc.)
    - Objective 2: Ensemble diversity (configurable via diversity_metric)

    Each solution is a weight vector over all models.  Models with weight
    below the threshold are excluded from the blend.

    The diversity_weight is only used for Pareto front selection, not
    during optimization. Use select_from_pareto() to re-select from
    the same Pareto front with different weights.

    Args:
        oof_list: List of OOF prediction arrays from all base models.
        test_list: List of test prediction arrays from all base models.
        y: True target values.
        n_trials: Approximate number of objective evaluations.  Mapped
                  to ``n_gen = max(n_trials // pop_size, 2)`` generations.
        metric: Primary metric name.
        diversity_weight: Weight for diversity in final selection
                          (0.0 = pure metric, 1.0 = pure diversity).
        seed: Random seed.
        labels: Model labels for logging.
        pop_size: NSGA-II population size per generation.
        diversity_metric: Diversity metric for NSGA-II's second objective.
            - 'pearson_neff': N_eff from Pearson error correlation eigenvalues
            - 'spearman_neff': N_eff from Spearman rank error correlation
            - 'ambiguity': Weighted prediction variance (ensemble disagreement)

    Returns:
        Tuple of (best_test_preds, info_dict):
        - best_test_preds: Test predictions from the best ensemble.
        - info_dict: Dictionary containing:
          - selected_models: indices of selected models
          - weights: blend weights
          - metric_score: primary metric value
          - effective_size: effective ensemble size
          - error_effective_size: error-based effective ensemble size
          - pareto_front: list of (metric, diversity) tuples
          - pareto_trials: dict with 'F' and 'X' arrays (for reuse
            with select_from_pareto)
          - diversity_metric: which diversity metric was used
    """
    n_models = len(oof_list)
    _labels = labels or [f"model[{i}]" for i in range(n_models)]
    n_gen = max(n_trials // pop_size, 2)

    import time as _time
    nsga2_start = _time.time()

    # --- Input diagnostics ---
    logger.info("=" * 60)
    logger.info("NSGA-II ENSEMBLE INPUT (pymoo)")
    logger.info("-" * 60)
    logger.info(f"  {n_models} prediction arrays, {len(oof_list[0])} samples each")
    logger.info(
        f"  pop_size={pop_size}, n_gen={n_gen} "
        f"(~{pop_size * n_gen} evals), diversity_weight={diversity_weight}, "
        f"diversity_metric={diversity_metric}"
    )

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
    logger.info(f"  Solo {metric} scores (best -> worst):")
    for idx, score in sorted_by_score:
        logger.info(f"    {_labels[idx]:<25s}  {score:.6f}")

    # Full correlation matrices — predictions vs errors
    full_corr = compute_correlation_matrix(oof_list)
    full_neff = effective_ensemble_size(full_corr)
    full_err_corr = compute_error_correlation_matrix(oof_list, y)
    full_err_neff = effective_ensemble_size(full_err_corr)
    logger.info(
        f"  Pred N_eff={full_neff:.2f}  Error N_eff={full_err_neff:.2f} "
        f"(all {n_models} arrays)"
    )
    if full_err_neff > full_neff + 0.3:
        logger.info(
            "  Error N_eff > Pred N_eff: models predict similarly but "
            "make mistakes on different samples — good ensemble potential."
        )

    # Correlation heatmap (top-left triangle, compact)
    logger.info(f"  Pairwise correlations (high=redundant):")
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

    # --- pymoo NSGA-II ---
    problem = _EnsembleProblem(oof_list, y, metric, diversity_metric=diversity_metric)

    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    result = pymoo_minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )

    nsga2_elapsed = _time.time() - nsga2_start

    # Extract Pareto front
    pareto_F = result.F  # (n_pareto, 2)  negated objectives
    pareto_X = result.X  # (n_pareto, n_models) weight vectors

    if pareto_F is None or len(pareto_F) == 0:
        # Fallback: use all models with equal weights
        equal_weights = [1.0 / n_models] * n_models
        test_preds = sum(w * p for w, p in zip(equal_weights, test_list))
        return np.array(test_preds), {
            "selected_models": list(range(n_models)),
            "weights": equal_weights,
            "metric_score": 0.0,
            "effective_size": 1.0,
            "error_effective_size": 1.0,
            "pareto_front": [],
            "pareto_trials": {"F": np.empty((0, 2)), "X": np.empty((0, n_models))},
            "diversity_metric": diversity_metric,
        }

    n_evals = pop_size * n_gen
    logger.info(
        f"NSGA-II (pymoo): {n_evals} evaluations in {nsga2_elapsed:.1f}s "
        f"({nsga2_elapsed / max(n_evals, 1) * 1000:.1f}ms/eval), "
        f"Pareto front: {len(pareto_F)} solutions"
    )

    # Select from Pareto front using knee-point by default
    test_preds, info_dict = select_from_pareto(
        pareto_F, pareto_X, oof_list, test_list, y,
        n_models=n_models,
        diversity_weight=diversity_weight,
        metric=metric,
        labels=_labels,
        use_knee=True,
    )

    # Include Pareto data for reuse with different diversity_weights
    info_dict["pareto_trials"] = {"F": pareto_F, "X": pareto_X}
    info_dict["diversity_metric"] = diversity_metric

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
