"""
Ensemble Blender — Weight optimization and blending strategies.

Provides multiple ensemble strategies:
- Optimized weighted blend (Optuna-optimized weights)
- Rank averaging (non-parametric blending)
- Meta-model stacking (LogisticRegression on OOF predictions)

All strategies work with OOF predictions to evaluate without leakage.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold

logger = logging.getLogger("maestro")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute metric score (always returns higher-is-better value)."""
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_pred))
    elif metric == "neg_rmse":
        return -float(np.sqrt(mean_squared_error(y_true, y_pred)))
    else:
        # Default: try roc_auc
        try:
            return float(roc_auc_score(y_true, y_pred))
        except Exception:
            return -float(np.sqrt(mean_squared_error(y_true, y_pred)))


def optimize_blend_weights(
    oof_list: list[np.ndarray],
    y: np.ndarray,
    n_trials: int = 500,
    metric: str = "roc_auc",
    seed: int = 42,
) -> list[float]:
    """Find optimal blend weights using Optuna.

    Optimizes weights for a weighted average of model predictions.
    Weights are constrained to sum to 1.0.

    Args:
        oof_list: List of OOF prediction arrays, one per model.
                  Each array has shape (n_samples,).
        y: True target values, shape (n_samples,).
        n_trials: Number of Optuna trials for weight search.
        metric: Metric to optimize ('roc_auc' for classification,
                'neg_rmse' for regression).
        seed: Random seed for reproducibility.

    Returns:
        List of optimal weights (one per model), summing to 1.0.

    Steps:
        1. Create an Optuna study (direction='maximize' for AUC).
        2. Define objective function:
           a. Suggest a weight for each model using
              trial.suggest_float(f'w_{i}', 0.0, 1.0).
           b. Normalize weights to sum to 1.0.
           c. Compute weighted average: blend = sum(w_i * oof_i).
           d. Compute and return the metric score.
        3. Run study.optimize for n_trials.
        4. Extract best weights and normalize.
        5. Return the weight list.
    """
    n_models = len(oof_list)
    oof_arr = np.column_stack(oof_list)  # shape (n_samples, n_models)

    def objective(trial: optuna.Trial) -> float:
        raw_weights = np.array(
            [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)]
        )
        total = raw_weights.sum()
        if total == 0:
            return -np.inf
        weights = raw_weights / total
        blend = oof_arr @ weights
        return _score(y, blend, metric)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    raw_weights = np.array([best_params[f"w_{i}"] for i in range(n_models)])
    total = raw_weights.sum()
    weights = (raw_weights / total).tolist() if total > 0 else [1.0 / n_models] * n_models
    return weights


def apply_blend(
    preds_list: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """Apply blend weights to a list of prediction arrays.

    Args:
        preds_list: List of prediction arrays (OOF or test).
        weights: List of weights (must sum to ~1.0).

    Returns:
        Blended prediction array: sum(weight_i * preds_i).

    Steps:
        1. Verify len(preds_list) == len(weights).
        2. Compute weighted sum: np.sum(w * p for w, p in zip).
        3. Return the blended array.
    """
    assert len(preds_list) == len(weights), (
        f"preds_list ({len(preds_list)}) and weights ({len(weights)}) must have same length"
    )
    blended = sum(w * p for w, p in zip(weights, preds_list))
    return np.array(blended)


def rank_average(preds_list: list[np.ndarray]) -> np.ndarray:
    """Blend predictions using rank averaging.

    Converts each model's predictions to ranks (percentiles), then
    averages the ranks. This is robust to different prediction scales.

    Args:
        preds_list: List of prediction arrays.

    Returns:
        Rank-averaged prediction array (values in [0, 1]).

    Steps:
        1. For each prediction array:
           a. Compute ranks using scipy.stats.rankdata.
           b. Normalize ranks to [0, 1] by dividing by (n + 1).
        2. Average all normalized rank arrays.
        3. Return the averaged ranks.
    """
    # REVIEW:BUG — raises IndexError if preds_list is empty; no guard for this edge case.
    # Fix: add `if not preds_list: raise ValueError("preds_list must not be empty")`.
    n = len(preds_list[0])
    normalized_ranks = []
    for preds in preds_list:
        ranks = rankdata(preds)
        normalized_ranks.append(ranks / (n + 1))
    return np.mean(normalized_ranks, axis=0)


def train_meta_model(
    oof_list: list[np.ndarray],
    test_list: list[np.ndarray],
    y: np.ndarray,
    n_folds: int = 10,
    seed: int = 42,
    C: float = 1.0,
    task_type: str = "binary_classification",
) -> tuple[np.ndarray, np.ndarray]:
    """Train a meta-model (stacking) on OOF predictions.

    Uses LogisticRegression as the meta-learner. Input features are
    the OOF predictions from base models, optionally with logit
    transforms appended for richer representation.

    The meta-model itself uses cross-validation on the OOF predictions
    to produce its own OOF predictions — preventing leakage.

    Args:
        oof_list: List of OOF prediction arrays from base models.
        test_list: List of test prediction arrays from base models.
        y: True target values.
        n_folds: Number of CV folds for the meta-model.
        seed: Random seed.
        C: Regularization strength for LogisticRegression.

    Returns:
        Tuple of (meta_oof_preds, meta_test_preds):
        - meta_oof_preds: OOF predictions from the meta-model.
        - meta_test_preds: Test predictions from the meta-model.

    Steps:
        1. Stack OOF predictions into a matrix: X_meta = np.column_stack(oof_list).
        2. Add logit transforms: for each OOF column, append
           log(p / (1 - p)) clipped to [-5, 5] to X_meta.
        3. Stack test predictions similarly: X_test_meta.
        4. Create StratifiedKFold(n_folds, seed).
        5. Initialize meta_oof (len(y),) and meta_test (len(test),).
        6. For each fold:
           a. Fit LogisticRegression(C=C) on training fold of X_meta.
           b. Predict on validation fold → meta_oof[val_idx].
           c. Predict on X_test_meta → meta_test += preds / n_folds.
        7. Return (meta_oof, meta_test).
    """
    is_regression = task_type == "regression"

    X_meta = np.column_stack(oof_list)  # (n_samples, n_models)

    if not is_regression:
        def _logit_transform(arr: np.ndarray) -> np.ndarray:
            p = np.clip(arr, 1e-6, 1 - 1e-6)
            return np.clip(np.log(p / (1 - p)), -5, 5)

        logit_meta = np.column_stack([_logit_transform(col) for col in oof_list])
        X_meta = np.hstack([X_meta, logit_meta])

    X_test_meta = np.column_stack(test_list)
    if not is_regression:
        logit_test = np.column_stack([_logit_transform(col) for col in test_list])
        X_test_meta = np.hstack([X_test_meta, logit_test])

    n_test = X_test_meta.shape[0]
    meta_oof = np.zeros(len(y))
    meta_test = np.zeros(n_test)

    if is_regression:
        cv_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = cv_splitter.split(X_meta)
    else:
        cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = cv_splitter.split(X_meta, y)

    for train_idx, val_idx in splits:
        X_tr, X_val = X_meta[train_idx], X_meta[val_idx]
        y_tr = y[train_idx]

        if is_regression:
            meta_model = Ridge(alpha=1.0 / C)
            meta_model.fit(X_tr, y_tr)
            meta_oof[val_idx] = meta_model.predict(X_val)
            meta_test += meta_model.predict(X_test_meta) / n_folds
        else:
            meta_model = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            meta_model.fit(X_tr, y_tr)
            meta_oof[val_idx] = meta_model.predict_proba(X_val)[:, 1]
            meta_test += meta_model.predict_proba(X_test_meta)[:, 1] / n_folds

    return meta_oof, meta_test


def pick_best_strategy(
    candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    metric: str = "roc_auc",
) -> tuple[np.ndarray, str, float]:
    """Evaluate multiple ensemble strategies and pick the best one.

    Args:
        candidates: Dict of {strategy_name: (oof_preds, test_preds)}.
                    Each strategy provides both OOF and test predictions.
        y: True target values.
        metric: Metric to evaluate ('roc_auc', 'neg_rmse', etc.).

    Returns:
        Tuple of (best_test_preds, best_name, best_score):
        - best_test_preds: Test predictions from the best strategy.
        - best_name: Name of the winning strategy.
        - best_score: CV score of the winning strategy.

    Steps:
        1. For each strategy in candidates:
           a. Compute the metric on (oof_preds, y).
           b. Log the strategy name and score.
        2. Select the strategy with the best score.
        3. Return its test predictions, name, and score.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")

    best_name = None
    best_score = -np.inf
    best_test_preds = None

    for name, (oof_preds, test_preds) in candidates.items():
        score = _score(y, oof_preds, metric)
        logger.info(f"Ensemble strategy '{name}': {metric}={score:.6f}")

        if score > best_score:
            best_score = score
            best_name = name
            best_test_preds = test_preds

    return best_test_preds, best_name, best_score
