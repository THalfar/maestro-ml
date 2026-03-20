"""
Model Trainer — Optuna-based hyperparameter optimization with CV.

Runs per-model Optuna studies with QMC warmup followed by TPE sampling.
Stores OOF predictions for every completed trial, which are later used
by the ensemble layer.

Each model gets its own independent study with its own trial budget,
pruning configuration, and search space (from model YAML + LLM overrides).
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
import optuna
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

from src.models.registry import ModelRegistry
from src.utils.io import PipelineConfig, parse_timeout


class FoldEntry(NamedTuple):
    """A single entry in the per-fold leaderboard."""
    score: float
    val_preds: np.ndarray
    val_idx: np.ndarray
    test_preds: np.ndarray
    trial_number: int
    params: dict

logger = logging.getLogger("maestro")

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Suppress pytorch_lightning "LOCAL_RANK" spam from RealMLP
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

# All supported scaler types for Optuna search
ALL_SCALER_CHOICES: list[str] = ["none", "standard", "robust", "quantile"]


@contextlib.contextmanager
def _suppress_catboost_gpu_warnings():
    # DISPUTE: Filtering stderr content to preserve non-CatBoost errors requires fragile regex
    # on C++ library output and is error-prone. The context manager wraps a single model.fit()
    # call, so any non-CatBoost stderr within that call is extremely unlikely. Low risk in practice.
    """Suppress CatBoost C++ GPU memory warnings from stderr."""
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr


@contextlib.contextmanager
def _redirect_stdout_to_log(model_name: str):
    """Redirect stdout to logger.debug() — hides skorch epoch tables from console but keeps them in log file."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        yield
    finally:
        sys.stdout = old_stdout
        captured = buf.getvalue()
        if captured.strip():
            _log = logging.getLogger(__name__)
            for line in captured.splitlines():
                _log.debug("[%s stdout] %s", model_name, line)


def _free_gpu_memory() -> None:
    """Release GPU memory between model training runs."""
    import gc
    gc.collect()
    # Only attempt torch cleanup if it was already imported (avoids Windows DLL issues)
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _rank_norm_1d(arr: np.ndarray) -> np.ndarray:
    """Rank-normalize a 1D array to (0, 1] preserving order.

    Maps values to their rank divided by n, giving a uniform distribution
    over (1/n, 1].  Used to remove scale differences between predictions
    from trials trained with different scalers or architectures before
    stitching per-fold predictions into a composite.

    Args:
        arr: 1D prediction array.

    Returns:
        Rank-normalized array of the same shape.
    """
    return rankdata(arr) / len(arr)


def _combine_fold_test_preds(
    fold_test_preds: list[np.ndarray],
    fold_scores: list[float],
    test_combine: str,
    n_test: int,
    is_multiclass: bool = False,
    maximize: bool = True,
) -> np.ndarray:
    """Combine per-fold test predictions into a single array.

    Args:
        fold_test_preds: List of test prediction arrays, one per fold.
        fold_scores: Validation score for each fold (used by score_weighted).
        test_combine: Combination method:
            ``"arithmetic"`` — uniform average (1/n_folds each).
            ``"score_weighted"`` — weight by fold validation score.
            ``"geomean"`` — geometric mean ``exp(mean(log(p)))``; better
            calibrated near 0/1 for binary classification.
        n_test: Number of test samples (for fallback initialization).
        is_multiclass: If True, skip geomean (not defined for prob vectors).
        maximize: If True, higher fold_scores are better (classification).
            If False (regression), fold_scores are errors — they are inverted
            (1/s) before weighting so better-scoring (lower-error) folds get
            more weight.

    Returns:
        Combined test prediction array of shape ``(n_test,)`` or
        ``(n_test, n_classes)`` for multiclass.
    """
    n = len(fold_test_preds)
    if n == 0:
        return np.zeros(n_test)

    if test_combine == "score_weighted":
        # For regression (maximize=False) fold_scores are errors (lower=better).
        # Invert so lower-error folds receive higher weight.
        if not maximize:
            weights = [1.0 / (s + 1e-9) for s in fold_scores]
        else:
            weights = list(fold_scores)
        total = float(sum(weights))
        if total > 0:
            return sum(w / total * p for w, p in zip(weights, fold_test_preds))
        # Fallback: all weights zero → uniform
        return sum(p for p in fold_test_preds) / n

    if test_combine == "geomean" and not is_multiclass:
        _eps = 1e-7
        return np.exp(
            np.mean([np.log(np.clip(p, _eps, 1.0 - _eps)) for p in fold_test_preds], axis=0)
        )

    # Default: arithmetic mean
    return sum(p for p in fold_test_preds) / n


def _deduplicate_composites(
    composites: list[dict[str, Any]],
    corr_threshold: float = 0.9999,
    maximize: bool = True,
) -> list[dict[str, Any]]:
    """Remove near-duplicate composites based on OOF correlation.

    Composites with pairwise Pearson correlation above ``corr_threshold``
    are considered duplicates.  Only the one with the best ``avg_score``
    is kept.

    Args:
        composites: List of composite dicts with ``oof_preds`` and ``avg_score``.
        corr_threshold: Correlation threshold for deduplication.
        maximize: Whether higher scores are better.

    Returns:
        Deduplicated list of composites (order preserved).
    """
    if len(composites) <= 1:
        return composites

    n = len(composites)
    keep = [True] * n

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            oof_i = composites[i]["oof_preds"].ravel()
            oof_j = composites[j]["oof_preds"].ravel()
            corr = abs(float(np.corrcoef(oof_i, oof_j)[0, 1]))
            if np.isnan(corr):
                corr = 1.0
            if corr >= corr_threshold:
                # Keep the one with better score
                i_better = (
                    composites[i]["avg_score"] >= composites[j]["avg_score"]
                    if maximize
                    else composites[i]["avg_score"] <= composites[j]["avg_score"]
                )
                if i_better:
                    keep[j] = False
                else:
                    keep[i] = False
                    break  # i is gone, stop comparing it

    original = n
    result = [c for c, k in zip(composites, keep) if k]
    if len(result) < original:
        logger.info(
            f"Composite deduplication: {original} → {len(result)} "
            f"(removed {original - len(result)} with corr ≥ {corr_threshold})"
        )
    return result


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base* in-place.

    For nested dicts, values are merged rather than replaced.
    For all other types, the override replaces the base value.

    Returns:
        The mutated *base* dict (also modified in-place).
    """
    for key, val in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def _make_scaler(scaler_type: str) -> StandardScaler | RobustScaler | QuantileTransformer | None:
    """Create a scaler instance from a type string.

    Args:
        scaler_type: One of "none", "standard", "robust", "quantile".

    Returns:
        A fitted-ready scaler instance, or None for "none".
    """
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "robust":
        return RobustScaler()
    elif scaler_type == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=42)
    return None


def _identify_scale_cols(X: pd.DataFrame) -> list[str]:
    """Identify columns suitable for scaling: numeric continuous only.

    Excludes binary columns (0/1) and low-cardinality ordinal-like columns
    (integers with <= 20 unique values). These don't benefit from scaling.

    Args:
        X: Feature DataFrame.

    Returns:
        List of column names to apply scaling to.
    """
    scale_cols = []
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            continue
        nunique = X[col].nunique(dropna=True)
        if nunique <= 2:
            continue  # binary
        if nunique <= 20:
            non_null = X[col].dropna()
            if len(non_null) > 0 and (non_null == non_null.round()).all():
                continue  # ordinal integer
        scale_cols.append(col)
    return scale_cols


def _apply_prescaling(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    feature_cols: list[str],
    scaler_type: str,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, bool]:
    """Pre-scale train and test DataFrames with a single locked scaler.

    Fits the scaler on the full training data and transforms both train
    and test.  Returns new DataFrames (no mutation).

    Args:
        train: Training DataFrame.
        test: Optional test DataFrame.
        feature_cols: Feature column names.
        scaler_type: Scaler type string ("standard", "robust", "quantile").
        model_name: For logging.

    Returns:
        Tuple of (train, test, was_scaled).  ``was_scaled`` is True when
        scaling was actually applied (scaler_type != "none" and scale_cols
        were found).
    """
    if scaler_type == "none":
        return train, test, False
    scale_cols = _identify_scale_cols(train[feature_cols])
    if not scale_cols:
        return train, test, False
    scaler_obj = _make_scaler(scaler_type)
    if scaler_obj is None:
        return train, test, False
    train = train.copy()
    train[scale_cols] = scaler_obj.fit_transform(train[scale_cols])
    if test is not None:
        test = test.copy()
        test[scale_cols] = scaler_obj.transform(test[scale_cols])
    logger.info(
        f"[{model_name}] Pre-scaled {len(scale_cols)} columns with "
        f"'{scaler_type}' (locked scaler — skipping per-fold scaling)"
    )
    return train, test, True


def _apply_scaler_fold(
    scaler_type: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame | None,
    scale_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Fit a scaler on X_train[scale_cols] and transform train/val/test.

    Returns new DataFrames (no mutation). If scaler_type is "none" or
    scale_cols is empty, returns inputs unchanged.

    Args:
        scaler_type: Scaler type string.
        X_train: Training features for this fold.
        X_val: Validation features for this fold.
        X_test: Test features (may be None).
        scale_cols: Columns to scale.

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled).
    """
    if scaler_type == "none" or not scale_cols:
        return X_train, X_val, X_test
    # Filter to cols that actually exist
    cols = [c for c in scale_cols if c in X_train.columns]
    if not cols:
        return X_train, X_val, X_test

    scaler = _make_scaler(scaler_type)
    if scaler is None:
        return X_train, X_val, X_test

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_val[cols] = scaler.transform(X_val[cols])
    if X_test is not None:
        X_test = X_test.copy()
        X_test[cols] = scaler.transform(X_test[cols])
    return X_train, X_val, X_test


def _greedy_pareto_select(
    composites: list[dict[str, Any]],
    n_select: int,
    diversity_metric: str,
    diversity_weight: float,
    maximize: bool,
) -> list[dict[str, Any]]:
    """Greedy diversity-aware selection from Pareto front composites.

    Selects ``n_select`` composites by iteratively adding the composite
    that maximises ``(1-dw)*norm_score + dw*norm_diversity``.  Diversity
    is computed using the configured metric (pearson_neff, spearman_neff,
    or ambiguity).

    .. note::
        For ``pearson_neff`` and ``spearman_neff``, **prediction** correlation
        is used (not error correlation as in the outer NSGA-II in
        ``diversity.py``) because ``y_true`` is unavailable here.  Prediction
        correlation is a reasonable proxy for composites from the same model.

    For ``spearman_neff`` the OOF arrays are pre-ranked once (O(n log n)
    per composite), then all subsequent pairwise computations are Pearson
    on ranks → O(n).  ``pearson_neff`` is O(n) per pair natively.
    ``ambiguity`` is O(n × k) variance.

    Args:
        composites: All Pareto front composites (each dict has
            ``oof_preds``, ``test_preds``, ``avg_score``, etc.).
        n_select: Number of composites to select.
        diversity_metric: One of ``pearson_neff``, ``spearman_neff``,
            ``ambiguity``.
        diversity_weight: Trade-off (0 = pure score, 1 = pure diversity).
        maximize: Whether higher scores are better.

    Returns:
        Selected composites in greedy-insertion order.
    """
    from src.ensemble.diversity import (
        compute_correlation_matrix,
        effective_ensemble_size,
        compute_ambiguity,
    )

    if len(composites) <= n_select:
        return list(composites)

    oofs = [c["oof_preds"] for c in composites]
    scores = np.array([c["avg_score"] for c in composites])

    # Normalise scores to [0, 1] (higher = better)
    s_range = scores.max() - scores.min()
    if s_range > 0:
        if maximize:
            norm_scores = (scores - scores.min()) / s_range
        else:
            norm_scores = (scores.max() - scores) / s_range
    else:
        norm_scores = np.ones_like(scores)

    # Pre-rank for spearman (done once, O(n log n) per composite)
    if diversity_metric == "spearman_neff":
        div_data = [rankdata(oof) for oof in oofs]
    else:
        div_data = oofs

    def _set_diversity(indices: list[int]) -> float:
        """Compute diversity for a set of composites."""
        if len(indices) <= 1:
            return 0.0
        if diversity_metric in ("pearson_neff", "spearman_neff"):
            selected = [div_data[i] for i in indices]
            corr = compute_correlation_matrix(selected)
            neff = effective_ensemble_size(corr)
            # Normalise: neff ∈ [1, k] → [0, 1]
            return (neff - 1.0) / max(len(indices) - 1.0, 1e-12)
        else:  # ambiguity
            selected = [oofs[i] for i in indices]
            w = np.ones(len(indices)) / len(indices)
            return compute_ambiguity(selected, w)

    # Start with best-scoring composite
    selected_idx: list[int] = [int(np.argmax(norm_scores))]
    remaining = set(range(len(composites))) - {selected_idx[0]}

    while len(selected_idx) < n_select and remaining:
        # Compute diversity for each candidate addition
        candidate_divs: dict[int, float] = {}
        for i in remaining:
            candidate_divs[i] = _set_diversity(selected_idx + [i])

        # Min-max normalise diversity across candidates at this step
        div_vals = np.array(list(candidate_divs.values()))
        d_range = div_vals.max() - div_vals.min()

        best_combined = -np.inf
        best_idx = -1
        for i in remaining:
            norm_div = (
                (candidate_divs[i] - div_vals.min()) / d_range
                if d_range > 0
                else 1.0
            )
            combined = (1 - diversity_weight) * norm_scores[i] + diversity_weight * norm_div
            if combined > best_combined:
                best_combined = combined
                best_idx = i

        selected_idx.append(best_idx)
        remaining.discard(best_idx)

    return [composites[i] for i in selected_idx]


class PerFoldTracker:
    """Track top-N predictions per CV fold during Optuna for per-fold selection.

    During each trial's fold training, the model also predicts on test data.
    This tracker maintains a bounded leaderboard per fold, keeping only
    the n_top best (score, oof_slice, test_preds) entries.  Works with
    pruned trials too — a trial pruned after fold 2 can still contribute
    its completed folds.

    After Optuna, ``assemble()`` builds composite prediction arrays where
    each fold's slice comes from the k-th best trial for that fold.

    Supports two modes controlled by ``diversity_mode``:

    - ``"vanilla"`` (default): Pure top-N by score.  Original behaviour.
    - ``"tiered"``: Two-tier insertion.  Tier-1 (``tier1_size`` slots) is
      protected — always top-K by pure score, never replaced for diversity
      reasons.  Tier-2 (remaining ``n_top - tier1_size`` slots) is
      diversity-aware: new entries with max correlation ≥ ``tier2_corr_threshold``
      to an existing tier-2 entry only replace *that closest entry* when
      the new score is better (cluster-best logic).  Entries with lower
      correlation are inserted normally.

    Attributes:
        n_top: Maximum entries to keep per fold.
        n_folds: Number of CV folds.
        maximize: True for metrics like AUC (higher=better), False for RMSE.
        fold_data: ``{fold_idx: [FoldEntry, ...]}`` — sorted best-first.
        diversity_mode: ``"vanilla"`` or ``"tiered"``.
        tier1_size: Number of score-protected slots (tiered mode only).
        tier2_corr_threshold: Correlation threshold for tier-2 cluster
            replacement (tiered mode only).  Default 0.99.
    """

    def __init__(
        self,
        n_top: int,
        n_folds: int,
        maximize: bool,
        diversity_mode: str = "vanilla",
        tier1_size: int = 5,
        tier2_corr_threshold: float = 0.99,
    ) -> None:
        self.n_top = n_top
        self.n_folds = n_folds
        self.maximize = maximize
        self.diversity_mode = diversity_mode
        self.tier1_size = min(tier1_size, n_top)
        self.tier2_corr_threshold = tier2_corr_threshold
        self.fold_data: dict[int, list[FoldEntry]] = {f: [] for f in range(n_folds)}

    # ------------------------------------------------------------------
    def n_entries(self, fold_idx: int) -> int:
        """Return the number of entries stored for a given fold."""
        return len(self.fold_data[fold_idx])

    # ------------------------------------------------------------------
    def _is_better(self, a: float, b: float) -> bool:
        """Return True if score *a* is strictly better than *b*."""
        return (a > b) if self.maximize else (a < b)

    # ------------------------------------------------------------------
    def update(
        self,
        fold_idx: int,
        score: float,
        val_preds: np.ndarray,
        val_idx: np.ndarray,
        test_preds: np.ndarray,
        trial_number: int,
        params: dict[str, Any],
    ) -> None:
        """Insert a fold result if it qualifies for the leaderboard.

        In ``"vanilla"`` mode, keeps the per-fold list sorted (best first)
        and bounded to n_top by pure score.

        In ``"tiered"`` mode, the first ``tier1_size`` slots are reserved
        for the best scores (always kept).  Remaining slots use
        diversity-aware insertion: if the new entry's max |correlation|
        with existing tier-2 entries exceeds ``tier2_corr_threshold``,
        it only replaces *that specific closest entry* when the new
        score is better.  Otherwise it is inserted normally (displacing
        the worst entry if full).

        All arrays are copied to avoid aliasing with trial-scope variables.
        """
        entry = FoldEntry(
            score=score,
            val_preds=val_preds.copy(),
            val_idx=val_idx.copy(),
            test_preds=test_preds.copy(),
            trial_number=trial_number,
            params=dict(params),
        )
        data = self.fold_data[fold_idx]

        if self.diversity_mode == "tiered" and len(data) >= self.tier1_size:
            self._tiered_insert(fold_idx, entry)
        else:
            # Vanilla mode (or tiered mode during warmup before tier1 is full)
            self._vanilla_insert(fold_idx, entry)

    # ------------------------------------------------------------------
    def _vanilla_insert(self, fold_idx: int, entry: FoldEntry) -> None:
        """Original top-N by score insertion."""
        data = self.fold_data[fold_idx]
        if len(data) < self.n_top:
            data.append(entry)
            data.sort(key=lambda x: x.score, reverse=self.maximize)
        else:
            worst_score = data[-1].score
            if self._is_better(entry.score, worst_score):
                data[-1] = entry
                data.sort(key=lambda x: x.score, reverse=self.maximize)

    # ------------------------------------------------------------------
    def _tiered_insert(self, fold_idx: int, entry: FoldEntry) -> None:
        """Two-tier diversity-aware insertion.

        Tier-1 (indices 0..tier1_size-1) is always sorted best-first by
        score.  If the new entry qualifies for tier-1 (better than the
        worst tier-1 entry), the displaced tier-1 entry cascades into
        tier-2 processing.

        Tier-2 (indices tier1_size..n_top-1) uses correlation-aware
        cluster logic: redundant entries (max |corr| ≥ threshold) only
        replace their closest match when the new score is better.
        Diverse entries are added normally.
        """
        data = self.fold_data[fold_idx]

        # --- Try tier-1 insertion first ---
        tier1 = data[:self.tier1_size]
        if len(tier1) < self.tier1_size:
            # Tier-1 not full yet — insert by score
            data.append(entry)
            data.sort(key=lambda x: x.score, reverse=self.maximize)
            return

        worst_tier1 = tier1[-1]
        if self._is_better(entry.score, worst_tier1.score):
            # New entry enters tier-1; displaced entry cascades to tier-2
            displaced = worst_tier1
            data[self.tier1_size - 1] = entry
            # Re-sort tier-1
            tier1_new = data[:self.tier1_size]
            tier1_new.sort(key=lambda x: x.score, reverse=self.maximize)
            data[:self.tier1_size] = tier1_new
            # Try to insert displaced into tier-2
            self._tier2_insert(fold_idx, displaced)
            return

        # --- Doesn't qualify for tier-1, try tier-2 ---
        self._tier2_insert(fold_idx, entry)

    # ------------------------------------------------------------------
    def _tier2_insert(self, fold_idx: int, entry: FoldEntry) -> None:
        """Insert into tier-2 with diversity-aware cluster logic."""
        data = self.fold_data[fold_idx]
        tier2_start = self.tier1_size
        tier2 = data[tier2_start:]
        tier2_capacity = self.n_top - self.tier1_size

        if tier2_capacity <= 0:
            return  # No tier-2 slots configured

        if not tier2:
            # Tier-2 empty — just add
            data.append(entry)
            return

        # Compute max |correlation| with existing tier-2 entries
        max_corr = -1.0
        closest_idx = -1  # index within full data list
        for i in range(tier2_start, len(data)):
            corr = abs(float(np.corrcoef(
                entry.val_preds.ravel(),
                data[i].val_preds.ravel(),
            )[0, 1]))
            if np.isnan(corr):
                corr = 1.0  # constant predictions → treat as identical
            if corr > max_corr:
                max_corr = corr
                closest_idx = i

        if max_corr >= self.tier2_corr_threshold:
            # Redundant — only replace the closest if score is better
            if self._is_better(entry.score, data[closest_idx].score):
                data[closest_idx] = entry
        else:
            # Diverse — insert normally
            if len(tier2) < tier2_capacity:
                data.append(entry)
            else:
                # Full — replace worst tier-2 entry
                worst_idx = tier2_start
                for i in range(tier2_start + 1, len(data)):
                    if self._is_better(data[worst_idx].score, data[i].score):
                        worst_idx = i
                if self._is_better(entry.score, data[worst_idx].score):
                    data[worst_idx] = entry

    # ------------------------------------------------------------------
    def assemble(
        self,
        n_samples: int,
        n_test: int,
        task_type: str = "binary_classification",
        rank_normalize: bool = True,
        test_combine: str = "arithmetic",
    ) -> list[dict[str, Any]]:
        """Build composite OOF + test arrays from per-fold bests.

        For the k-th composite:
        - OOF: ``oof[val_idx] = k-th best trial's val_preds`` for each fold.
        - Test: combine per-fold test predictions according to ``test_combine``.

        When ``rank_normalize=True`` (default), per-fold predictions are
        rank-normalized to (0, 1] before stitching.  This removes scale
        differences caused by different scalers or architectures across
        trials, which would otherwise distort cross-fold comparisons and
        metric computation on the assembled OOF.

        Args:
            n_samples: Total training samples (OOF length).
            n_test: Test set size.
            task_type: Task type string.
            rank_normalize: If True, rank-normalize per-fold predictions to
                (0, 1] before stitching (binary/regression only).
            test_combine: How to combine per-fold test predictions.
                ``"arithmetic"`` — uniform average (``+= test_p / n_folds``).
                ``"score_weighted"`` — weight each fold's test predictions by
                that fold's validation score (higher score = more weight).
                ``"geomean"`` — geometric mean ``exp(mean(log(p)))`` across
                folds; better calibrated near 0/1 for binary classification.

        Returns:
            List of dicts, each with keys: ``oof_preds``, ``test_preds``,
            ``fold_trials``, ``fold_scores``, ``avg_score``.
        """
        n_composites = min(
            self.n_top,
            min(len(d) for d in self.fold_data.values()) if self.fold_data else 0,
        )

        is_multiclass = task_type == "multiclass"
        do_rank = rank_normalize and not is_multiclass

        results: list[dict[str, Any]] = []
        for k in range(n_composites):
            if is_multiclass:
                sample_preds = self.fold_data[0][k].val_preds
                n_classes = sample_preds.shape[1] if sample_preds.ndim > 1 else 1
                oof = np.zeros((n_samples, n_classes))
            else:
                oof = np.zeros(n_samples)

            fold_trials: list[int] = []
            fold_scores: list[float] = []
            fold_test_preds: list[np.ndarray] = []

            for fold_idx in range(self.n_folds):
                entry = self.fold_data[fold_idx][k]
                val_p = entry.val_preds
                test_p = entry.test_preds
                if do_rank:
                    val_p = _rank_norm_1d(val_p)
                    test_p = _rank_norm_1d(test_p)
                oof[entry.val_idx] = val_p
                fold_trials.append(entry.trial_number)
                fold_scores.append(entry.score)
                fold_test_preds.append(test_p)

            test_preds = _combine_fold_test_preds(
                fold_test_preds, fold_scores, test_combine, n_test,
                is_multiclass=is_multiclass,
                maximize=self.maximize,
            )

            results.append({
                "oof_preds": oof,
                "test_preds": test_preds,
                "fold_trials": fold_trials,
                "fold_scores": fold_scores,
                "avg_score": float(np.mean(fold_scores)),
            })

        return results

    # ------------------------------------------------------------------
    def assemble_nsga2(
        self,
        n_samples: int,
        n_test: int,
        task_type: str = "binary_classification",
        n_composites: int = 20,
        n_generations: int = 50,
        pop_size: int = 100,
        diversity_metric: str = "pearson_neff",
        diversity_weight: float = 0.3,
        seed: int = 42,
        rank_normalize: bool = True,
        test_combine: str = "arithmetic",
    ) -> list[dict[str, Any]]:
        """Build composite arrays via NSGA-II fold-level optimization.

        Instead of rank-based assembly (composite k = k-th best per fold),
        uses NSGA-II to find diverse fold combinations.  Each individual
        is a vector of ``n_folds`` continuous variables mapped to integer
        indices, each selecting which of the top-N candidates to use for
        that fold.

        Two objectives (both maximized via negation for pymoo):
            1. Average fold score of the assembled composite.
            2. Trial source diversity proxy (unique trials + index spread).

        After NSGA-II, all Pareto front composites are built and then
        greedy-selected using the actual ``diversity_metric``
        (pearson_neff, spearman_neff, or ambiguity).  At each step the
        composite maximising ``(1-dw)*norm_score + dw*norm_diversity``
        is added.  For spearman_neff, OOF arrays are pre-ranked once
        so pairwise computations are O(n).

        Args:
            n_samples: Number of training samples (OOF array length).
            n_test: Number of test samples.
            task_type: Task type for metric computation.
            n_composites: How many composites to select from Pareto front.
            n_generations: NSGA-II generations.
            pop_size: NSGA-II population size.
            diversity_metric: Diversity metric (pearson_neff, spearman_neff,
                ambiguity).  Used in greedy Pareto selection (the NSGA-II
                objective uses a fast trial-source proxy).
            diversity_weight: Trade-off for Pareto front selection
                (0=pure score, 1=pure diversity).
            seed: Random seed for reproducibility.

        Returns:
            Same format as ``assemble()``: list of dicts with keys
            ``oof_preds``, ``test_preds``, ``fold_trials``, ``fold_scores``,
            ``avg_score``.
        """
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize as pymoo_minimize

        # How many candidates per fold are available
        n_per_fold = min(len(d) for d in self.fold_data.values())
        if n_per_fold == 0:
            return []

        is_multiclass = task_type == "multiclass"
        do_rank = rank_normalize and not is_multiclass

        def _build_composite(indices: list[int]) -> tuple[np.ndarray, np.ndarray, list[int], list[float]]:
            """Build a single composite from fold-index selections.

            Rank-normalizes per-fold predictions to (0, 1] before stitching
            when ``rank_normalize=True``, removing scale differences between
            trials trained with different scalers or architectures.
            """
            if is_multiclass:
                sample_preds = self.fold_data[0][0].val_preds
                n_classes = sample_preds.shape[1] if sample_preds.ndim > 1 else 1
                oof = np.zeros((n_samples, n_classes))
            else:
                oof = np.zeros(n_samples)

            fold_trials: list[int] = []
            fold_scores: list[float] = []
            fold_test_preds: list[np.ndarray] = []

            for fold_idx in range(self.n_folds):
                entry = self.fold_data[fold_idx][indices[fold_idx]]
                val_p = entry.val_preds
                test_p = entry.test_preds
                if do_rank:
                    val_p = _rank_norm_1d(val_p)
                    test_p = _rank_norm_1d(test_p)
                oof[entry.val_idx] = val_p
                fold_trials.append(entry.trial_number)
                fold_scores.append(entry.score)
                fold_test_preds.append(test_p)

            test_preds = _combine_fold_test_preds(
                fold_test_preds, fold_scores, test_combine, n_test,
                is_multiclass=is_multiclass,
                maximize=self.maximize,
            )
            return oof, test_preds, fold_trials, fold_scores

        # --- pymoo problem: each individual = n_folds continuous vars mapped to int indices ---
        tracker = self

        class _FoldAssemblyProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(
                    n_var=tracker.n_folds,
                    n_obj=2,
                    xl=np.zeros(tracker.n_folds),
                    xu=np.ones(tracker.n_folds) * (n_per_fold - 1e-9),
                )

            def _evaluate(self, x, out, *args, **kwargs):
                indices = [int(xi) for xi in x]
                # Clamp to valid range
                indices = [min(i, n_per_fold - 1) for i in indices]

                # Only fold scores are needed here — skip full array allocation.
                fold_scores = [tracker.fold_data[f][indices[f]].score for f in range(tracker.n_folds)]
                avg_score = float(np.mean(fold_scores))

                # Diversity proxy (obj2): unique-trial ratio + index spread.
                # Encourages solutions that draw from different trial sources
                # across folds, without requiring population-level comparisons.
                unique_trials = set()
                for fold_idx, k in enumerate(indices):
                    unique_trials.add(self._get_trial_num(fold_idx, k))
                uniqueness = len(unique_trials) / tracker.n_folds

                # Combine: spread of indices (avoid all-same-rank)
                idx_variance = float(np.var(indices)) / max(1, (n_per_fold - 1))

                # Diversity proxy = uniqueness + index spread (both 0-1 range)
                diversity_proxy = 0.7 * uniqueness + 0.3 * idx_variance

                # Negate both (pymoo minimizes)
                if tracker.maximize:
                    out["F"] = np.array([-avg_score, -diversity_proxy])
                else:
                    out["F"] = np.array([avg_score, -diversity_proxy])

            def _get_trial_num(self, fold_idx: int, k: int) -> int:
                return tracker.fold_data[fold_idx][k].trial_number

        problem = _FoldAssemblyProblem()

        algorithm = NSGA2(
            pop_size=pop_size,
            n_offsprings=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=3),   # low eta = more exploration
            mutation=PM(eta=5),                # low eta = larger mutations
            eliminate_duplicates=True,
        )

        nsga2_start = time.time()
        result = pymoo_minimize(
            problem,
            algorithm,
            ("n_gen", n_generations),
            seed=seed,
            verbose=False,
        )
        nsga2_elapsed = time.time() - nsga2_start

        if result.F is None or len(result.F) == 0:
            logger.warning("Fold-NSGA-II produced no solutions, falling back to rank assembly")
            return self.assemble(n_samples, n_test, task_type, rank_normalize, test_combine)

        # Extract Pareto front solutions
        pareto_F = result.F  # (n_pareto, 2)
        pareto_X = result.X  # (n_pareto, n_folds)

        # Build ALL Pareto composites (needed for real diversity computation)
        all_pareto: list[dict[str, Any]] = []
        for i in range(len(pareto_X)):
            indices = [min(int(xi), n_per_fold - 1) for xi in pareto_X[i]]
            oof, test_p, fold_t, fold_s = _build_composite(indices)
            all_pareto.append({
                "oof_preds": oof,
                "test_preds": test_p,
                "fold_trials": fold_t,
                "fold_scores": fold_s,
                "avg_score": float(np.mean(fold_s)),
            })

        # Greedy selection using actual diversity metric
        n_select = min(n_composites, len(all_pareto))
        results = _greedy_pareto_select(
            all_pareto,
            n_select=n_select,
            diversity_metric=diversity_metric,
            diversity_weight=diversity_weight,
            maximize=self.maximize,
        )

        logger.info(
            f"Fold-NSGA-II: {len(pareto_F)} Pareto solutions in {nsga2_elapsed:.1f}s, "
            f"greedy-selected {len(results)} composites ({diversity_metric}, "
            f"dw={diversity_weight})"
        )
        if results:
            scores = [r["avg_score"] for r in results]
            logger.info(
                f"  Score range: best={max(scores):.6f}, "
                f"worst={min(scores):.6f}"
            )
            # Count unique trials across all selected composites
            all_trials: set[int] = set()
            for r in results:
                all_trials.update(r["fold_trials"])
            logger.info(f"  Unique trials used: {len(all_trials)}")

        return results

    # ------------------------------------------------------------------
    def log_summary(self, model_name: str) -> None:
        """Log per-fold best/worst scores and composite statistics."""
        if not self.fold_data or not self.fold_data[0]:
            return

        n_composites = min(
            self.n_top,
            min(len(d) for d in self.fold_data.values()),
        )

        # Count unique trials contributed
        all_trial_nums: set[int] = set()
        total_entries = 0
        for data in self.fold_data.values():
            for entry in data:
                all_trial_nums.add(entry.trial_number)
                total_entries += 1

        mode_str = f"mode={self.diversity_mode}"
        if self.diversity_mode == "tiered":
            mode_str += f" (tier1={self.tier1_size}, tier2_corr={self.tier2_corr_threshold})"

        logger.info(
            f"[{model_name}] Per-fold selection ({mode_str}): "
            f"{n_composites} composites from {len(all_trial_nums)} unique trials"
        )

        # Per-fold stats
        all_scores: list[float] = []
        for fold_idx in range(self.n_folds):
            data = self.fold_data[fold_idx]
            if not data:
                continue
            best = data[0]
            worst = data[-1]
            all_scores.extend(e.score for e in data)

            # Tier info for tiered mode
            tier_info = ""
            if self.diversity_mode == "tiered" and len(data) > self.tier1_size:
                tier1_trials = {e.trial_number for e in data[:self.tier1_size]}
                tier2_trials = {e.trial_number for e in data[self.tier1_size:]}
                tier_info = f" [T1:{len(tier1_trials)} T2:{len(tier2_trials)} trials]"

            logger.info(
                f"[{model_name}]   Fold {fold_idx:2d}: "
                f"best={best.score:.6f} (trial #{best.trial_number}), "
                f"{len(data)}th={worst.score:.6f} (trial #{worst.trial_number})"
                f"{tier_info}"
            )

        if all_scores:
            logger.info(
                f"[{model_name}] Per-fold score range: "
                f"best={max(all_scores):.6f}, worst={min(all_scores):.6f}"
            )

        # Diversity stats for tiered mode
        if self.diversity_mode == "tiered":
            # Compute average pairwise correlation within tier-2 for fold 0
            fold0 = self.fold_data[0]
            if len(fold0) > self.tier1_size:
                tier2_preds = [e.val_preds for e in fold0[self.tier1_size:]]
                if len(tier2_preds) >= 2:
                    corrs = []
                    for i in range(len(tier2_preds)):
                        for j in range(i + 1, len(tier2_preds)):
                            c = abs(float(np.corrcoef(
                                tier2_preds[i].ravel(), tier2_preds[j].ravel()
                            )[0, 1]))
                            if not np.isnan(c):
                                corrs.append(c)
                    if corrs:
                        logger.info(
                            f"[{model_name}] Tier-2 avg |corr| (fold 0): "
                            f"{np.mean(corrs):.4f} (min={min(corrs):.4f}, max={max(corrs):.4f})"
                        )

        logger.info(
            f"[{model_name}] Unique trials across all composites: "
            f"{len(all_trial_nums)}"
        )

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Pickle the tracker to disk for resume between interrupted runs."""
        import pickle
        data = {
            "fold_data": self.fold_data,
            "n_top": self.n_top,
            "n_folds": self.n_folds,
            "maximize": self.maximize,
            "diversity_mode": self.diversity_mode,
            "tier1_size": self.tier1_size,
            "tier2_corr_threshold": self.tier2_corr_threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "PerFoldTracker":
        """Load a previously saved tracker from disk.

        Warning:
            Uses ``pickle.load`` — only load files written by this process
            in a trusted ``storage_dir``.  Do not load tracker files from
            untrusted or externally-supplied sources.
        """
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        for k, v in data.items():
            setattr(obj, k, v)
        return obj


def _compute_cv_metric(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> float:
    """Compute the cross-validation metric for a given task type."""
    if task_type == "binary_classification":
        return float(roc_auc_score(y_true, y_pred))
    elif task_type == "multiclass":
        return float(roc_auc_score(y_true, y_pred, multi_class="ovr"))
    else:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _extract_sample_weights(
    train: pd.DataFrame,
    training_cfg: dict[str, Any],
) -> np.ndarray | None:
    """Extract sample_weight array from train if the model supports it.

    Returns None if the model does not support sample_weight, if the
    _sample_weight column is missing, or if all weights are uniform (1.0).
    """
    if not training_cfg.get("supports_sample_weight", False):
        return None
    if "_sample_weight" not in train.columns:
        return None
    sw = train["_sample_weight"].values.astype(np.float64)
    if np.all(sw == 1.0):
        return None
    return sw


def _get_eval_metric_value(training: dict, task_type: str, gpu: bool) -> str | None:
    """Extract the eval metric value from training config for a given task/device."""
    em = training.get("eval_metric")
    if not em:
        return None
    # Level 1: by task_type
    if isinstance(em, dict):
        task_em = em.get(task_type, em)
        if isinstance(task_em, str):
            return task_em
        if isinstance(task_em, dict):
            # Has gpu/cpu variants
            return task_em.get("gpu" if gpu else "cpu", next(iter(task_em.values()), None))
    return str(em)


def _build_pruner(pruner_cfg: dict[str, Any]) -> optuna.pruners.BasePruner:
    """Build an Optuna pruner from a config dict.

    Supported types:
        - ``median``: MedianPruner (default). Prunes trials below the median
          of intermediate reports at the same step.
        - ``percentile``: PercentilePruner. Prunes trials in the bottom
          ``percentile`` % — useful when substudy warm-start raises the
          baseline and you want to keep more of the "ok" range for TPE.
        - ``hyperband``: HyperbandPruner.
        - ``none``: NopPruner (no pruning).

    Config keys (all optional):
        - ``type`` (str): Pruner type. Default ``"median"``.
        - ``n_warmup_steps`` (int): Steps before pruning activates.
          Default 1 for median, 0 for percentile.
        - ``n_startup_trials`` (int): Trials that always run in full.
          Default 10 for median, 20 for percentile.
        - ``percentile`` (float): Bottom % to prune (percentile type only).
          Default 25.0.
    """
    pruner_type = pruner_cfg.get("type", "median")

    if pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 1),
            n_startup_trials=pruner_cfg.get("n_startup_trials", 10),
        )
    elif pruner_type == "percentile":
        return optuna.pruners.PercentilePruner(
            percentile=pruner_cfg.get("percentile", 25.0),
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 0),
            n_startup_trials=pruner_cfg.get("n_startup_trials", 20),
        )
    elif pruner_type == "hyperband":
        return optuna.pruners.HyperbandPruner()
    else:
        return optuna.pruners.NopPruner()


def generate_targeted_enqueue(
    study: optuna.Study,
    target_param: str,
    *,
    values: list[int | float] | None = None,
    target_range: tuple[float, float] | None = None,
    n_points: int = 5,
    log: bool = False,
    n_base: int = 3,
    temperature: float = 0.3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate targeted enqueue trials for a specific hyperparameter.

    For each target value, pairs it with other params from the study's
    top trials (sampled via rank-weighted exponential distribution).
    This lets the AI guide TPE exploration **without** changing the search
    space — preserving ``multivariate=True`` compatibility on persistent DBs.

    Args:
        study: Optuna study with completed trials to sample from.
        target_param: Name of the hyperparameter to explore.
        values: Explicit list of values to try.  Mutually exclusive with
            *target_range*.
        target_range: ``(low, high)`` tuple — generates *n_points* evenly
            spaced values (linear or log).  Mutually exclusive with *values*.
        n_points: Number of points to generate from *target_range*.
        log: If ``True``, use ``np.geomspace`` (log-spaced).  If ``False``,
            use ``np.linspace``.  Also affects int rounding.
        n_base: Number of top trials to use as "base" configs.  Each target
            value is paired with each base trial → up to
            ``len(values) × n_base`` candidates.
        temperature: Exponential sampling sharpness for base trial selection.
            Lower = prefer best trials, higher = more diverse.
        seed: Random seed for reproducible sampling.

    Returns:
        List of param dicts ready for ``study.enqueue_trial()``.
        Deduplicated against existing completed trials in the study.

    Raises:
        ValueError: If neither *values* nor *target_range* is provided,
            or if both are provided.
    """
    if (values is None) == (target_range is None):
        raise ValueError(
            "Exactly one of 'values' or 'target_range' must be provided"
        )

    # Completed trials sorted by score (best first).
    maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        return []

    completed.sort(key=lambda t: t.value, reverse=maximize)

    # Resolve target values.
    if values is not None:
        target_vals = list(values)
    else:
        low, high = target_range  # type: ignore[misc]
        if log:
            target_vals = list(np.geomspace(low, high, n_points))
        else:
            target_vals = list(np.linspace(low, high, n_points))

    # Detect whether target param is int-typed (cast generated values).
    existing_target_vals = [
        t.params[target_param]
        for t in completed
        if target_param in t.params
    ]
    is_int_param = (
        existing_target_vals
        and all(isinstance(v, int) for v in existing_target_vals)
    )

    # Sample base trials via exp-distribution.
    effective_n_base = min(n_base, len(completed))
    rng = np.random.default_rng(seed)

    if effective_n_base < len(completed):
        n_candidates = len(completed)
        ranks = np.arange(n_candidates)
        weights = np.exp(-temperature * ranks / n_candidates)
        weights /= weights.sum()
        chosen_idx = rng.choice(
            n_candidates, size=effective_n_base, replace=False, p=weights
        )
    else:
        chosen_idx = np.arange(effective_n_base)

    base_trials = [completed[i] for i in chosen_idx]

    # Build dedup set from existing completed trials.
    existing_keys: set[str] = {
        str(sorted(t.params.items()))
        for t in completed
    }

    # Generate candidates: each target value × each base trial.
    result: list[dict[str, Any]] = []
    for tv in target_vals:
        val = int(round(tv)) if is_int_param else tv
        for bt in base_trials:
            params = dict(bt.params)
            params.pop("scaler", None)
            params[target_param] = val
            key = str(sorted(params.items()))
            if key not in existing_keys:
                result.append(params)
                existing_keys.add(key)

    return result


def _run_substudy(
    model_name: str,
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    registry: ModelRegistry,
    task_type: str,
    gpu: bool,
    results_dir: Path,
    substudy_cfg: dict[str, Any],
    search_space: dict[str, dict[str, Any]],
    scaler_choices: list[str] | None,
    global_seed: int,
    monotone_constraints: list[int] | None = None,
    run_name: str = "run",
    storage_dir: str | None = None,
    existing_main_params: set[str] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Run a substudy on a stratified data subset to find good starting configs.

    Creates a stratified subsample of training data, runs a pure QMC Optuna
    study (no TPE) with fewer CV folds and its own timeout, then rank-samples
    trial configurations for enqueue into the main study.  No pruning — trials
    are fast on the small subset and should always complete.

    Also determines the best scaler when ``lock_scaler`` is True.

    Args:
        model_name: Model name for logging.
        train: Full training DataFrame.
        feature_cols: Feature columns.
        target_col: Target column name.
        registry: ModelRegistry instance.
        task_type: Task type string.
        gpu: Whether to use GPU.
        results_dir: For model artifacts.
        substudy_cfg: Substudy configuration dict.
        search_space: Optuna search space (same as main study).
        scaler_choices: Scaler choices (None if no scaling needed).
        global_seed: Seed for reproducibility.
        monotone_constraints: Optional monotone constraints list.
        existing_main_params: Set of ``str(sorted(params.items()))`` keys for
            trials already completed in the main study.  When provided, the
            top-N guarantee selects the top-N configs that are *not* yet known
            to the main study — sliding past already-known entries so the main
            study always receives genuinely new configs.

    Returns:
        Tuple of:
        - List of trial param dicts (rank-sampled configs to enqueue).
        - Best scaler string (or None if lock_scaler is False or no scaling).
    """
    from sklearn.model_selection import train_test_split

    # Parse substudy config with defaults
    sample_fraction = substudy_cfg.get("sample_fraction", 0.10)
    n_folds = substudy_cfg.get("n_folds", 3)
    timeout_raw = substudy_cfg.get("timeout", "15m")
    timeout = parse_timeout(timeout_raw)
    n_trials = substudy_cfg.get("n_trials")  # None = unlimited, substudy timeout controls
    n_enqueue = substudy_cfg.get("n_enqueue", 20)
    top_n = substudy_cfg.get("top_n", 3)
    temperature = substudy_cfg.get("temperature", 0.3)
    lock_scaler = substudy_cfg.get("lock_scaler", True)

    # Stratified subsample (or full data when fraction >= 1.0)
    if sample_fraction >= 1.0:
        sub_train = train.reset_index(drop=True)
    else:
        y = train[target_col].values
        subsample_seed = global_seed + 9999
        if task_type != "regression":
            sub_train, _ = train_test_split(
                train, train_size=sample_fraction,
                stratify=y, random_state=subsample_seed,
            )
        else:
            sub_train, _ = train_test_split(
                train, train_size=sample_fraction,
                random_state=subsample_seed,
            )
        sub_train = sub_train.reset_index(drop=True)

    # Min size check
    min_rows = n_folds * 10
    if len(sub_train) < min_rows:
        logger.warning(
            f"[{model_name}] Substudy skipped: subsample has {len(sub_train)} rows "
            f"(need ≥{min_rows} for {n_folds}-fold CV)"
        )
        return [], None

    logger.info(
        f"[{model_name}] Substudy: {len(sub_train)} rows ({sample_fraction:.0%} of "
        f"{len(train)}), {n_folds} folds, {n_trials} QMC trials, "
        f"timeout={timeout_raw}"
    )

    # Create lightweight CV splitter
    if task_type != "regression":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=global_seed)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=global_seed)

    # Create objective — global mode, no per-fold tracker, no test, no diversity pruning
    objective = _create_objective(
        model_name=model_name,
        train=sub_train,
        feature_cols=feature_cols,
        target_col=target_col,
        cv=cv,
        search_space=search_space,
        registry=registry,
        task_type=task_type,
        gpu=gpu,
        results_dir=results_dir,
        test=None,
        per_fold_tracker=None,
        fold_timeout=None,
        n_top_trials=n_enqueue,
        scaler_choices=scaler_choices,
        prescaled_scaler=None,
        monotone_constraints=monotone_constraints,
        diversity_pruning=None,
    )

    # Substudy pruner: default NopPruner (no pruning). Substudy trials are
    # fast (15-30s) so pruning saves little time but loses data points for
    # importance analysis and rank-weighted sampling. Configurable via YAML.
    substudy_pruner_cfg = substudy_cfg.get("pruner", {}) or {}
    if not substudy_pruner_cfg.get("type"):
        substudy_pruner_cfg["type"] = "none"
    pruner = _build_pruner(substudy_pruner_cfg)

    # Study direction + persistent storage (substudy gets own __sub DB)
    direction = "minimize" if task_type == "regression" else "maximize"
    sub_study_name = f"{run_name}__{model_name}__sub"
    sub_storage: str | None = None
    if storage_dir:
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        sub_db_path = Path(storage_dir) / f"{sub_study_name}.db"
        sub_storage = f"sqlite:///{sub_db_path}"
    study = optuna.create_study(
        study_name=sub_study_name,
        storage=sub_storage,
        direction=direction,
        pruner=pruner,
        load_if_exists=True,
    )

    # Pure QMC — no TPE phase
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="QMCSampler is experimental")
        study.sampler = optuna.samplers.QMCSampler(
            seed=global_seed,
            warn_independent_sampling=False,
            independent_sampler=optuna.samplers.RandomSampler(seed=global_seed),
        )

    # Enable Optuna's built-in trial logging for substudies
    # (shows "Trial X finished with value: ... and parameters: {...}")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    try:
        study.optimize(
            objective, n_trials=n_trials, timeout=timeout,
            callbacks=[_duration_callback],
        )
    finally:
        # Restore quiet mode for main study
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    elapsed = time.time() - start_time

    # Extract completed trials
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    n_completed = len(completed)

    if n_completed == 0:
        logger.warning(
            f"[{model_name}] Substudy: 0 completed trials in {elapsed:.0f}s — "
            f"no configs to enqueue"
        )
        return [], None

    # Sort by score (best first)
    maximize = direction == "maximize"
    completed.sort(key=lambda t: t.value, reverse=maximize)

    logger.info(
        f"[{model_name}] Substudy done: {n_completed} completed, "
        f"best={completed[0].value:.6f}, {elapsed:.0f}s"
    )

    # Hyperparameter importance (fANOVA) — useful signal before main study
    try:
        importances = optuna.importance.get_param_importances(study)
        if importances:
            imp_lines = [
                f"  {name}: {val:.3f}" for name, val in importances.items()
            ]
            logger.info(
                f"[{model_name}] Substudy hyperparameter importance:\n"
                + "\n".join(imp_lines)
            )
    except Exception:
        pass  # Not enough completed trials or other issue

    # Top-N: pick the best N trials that are NOT already known to the main study.
    # If existing_main_params is provided (transfer_from scenario), we slide past
    # already-known entries so the main study always gets genuinely new configs.
    # Example: top-3 are all from r2 → take indices 3, 4, 5 instead.
    top_indices: set[int] = set()
    for i in range(n_completed):
        if len(top_indices) >= top_n:
            break
        params_key = str(sorted(
            {k: v for k, v in completed[i].params.items() if k != "scaler"}.items()
        ))
        if existing_main_params and params_key in existing_main_params:
            continue  # already known → slide to next best
        top_indices.add(i)
    actual_top_n = len(top_indices)

    # Rank-weighted importance sampling for remaining slots.
    # Candidates: all non-top-N trials that are also not already known.
    n_remaining = n_enqueue - actual_top_n
    novel_candidates = [
        i for i in range(n_completed)
        if i not in top_indices and (
            not existing_main_params or str(sorted(
                {k: v for k, v in completed[i].params.items() if k != "scaler"}.items()
            )) not in existing_main_params
        )
    ]
    effective_n_enqueue = min(n_remaining, max(len(novel_candidates) // 3, 1))
    if effective_n_enqueue < 1:
        effective_n_enqueue = min(n_remaining, len(novel_candidates))
    effective_n_enqueue = max(effective_n_enqueue, 0)

    sampled_indices: list[int] = []
    if effective_n_enqueue > 0:
        # Build weights over novel (non-top-N, non-already-known) candidates only
        candidate_indices = novel_candidates
        if candidate_indices:
            n_candidates = len(candidate_indices)
            rng = np.random.default_rng(global_seed)
            ranks = np.arange(n_candidates)
            weights = np.exp(-temperature * ranks / n_candidates)
            weights /= weights.sum()
            n_sample = min(effective_n_enqueue, n_candidates)
            chosen = rng.choice(n_candidates, size=n_sample, replace=False, p=weights)
            sampled_indices = [candidate_indices[j] for j in chosen]

    all_indices = sorted(top_indices) + sorted(sampled_indices)

    enqueue_params = []
    for i in all_indices:
        params = dict(completed[i].params)
        params.pop("scaler", None)  # Scaler handled separately
        enqueue_params.append(params)

    n_already_known = sum(
        1 for i in range(n_completed)
        if existing_main_params and str(sorted(
            {k: v for k, v in completed[i].params.items() if k != "scaler"}.items()
        )) in existing_main_params
    ) if existing_main_params else 0
    logger.info(
        f"[{model_name}] Substudy sampled {len(enqueue_params)} trials "
        f"(top_n={actual_top_n} novel-guaranteed + {len(sampled_indices)} rank-weighted"
        + (f", {n_already_known} skipped (already in main study)" if n_already_known else "")
        + f", temperature={temperature}, from {n_completed} completed)"
    )

    # Determine best scaler
    best_scaler: str | None = None
    if lock_scaler and scaler_choices:
        try:
            best_scaler = study.best_trial.params.get("scaler")
        except ValueError:
            pass  # No best trial

    return enqueue_params, best_scaler


def run_optuna_study(
    model_name: str,
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    registry: ModelRegistry,
    pipeline_config: PipelineConfig,
    strategy: dict,
    gpu: bool = False,
    timeout_override: int | None = None,
    test: pd.DataFrame | None = None,
) -> tuple[optuna.Study, PerFoldTracker | None, TrialOOFStore | None]:
    """Run a complete Optuna study for a single model.

    Creates an Optuna study with the model's configured pruner, runs
    QMC warmup trials followed by TPE trials, and returns the study
    with all trial results.

    When the model's ``selection_mode`` is ``per_fold``, a
    :class:`PerFoldTracker` is created and passed into the objective.
    During each trial's fold training the model also predicts on test
    data and the tracker records per-fold results (including from
    pruned trials).

    When ``pipeline_config.optuna.persist_trackers`` is True and
    ``storage_dir`` is set, the tracker/oof_store is loaded from disk
    at startup (if a file exists) and saved after every trial via an
    Optuna callback. File names: ``{storage_dir}/{study_name}__tracker.pkl``
    and ``{storage_dir}/{study_name}__oof_store.pkl``.

    Args:
        model_name: Name of the model (must be registered in registry).
        train: Training DataFrame with features and target.
        feature_cols: List of feature column names to use.
        target_col: Name of the target column.
        registry: ModelRegistry instance with loaded configs.
        pipeline_config: Pipeline configuration for CV settings, etc.
        strategy: Strategy dict (may contain per-model overrides).
        gpu: Whether to use GPU for this model.
        timeout_override: Optional timeout in seconds.
        test: Test DataFrame (required for ``selection_mode: per_fold``).

    Returns:
        3-tuple of (study, tracker, trial_oof_store).  ``tracker`` is None
        when the model uses global or fold_coverage selection mode.
        ``trial_oof_store`` is None when the model uses global or per_fold
        selection mode.

    Steps:
        1. Get the model's Optuna config from registry.get_optuna_config.
        2. Get the search space from registry.get_search_space, applying
           any LLM overrides from strategy['overrides'].get(model_name).
        3. Create the CV splitter (StratifiedKFold or KFold) using
           pipeline_config.cv settings.
        4. If selection_mode == 'per_fold', create a PerFoldTracker.
        5. Create the objective function via _create_objective.
        6. Configure the pruner based on model's optuna.pruner settings:
           - 'median' → optuna.pruners.MedianPruner
           - 'hyperband' → optuna.pruners.HyperbandPruner
           - 'none' → optuna.pruners.NopPruner
        7. Create the study with direction='maximize' (for AUC-ROC)
           or 'minimize' (for RMSE) based on task_type.
        8. Call _run_two_phase_study to execute QMC + TPE trials.
        9. Log the best trial value and parameters.
        10. Return (study, tracker).
    """
    optuna_cfg = registry.get_optuna_config(model_name)

    # CV splitter
    cv_cfg = pipeline_config.cv
    task_type = pipeline_config.task_type

    # Search space with optional LLM overrides
    overrides_all = strategy.get("overrides", {}) or {}
    # Copy so that .pop() calls below don't mutate the caller's strategy dict.
    model_overrides_raw = dict(overrides_all.get(model_name, None) or {})

    # Separate optuna config overrides (n_trials, n_top_trials, n_seeds,
    # assembly, selection_mode, fold_timeout, etc.) from hyperparameter overrides.
    # Deep merge so partial assembly overrides don't drop unspecified keys.
    optuna_overrides = model_overrides_raw.pop("optuna", None)
    if optuna_overrides:
        _deep_merge(optuna_cfg, optuna_overrides)

    # Per-model monotone_constraints override.  An empty dict {} means "disable
    # constraints for this model" — useful when a model (e.g. XGBoost) underperforms
    # with global constraints but siblings (LightGBM) should keep them.
    # None means "use global strategy['monotone_constraints']" (default).
    mc_per_model = model_overrides_raw.pop("monotone_constraints", None)

    # Separate hyperparameters key if present (explicit hyperparameter overrides)
    hp_overrides = model_overrides_raw.pop("hyperparameters", None)
    # Remaining keys are treated as hyperparameter overrides too
    if hp_overrides:
        model_overrides_raw.update(hp_overrides)

    search_space = registry.get_search_space(
        model_name, overrides=model_overrides_raw or None, task_type=task_type,
    )
    if cv_cfg.stratified and task_type != "regression":
        cv = StratifiedKFold(n_splits=cv_cfg.n_folds, shuffle=True, random_state=cv_cfg.seed)
    else:
        cv = KFold(n_splits=cv_cfg.n_folds, shuffle=True, random_state=cv_cfg.seed)

    results_dir = Path(pipeline_config.output.results_dir)

    # Selection mode: global / per_fold / fold_coverage
    selection_mode = optuna_cfg.get("selection_mode", "global")
    tracker: PerFoldTracker | None = None
    trial_oof_store: TrialOOFStore | None = None

    if selection_mode == "fold_coverage":
        if test is None:
            logger.warning(
                f"[{model_name}] selection_mode='fold_coverage' requires test "
                f"DataFrame for test predictions, but test=None was passed — "
                f"test_preds will be zeros."
            )
        n_test_rows = len(test) if test is not None else 0
        maximize = task_type != "regression"
        trial_oof_store = TrialOOFStore(
            n_samples=len(train),
            n_test=n_test_rows,
            n_folds=cv_cfg.n_folds,
            maximize=maximize,
        )
        logger.info(
            f"[{model_name}] Fold-coverage selection enabled: "
            f"{len(train)} samples, {cv_cfg.n_folds} folds"
        )

    if selection_mode == "per_fold":
        if test is None:
            logger.warning(
                f"[{model_name}] selection_mode='per_fold' requires test DataFrame "
                f"for test predictions, but test=None was passed. "
                f"The tracker will be created but never populated — "
                f"assemble() will return an empty list. Pass test= to fix this."
            )
        maximize = task_type != "regression"

        # Tracker diversity mode from optuna config
        tracker_cfg = optuna_cfg.get("tracker", {}) or {}
        diversity_mode = tracker_cfg.get("diversity_mode", "vanilla")
        tier1_size = tracker_cfg.get("tier1_size", 5)
        tier2_corr_threshold = tracker_cfg.get("tier2_corr_threshold", 0.99)

        tracker = PerFoldTracker(
            n_top=optuna_cfg["n_top_trials"],
            n_folds=cv_cfg.n_folds,
            maximize=maximize,
            diversity_mode=diversity_mode,
            tier1_size=tier1_size,
            tier2_corr_threshold=tier2_corr_threshold,
        )
        mode_label = f"{diversity_mode}" if diversity_mode == "vanilla" else (
            f"{diversity_mode} (tier1={tier1_size}, tier2_corr={tier2_corr_threshold})"
        )
        logger.info(
            f"[{model_name}] Per-fold selection enabled: "
            f"tracking top {optuna_cfg['n_top_trials']} per fold, mode={mode_label}"
        )

    fold_timeout = optuna_cfg.get("fold_timeout")

    # Determine scaler choices for this model (None = no scaling)
    feat_reqs = registry.get_feature_requirements(model_name)
    needs_scaling = feat_reqs.get("needs_scaling", False)

    # Strategy can override needs_scaling per model and constrain scaler choices
    preprocessing = strategy.get("preprocessing", {}) or {}
    per_model_preproc = preprocessing.get("per_model", {}) or {}
    model_preproc = per_model_preproc.get(model_name, {}) or {}
    # Strategy can force scaling on/off per model
    if "needs_scaling" in model_preproc:
        needs_scaling = bool(model_preproc["needs_scaling"])

    scaler_choices: list[str] | None = None
    prescaled_scaler: str | None = None  # Set when we pre-scale once upfront
    if needs_scaling:
        # Default: all scalers. Strategy can narrow to a subset.
        scaler_choices = model_preproc.get("scaler_choices") or preprocessing.get("scaler_choices") or list(ALL_SCALER_CHOICES)
        # Add "none" so Optuna can opt out — UNLESS strategy locked to a single scaler
        if len(scaler_choices) > 1 and "none" not in scaler_choices:
            scaler_choices = ["none"] + list(scaler_choices)

        # Optimisation: if locked to a single scaler (no Optuna choice), pre-scale
        # train+test ONCE upfront instead of per-fold × per-trial.
        # With 595k rows × 10 folds × 100+ trials this saves thousands of scaler fits.
        # Slight leakage (fit on full train, not 90%) is negligible at large N.
        # Guard: skip pre-scaling on small datasets (<1000 rows) where val fold is
        # 10-20% of train and scaler statistics leak would be meaningful.
        if len(scaler_choices) == 1 and scaler_choices[0] != "none":
            if len(train) >= 1000:
                prescaled_scaler = scaler_choices[0]
                train, test, was_scaled = _apply_prescaling(
                    train, test, feature_cols, prescaled_scaler, model_name,
                )
                if was_scaled:
                    scaler_choices = None  # Don't pass to objective — already scaled
                else:
                    prescaled_scaler = None
            else:
                logger.info(
                    f"[{model_name}] Pre-scaling skipped (n_train={len(train)} < 1000) "
                    f"— using per-fold scaler fit to avoid leakage"
                )
        else:
            logger.info(f"[{model_name}] Scaler search: {scaler_choices}")

    # Resolve monotone constraints from strategy (feature_name → direction)
    # into a positional list matching feature_cols order.
    # NOTE: moved before substudy block so constraints are available there too.
    # Per-model override (mc_per_model) takes precedence over global constraints.
    # An empty dict {} from the override means "disable constraints for this model".
    monotone_constraints_list: list[int] | None = None
    mc_dict = mc_per_model if mc_per_model is not None else (strategy.get("monotone_constraints", {}) or {})
    if mc_dict and model_name in ("catboost", "xgboost", "lightgbm"):
        monotone_constraints_list = [
            int(mc_dict.get(col, 0)) for col in feature_cols
        ]
        n_constrained = sum(1 for v in monotone_constraints_list if v != 0)
        if n_constrained > 0:
            if model_name == "catboost" and gpu:
                logger.warning(
                    f"[{model_name}] Monotone constraints ({n_constrained} features) "
                    "skipped — CatBoost GPU does not support monotone_constraints"
                )
                monotone_constraints_list = None
            else:
                logger.info(
                    f"[{model_name}] Monotone constraints: {n_constrained}/{len(feature_cols)} features constrained"
                )
        else:
            monotone_constraints_list = None

    # --- Substudy: warm-start from small-data QMC exploration ---
    substudy_cfg = optuna_cfg.get("substudy")
    substudy_ran = False
    if substudy_cfg and substudy_cfg.get("enabled", False):
        # Substudy reset: delete existing DB to start fresh (e.g. when AI
        # needs new substudy with different ranges while keeping main study
        # intact for multivariate TPE compatibility).
        if substudy_cfg.get("reset", False):
            _stor = pipeline_config.optuna.storage_dir
            if _stor:
                _sub_db = Path(_stor) / f"{pipeline_config.run_name}__{model_name}__sub.db"
                if _sub_db.exists():
                    _sub_db.unlink()
                    logger.info(f"[{model_name}] Substudy reset: deleted {_sub_db}")

        # Build existing_main_params so substudy can skip already-known configs
        # in its top-N guarantee (slides past them to find genuinely new ones).
        # Note: study_name is defined later; construct it here directly.
        existing_main_params: set[str] | None = None
        _main_storage = pipeline_config.optuna.storage_dir
        _main_study_name = f"{pipeline_config.run_name}__{model_name}"
        if _main_storage:
            _main_db = Path(_main_storage) / f"{_main_study_name}.db"
            if _main_db.exists():
                try:
                    _tmp = optuna.load_study(
                        study_name=_main_study_name,
                        storage=f"sqlite:///{_main_db}",
                    )
                    existing_main_params = {
                        str(sorted(
                            {k: v for k, v in t.params.items() if k != "scaler"}.items()
                        ))
                        for t in _tmp.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    }
                    del _tmp
                except Exception:
                    pass  # DB exists but study not yet created — will be empty

        substudy_enqueue, substudy_scaler = _run_substudy(
            model_name=model_name,
            train=train,
            feature_cols=feature_cols,
            target_col=target_col,
            registry=registry,
            task_type=task_type,
            gpu=gpu,
            results_dir=results_dir,
            substudy_cfg=substudy_cfg,
            search_space=search_space,
            scaler_choices=scaler_choices,
            global_seed=pipeline_config.optuna.global_seed,
            monotone_constraints=monotone_constraints_list,
            run_name=pipeline_config.run_name,
            storage_dir=pipeline_config.optuna.storage_dir,
            existing_main_params=existing_main_params,
        )

        # Scaler lock: replace scaler_choices with single winner
        if substudy_scaler and scaler_choices:
            logger.info(
                f"[{model_name}] Substudy scaler lock: '{substudy_scaler}' "
                f"(was: {scaler_choices})"
            )
            scaler_choices = [substudy_scaler]
            train, test, was_scaled = _apply_prescaling(
                train, test, feature_cols, substudy_scaler, model_name,
            )
            if was_scaled:
                scaler_choices = None
                prescaled_scaler = substudy_scaler

        # Prepend substudy configs to enqueue_trials
        if substudy_enqueue:
            existing = optuna_cfg.get("enqueue_trials", []) or []
            optuna_cfg["enqueue_trials"] = substudy_enqueue + list(existing)
            logger.info(
                f"[{model_name}] Substudy enqueued {len(substudy_enqueue)} configs"
            )
            substudy_ran = True

    # Diversity pruning config (only meaningful with per-fold tracker)
    dp_cfg = optuna_cfg.get("diversity_pruning") if selection_mode == "per_fold" else None

    # Configure pruner
    pruner_cfg = optuna_cfg.get("pruner", {}) or {}
    pruner = _build_pruner(pruner_cfg)

    # Study direction + persistent storage (optional SQLite per model)
    direction = "minimize" if task_type == "regression" else "maximize"
    run_name = pipeline_config.run_name
    storage_dir = pipeline_config.optuna.storage_dir
    study_name = f"{run_name}__{model_name}"
    storage: str | None = None
    if storage_dir:
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        db_path = Path(storage_dir) / f"{study_name}.db"
        storage = f"sqlite:///{db_path}"
        logger.info(f"[{model_name}] Optuna storage: {db_path}")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
        load_if_exists=True,
    )

    # Tracker/OOF-store persistence: load from disk if available
    persist = pipeline_config.optuna.persist_trackers and bool(storage_dir)
    reset_trackers = pipeline_config.optuna.reset_trackers
    tracker_pkl: Path | None = None
    oof_store_pkl: Path | None = None
    if persist:
        tracker_pkl = Path(storage_dir) / f"{study_name}__tracker.pkl"
        oof_store_pkl = Path(storage_dir) / f"{study_name}__oof_store.pkl"
        if reset_trackers:
            logger.info(
                f"[{model_name}] reset_trackers=true — skipping OOF pkl load, starting fresh"
            )
        if tracker is not None and tracker_pkl.exists() and not reset_trackers:
            try:
                tracker = PerFoldTracker.load(tracker_pkl)
                # Shape check: invalidate if n_folds changed (e.g. config edited between runs)
                if tracker.n_folds != cv_cfg.n_folds:
                    logger.warning(
                        f"[{model_name}] Loaded PerFoldTracker n_folds mismatch "
                        f"(stored={tracker.n_folds}, current={cv_cfg.n_folds}) "
                        f"— discarding stale cache, starting fresh"
                    )
                    tracker = PerFoldTracker(
                        n_top=optuna_cfg["n_top_trials"],
                        n_folds=cv_cfg.n_folds,
                        maximize=maximize,
                        diversity_mode=diversity_mode,
                        tier1_size=tier1_size,
                        tier2_corr_threshold=tier2_corr_threshold,
                    )
                else:
                    n_entries = sum(len(v) for v in tracker.fold_data.values())
                    logger.info(
                        f"[{model_name}] Loaded PerFoldTracker from {tracker_pkl} "
                        f"({n_entries} fold entries)"
                    )
            except Exception as exc:
                logger.warning(
                    f"[{model_name}] Failed to load tracker from {tracker_pkl}: {exc}"
                )
        if trial_oof_store is not None and oof_store_pkl.exists() and not reset_trackers:
            try:
                trial_oof_store = TrialOOFStore.load(oof_store_pkl)
                logger.info(
                    f"[{model_name}] Loaded TrialOOFStore from {oof_store_pkl} "
                    f"({len(trial_oof_store._oof)} committed trials)"
                )
                # Shape check: invalidate if training set size changed (e.g. extra_data added)
                if trial_oof_store.n_samples != len(train):
                    logger.warning(
                        f"[{model_name}] TrialOOFStore n_samples mismatch "
                        f"(stored={trial_oof_store.n_samples}, current={len(train)}) "
                        f"— discarding stale cache, starting fresh"
                    )
                    trial_oof_store = TrialOOFStore(
                        n_samples=len(train),
                        n_test=n_test_rows,
                        n_folds=cv_cfg.n_folds,
                        maximize=maximize,
                    )
            except Exception as exc:
                logger.warning(
                    f"[{model_name}] Failed to load oof_store from {oof_store_pkl}: {exc}"
                )

    # Objective must be created AFTER tracker loading so it holds the
    # up-to-date tracker/trial_oof_store (possibly loaded from disk above).
    objective = _create_objective(
        model_name=model_name,
        train=train,
        feature_cols=feature_cols,
        target_col=target_col,
        cv=cv,
        search_space=search_space,
        registry=registry,
        task_type=task_type,
        gpu=gpu,
        results_dir=results_dir,
        test=test,
        per_fold_tracker=tracker,
        trial_oof_store=trial_oof_store,
        fold_timeout=fold_timeout,
        n_top_trials=optuna_cfg["n_top_trials"],
        scaler_choices=scaler_choices,
        prescaled_scaler=prescaled_scaler,
        monotone_constraints=monotone_constraints_list,
        diversity_pruning=dp_cfg,
    )

    # --- Targeted enqueue: AI-guided HP exploration without changing search space ---
    # Generates trials that explore specific HP values paired with top trial configs.
    # Preserves multivariate TPE compatibility on persistent DBs.
    targeted_cfgs = optuna_cfg.get("targeted_enqueue", []) or []
    if targeted_cfgs:
        n_complete = sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        if n_complete > 0:
            for tc in targeted_cfgs:
                generated = generate_targeted_enqueue(
                    study=study,
                    target_param=tc["param"],
                    values=tc.get("values"),
                    target_range=tuple(tc["range"]) if "range" in tc else None,
                    n_points=tc.get("n_points", 5),
                    log=tc.get("log", False),
                    n_base=tc.get("n_base", 3),
                    temperature=tc.get("temperature", 0.3),
                    seed=pipeline_config.optuna.global_seed,
                )
                existing = optuna_cfg.get("enqueue_trials", []) or []
                optuna_cfg["enqueue_trials"] = list(existing) + generated
                logger.info(
                    f"[{model_name}] Targeted enqueue '{tc['param']}': "
                    f"{len(generated)} trials"
                )
        else:
            logger.info(
                f"[{model_name}] Targeted enqueue skipped: "
                f"no completed trials in study"
            )

    # Enqueue pre-specified trials (e.g., substudy warm-start, LLM-suggested configs).
    # Skip any params that exactly match an already-completed trial — avoids re-running
    # configs transferred from a previous round via transfer_from or substudy overlap.
    enqueue_trials = optuna_cfg.get("enqueue_trials", []) or []
    if enqueue_trials:
        existing_params_set: set[str] = {
            str(sorted(t.params.items()))
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        }
        n_skipped = 0
        for trial_params in enqueue_trials:
            key = str(sorted(dict(trial_params).items()))
            if key in existing_params_set:
                n_skipped += 1
            else:
                study.enqueue_trial(dict(trial_params))
        n_queued = len(enqueue_trials) - n_skipped
        if n_queued > 0:
            logger.info(
                f"[{model_name}] Enqueued {n_queued} trial(s) "
                f"(run before QMC/TPE)"
                + (f", skipped {n_skipped} duplicate(s)" if n_skipped else "")
            )
        elif n_skipped:
            logger.info(
                f"[{model_name}] All {n_skipped} enqueue_trials already in study — skipped"
            )

    # Pipeline-level timeout override takes precedence over model YAML timeout
    effective_timeout = timeout_override if timeout_override is not None else optuna_cfg.get("timeout")
    if effective_timeout:
        logger.info(f"[{model_name}] Timeout: {effective_timeout}s ({effective_timeout / 3600:.1f}h)")

    # When substudy provided warm-start, skip QMC → pure TPE
    effective_qmc = 0 if substudy_ran else optuna_cfg["qmc_warmup_trials"]
    tpe_cfg = optuna_cfg.get("tpe", {}) or {}

    # Build persist callback if tracker persistence is enabled
    persist_callbacks: list = []
    if persist:
        _t = tracker
        _os = trial_oof_store
        _tp = tracker_pkl
        _op = oof_store_pkl

        def _persist_callback(
            _study: optuna.Study, _trial: optuna.trial.FrozenTrial
        ) -> None:
            if _t is not None and _tp is not None:
                _t.save(_tp)
            if _os is not None and _op is not None:
                _os.save(_op)

        persist_callbacks = [_persist_callback]

    # Build collapse restart callback if configured (opt-in).
    collapse_cfg = optuna_cfg.get("collapse_restart")
    if collapse_cfg:
        sub_db_path: str | None = None
        sub_study_name: str | None = None
        _stor = pipeline_config.optuna.storage_dir
        if _stor and substudy_cfg and substudy_cfg.get("enabled", False):
            _sub_db = Path(_stor) / f"{pipeline_config.run_name}__{model_name}__sub.db"
            if _sub_db.exists():
                sub_db_path = str(_sub_db)
                sub_study_name = f"{pipeline_config.run_name}__{model_name}__sub"

        collapse_cb = _CollapseRestartCallback(
            search_space=search_space,
            window=collapse_cfg.get("window", 20),
            threshold=collapse_cfg.get("threshold", 0.05),
            n_restart=collapse_cfg.get("n_restart", 5),
            cooldown=collapse_cfg.get("cooldown", 10),
            top_n=collapse_cfg.get("top_n", 2),
            temperature=collapse_cfg.get("temperature", 0.3),
            maximize=(task_type != "regression"),
            substudy_db_path=sub_db_path,
            substudy_study_name=sub_study_name,
            global_seed=pipeline_config.optuna.global_seed,
        )
        persist_callbacks.append(collapse_cb)

    _run_two_phase_study(
        study=study,
        objective=objective,
        n_trials=optuna_cfg.get("n_trials"),  # None = unlimited, timeout controls duration
        qmc_warmup_trials=effective_qmc,
        timeout=effective_timeout,
        global_seed=pipeline_config.optuna.global_seed,
        tpe_cfg=tpe_cfg,
        extra_callbacks=persist_callbacks or None,
    )

    try:
        best = study.best_trial
        display_params = _reassemble_int_lists(dict(best.params))
        logger.info(
            f"[{model_name}] Best trial #{best.number}: "
            f"value={best.value:.6f}, params={display_params}"
        )
    except ValueError:
        logger.warning(
            f"[{model_name}] No completed trials — all were pruned or failed."
        )

    # Pruning summary
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_failed = len(study.trials) - n_complete - n_pruned
    # Count diversity-pruned trials (tagged by the objective)
    n_div_pruned = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
        and hasattr(t, "user_attrs")
        and t.user_attrs.get("diversity_pruned", False)
    )
    summary = (
        f"[{model_name}] Study complete: {n_complete} completed, "
        f"{n_pruned} pruned"
    )
    if dp_cfg:
        summary += f" ({n_div_pruned} diversity-pruned)"
    if n_failed > 0:
        summary += f", {n_failed} failed"
    logger.info(summary)

    # Attach in-memory OOF store to the study for downstream access
    # (e.g., global-mode retrain can skip re-fitting if OOF is cached).
    study._oof_store = objective.oof_store  # type: ignore[attr-defined]

    return study, tracker, trial_oof_store


class TrialOOFStore:
    """Store complete OOF arrays per trial for fold-coverage selection.

    Used with ``selection_mode='fold_coverage'``.  Every completed trial's
    full OOF vector (all folds, same config, no stitching) is stored during
    Optuna.  After the study, :meth:`select` applies fold-coverage selection:

    - **Round 1**: For each fold 0 → n_folds-1, pick the best-scoring trial
      for that fold not yet selected (round-robin, deduplicated).  Continues
      until ``n_fold_best`` unique trials are collected or all folds are
      exhausted.
    - **Round 2**: Add the top ``n_mean_best`` trials by mean CV score,
      skipping already selected ones.

    Because each composite is one trial's complete OOF there is no scale
    mismatch across folds — rank normalisation is not needed.

    Memory: ~2.4 MB per trial on 595k rows (float32).  Only committed
    (all-folds-complete, non-pruned) trials are stored long-term; partial
    OOF arrays from pruned trials are discarded at commit time (never called).
    """

    def __init__(
        self,
        n_samples: int,
        n_test: int,
        n_folds: int,
        maximize: bool = True,
    ) -> None:
        self.n_samples = n_samples
        self.n_test = n_test
        self.n_folds = n_folds
        self.maximize = maximize

        # In-progress (pruned trials never reach commit, so partial data is
        # implicitly discarded when they fall out of scope)
        self._partial_oof: dict[int, np.ndarray] = {}
        self._partial_test: dict[int, np.ndarray] = {}
        self._partial_fold_scores: dict[int, dict[int, float]] = {}

        # Finalized after commit_trial()
        self._oof: dict[int, np.ndarray] = {}
        self._test: dict[int, np.ndarray] = {}
        self._fold_scores: dict[int, list[float]] = {}
        self._mean_scores: dict[int, float] = {}

        # Best trial (and score) seen per fold across ALL updates
        self._fold_best: dict[int, tuple[int, float]] = {}  # fold → (trial_num, score)

    def update(
        self,
        trial_num: int,
        fold_idx: int,
        fold_score: float,
        val_preds: np.ndarray,
        val_idx: np.ndarray,
        test_preds: np.ndarray,
    ) -> None:
        """Accumulate one fold's predictions. Called once per fold per trial."""
        if trial_num not in self._partial_oof:
            self._partial_oof[trial_num] = np.zeros(self.n_samples)
            self._partial_test[trial_num] = np.zeros(self.n_test)
            self._partial_fold_scores[trial_num] = {}

        self._partial_oof[trial_num][val_idx] = val_preds
        self._partial_test[trial_num] += test_preds / self.n_folds
        self._partial_fold_scores[trial_num][fold_idx] = fold_score

    def commit_trial(self, trial_num: int) -> None:
        """Finalise a completed (non-pruned) trial. Must be called once after
        all folds finish, just before the objective returns its score."""
        if trial_num not in self._partial_oof:
            return
        scores_dict = self._partial_fold_scores.pop(trial_num, {})
        fold_scores = [scores_dict.get(f, 0.0) for f in range(self.n_folds)]
        self._oof[trial_num] = self._partial_oof.pop(trial_num)
        self._test[trial_num] = self._partial_test.pop(trial_num)
        self._fold_scores[trial_num] = fold_scores
        self._mean_scores[trial_num] = float(np.mean(fold_scores))
        # Update fold_best only for committed (non-pruned) trials so that
        # _fold_best never points to a pruned trial's number.
        for fold_idx, fold_score in scores_dict.items():
            better = (
                fold_idx not in self._fold_best
                or (self.maximize and fold_score > self._fold_best[fold_idx][1])
                or (not self.maximize and fold_score < self._fold_best[fold_idx][1])
            )
            if better:
                self._fold_best[fold_idx] = (trial_num, fold_score)

    def select(
        self,
        n_fold_best: int = 10,
        n_mean_best: int = 5,
        min_quality_percentile: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Fold-coverage selection → composites in PerFoldTracker.assemble() format.

        Returns list of dicts with keys: ``oof_preds``, ``test_preds``,
        ``fold_trials``, ``fold_scores``, ``avg_score``.

        Args:
            n_fold_best: Max trials to select via round-robin fold-best (Round 1).
            n_mean_best: Additional trials to add by mean CV score (Round 2).
            min_quality_percentile: Quality gate — a trial is eligible only if its
                mean CV score is at or above this percentile across all committed
                trials (0.0 = no gate, 50.0 = must be above median, 25.0 = top 75%).
                Prevents catastrophically weak fold-champions from entering the pool
                solely due to fold specialization. Applies to both rounds.
        """
        if not self._oof:
            return []

        # Quality gate: compute threshold from all committed trials' mean scores
        eligible: set[int] = set(self._oof.keys())
        if min_quality_percentile > 0.0 and len(self._mean_scores) > 1:
            all_scores = list(self._mean_scores.values())
            threshold = float(np.percentile(all_scores, min_quality_percentile))
            if self.maximize:
                eligible = {t for t in eligible if self._mean_scores[t] >= threshold}
            else:
                eligible = {t for t in eligible if self._mean_scores[t] <= threshold}
            n_filtered = len(self._oof) - len(eligible)
            if n_filtered > 0:
                import logging as _log
                _log.getLogger(__name__).info(
                    f"[fold_coverage] Quality gate (p{min_quality_percentile:.0f}): "
                    f"filtered {n_filtered}/{len(self._oof)} trials below threshold "
                    f"{threshold:.6f}"
                )

        selected: list[int] = []
        selected_set: set[int] = set()

        # Round 1: per-fold best — round-robin, deduplicated, quality-gated
        for fold_idx in range(self.n_folds):
            if len(selected) >= n_fold_best:
                break
            # Find best COMMITTED eligible trial for this fold
            fold_candidates = [
                (t, self._fold_scores[t][fold_idx])
                for t in eligible
                if fold_idx < len(self._fold_scores.get(t, []))
            ]
            if not fold_candidates:
                continue
            fold_candidates.sort(key=lambda x: x[1], reverse=self.maximize)
            for t, _ in fold_candidates:
                if t not in selected_set:
                    selected.append(t)
                    selected_set.add(t)
                    break

        # Round 2: top by mean CV score, skip already selected, quality-gated
        remaining = sorted(
            [(t, s) for t, s in self._mean_scores.items()
             if t not in selected_set and t in eligible],
            key=lambda x: x[1],
            reverse=self.maximize,
        )
        for t, _ in remaining[:n_mean_best]:
            selected.append(t)
            selected_set.add(t)

        results: list[dict[str, Any]] = []
        for t in selected:
            results.append({
                "oof_preds": self._oof[t].copy(),
                "test_preds": self._test[t].copy(),
                "fold_trials": [t] * self.n_folds,
                "fold_scores": self._fold_scores[t],
                "avg_score": self._mean_scores[t],
            })
        return results

    def log_summary(self, model_name: str) -> None:
        """Log fold-best trial per fold and overall statistics."""
        n = len(self._oof)
        if n == 0:
            logger.info(f"[{model_name}] TrialOOFStore: no completed trials")
            return
        scores = sorted(self._mean_scores.values(), reverse=self.maximize)
        logger.info(
            f"[{model_name}] Fold-coverage store: {n} completed trials, "
            f"best={scores[0]:.6f}, worst={scores[-1]:.6f}"
        )
        for fold_idx in sorted(self._fold_best.keys()):
            trial_num, score = self._fold_best[fold_idx]
            if trial_num in self._oof:
                logger.info(
                    f"  Fold {fold_idx}: best trial #{trial_num} score={score:.6f}"
                )

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Pickle the finalized store to disk for resume between interrupted runs.

        Only committed (non-pruned) trial data is saved; in-progress
        partial data from the current run is always in RAM anyway.
        """
        import pickle
        data = {
            "n_samples": self.n_samples,
            "n_test": self.n_test,
            "n_folds": self.n_folds,
            "maximize": self.maximize,
            "_oof": self._oof,
            "_test": self._test,
            "_fold_scores": self._fold_scores,
            "_mean_scores": self._mean_scores,
            "_fold_best": self._fold_best,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "TrialOOFStore":
        """Load a previously saved store from disk.

        Warning:
            Uses ``pickle.load`` — only load files written by this process
            in a trusted ``storage_dir``.  Do not load store files from
            untrusted or externally-supplied sources.
        """
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        # Restore finalized fields
        for k, v in data.items():
            setattr(obj, k, v)
        # Initialize empty partial dicts (in-progress state is not persisted)
        obj._partial_oof = {}
        obj._partial_test = {}
        obj._partial_fold_scores = {}
        return obj


def _create_objective(
    model_name: str,
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cv: StratifiedKFold | KFold,
    search_space: dict[str, dict[str, Any]],
    registry: ModelRegistry,
    task_type: str,
    gpu: bool,
    results_dir: Path,
    test: pd.DataFrame | None = None,
    per_fold_tracker: PerFoldTracker | None = None,
    trial_oof_store: TrialOOFStore | None = None,
    fold_timeout: int | None = None,
    n_top_trials: int = 5,
    scaler_choices: list[str] | None = None,
    prescaled_scaler: str | None = None,
    monotone_constraints: list[int] | None = None,
    diversity_pruning: dict[str, Any] | None = None,
) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective function for a model.

    The returned callable samples hyperparameters from the search space,
    runs cross-validation, optionally prunes early, and stores OOF
    predictions as a trial user attribute.

    When ``per_fold_tracker`` is provided (per-fold selection mode), each
    fold also predicts on the test set immediately after training, and the
    tracker records per-fold (score, oof_slice, test_preds).  This happens
    **before** the prune check, so pruned trials still contribute their
    completed folds to the leaderboard.

    When ``fold_timeout`` is set, each fold's training time is measured.
    If a single fold exceeds the timeout, the trial is pruned immediately.

    When ``diversity_pruning`` is provided, after each fold the trial's
    predictions are compared (Pearson |correlation|) against existing
    tracker entries for that fold.  If the trial is redundant (high corr)
    across ``n_consecutive`` consecutive folds AND its score is not the
    new best, the trial is pruned to save compute.  This is especially
    useful for low-signal data where neural nets converge to nearly
    identical predictions.

    Args:
        model_name: Name of the model.
        train: Training DataFrame.
        feature_cols: Feature column names.
        target_col: Target column name.
        cv: Cross-validation splitter.
        search_space: Optuna search space definition.
        registry: ModelRegistry for creating model instances.
        task_type: 'binary_classification', 'multiclass', or 'regression'.
        gpu: Whether to use GPU.
        results_dir: Directory for saving model artifacts.
        test: Test DataFrame (required when per_fold_tracker is set).
        per_fold_tracker: PerFoldTracker for per-fold selection mode.
        fold_timeout: Max seconds per fold; exceeding triggers TrialPruned.
        diversity_pruning: Optional dict with keys:
            - ``corr_threshold`` (float): Max |correlation| to consider
              redundant (default 0.995).
            - ``warmup_entries`` (int): Min entries per fold before
              diversity pruning activates (default 5).
            - ``n_consecutive`` (int): Consecutive redundant folds
              required to prune (default 2).
            - ``score_tolerance`` (float): Fraction of best score below
              which diversity pruning applies (default 0.001).

    Returns:
        Callable that takes an optuna.Trial and returns the CV score.

    Steps:
        1. Define the inner objective(trial) function:
           a. For each param in search_space, call the appropriate
              trial.suggest_* method based on param['type']:
              - 'int' → trial.suggest_int(name, low, high, log=log)
              - 'float' → trial.suggest_float(name, low, high, log=log)
              - 'categorical' → trial.suggest_categorical(name, choices)
           b. Create the model via registry.get_model with sampled hparams.
           c. Initialize OOF prediction array (len(train),).
           d. For each fold_idx, (train_idx, val_idx) in enumerate(cv.split):
              - Split features and target.
              - If model needs eval_set (early stopping): fit with
                eval_set=[(X_val, y_val)] and early_stopping_rounds.
              - Else: fit normally.
              - Predict on validation set. For classification, use
                predict_proba[:, 1].
              - Store predictions in oof[val_idx].
              - Compute fold metric. Report to trial for pruning:
                trial.report(fold_metric, fold_idx).
              - If per_fold_tracker: predict on test, update tracker.
              - Check if trial should be pruned:
                if trial.should_prune(): raise TrialPruned.
           e. Compute overall CV metric from OOF predictions.
           f. Store OOF predictions in an in-memory dict (``_oof_store``)
              keyed by trial number, only if this trial ranks in the top
              n_top_trials (bounds memory). Attached to the returned
              objective as ``objective.oof_store`` and to the study as
              ``study._oof_store`` after optimization.
           g. Return the overall CV metric.
        2. Return the objective function.
    """
    X = train[feature_cols]
    y = train[target_col].values
    X_test = test[feature_cols] if test is not None else None

    # Impute NaN for models that don't handle missing values (e.g. RealMLP, TabM).
    # Fitted per fold (on train fold only) to avoid leakage into the validation set.
    feat_reqs = registry.get_feature_requirements(model_name)
    has_nan_in_train = X.isna().any().any()
    has_nan_in_test = X_test is not None and X_test.isna().any().any()
    needs_imputation = not feat_reqs.get("handles_missing", False) and (has_nan_in_train or has_nan_in_test)
    if needs_imputation:
        from sklearn.impute import SimpleImputer
        nan_cols = [
            c for c in X.columns
            if X[c].isna().any() or (X_test is not None and X_test[c].isna().any())
        ]
        logger.info(f"Will impute NaN values per-fold (median) for {model_name}")

    # Identify columns to scale (computed once, reused by all trials)
    scale_cols = _identify_scale_cols(X) if scaler_choices else []

    training_cfg = registry.get_training_config(model_name)
    sample_weights = _extract_sample_weights(train, training_cfg)

    # Compute CV splits once — deterministic (fixed seed), same every trial.
    try:
        cv_splits = list(cv.split(X, y))
    except Exception:
        cv_splits = list(cv.split(X))

    # Suppress pytabkit PWL embedding warnings for binary features (once per study)
    warnings.filterwarnings(
        "ignore",
        message=".*has just two bin edges.*",
        module=r"pytabkit\.models\.nn_models\.rtdl_num_embeddings",
    )

    # Diversity pruning config (only active with per_fold_tracker)
    dp_enabled = diversity_pruning is not None and per_fold_tracker is not None
    dp_corr_threshold = diversity_pruning.get("corr_threshold", 0.995) if dp_enabled else 0.995
    dp_warmup = diversity_pruning.get("warmup_entries", 5) if dp_enabled else 5
    dp_n_consecutive = diversity_pruning.get("n_consecutive", 2) if dp_enabled else 2
    dp_score_tolerance = diversity_pruning.get("score_tolerance", 0.001) if dp_enabled else 0.001

    # In-memory OOF storage — avoids Optuna's JSON serialization (ndarray
    # is not JSON-serializable with SQLite backend).  Top-N bounding is
    # applied inside the objective so memory stays bounded.
    _oof_store: dict[int, np.ndarray] = {}
    # Parallel score cache so top-N admission/eviction runs in O(n_top_trials)
    # instead of O(n_all_trials) by avoiding iteration of study.trials.
    _score_cache: dict[int, float] = {}

    def objective(trial: optuna.Trial) -> float:
        # Scaler selection (Optuna parameter when model needs scaling)
        if scaler_choices and scale_cols:
            scaler_type = trial.suggest_categorical("scaler", scaler_choices)
        elif prescaled_scaler:
            # Data was pre-scaled upfront — record the scaler type for
            # train_with_config() but don't apply per-fold.
            scaler_type = "none"
            trial.set_user_attr("prescaled_scaler", prescaled_scaler)
        else:
            scaler_type = "none"

        # Sample hyperparameters
        hparams: dict[str, Any] = {}
        for param_name, spec in search_space.items():
            param_type = spec.get("type", "float")
            use_log = spec.get("log", False)
            if param_type == "fixed":
                hparams[param_name] = spec["value"]
            elif param_type == "int":
                int_step = spec.get("step")
                if int_step is not None and not use_log:
                    hparams[param_name] = trial.suggest_int(
                        param_name, int(spec["low"]), int(spec["high"]), step=int(int_step)
                    )
                else:
                    hparams[param_name] = trial.suggest_int(
                        param_name, int(spec["low"]), int(spec["high"]), log=use_log
                    )
            elif param_type == "float":
                float_step = spec.get("step")
                if float_step is not None and not use_log:
                    hparams[param_name] = trial.suggest_float(
                        param_name, float(spec["low"]), float(spec["high"]), step=float(float_step)
                    )
                else:
                    hparams[param_name] = trial.suggest_float(
                        param_name, float(spec["low"]), float(spec["high"]), log=use_log
                    )
            elif param_type == "categorical":
                raw_choices = spec["choices"]
                # Optuna categorical only supports None/bool/int/float/str.
                # For list/tuple choices (e.g. RealMLP hidden_sizes),
                # convert to JSON strings for Optuna, then decode back.
                has_complex = any(isinstance(c, (list, tuple)) for c in raw_choices)
                if has_complex:
                    import json
                    str_choices = [json.dumps(c) for c in raw_choices]
                    picked = trial.suggest_categorical(param_name, str_choices)
                    hparams[param_name] = json.loads(picked)
                else:
                    hparams[param_name] = trial.suggest_categorical(
                        param_name, raw_choices
                    )
            elif param_type == "int_list":
                # Suggest N independent int params, combine into a list.
                # YAML: hidden_sizes: {type: int_list, n: 2, low: 8, high: 128}
                n_elements = int(spec["n"])
                low = int(spec["low"])
                high = int(spec["high"])
                hparams[param_name] = [
                    trial.suggest_int(
                        f"{param_name}_{i}", low, high, log=use_log
                    )
                    for i in range(n_elements)
                ]
            elif param_type == "dynamic_int_list":
                # Suggest variable-length list: first choose length, then values.
                # YAML: hidden_sizes: {type: dynamic_int_list, n_min: 1, n_max: 3, low: 4, high: 256}
                n_min = int(spec["n_min"])
                n_max = int(spec["n_max"])
                low = int(spec["low"])
                high = int(spec["high"])
                n_layers = trial.suggest_int(f"{param_name}_n", n_min, n_max)
                hparams[param_name] = [
                    trial.suggest_int(
                        f"{param_name}_{i}", low, high, log=use_log
                    )
                    for i in range(n_layers)
                ]

        # Add eval_metric to constructor params where needed
        eval_metric_val = _get_eval_metric_value(training_cfg, task_type, gpu)
        eval_metric_param = training_cfg.get("eval_metric_param")
        if eval_metric_val and eval_metric_param:
            hparams[eval_metric_param] = eval_metric_val

        needs_eval_set = training_cfg.get("needs_eval_set", False)
        early_stopping_rounds = training_cfg.get("early_stopping_rounds")
        is_lgbm = training_cfg.get("uses_callbacks_for_early_stopping", False)
        es_in_constructor = training_cfg.get("early_stopping_in_constructor", False)

        if task_type == "multiclass":
            n_classes = len(np.unique(y))
            oof = np.zeros((len(X), n_classes))
        else:
            oof = np.zeros(len(X))

        # Per-fold test reference for scaler transform (avoid mutating shared X_test)
        X_test_base = X_test

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            fold_weights = sample_weights[train_idx] if sample_weights is not None else None

            # Impute NaN per fold (fit on train fold only — no leakage from val)
            X_test_fold = X_test_base
            if needs_imputation:
                imputer = SimpleImputer(strategy="median")
                X_train = X_train.copy()
                X_val = X_val.copy()
                X_train[nan_cols] = imputer.fit_transform(X_train[nan_cols])
                X_val[nan_cols] = imputer.transform(X_val[nan_cols])
                if X_test_fold is not None:
                    X_test_fold = X_test_fold.copy()
                    X_test_fold[nan_cols] = imputer.transform(X_test_fold[nan_cols])

            # Apply scaler per fold (fit on train fold, transform val + test)
            if scaler_type != "none" and scale_cols:
                X_train, X_val, X_test_fold = _apply_scaler_fold(
                    scaler_type, X_train, X_val, X_test_fold, scale_cols
                )

            # Early stopping via constructor (XGBoost v2.0+)
            model_hparams = dict(hparams)
            if es_in_constructor and early_stopping_rounds:
                model_hparams["early_stopping_rounds"] = early_stopping_rounds

            # Monotone constraints (CatBoost/XGBoost/LightGBM)
            if monotone_constraints is not None:
                if model_name == "catboost" and gpu:
                    pass  # CatBoost GPU doesn't support monotone_constraints
                elif model_name == "catboost":
                    model_hparams["monotone_constraints"] = list(monotone_constraints)
                else:
                    model_hparams["monotone_constraints"] = tuple(monotone_constraints)

            # Fresh model per fold — each fold gets an independent RandomState,
            # matching the behaviour of train_with_config for reproducibility.
            model = registry.get_model(
                model_name, hparams=model_hparams, task_type=task_type,
                gpu=gpu, results_dir=results_dir
            )

            fold_start = time.monotonic()

            # Suppress CatBoost C++ GPU memory warnings on stderr;
            # redirect FTT/skorch stdout epoch tables to log file (not console)
            _stderr_ctx = _suppress_catboost_gpu_warnings() if (model_name == "catboost" and gpu) else contextlib.nullcontext()
            _stdout_ctx = _redirect_stdout_to_log(model_name) if model_name == "ftt" else contextlib.nullcontext()

            with _stderr_ctx, _stdout_ctx:
                if needs_eval_set:
                    fit_params: dict[str, Any] = {
                        "eval_set": [(X_val, y_val)],
                        "verbose": False,  # suppress per-iteration eval output
                    }
                    if fold_weights is not None:
                        fit_params["sample_weight"] = fold_weights
                    if is_lgbm:
                        import lightgbm as lgb
                        del fit_params["verbose"]  # LightGBM uses callbacks instead
                        fit_params["callbacks"] = [
                            lgb.early_stopping(
                                stopping_rounds=early_stopping_rounds or 50,
                                verbose=False,
                            ),
                            lgb.log_evaluation(-1),
                        ]
                    elif early_stopping_rounds and not es_in_constructor:
                        fit_params["early_stopping_rounds"] = early_stopping_rounds
                    model.fit(X_train, y_train, **fit_params)
                else:
                    if fold_weights is not None:
                        model.fit(X_train, y_train, sample_weight=fold_weights)
                    else:
                        model.fit(X_train, y_train)

            fold_elapsed = time.monotonic() - fold_start

            if task_type == "binary_classification":
                fold_preds = model.predict_proba(X_val)[:, 1]
            elif task_type == "multiclass":
                fold_preds = model.predict_proba(X_val)  # full (n_val, n_classes) matrix
            else:
                fold_preds = model.predict(X_val)

            oof[val_idx] = fold_preds

            fold_metric = _compute_cv_metric(y_val, fold_preds, task_type)
            trial.report(fold_metric, fold_idx)

            # Per-fold tracking: predict test and update leaderboard.
            # Runs BEFORE prune check so pruned trials still contribute.
            # Use X_test_fold (scaled with this fold's scaler) for predictions.
            test_fold_preds: np.ndarray | None = None
            if per_fold_tracker is not None and X_test_fold is not None:
                if task_type == "binary_classification":
                    test_fold_preds = model.predict_proba(X_test_fold)[:, 1]
                elif task_type == "multiclass":
                    test_fold_preds = model.predict_proba(X_test_fold)
                else:
                    test_fold_preds = model.predict(X_test_fold)
                per_fold_tracker.update(
                    fold_idx=fold_idx,
                    score=fold_metric,
                    val_preds=fold_preds,
                    val_idx=val_idx,
                    test_preds=test_fold_preds,
                    trial_number=trial.number,
                    params=hparams,
                )

            # Fold-coverage store: accumulate full OOF per trial.
            # Reuses test_fold_preds from per_fold_tracker block if already computed.
            if trial_oof_store is not None and X_test_fold is not None:
                if test_fold_preds is None:
                    if task_type == "binary_classification":
                        test_fold_preds_fc = model.predict_proba(X_test_fold)[:, 1]
                    elif task_type == "multiclass":
                        test_fold_preds_fc = model.predict_proba(X_test_fold)
                    else:
                        test_fold_preds_fc = model.predict(X_test_fold)
                else:
                    test_fold_preds_fc = test_fold_preds
                trial_oof_store.update(
                    trial_num=trial.number,
                    fold_idx=fold_idx,
                    fold_score=fold_metric,
                    val_preds=fold_preds,
                    val_idx=val_idx,
                    test_preds=test_fold_preds_fc,
                )

            # Diversity pruning: check if this trial's fold predictions
            # are redundant (highly correlated) with existing tracker entries.
            # Only active when diversity_pruning config is provided AND
            # per_fold_tracker is set AND enough warmup entries exist.
            if dp_enabled and per_fold_tracker is not None:
                tracker_entries = per_fold_tracker.fold_data[fold_idx]
                if per_fold_tracker.n_entries(fold_idx) >= dp_warmup:
                    # Check if this fold is redundant
                    max_corr = 0.0
                    for existing in tracker_entries:
                        if existing.trial_number == trial.number:
                            continue  # skip self (just inserted)
                        corr = abs(float(np.corrcoef(
                            fold_preds.ravel(),
                            existing.val_preds.ravel(),
                        )[0, 1]))
                        if np.isnan(corr):
                            corr = 1.0
                        max_corr = max(max_corr, corr)

                    # Check score gate: never diversity-prune if best score.
                    # Compare fold_metric against the best fold-level score from the
                    # tracker for this specific fold (apples-to-apples comparison).
                    is_best_so_far = True
                    try:
                        if tracker_entries:
                            best_fold_score = tracker_entries[0].score  # sorted best-first
                            if per_fold_tracker.maximize:
                                is_best_so_far = fold_metric > best_fold_score * (1 + dp_score_tolerance)
                            else:
                                is_best_so_far = fold_metric < best_fold_score * (1 - dp_score_tolerance)
                    except Exception:
                        is_best_so_far = True  # no entries yet or other issue

                    if max_corr >= dp_corr_threshold and not is_best_so_far:
                        count = trial.user_attrs.get("_div_flag_count", 0) + 1
                        trial.set_user_attr("_div_flag_count", count)

                        if count >= dp_n_consecutive:
                            logger.info(
                                f"[{model_name}] Trial #{trial.number} diversity-pruned at fold "
                                f"{fold_idx} (max |corr|={max_corr:.4f} >= {dp_corr_threshold})"
                            )
                            trial.set_user_attr("diversity_pruned", True)
                            raise optuna.exceptions.TrialPruned()
                    else:
                        # Reset counter — must be consecutive
                        trial.set_user_attr("_div_flag_count", 0)

            # Fold timeout: if this fold took too long, prune to skip
            # remaining folds.  Completed fold's predictions are already saved.
            if fold_timeout is not None and fold_elapsed > fold_timeout:
                logger.warning(
                    f"[{model_name}] Trial #{trial.number} fold {fold_idx} "
                    f"exceeded fold_timeout ({fold_elapsed:.0f}s > {fold_timeout}s) — pruning"
                )
                raise optuna.exceptions.TrialPruned()

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        overall_metric = _compute_cv_metric(y, oof, task_type)

        # Store OOF only for top-N trials to bound memory usage.
        # Uses an in-memory dict (not Optuna user_attrs) to avoid JSON
        # serialization issues with SQLite storage backend.
        maximize = task_type != "regression"
        # Admission/eviction using _score_cache — O(n_top_trials) per trial,
        # not O(n_all_trials).  _score_cache mirrors _oof_store: both always
        # contain the same set of trial numbers (the current top-N).
        # Tiebreaker: on equal cutoff score, evict the newest trial (highest
        # number) so earlier (often better) trials are retained.
        if len(_oof_store) < n_top_trials:
            _oof_store[trial.number] = oof.copy()
            _score_cache[trial.number] = overall_metric
        else:
            worst_num = min(
                _score_cache,
                key=lambda k: (_score_cache[k] if maximize else -_score_cache[k], -k),
            )
            cutoff = _score_cache[worst_num]
            if (maximize and overall_metric >= cutoff) or (not maximize and overall_metric <= cutoff):
                _oof_store[trial.number] = oof.copy()
                _score_cache[trial.number] = overall_metric
                del _oof_store[worst_num]
                del _score_cache[worst_num]

        # Fold-coverage: commit completed trial (must come after all folds finish)
        if trial_oof_store is not None:
            trial_oof_store.commit_trial(trial.number)

        return overall_metric

    objective.oof_store = _oof_store  # type: ignore[attr-defined]
    return objective


def _duration_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Log trial duration after each completed/pruned trial."""
    if trial.duration is not None:
        secs = trial.duration.total_seconds()
        if secs >= 60:
            mins = int(secs // 60)
            remaining = secs - mins * 60
            dur_str = f"{mins}m {remaining:.0f}s"
        else:
            dur_str = f"{secs:.1f}s"
        logger.info(f"  Trial {trial.number} duration: {dur_str}")


class _CollapseRestartCallback:
    """Optuna callback that detects TPE parameter collapse and injects restarts.

    After each trial, checks if the last ``window`` completed trials have
    low parameter variance (normalised std ratio below ``threshold``).
    If collapsed **and** the current score is not a new best, enqueues
    ``n_restart`` trials sampled from one of two sources:

    * **Substudy history** (preferred) — loads the substudy SQLite DB and
      samples via rank-weighted exponential distribution (same algorithm as
      substudy enqueue: ``top_n`` guaranteed best + remaining via
      ``exp(-temperature * rank / n)``).  This reuses the substudy's
      cheap exploration as a "map" to escape the local minimum.
    * **Random from search space** (fallback when no substudy exists) —
      samples uniformly (or log-uniformly for log-scale params) from the
      search space bounds.

    All injected trials are deduplicated against the main study's completed
    trial history so the same config is never enqueued twice.

    Attributes:
        search_space: Model search space dict from registry.
        window: Number of recent trials to check for collapse.
        threshold: Param variance ratio below which collapse is flagged.
        n_restart: Number of restart trials to inject per event.
        cooldown: Minimum trials between restart injections.
        top_n: Guaranteed best trials from substudy in each restart batch.
        temperature: Exponential sampling sharpness for substudy history.
        maximize: Whether higher scores are better.
        substudy_db_path: Path to substudy SQLite DB (None if no substudy).
        substudy_study_name: Optuna study name inside the DB.
        global_seed: Base seed for reproducible random sampling.
    """

    def __init__(
        self,
        search_space: dict[str, dict[str, Any]],
        *,
        window: int = 20,
        threshold: float = 0.05,
        n_restart: int = 5,
        cooldown: int = 10,
        top_n: int = 2,
        temperature: float = 0.3,
        maximize: bool = True,
        substudy_db_path: str | None = None,
        substudy_study_name: str | None = None,
        global_seed: int = 42,
    ) -> None:
        self.search_space = search_space
        self.window = window
        self.threshold = threshold
        self.n_restart = n_restart
        self.cooldown = cooldown
        self.top_n = top_n
        self.temperature = temperature
        self.maximize = maximize
        self.substudy_db_path = substudy_db_path
        self.substudy_study_name = substudy_study_name
        self.global_seed = global_seed

        self._cooldown_counter: int = 0
        self._n_injected: int = 0
        self._rng = np.random.default_rng(global_seed)

    # ------------------------------------------------------------------
    # Public callback interface
    # ------------------------------------------------------------------

    def __call__(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        # Cooldown — skip if we recently injected.
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed) < self.window:
            return

        is_collapsed, ratio = self._is_collapsed(completed)
        if not is_collapsed:
            return

        # Score-gate: never restart when the current trial is a new best.
        best_value = study.best_value
        if trial.state == optuna.trial.TrialState.COMPLETE:
            if trial.value == best_value:
                return

        # Build set of existing param keys for deduplication.
        existing_keys: set[str] = {
            str(sorted(t.params.items()))
            for t in completed
        }

        # Choose restart source.  Both methods return pre-deduplicated
        # candidates (they check against existing_keys internally).
        if self.substudy_db_path is not None:
            candidates = self._sample_from_substudy(existing_keys)
            source = "substudy"
        else:
            candidates = self._sample_random_from_space(existing_keys)
            source = "random"

        n_enqueued = 0
        for params in candidates:
            study.enqueue_trial(params)
            n_enqueued += 1

        if n_enqueued > 0:
            self._n_injected += n_enqueued
            # Cooldown does NOT include the restart trials themselves.
            # The restart trials are exploration — we want to re-check
            # for collapse shortly after they complete, not after a long delay.
            self._cooldown_counter = self.cooldown
            logger.warning(
                f"[collapse_restart] Injected {n_enqueued} restarts from "
                f"{source} (param_std_ratio={ratio:.3f}, "
                f"total_injected={self._n_injected})"
            )

    # ------------------------------------------------------------------
    # Collapse detection
    # ------------------------------------------------------------------

    def _is_collapsed(
        self, completed: list[optuna.trial.FrozenTrial]
    ) -> tuple[bool, float]:
        """Check if the last ``window`` trials have collapsed parameter variance.

        Returns:
            (is_collapsed, param_std_ratio) where param_std_ratio is the mean
            normalised std across all numeric HPs.
        """
        sorted_trials = sorted(completed, key=lambda t: t.number)
        last_window = sorted_trials[-self.window:]

        ratios: list[float] = []
        for param_name, spec in self.search_space.items():
            ptype = spec.get("type", "float")
            if ptype not in ("int", "float"):
                continue

            # Collect values from the window and full history.
            window_vals = [
                t.params[param_name]
                for t in last_window
                if param_name in t.params
            ]
            all_vals = [
                t.params[param_name]
                for t in sorted_trials
                if param_name in t.params
            ]
            if len(window_vals) < 2 or len(all_vals) < 2:
                continue

            use_log = spec.get("log", False)
            if use_log:
                # Work in log-space for log-scale params.
                window_vals = [np.log(max(v, 1e-30)) for v in window_vals]
                all_vals = [np.log(max(v, 1e-30)) for v in all_vals]

            full_range = max(all_vals) - min(all_vals)
            if full_range < 1e-10:
                continue

            window_std = float(np.std(window_vals))
            ratios.append(window_std / full_range)

        if not ratios:
            return False, 1.0

        mean_ratio = float(np.mean(ratios))
        return mean_ratio < self.threshold, mean_ratio

    # ------------------------------------------------------------------
    # Restart trial sampling — substudy history
    # ------------------------------------------------------------------

    def _sample_from_substudy(
        self, existing_keys: set[str]
    ) -> list[dict[str, Any]]:
        """Sample restart trials from substudy history (exp-distribution).

        Mirrors the two-tier sampling in ``_run_substudy()``:
        tier-1 = top_n best novel trials (unconditional),
        tier-2 = remaining slots via rank-weighted exponential sampling.
        """
        try:
            sub_study = optuna.load_study(
                study_name=self.substudy_study_name,
                storage=f"sqlite:///{self.substudy_db_path}",
            )
        except Exception:
            logger.debug("[collapse_restart] Could not load substudy DB, falling back to random")
            return self._sample_random_from_space(existing_keys)

        sub_completed = [
            t for t in sub_study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not sub_completed:
            return self._sample_random_from_space(existing_keys)

        # Sort by score (best first).
        sub_completed.sort(key=lambda t: t.value, reverse=self.maximize)

        def _params_key(t: optuna.trial.FrozenTrial) -> str:
            p = {k: v for k, v in t.params.items() if k != "scaler"}
            return str(sorted(p.items()))

        # Tier 1: top_n novel trials.
        top_indices: set[int] = set()
        for i in range(len(sub_completed)):
            if len(top_indices) >= self.top_n:
                break
            if _params_key(sub_completed[i]) not in existing_keys:
                top_indices.add(i)

        # Tier 2: rank-weighted exponential sampling over remaining novel candidates.
        n_remaining = self.n_restart - len(top_indices)
        novel_candidates = [
            i for i in range(len(sub_completed))
            if i not in top_indices and _params_key(sub_completed[i]) not in existing_keys
        ]

        sampled_indices: list[int] = []
        if n_remaining > 0 and novel_candidates:
            n_candidates = len(novel_candidates)
            effective_n = min(n_remaining, max(n_candidates // 3, 1))
            effective_n = max(effective_n, 0)
            if effective_n > 0:
                ranks = np.arange(n_candidates)
                weights = np.exp(-self.temperature * ranks / n_candidates)
                weights /= weights.sum()
                n_sample = min(effective_n, n_candidates)
                chosen = self._rng.choice(
                    n_candidates, size=n_sample, replace=False, p=weights
                )
                sampled_indices = [novel_candidates[j] for j in chosen]

        all_indices = sorted(top_indices) + sorted(sampled_indices)
        result: list[dict[str, Any]] = []
        for i in all_indices:
            params = dict(sub_completed[i].params)
            params.pop("scaler", None)
            result.append(params)

        return result

    # ------------------------------------------------------------------
    # Restart trial sampling — random from search space
    # ------------------------------------------------------------------

    def _sample_random_from_space(
        self, existing_keys: set[str]
    ) -> list[dict[str, Any]]:
        """Sample random trials uniformly from search space bounds.

        Used as fallback when no substudy DB is available.
        """
        import json

        result: list[dict[str, Any]] = []
        max_attempts = self.n_restart * 3  # avoid infinite loop on tiny spaces
        attempts = 0
        while len(result) < self.n_restart and attempts < max_attempts:
            attempts += 1
            params: dict[str, Any] = {}
            for name, spec in self.search_space.items():
                ptype = spec.get("type", "float")
                use_log = spec.get("log", False)

                if ptype == "fixed":
                    continue  # Optuna fills fixed params automatically
                elif ptype == "int":
                    low, high = int(spec["low"]), int(spec["high"])
                    step = spec.get("step")
                    if use_log:
                        val = int(
                            np.exp(self._rng.uniform(np.log(max(low, 1)), np.log(high)))
                        )
                        val = max(low, min(val, high))
                    elif step:
                        step = int(step)
                        choices = list(range(low, high + 1, step))
                        val = int(self._rng.choice(choices))
                    else:
                        val = int(self._rng.integers(low, high + 1))
                    params[name] = val
                elif ptype == "float":
                    low, high = float(spec["low"]), float(spec["high"])
                    step = spec.get("step")
                    if use_log:
                        val = float(
                            np.exp(self._rng.uniform(np.log(max(low, 1e-30)), np.log(high)))
                        )
                    elif step:
                        step = float(step)
                        n_steps = int((high - low) / step)
                        val = low + float(self._rng.integers(0, n_steps + 1)) * step
                    else:
                        val = float(self._rng.uniform(low, high))
                    params[name] = val
                elif ptype == "categorical":
                    raw_choices = spec["choices"]
                    has_complex = any(isinstance(c, (list, tuple)) for c in raw_choices)
                    if has_complex:
                        idx = int(self._rng.integers(0, len(raw_choices)))
                        params[name] = json.dumps(raw_choices[idx])
                    else:
                        idx = int(self._rng.integers(0, len(raw_choices)))
                        params[name] = raw_choices[idx]
                elif ptype == "int_list":
                    n_elements = int(spec["n"])
                    low, high = int(spec["low"]), int(spec["high"])
                    for i in range(n_elements):
                        if use_log:
                            val = int(
                                np.exp(self._rng.uniform(np.log(max(low, 1)), np.log(high)))
                            )
                            val = max(low, min(val, high))
                        else:
                            val = int(self._rng.integers(low, high + 1))
                        params[f"{name}_{i}"] = val
                elif ptype == "dynamic_int_list":
                    n_min = int(spec["n_min"])
                    n_max = int(spec["n_max"])
                    low, high = int(spec["low"]), int(spec["high"])
                    n_layers = int(self._rng.integers(n_min, n_max + 1))
                    params[f"{name}_n"] = n_layers
                    for i in range(n_layers):
                        if use_log:
                            val = int(
                                np.exp(self._rng.uniform(np.log(max(low, 1)), np.log(high)))
                            )
                            val = max(low, min(val, high))
                        else:
                            val = int(self._rng.integers(low, high + 1))
                        params[f"{name}_{i}"] = val

            key = str(sorted(params.items()))
            if key not in existing_keys:
                result.append(params)
                existing_keys.add(key)

        return result


def _run_two_phase_study(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], float],
    n_trials: int | None,
    qmc_warmup_trials: int,
    timeout: int | None = None,
    global_seed: int = 42,
    tpe_cfg: dict[str, Any] | None = None,
    extra_callbacks: list | None = None,
) -> None:
    """Run a two-phase Optuna study: QMC warmup then TPE.

    Phase 1 (QMC): Uses Quasi-Monte Carlo sampling for space-filling
    exploration of the search space. This provides better initial coverage
    than random sampling.  Skipped when ``qmc_warmup_trials`` is 0
    (e.g. when a substudy already provided warm-start configs).

    Phase 2 (TPE): Switches to Tree-structured Parzen Estimator for
    Bayesian optimization guided by Phase 1 results (or enqueued warm-start).

    Args:
        study: Optuna study object (already created with pruner).
        objective: Objective function to optimize.
        n_trials: Total number of trials (QMC + TPE combined). None = unlimited
            (timeout is the only stopping criterion).
        qmc_warmup_trials: Number of QMC warmup trials. Set to 0 to skip
            QMC entirely (e.g. when substudy provides warm-start).
        timeout: Optional timeout in seconds for the entire study.
        global_seed: Seed for reproducibility.
        tpe_cfg: Optional TPE sampler configuration dict with keys:
            - ``gamma`` (int): Fixed number of trials classified as "good"
              in TPE's surrogate model (default 25). AI sets per model via
              strategy YAML based on round report analysis.
            - ``n_startup_trials`` (int): TPE startup random trials
              (default 0 since QMC/substudy already explored).
            - ``multivariate`` defaults to ``True`` (considers HP correlations).
            - Any additional ``TPESampler`` kwargs (e.g. ``n_ei_candidates``)
              are passed through directly.
        extra_callbacks: Optional list of additional Optuna callbacks to
            run after each trial (e.g. a persist callback that saves the
            tracker/oof_store to disk for interrupt recovery).

    Steps:
        1. Use n_qmc = qmc_warmup_trials directly (0 = skip QMC).
        2. Calculate n_tpe = n_trials - n_qmc.
        3. Phase 1: Set study.sampler to QMCSampler(seed=global_seed).
           Run study.optimize(objective, n_trials=n_qmc, timeout=timeout).
        4. Phase 2: Set study.sampler to TPESampler with custom gamma.
           Run study.optimize(objective, n_trials=n_tpe, timeout=remaining).
        5. Log phase completion summaries.
    """
    if qmc_warmup_trials <= 0:
        n_qmc = 0
        n_tpe = n_trials  # None = unlimited
    else:
        if n_trials is None:
            n_qmc = qmc_warmup_trials
            n_tpe = None  # TPE runs until timeout after QMC
        else:
            n_qmc = max(1, min(qmc_warmup_trials, n_trials - 1))
            n_tpe = n_trials - n_qmc

    callbacks = [_duration_callback] + (extra_callbacks or [])

    start_time = time.time()

    # Enable Optuna's built-in trial logging (shows params + value per trial)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    try:
        # Phase 1: QMC — categorical params fall back to RandomSampler automatically
        # Skipped when qmc_warmup_trials=0 (e.g. substudy already provided warm-start)
        qmc_elapsed = 0.0
        if n_qmc > 0:
            logger.info(f"Phase 1 (QMC): {n_qmc} trials")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="QMCSampler is experimental")
                study.sampler = optuna.samplers.QMCSampler(
                    seed=global_seed,
                    independent_sampler=optuna.samplers.RandomSampler(seed=global_seed),
                    warn_independent_sampling=False,
                )
            study.optimize(
                objective,
                n_trials=n_qmc,
                timeout=timeout,
                show_progress_bar=False,
                callbacks=callbacks,
            )
            qmc_elapsed = time.time() - start_time
            logger.info(
                f"Phase 1 done: {len(study.trials)} trials completed in {qmc_elapsed:.1f}s"
            )
        else:
            logger.info("Phase 1 (QMC): skipped (substudy warm-start)")

        # Phase 2: TPE — skipped when all trials were consumed by QMC (e.g. n_trials=1)
        if n_tpe is None or n_tpe > 0:
            remaining_timeout = None
            if timeout is not None:
                remaining_timeout = max(1, timeout - int(qmc_elapsed))

            # Build TPE sampler with fixed gamma from config.
            # gamma is a fixed integer — AI sets it per model in strategy YAML.
            # No dynamic scaling; the AI adjusts between rounds based on
            # the round report's TPE HEALTH section.
            tpe_cfg = tpe_cfg or {}
            gamma = tpe_cfg.get("gamma", 25)
            n_startup = tpe_cfg.get("n_startup_trials", 0)

            tpe_kwargs: dict[str, Any] = {
                "seed": global_seed,
                "n_startup_trials": n_startup,
                "gamma": lambda x, _g=gamma: _g,
                "multivariate": True,  # project default: always consider HP correlations
            }

            # Pass through additional TPE params from config so the AI
            # can tune any TPESampler knob via strategy YAML.
            _tpe_passthrough = (
                "n_ei_candidates", "prior_weight", "consider_prior",
                "consider_magic_clip", "consider_endpoints",
                "multivariate", "group", "constant_liar",
            )
            for _k in _tpe_passthrough:
                if _k in tpe_cfg:
                    tpe_kwargs[_k] = tpe_cfg[_k]

            logger.info(
                f"Phase 2 (TPE): {n_tpe} trials "
                f"(gamma={gamma}, n_startup={n_startup})"
            )
            study.sampler = optuna.samplers.TPESampler(**tpe_kwargs)
            study.optimize(
                objective,
                n_trials=n_tpe,
                timeout=remaining_timeout,
                show_progress_bar=False,
                callbacks=callbacks,
            )
            total_elapsed = time.time() - start_time
            logger.info(
                f"Phase 2 done: {len(study.trials)} total trials in {total_elapsed:.1f}s"
            )
    finally:
        # Restore quiet mode after study completes
        optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_with_config(
    model_name: str,
    hparams: dict[str, Any],
    feature_cols: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    cv: StratifiedKFold | KFold,
    registry: ModelRegistry,
    task_type: str,
    gpu: bool,
    seeds: list[int],
    results_dir: Path,
    monotone_constraints: list[int] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Train a model with fixed hyperparameters across multiple seeds.

    Used after Optuna selects top configs. Trains with multiple seeds
    for stability and produces OOF + test predictions.

    Args:
        model_name: Name of the model.
        hparams: Fixed hyperparameters (from a top Optuna trial).
        feature_cols: Feature column names.
        train: Training DataFrame.
        test: Test DataFrame.
        target_col: Target column name.
        cv: Cross-validation splitter.
        registry: ModelRegistry instance.
        task_type: Task type string.
        gpu: Whether to use GPU.
        seeds: List of random seeds for stability.
        results_dir: Directory for model artifacts.

    Returns:
        Tuple of:
        - oof_preds_list: List of OOF prediction arrays (one per seed).
        - test_preds_list: List of test prediction arrays (one per seed).
        - labels: The true target values (y_train).

    Steps:
        1. Extract X_train, y_train, X_test from DataFrames.
        2. For each seed in seeds:
           a. Set the random state in hparams to the current seed.
           b. Create the model via registry.get_model.
           c. Initialize oof_preds (len(train),) and test_preds (len(test),).
           d. For each fold (train_idx, val_idx):
              - Fit model on training fold (with eval_set if applicable).
              - Predict on validation set → oof_preds[val_idx].
              - Predict on test set → test_preds += preds / n_folds.
           e. Append oof_preds to oof_preds_list.
           f. Append test_preds to test_preds_list.
        3. Return (oof_preds_list, test_preds_list, y_train).
    """
    X_train = train[feature_cols]
    y_train = train[target_col].values
    X_test = test[feature_cols]

    # Impute NaN for models that don't handle missing values.
    # Fitted per fold (on train fold only) to avoid leakage into the validation set.
    feat_reqs = registry.get_feature_requirements(model_name)
    has_nan_in_train = X_train.isna().any().any()
    has_nan_in_test = X_test.isna().any().any()
    needs_imputation = not feat_reqs.get("handles_missing", False) and (has_nan_in_train or has_nan_in_test)
    if needs_imputation:
        from sklearn.impute import SimpleImputer
        nan_cols = [c for c in X_train.columns if X_train[c].isna().any() or X_test[c].isna().any()]

    # Extract scaler type from trial params (Optuna stored it during search).
    # Copy first so we don't mutate the caller's dict.
    hparams = dict(hparams)
    scaler_type = hparams.pop("scaler", "none")
    scale_cols = _identify_scale_cols(X_train) if scaler_type != "none" else []

    training_cfg = registry.get_training_config(model_name)
    needs_eval_set = training_cfg.get("needs_eval_set", False)
    early_stopping_rounds = training_cfg.get("early_stopping_rounds")
    is_lgbm = training_cfg.get("uses_callbacks_for_early_stopping", False)
    es_in_constructor = training_cfg.get("early_stopping_in_constructor", False)
    sample_weights = _extract_sample_weights(train, training_cfg)

    # Eval metric for constructor
    eval_metric_val = _get_eval_metric_value(training_cfg, task_type, gpu)
    eval_metric_param = training_cfg.get("eval_metric_param")

    # seed_param is declared per-model in the YAML training section.
    seed_param = training_cfg.get("seed_param", "random_state")

    oof_preds_list: list[np.ndarray] = []
    test_preds_list: list[np.ndarray] = []

    try:
        splits = list(cv.split(X_train, y_train))
    except Exception:
        splits = list(cv.split(X_train))

    n_folds = len(splits)

    if task_type == "multiclass":
        n_classes = len(np.unique(y_train))

    # Suppress pytabkit PWL embedding warnings for binary features (once per call)
    warnings.filterwarnings(
        "ignore",
        message=".*has just two bin edges.*",
        module=r"pytabkit\.models\.nn_models\.rtdl_num_embeddings",
    )

    for seed_idx, seed in enumerate(seeds):
        hparams_seeded = {**hparams, seed_param: seed} if seed_param else dict(hparams)
        if eval_metric_val and eval_metric_param:
            hparams_seeded[eval_metric_param] = eval_metric_val
        if es_in_constructor and early_stopping_rounds:
            hparams_seeded["early_stopping_rounds"] = early_stopping_rounds
        if monotone_constraints is not None:
            if model_name == "catboost" and gpu:
                pass  # CatBoost GPU doesn't support monotone_constraints
            elif model_name == "catboost":
                hparams_seeded["monotone_constraints"] = list(monotone_constraints)
            else:
                hparams_seeded["monotone_constraints"] = tuple(monotone_constraints)

        if task_type == "multiclass":
            oof_preds = np.zeros((len(X_train), n_classes))
            test_preds = np.zeros((len(X_test), n_classes))
        else:
            oof_preds = np.zeros(len(X_train))
            test_preds = np.zeros(len(X_test))

        seed_start = time.time()
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_start = time.time()
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            fold_weights = sample_weights[train_idx] if sample_weights is not None else None

            # Impute NaN per fold (fit on train fold only — no leakage from val)
            X_test_fold = X_test
            if needs_imputation:
                imputer = SimpleImputer(strategy="median")
                X_fold_train = X_fold_train.copy()
                X_fold_val = X_fold_val.copy()
                X_fold_train[nan_cols] = imputer.fit_transform(X_fold_train[nan_cols])
                X_fold_val[nan_cols] = imputer.transform(X_fold_val[nan_cols])
                X_test_fold = X_test.copy()
                X_test_fold[nan_cols] = imputer.transform(X_test_fold[nan_cols])

            # Apply scaler per fold (same type as Optuna trial selected)
            if scaler_type != "none" and scale_cols:
                X_fold_train, X_fold_val, X_test_fold = _apply_scaler_fold(
                    scaler_type, X_fold_train, X_fold_val, X_test_fold, scale_cols
                )

            model = registry.get_model(
                model_name, hparams=hparams_seeded,
                task_type=task_type, gpu=gpu, results_dir=results_dir
            )

            # Suppress CatBoost C++ GPU memory warnings on stderr;
            # redirect FTT/skorch stdout epoch tables to log file (not console)
            _stderr_ctx = _suppress_catboost_gpu_warnings() if (model_name == "catboost" and gpu) else contextlib.nullcontext()
            _stdout_ctx = _redirect_stdout_to_log(model_name) if model_name == "ftt" else contextlib.nullcontext()

            with _stderr_ctx, _stdout_ctx:
                if needs_eval_set:
                    fit_params: dict[str, Any] = {
                        "eval_set": [(X_fold_val, y_fold_val)],
                        "verbose": False,  # suppress per-iteration eval output
                    }
                    if fold_weights is not None:
                        fit_params["sample_weight"] = fold_weights
                    if is_lgbm:
                        import lightgbm as lgb
                        del fit_params["verbose"]  # LightGBM uses callbacks instead
                        fit_params["callbacks"] = [
                            lgb.early_stopping(
                                stopping_rounds=early_stopping_rounds or 50,
                                verbose=False,
                            ),
                            lgb.log_evaluation(-1),
                        ]
                    elif early_stopping_rounds and not es_in_constructor:
                        fit_params["early_stopping_rounds"] = early_stopping_rounds
                    model.fit(X_fold_train, y_fold_train, **fit_params)
                else:
                    if fold_weights is not None:
                        model.fit(X_fold_train, y_fold_train, sample_weight=fold_weights)
                    else:
                        model.fit(X_fold_train, y_fold_train)

            if task_type == "binary_classification":
                val_preds = model.predict_proba(X_fold_val)[:, 1]
                tst_preds = model.predict_proba(X_test_fold)[:, 1]
            elif task_type == "multiclass":
                val_preds = model.predict_proba(X_fold_val)  # (n_val, n_classes)
                tst_preds = model.predict_proba(X_test_fold)  # (n_test, n_classes)
            else:
                val_preds = model.predict(X_fold_val)
                tst_preds = model.predict(X_test_fold)

            oof_preds[val_idx] = val_preds
            test_preds += tst_preds / n_folds

            fold_metric = _compute_cv_metric(y_fold_val, val_preds, task_type)
            logger.debug(
                f"    Fold {fold_idx + 1}/{n_folds}: "
                f"score={fold_metric:.6f} ({time.time() - fold_start:.1f}s)"
            )

        seed_elapsed = time.time() - seed_start
        oof_metric = _compute_cv_metric(y_train, oof_preds, task_type)
        logger.debug(
            f"  Seed {seed} (#{seed_idx + 1}/{len(seeds)}): "
            f"CV={oof_metric:.6f} in {seed_elapsed:.1f}s"
        )
        oof_preds_list.append(oof_preds)
        test_preds_list.append(test_preds)

    return oof_preds_list, test_preds_list, y_train


def _reassemble_int_lists(params: dict[str, Any]) -> dict[str, Any]:
    """Reassemble int_list / dynamic_int_list params back into lists.

    Optuna stores them as separate keys. This function detects the pattern
    ``<name>_0, <name>_1, ...`` and combines them into ``<name>: [v0, v1, ...]``.
    For dynamic_int_list, ``<name>_n`` holds the layer count and is removed.
    """
    import re
    # Find all keys matching the pattern <name>_<digit> or <name>_n
    list_keys: dict[str, dict[int, Any]] = {}
    n_keys: dict[str, int] = {}  # <name>_n from dynamic_int_list
    plain_keys: dict[str, Any] = {}
    for k, v in params.items():
        m = re.match(r"^(.+)_(\d+)$", k)
        m_n = re.match(r"^(.+)_n$", k)
        if m:
            name, idx = m.group(1), int(m.group(2))
            list_keys.setdefault(name, {})[idx] = v
        elif m_n:
            n_keys[m_n.group(1)] = v
        else:
            plain_keys[k] = v

    # Only reassemble if ALL indices 0..N-1 are present (avoid false positives)
    result = dict(plain_keys)
    for name, idx_map in list_keys.items():
        n = len(idx_map)
        if all(i in idx_map for i in range(n)):
            result[name] = [idx_map[i] for i in range(n)]
        else:
            # Not a complete sequence, keep as separate keys
            for idx, val in idx_map.items():
                result[f"{name}_{idx}"] = val
    # Drop _n keys that were already consumed (dynamic_int_list)
    for name in n_keys:
        if name not in result:
            result[f"{name}_n"] = n_keys[name]
    return result


def get_top_configs(
    study: optuna.Study,
    n_top: int = 5,
) -> list[dict[str, Any]]:
    """Extract the top N trial configurations from a completed study.

    Args:
        study: Completed Optuna study.
        n_top: Number of top configurations to extract.

    Returns:
        List of dicts, each containing:
        - params: dict of hyperparameter values
        - value: CV score achieved
        - trial_number: original trial number

    Steps:
        1. Get all completed (non-pruned) trials from the study.
        2. Sort by value (descending for maximize, ascending for minimize).
        3. Take the top n_top trials.
        4. For each, extract params, value, and number into a dict.
        5. Return the list.
    """
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]

    descending = (study.direction == optuna.study.StudyDirection.MAXIMIZE)
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=descending)
    top = sorted_trials[:n_top]

    return [
        {
            "params": _reassemble_int_lists(dict(t.params)),
            "value": t.value,
            "trial_number": t.number,
        }
        for t in top
    ]


def run_all_studies(
    pipeline_config: PipelineConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    strategy: dict,
    registry: ModelRegistry,
    gpu_status: dict[str, bool],
) -> dict[str, dict[str, Any]]:
    """Run Optuna studies for all models in the pipeline, then retrain
    top configs with multiple seeds.

    This is the main orchestrator for Layer 3 model training.

    Args:
        pipeline_config: Pipeline configuration.
        train: Training DataFrame with engineered features.
        test: Test DataFrame with engineered features.
        feature_cols: Feature column names to use.
        strategy: Strategy dict with optional per-model overrides.
        registry: ModelRegistry with loaded configs.
        gpu_status: Dict of {model_name: gpu_available_bool}.

    Returns:
        Dictionary keyed by model name, each value containing:
        - study: the Optuna Study object
        - top_configs: list of top config dicts
        - oof_preds: list of OOF prediction arrays (all seeds, all configs)
        - test_preds: list of test prediction arrays
        - labels: true target values

    Steps:
        1. Create the CV splitter from pipeline_config.cv.
        2. For each model_name in pipeline_config.models:
           a. Log the model name and start.
           b. Run run_optuna_study to get the study.
           c. Call get_top_configs to get the best N configs.
           d. For each top config, call train_with_config with
              the configured seeds.
           e. Collect all OOF and test prediction arrays.
           f. Store results in the output dict.
        3. Return the complete results dict.
    """
    cv_cfg = pipeline_config.cv
    task_type = pipeline_config.task_type

    if cv_cfg.stratified and task_type != "regression":
        cv = StratifiedKFold(n_splits=cv_cfg.n_folds, shuffle=True, random_state=cv_cfg.seed)
    else:
        cv = KFold(n_splits=cv_cfg.n_folds, shuffle=True, random_state=cv_cfg.seed)

    results: dict[str, dict[str, Any]] = {}
    global_seed = pipeline_config.optuna.global_seed
    results_dir = Path(pipeline_config.output.results_dir)

    n_models = len(pipeline_config.models)
    all_models_start = time.time()

    for model_idx, model_name in enumerate(pipeline_config.models, 1):
        model_start = time.time()
        logger.info(
            f"[{model_idx}/{n_models}] Starting model: {model_name}"
        )
        gpu = gpu_status.get(model_name, False)

        # Resolve per-model timeout from pipeline config
        model_timeout = pipeline_config.optuna.model_timeouts.get(model_name)

        try:
            study, tracker, oof_store = run_optuna_study(
                model_name=model_name,
                train=train,
                feature_cols=feature_cols,
                target_col=pipeline_config.target_column,
                registry=registry,
                pipeline_config=pipeline_config,
                strategy=strategy,
                gpu=gpu,
                timeout_override=model_timeout,
                test=test,
            )
            optuna_elapsed = time.time() - model_start

            # Study statistics
            n_completed = sum(
                1 for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            )
            n_pruned = sum(
                1 for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            )
            n_failed = len(study.trials) - n_completed - n_pruned
            avg_trial_time = optuna_elapsed / max(len(study.trials), 1)
            try:
                best_val_str = f"{study.best_value:.6f}"
            except ValueError:
                best_val_str = "n/a"
            logger.info(
                f"[{model_name}] Optuna done in {optuna_elapsed:.0f}s "
                f"({optuna_elapsed / 60:.1f}min) | "
                f"{len(study.trials)} trials "
                f"({n_completed} ok, {n_pruned} pruned, {n_failed} failed) | "
                f"avg {avg_trial_time:.1f}s/trial | "
                f"best={best_val_str}"
            )

            # Hyperparameter importance (fANOVA)
            hp_importance: dict = {}
            try:
                importances = optuna.importance.get_param_importances(study)
                if importances:
                    hp_importance = dict(importances)
                    imp_lines = [
                        f"  {name}: {val:.3f}" for name, val in importances.items()
                    ]
                    logger.info(
                        f"[{model_name}] Hyperparameter importance:\n"
                        + "\n".join(imp_lines)
                    )
            except Exception:
                pass  # Not enough completed trials or other issue

            optuna_cfg = registry.get_optuna_config(model_name)
            # Same deep merge as run_optuna_study — recurses into nested dicts so
            # doubly-nested keys like assembly.nsga2.* are merged, not replaced.
            strategy_overrides = strategy.get("overrides", {}).get(model_name, {}) or {}
            optuna_overrides = strategy_overrides.get("optuna", {}) or {}
            _deep_merge(optuna_cfg, optuna_overrides)

            n_top = optuna_cfg["n_top_trials"]
            n_seeds = optuna_cfg["n_seeds"]
            seeds = [global_seed + i for i in range(n_seeds)]
            selection_mode = optuna_cfg.get("selection_mode", "global")

            all_oof: list[np.ndarray] = []
            all_test: list[np.ndarray] = []
            labels: np.ndarray | None = None
            retrain_elapsed = 0.0
            n_composites_raw: int | None = None
            best_fold_scores: list[float] = []

            if selection_mode == "fold_coverage" and oof_store is not None:
                # --- Fold-coverage path: complete OOF per trial, no retraining ---
                assembly_cfg = optuna_cfg.get("assembly", {}) or {}
                n_fold_best = assembly_cfg.get("n_fold_best", pipeline_config.cv.n_folds)
                n_mean_best = assembly_cfg.get("n_mean_best", 5)
                min_quality_percentile = assembly_cfg.get("min_quality_percentile", 0.0)

                oof_store.log_summary(model_name)
                composites = oof_store.select(
                    n_fold_best=n_fold_best,
                    n_mean_best=n_mean_best,
                    min_quality_percentile=min_quality_percentile,
                )

                if not composites:
                    logger.warning(
                        f"[{model_name}] Fold-coverage: no completed trials — "
                        f"skipping model"
                    )
                else:
                    best_fold_scores = composites[0]["fold_scores"]
                    all_oof = [c["oof_preds"] for c in composites]
                    all_test = [c["test_preds"] for c in composites]
                    labels = train[pipeline_config.target_column].values
                    top_configs = [
                        {
                            "params": {},
                            "value": c["avg_score"],
                            "trial_number": c["fold_trials"][0],
                        }
                        for c in composites
                    ]
                    logger.info(
                        f"[{model_name}] Fold-coverage assembly: "
                        f"{len(all_oof)} composites (no retraining needed)"
                    )

            elif selection_mode == "per_fold" and tracker is not None:
                # --- Per-fold path: OOF + test from tracker, no retraining ---
                assembly_cfg = optuna_cfg.get("assembly", {"mode": "rank"})
                assembly_mode = assembly_cfg.get("mode", "rank")

                rank_normalize = assembly_cfg.get("rank_normalize", True)
                test_combine = assembly_cfg.get("test_combine", "arithmetic")

                if assembly_mode == "nsga2":
                    composites = tracker.assemble_nsga2(
                        n_samples=len(train),
                        n_test=len(test),
                        task_type=task_type,
                        n_composites=assembly_cfg.get("n_composites", n_top),
                        n_generations=assembly_cfg.get("n_generations", 50),
                        pop_size=assembly_cfg.get("pop_size", 100),
                        diversity_metric=assembly_cfg.get("diversity_metric", "pearson_neff"),
                        diversity_weight=assembly_cfg.get("diversity_weight", 0.3),
                        seed=global_seed,
                        rank_normalize=rank_normalize,
                        test_combine=test_combine,
                    )
                else:
                    composites = tracker.assemble(
                        n_samples=len(train),
                        n_test=len(test),
                        task_type=task_type,
                        rank_normalize=rank_normalize,
                        test_combine=test_combine,
                    )
                tracker.log_summary(model_name)

                # Deduplicate near-identical composites (corr ≥ 0.9999)
                n_composites_raw = len(composites)
                composites = _deduplicate_composites(
                    composites,
                    corr_threshold=0.9999,
                    maximize=(task_type != "regression"),
                )
                if composites:
                    best_fold_scores = composites[0]["fold_scores"]

                all_oof = [c["oof_preds"] for c in composites]
                all_test = [c["test_preds"] for c in composites]
                labels = train[pipeline_config.target_column].values

                top_configs = [
                    {
                        "params": {"assembly_mode": assembly_mode},
                        "value": c["avg_score"],
                        "trial_number": c["fold_trials"],
                    }
                    for c in composites
                ]

                logger.info(
                    f"[{model_name}] Per-fold assembly ({assembly_mode}): "
                    f"{len(all_oof)} composites (no retraining needed)"
                )
                # Log actual assembled OOF AUC (vs optimistic avg_score from tracker)
                if all_oof and task_type != "regression":
                    from sklearn.metrics import roc_auc_score
                    assembled_aucs = [
                        roc_auc_score(labels, oof) for oof in all_oof
                    ]
                    logger.info(
                        f"[{model_name}] Assembled OOF AUC: "
                        f"best={max(assembled_aucs):.6f}, "
                        f"mean={sum(assembled_aucs)/len(assembled_aucs):.6f}, "
                        f"worst={min(assembled_aucs):.6f} "
                        f"(tracker avg_score={composites[0]['avg_score']:.6f})"
                    )

            else:
                # --- Global path: retrain top configs (existing behaviour) ---
                top_configs = get_top_configs(study, n_top=n_top)

                retrain_start = time.time()
                total_retrain_fits = len(top_configs) * n_seeds * cv_cfg.n_folds
                logger.info(
                    f"[{model_name}] Retraining top {len(top_configs)} configs "
                    f"x {n_seeds} seeds x {cv_cfg.n_folds} folds = "
                    f"{total_retrain_fits} fits"
                )

                # Resolve monotone constraints for retraining.
                # Per-model override in strategy["overrides"][model]["monotone_constraints"]
                # takes precedence over global strategy["monotone_constraints"].
                # An empty dict {} from the override disables constraints for this model.
                _model_ov = (strategy.get("overrides") or {}).get(model_name) or {}
                if "monotone_constraints" in _model_ov:
                    mc_dict = _model_ov["monotone_constraints"] or {}
                else:
                    mc_dict = strategy.get("monotone_constraints", {}) or {}
                mc_list: list[int] | None = None
                if mc_dict and model_name in ("catboost", "xgboost", "lightgbm"):
                    if model_name == "catboost" and gpu:
                        mc_list = None  # CatBoost GPU doesn't support monotone_constraints
                    else:
                        mc_list = [int(mc_dict.get(col, 0)) for col in feature_cols]
                        if not any(v != 0 for v in mc_list):
                            mc_list = None

                # When run_optuna_study prescaled train/test upfront (single locked scaler),
                # the objective stored prescaled_scaler as a trial user attr instead of a param.
                # Inject it into cfg["params"] so train_with_config applies per-fold scaling
                # on the original DataFrames (per-fold scaling is correct — no leakage).
                trial_by_number = {t.number: t for t in study.trials}
                for cfg in top_configs:
                    if "scaler" not in cfg["params"]:
                        trial = trial_by_number.get(cfg["trial_number"])
                        if trial is not None:
                            ps = trial.user_attrs.get("prescaled_scaler")
                            if ps:
                                cfg["params"]["scaler"] = ps

                for cfg_idx, cfg in enumerate(top_configs, 1):
                    logger.debug(
                        f"[{model_name}] Retrain config {cfg_idx}/{len(top_configs)} "
                        f"(score={cfg['value']:.6f})"
                    )
                    oof_list, test_list, y = train_with_config(
                        model_name=model_name,
                        hparams=cfg["params"],
                        feature_cols=feature_cols,
                        train=train,
                        test=test,
                        target_col=pipeline_config.target_column,
                        cv=cv,
                        registry=registry,
                        task_type=task_type,
                        gpu=gpu,
                        seeds=seeds,
                        results_dir=results_dir,
                        monotone_constraints=mc_list,
                    )
                    all_oof.extend(oof_list)
                    all_test.extend(test_list)
                    if labels is None:
                        labels = y

                retrain_elapsed = time.time() - retrain_start

            model_elapsed = time.time() - model_start

            results[model_name] = {
                "study": study,
                "top_configs": top_configs,
                "oof_preds": all_oof,
                "test_preds": all_test,
                "labels": labels,
                "elapsed": model_elapsed,
                "optuna_elapsed": optuna_elapsed,
                "retrain_elapsed": retrain_elapsed,
                "n_trials": len(study.trials),
                "avg_trial_time": avg_trial_time,
                "selection_mode": selection_mode,
                "n_committed": len(oof_store._oof) if oof_store is not None else None,
                "n_composites_raw": n_composites_raw,
                "best_fold_scores": best_fold_scores,
                "hp_importance": hp_importance,
            }
            logger.info(
                f"[{model_idx}/{n_models}] {model_name} done: "
                f"{len(all_oof)} arrays | "
                f"optuna={optuna_elapsed:.0f}s retrain={retrain_elapsed:.0f}s "
                f"total={model_elapsed:.0f}s ({model_elapsed / 60:.1f}min)"
            )

        except Exception as exc:
            logger.error(f"Model '{model_name}' failed: {exc}", exc_info=True)

        # Free GPU memory between models to avoid OOM
        _free_gpu_memory()

    total_training = time.time() - all_models_start
    total_m, total_s = divmod(int(total_training), 60)
    total_h, total_m = divmod(total_m, 60)
    if total_h > 0:
        fmt = f"{total_h}h {total_m:02d}m {total_s:02d}s"
    elif total_m > 0:
        fmt = f"{total_m}m {total_s:02d}s"
    else:
        fmt = f"{total_training:.1f}s"
    logger.info(
        f"All model training complete: {len(results)}/{n_models} models "
        f"in {fmt}"
    )

    return results
