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
from src.utils.io import PipelineConfig


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
    """Suppress CatBoost C++ GPU memory warnings from stderr."""
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr


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
    ) -> list[dict[str, Any]]:
        """Build composite OOF + test arrays from per-fold bests.

        For the k-th composite:
        - OOF: ``oof[val_idx] = k-th best trial's val_preds`` for each fold.
        - Test: average of per-fold test predictions (``+= fold_test / n_folds``).

        Returns:
            List of dicts, each with keys: ``oof_preds``, ``test_preds``,
            ``fold_trials``, ``fold_scores``, ``avg_score``.
        """
        n_composites = min(
            self.n_top,
            min(len(d) for d in self.fold_data.values()) if self.fold_data else 0,
        )

        is_multiclass = task_type == "multiclass"

        results: list[dict[str, Any]] = []
        for k in range(n_composites):
            if is_multiclass:
                # Infer n_classes from stored predictions
                sample_preds = self.fold_data[0][k].val_preds
                n_classes = sample_preds.shape[1] if sample_preds.ndim > 1 else 1
                oof = np.zeros((n_samples, n_classes))
                test_preds = np.zeros((n_test, n_classes))
            else:
                oof = np.zeros(n_samples)
                test_preds = np.zeros(n_test)

            fold_trials: list[int] = []
            fold_scores: list[float] = []

            for fold_idx in range(self.n_folds):
                entry = self.fold_data[fold_idx][k]
                oof[entry.val_idx] = entry.val_preds
                test_preds += entry.test_preds / self.n_folds
                fold_trials.append(entry.trial_number)
                fold_scores.append(entry.score)

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

        def _build_composite(indices: list[int]) -> tuple[np.ndarray, np.ndarray, list[int], list[float]]:
            """Build a single composite from fold-index selections."""
            is_multiclass = task_type == "multiclass"
            if is_multiclass:
                sample_preds = self.fold_data[0][0].val_preds
                n_classes = sample_preds.shape[1] if sample_preds.ndim > 1 else 1
                oof = np.zeros((n_samples, n_classes))
                test_preds = np.zeros((n_test, n_classes))
            else:
                oof = np.zeros(n_samples)
                test_preds = np.zeros(n_test)

            fold_trials: list[int] = []
            fold_scores: list[float] = []

            for fold_idx in range(self.n_folds):
                entry = self.fold_data[fold_idx][indices[fold_idx]]
                oof[entry.val_idx] = entry.val_preds
                test_preds += entry.test_preds / self.n_folds
                fold_trials.append(entry.trial_number)
                fold_scores.append(entry.score)

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
            return self.assemble(n_samples, n_test, task_type)

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
) -> tuple[optuna.Study, PerFoldTracker | None]:
    """Run a complete Optuna study for a single model.

    Creates an Optuna study with the model's configured pruner, runs
    QMC warmup trials followed by TPE trials, and returns the study
    with all trial results.

    When the model's ``selection_mode`` is ``per_fold``, a
    :class:`PerFoldTracker` is created and passed into the objective.
    During each trial's fold training the model also predicts on test
    data and the tracker records per-fold results (including from
    pruned trials).

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
        Tuple of (study, tracker).  ``tracker`` is None when the model
        uses global selection mode.

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

    # Per-fold selection: create tracker if enabled
    selection_mode = optuna_cfg.get("selection_mode", "global")
    tracker: PerFoldTracker | None = None
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
    if needs_scaling:
        # Default: all scalers. Strategy can narrow to a subset.
        scaler_choices = model_preproc.get("scaler_choices") or preprocessing.get("scaler_choices") or list(ALL_SCALER_CHOICES)
        # Ensure "none" is always an option so Optuna can decide
        if "none" not in scaler_choices:
            scaler_choices = ["none"] + list(scaler_choices)
        logger.info(f"[{model_name}] Scaler search: {scaler_choices}")

    # Resolve monotone constraints from strategy (feature_name → direction)
    # into a positional list matching feature_cols order.
    monotone_constraints_list: list[int] | None = None
    mc_dict = strategy.get("monotone_constraints", {}) or {}
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

    # Diversity pruning config (only meaningful with per-fold tracker)
    dp_cfg = optuna_cfg.get("diversity_pruning") if selection_mode == "per_fold" else None

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
        fold_timeout=fold_timeout,
        n_top_trials=optuna_cfg["n_top_trials"],
        scaler_choices=scaler_choices,
        monotone_constraints=monotone_constraints_list,
        diversity_pruning=dp_cfg,
    )

    # Configure pruner
    pruner_cfg = optuna_cfg.get("pruner", {}) or {}
    pruner_type = pruner_cfg.get("type", "median")
    if pruner_type == "median":
        pruner = optuna.pruners.MedianPruner(
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 3),
            n_startup_trials=pruner_cfg.get("n_startup_trials", 10),
        )
    elif pruner_type == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    # Study direction
    direction = "minimize" if task_type == "regression" else "maximize"

    study = optuna.create_study(direction=direction, pruner=pruner)

    # Pipeline-level timeout override takes precedence over model YAML timeout
    effective_timeout = timeout_override if timeout_override is not None else optuna_cfg.get("timeout")
    if effective_timeout:
        logger.info(f"[{model_name}] Timeout: {effective_timeout}s ({effective_timeout / 3600:.1f}h)")

    _run_two_phase_study(
        study=study,
        objective=objective,
        n_trials=optuna_cfg["n_trials"],
        qmc_warmup_trials=optuna_cfg["qmc_warmup_trials"],
        timeout=effective_timeout,
        global_seed=pipeline_config.optuna.global_seed,
        verbose=pipeline_config.runtime.verbose,
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

    return study, tracker


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
    fold_timeout: int | None = None,
    n_top_trials: int = 5,
    scaler_choices: list[str] | None = None,
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
           f. Store OOF predictions via trial.set_user_attr('oof_preds', oof)
              only if this trial ranks in the top n_top_trials among all
              completed trials so far (bounds memory usage).
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

    def objective(trial: optuna.Trial) -> float:
        # Scaler selection (Optuna parameter when model needs scaling)
        if scaler_choices and scale_cols:
            scaler_type = trial.suggest_categorical("scaler", scaler_choices)
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
                else:
                    model_hparams["monotone_constraints"] = tuple(monotone_constraints)

            # Fresh model per fold — each fold gets an independent RandomState,
            # matching the behaviour of train_with_config for reproducibility.
            model = registry.get_model(
                model_name, hparams=model_hparams, task_type=task_type,
                gpu=gpu, results_dir=results_dir
            )

            fold_start = time.monotonic()

            # Suppress CatBoost C++ GPU memory warnings on stderr
            _fit_ctx = _suppress_catboost_gpu_warnings() if (model_name == "catboost" and gpu) else contextlib.nullcontext()

            with _fit_ctx:
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

                    # Check score gate: never diversity-prune if best score
                    is_best_so_far = True
                    try:
                        best_val = trial.study.best_value
                        if per_fold_tracker.maximize:
                            is_best_so_far = fold_metric > best_val * (1 + dp_score_tolerance)
                        else:
                            is_best_so_far = fold_metric < best_val * (1 - dp_score_tolerance)
                    except ValueError:
                        is_best_so_far = True  # no completed trials yet

                    if max_corr >= dp_corr_threshold and not is_best_so_far:
                        if not hasattr(trial, "_diversity_flag_count"):
                            trial._diversity_flag_count = 0  # type: ignore[attr-defined]
                        trial._diversity_flag_count += 1  # type: ignore[attr-defined]

                        if trial._diversity_flag_count >= dp_n_consecutive:  # type: ignore[attr-defined]
                            logger.info(
                                f"[{model_name}] Trial #{trial.number} diversity-pruned at fold "
                                f"{fold_idx} (max |corr|={max_corr:.4f} >= {dp_corr_threshold})"
                            )
                            trial.set_user_attr("diversity_pruned", True)
                            raise optuna.exceptions.TrialPruned()
                    else:
                        # Reset counter — must be consecutive
                        trial._diversity_flag_count = 0  # type: ignore[attr-defined]

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
        # At the time of completion, the current trial is RUNNING, so
        # study.trials only returns previously completed trials — making
        # this an accurate, race-free rank check.
        maximize = task_type != "regression"
        prior_values = [
            t.value for t in trial.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        if len(prior_values) < n_top_trials:
            trial.set_user_attr("oof_preds", oof)
        else:
            cutoff = sorted(prior_values, reverse=maximize)[n_top_trials - 1]
            if (maximize and overall_metric >= cutoff) or (not maximize and overall_metric <= cutoff):
                trial.set_user_attr("oof_preds", oof)
        return overall_metric

    return objective


def _run_two_phase_study(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    qmc_warmup_trials: int,
    timeout: int | None = None,
    global_seed: int = 42,
    verbose: int = 1,
) -> None:
    """Run a two-phase Optuna study: QMC warmup then TPE.

    Phase 1 (QMC): Uses Quasi-Monte Carlo sampling for space-filling
    exploration of the search space. This provides better initial coverage
    than random sampling.

    Phase 2 (TPE): Switches to Tree-structured Parzen Estimator for
    Bayesian optimization guided by Phase 1 results.

    Args:
        study: Optuna study object (already created with pruner).
        objective: Objective function to optimize.
        n_trials: Total number of trials (QMC + TPE combined).
        qmc_warmup_trials: Number of QMC warmup trials (e.g. 50).
        timeout: Optional timeout in seconds for the entire study.
        global_seed: Seed for reproducibility.

    Steps:
        1. Use n_qmc = qmc_warmup_trials directly.
        2. Calculate n_tpe = n_trials - n_qmc.
        3. Phase 1: Set study.sampler to QMCSampler(seed=global_seed).
           Run study.optimize(objective, n_trials=n_qmc, timeout=timeout).
        4. Phase 2: Set study.sampler to TPESampler(seed=global_seed,
           n_startup_trials=0) (startup=0 because QMC already explored).
           Run study.optimize(objective, n_trials=n_tpe, timeout=remaining).
        5. Log phase completion summaries.
    """
    n_qmc = max(1, min(qmc_warmup_trials, n_trials - 1))
    n_tpe = n_trials - n_qmc  # QMC + TPE = n_trials exactly

    start_time = time.time()

    # Progress callback
    log_interval = max(1, n_trials // 8)  # ~8 progress updates per study (verbose=1)

    def _progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        n_done = len(study.trials)
        try:
            best_val = study.best_value
        except ValueError:
            best_val = float("nan")
        elapsed = time.time() - start_time
        if verbose >= 2:
            # Show every trial with its own score — check state, not value,
            # because Optuna 3.x sets trial.value to last intermediate for pruned trials
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_val = f"{trial.value:.6f}"
            elif trial.state == optuna.trial.TrialState.PRUNED:
                trial_val = "pruned"
            else:
                trial_val = "failed"
            logger.info(
                f"  Trial {n_done}/{n_trials} | "
                f"score={trial_val} | "
                f"best={best_val:.6f} | "
                f"{elapsed:.0f}s"
            )
        elif n_done % log_interval == 0 or n_done == n_qmc:
            logger.info(
                f"  Trial {n_done}/{n_trials} | "
                f"best={best_val:.6f} | "
                f"elapsed={elapsed:.0f}s"
            )

    # Phase 1: QMC — categorical params fall back to RandomSampler automatically
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
        callbacks=[_progress_callback],
        show_progress_bar=False,
    )
    qmc_elapsed = time.time() - start_time
    logger.info(
        f"Phase 1 done: {len(study.trials)} trials completed in {qmc_elapsed:.1f}s"
    )

    # Phase 2: TPE — skipped when all trials were consumed by QMC (e.g. n_trials=1)
    if n_tpe > 0:
        remaining_timeout = None
        if timeout is not None:
            remaining_timeout = max(1, timeout - int(qmc_elapsed))

        logger.info(f"Phase 2 (TPE): {n_tpe} trials")
        study.sampler = optuna.samplers.TPESampler(
            seed=global_seed, n_startup_trials=0
        )
        study.optimize(
            objective,
            n_trials=n_tpe,
            timeout=remaining_timeout,
            callbacks=[_progress_callback],
            show_progress_bar=False,
        )
        total_elapsed = time.time() - start_time
        logger.info(
            f"Phase 2 done: {len(study.trials)} total trials in {total_elapsed:.1f}s"
        )


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

            # Suppress CatBoost C++ GPU memory warnings on stderr
            _fit_ctx = _suppress_catboost_gpu_warnings() if (model_name == "catboost" and gpu) else contextlib.nullcontext()

            with _fit_ctx:
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
            study, tracker = run_optuna_study(
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
            try:
                importances = optuna.importance.get_param_importances(study)
                if importances:
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
            n_top = optuna_cfg["n_top_trials"]
            n_seeds = optuna_cfg["n_seeds"]
            seeds = [global_seed + i for i in range(n_seeds)]
            selection_mode = optuna_cfg.get("selection_mode", "global")

            all_oof: list[np.ndarray] = []
            all_test: list[np.ndarray] = []
            labels: np.ndarray | None = None
            retrain_elapsed = 0.0

            if selection_mode == "per_fold" and tracker is not None:
                # --- Per-fold path: OOF + test from tracker, no retraining ---
                assembly_cfg = optuna_cfg.get("assembly", {"mode": "rank"})
                assembly_mode = assembly_cfg.get("mode", "rank")

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
                    )
                else:
                    composites = tracker.assemble(
                        n_samples=len(train),
                        n_test=len(test),
                        task_type=task_type,
                    )
                tracker.log_summary(model_name)

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

                # Resolve monotone constraints for retraining
                mc_dict = strategy.get("monotone_constraints", {}) or {}
                mc_list: list[int] | None = None
                if mc_dict and model_name in ("catboost", "xgboost", "lightgbm"):
                    if model_name == "catboost" and gpu:
                        mc_list = None  # CatBoost GPU doesn't support monotone_constraints
                    else:
                        mc_list = [int(mc_dict.get(col, 0)) for col in feature_cols]
                        if not any(v != 0 for v in mc_list):
                            mc_list = None

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
            }
            logger.info(
                f"[{model_idx}/{n_models}] {model_name} done: "
                f"{len(all_oof)} arrays | "
                f"optuna={optuna_elapsed:.0f}s retrain={retrain_elapsed:.0f}s "
                f"total={model_elapsed:.0f}s ({model_elapsed / 60:.1f}min)"
            )

        except Exception as exc:
            logger.error(f"Model '{model_name}' failed: {exc}", exc_info=True)

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
