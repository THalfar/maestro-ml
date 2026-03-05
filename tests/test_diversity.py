"""Tests for src/ensemble/diversity.py — pymoo NSGA-II ensemble diversity."""
from __future__ import annotations

import numpy as np
import pytest

from src.ensemble.diversity import (
    compute_correlation_matrix,
    compute_error_correlation_matrix,
    compute_spearman_correlation_matrix,
    compute_spearman_error_correlation_matrix,
    compute_ambiguity,
    _compute_diversity,
    effective_ensemble_size,
    greedy_diverse_select,
    run_nsga2_ensemble,
    select_from_pareto,
    _normalize,
    _weights_to_selection,
    DIVERSITY_METRICS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def diverse_preds():
    """Generate diverse OOF predictions."""
    rng = np.random.default_rng(42)
    n = 200
    y = rng.choice([0, 1], n)
    # Model A: good, based on feature 1
    a = y * 0.7 + rng.normal(0, 0.2, n)
    a = np.clip(a, 0.01, 0.99)
    # Model B: good, based on different features (low correlation with A)
    b = y * 0.65 + rng.normal(0, 0.25, n)
    b = np.clip(b, 0.01, 0.99)
    # Model C: near-identical to A (high correlation)
    c = a + rng.normal(0, 0.05, n)
    c = np.clip(c, 0.01, 0.99)
    return [a, b, c], y


@pytest.fixture
def synthetic_pareto():
    """Synthetic Pareto front data for select_from_pareto tests.

    5 solutions with negated objectives (pymoo minimises):
      sol 0: best metric (-0.90), worst diversity (-1.0)
      sol 1: good metric (-0.88), ok diversity (-1.5)
      sol 2: balanced (-0.85), good diversity (-2.0)  <-- likely knee
      sol 3: poor metric (-0.80), great diversity (-2.8)
      sol 4: worst metric (-0.75), best diversity (-3.0)
    """
    pareto_F = np.array([
        [-0.90, -1.0],
        [-0.88, -1.5],
        [-0.85, -2.0],
        [-0.80, -2.8],
        [-0.75, -3.0],
    ])
    # Weight vectors (3 models)
    pareto_X = np.array([
        [0.8, 0.1, 0.1],   # mostly model 0
        [0.5, 0.3, 0.2],
        [0.4, 0.35, 0.25],
        [0.3, 0.35, 0.35],
        [0.33, 0.33, 0.34],
    ])
    return pareto_F, pareto_X


# ---------------------------------------------------------------------------
# compute_correlation_matrix
# ---------------------------------------------------------------------------

class TestComputeCorrelationMatrix:
    def test_shape(self, diverse_preds):
        oof_list, _ = diverse_preds
        corr = compute_correlation_matrix(oof_list)
        assert corr.shape == (3, 3)

    def test_diagonal_is_one(self, diverse_preds):
        oof_list, _ = diverse_preds
        corr = compute_correlation_matrix(oof_list)
        np.testing.assert_array_almost_equal(np.diag(corr), [1.0, 1.0, 1.0])

    def test_symmetric(self, diverse_preds):
        oof_list, _ = diverse_preds
        corr = compute_correlation_matrix(oof_list)
        np.testing.assert_array_almost_equal(corr, corr.T)

    def test_identical_predictions_corr_one(self):
        preds = np.random.default_rng(0).random(100)
        corr = compute_correlation_matrix([preds, preds])
        np.testing.assert_almost_equal(corr[0, 1], 1.0)

    def test_constant_oof_produces_zero_not_nan(self):
        """A constant OOF array (zero variance) must not produce NaN.
        The fix replaces NaN with 0.0 (treat as uncorrelated) and keeps
        the diagonal at 1.0 so effective_ensemble_size stays well-defined."""
        constant = np.full(100, 0.5)
        varying = np.random.default_rng(0).random(100)
        corr = compute_correlation_matrix([constant, varying])
        assert corr.shape == (2, 2)
        assert not np.any(np.isnan(corr)), "No NaN should remain after fix"
        assert corr[0, 1] == 0.0, "Undefined correlation should be treated as 0.0"
        np.testing.assert_array_almost_equal(np.diag(corr), [1.0, 1.0])


# ---------------------------------------------------------------------------
# compute_error_correlation_matrix
# ---------------------------------------------------------------------------

class TestComputeErrorCorrelationMatrix:
    def test_shape(self, diverse_preds):
        oof_list, y = diverse_preds
        corr = compute_error_correlation_matrix(oof_list, y)
        assert corr.shape == (3, 3)

    def test_diagonal_is_one(self, diverse_preds):
        oof_list, y = diverse_preds
        corr = compute_error_correlation_matrix(oof_list, y)
        np.testing.assert_array_almost_equal(np.diag(corr), [1.0, 1.0, 1.0])

    def test_error_neff_higher_than_pred_neff_when_errors_differ(self):
        """Models that predict similarly but err on different rows should show
        higher error N_eff than prediction N_eff."""
        rng = np.random.default_rng(7)
        n = 300
        y = rng.choice([0, 1], n).astype(float)

        # Both models learn the signal well — predictions are correlated
        signal = y * 0.8
        a = np.clip(signal + rng.normal(0, 0.1, n), 0.01, 0.99)
        b = np.clip(signal + rng.normal(0, 0.1, n), 0.01, 0.99)
        # But errors are uncorrelated (different noise sources)
        oof_list = [a, b]

        pred_corr = compute_correlation_matrix(oof_list)
        err_corr = compute_error_correlation_matrix(oof_list, y)

        pred_neff = effective_ensemble_size(pred_corr)
        err_neff = effective_ensemble_size(err_corr)
        # Error diversity should be >= prediction diversity
        assert err_neff >= pred_neff - 0.1  # allow tiny numeric tolerance

    def test_identical_errors_corr_one(self):
        rng = np.random.default_rng(0)
        y = rng.choice([0, 1], 100).astype(float)
        oof = rng.random(100)
        corr = compute_error_correlation_matrix([oof, oof], y)
        np.testing.assert_almost_equal(corr[0, 1], 1.0)


# ---------------------------------------------------------------------------
# effective_ensemble_size
# ---------------------------------------------------------------------------

class TestEffectiveEnsembleSize:
    def test_identical_models(self):
        """N identical models -> N_eff = 1."""
        corr = np.ones((3, 3))
        n_eff = effective_ensemble_size(corr)
        np.testing.assert_almost_equal(n_eff, 1.0, decimal=3)

    def test_uncorrelated_models(self):
        """N uncorrelated models -> N_eff = N."""
        corr = np.eye(4)
        n_eff = effective_ensemble_size(corr)
        np.testing.assert_almost_equal(n_eff, 4.0, decimal=3)

    def test_partial_correlation(self):
        """Partially correlated -> 1 < N_eff < N."""
        corr = np.array([
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        n_eff = effective_ensemble_size(corr)
        assert 1.0 < n_eff < 3.0

    def test_single_model(self):
        corr = np.array([[1.0]])
        n_eff = effective_ensemble_size(corr)
        np.testing.assert_almost_equal(n_eff, 1.0)


# ---------------------------------------------------------------------------
# greedy_diverse_select
# ---------------------------------------------------------------------------

class TestGreedyDiverseSelect:
    def test_selects_diverse_pair(self, diverse_preds):
        oof_list, y = diverse_preds
        # B score must be >= max * min_score_ratio (default 0.95) to be eligible
        scores = [0.8, 0.78, 0.79]  # all above 0.8*0.95=0.76
        selected = greedy_diverse_select(oof_list, scores, n_select=2, y_true=y)
        assert len(selected) == 2
        assert 0 in selected  # best score
        # Should prefer B (diverse from A) over C (near-duplicate of A)
        assert 1 in selected

    def test_selects_diverse_pair_without_y_true(self, diverse_preds):
        """Falls back to prediction correlation when y_true is not provided."""
        oof_list, _ = diverse_preds
        scores = [0.8, 0.78, 0.79]
        selected = greedy_diverse_select(oof_list, scores, n_select=2)
        assert len(selected) == 2
        assert 0 in selected

    def test_respects_min_score_ratio(self, diverse_preds):
        oof_list, _ = diverse_preds
        scores = [0.9, 0.5, 0.88]
        selected = greedy_diverse_select(
            oof_list, scores, n_select=3, min_score_ratio=0.95
        )
        # Model B (score=0.5) is below 95% of 0.9 -> excluded
        assert 1 not in selected

    def test_starts_with_best_score(self, diverse_preds):
        oof_list, _ = diverse_preds
        scores = [0.7, 0.9, 0.8]
        selected = greedy_diverse_select(oof_list, scores, n_select=1)
        assert selected == [1]  # index 1 has best score

    def test_handles_single_model(self):
        preds = [np.array([0.1, 0.5, 0.9])]
        selected = greedy_diverse_select(preds, [0.8], n_select=1)
        assert selected == [0]


# ---------------------------------------------------------------------------
# _weights_to_selection
# ---------------------------------------------------------------------------

class TestWeightsToSelection:
    def test_excludes_below_threshold(self):
        x = np.array([0.5, 0.005, 0.3])
        included, weights = _weights_to_selection(x)
        assert included == [0, 2]
        np.testing.assert_almost_equal(sum(weights), 1.0)

    def test_all_above_threshold(self):
        x = np.array([0.3, 0.4, 0.3])
        included, weights = _weights_to_selection(x)
        assert included == [0, 1, 2]
        np.testing.assert_almost_equal(sum(weights), 1.0)

    def test_all_below_threshold_fallback(self):
        x = np.array([0.001, 0.002, 0.003])
        included, weights = _weights_to_selection(x)
        # Fallback: include all
        assert included == [0, 1, 2]
        np.testing.assert_almost_equal(sum(weights), 1.0)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_basic(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _normalize(arr)
        np.testing.assert_almost_equal(result, [0.0, 0.5, 1.0])

    def test_constant_returns_ones(self):
        arr = np.array([5.0, 5.0, 5.0])
        result = _normalize(arr)
        np.testing.assert_almost_equal(result, [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# select_from_pareto
# ---------------------------------------------------------------------------

class TestSelectFromPareto:
    def test_returns_valid_structure(self, diverse_preds, synthetic_pareto):
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]
        pareto_F, pareto_X = synthetic_pareto

        test_preds, info = select_from_pareto(
            pareto_F, pareto_X,
            oof_list, test_list, y,
            n_models=3,
            diversity_weight=0.3,
            metric="roc_auc",
        )
        assert test_preds.shape == (50,)
        assert "selected_models" in info
        assert "weights" in info
        assert "metric_score" in info
        assert "effective_size" in info
        assert "error_effective_size" in info
        assert "pareto_front" in info

    def test_weights_sum_to_one(self, diverse_preds, synthetic_pareto):
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]
        pareto_F, pareto_X = synthetic_pareto

        _, info = select_from_pareto(
            pareto_F, pareto_X,
            oof_list, test_list, y,
            n_models=3,
            diversity_weight=0.5,
            metric="roc_auc",
        )
        np.testing.assert_almost_equal(sum(info["weights"]), 1.0, decimal=5)

    def test_diversity_weight_0_prefers_metric(self, diverse_preds, synthetic_pareto):
        """diversity_weight=0 should pick the solution with best metric."""
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]
        pareto_F, pareto_X = synthetic_pareto

        _, info_metric = select_from_pareto(
            pareto_F, pareto_X,
            oof_list, test_list, y,
            n_models=3,
            diversity_weight=0.0,
            metric="roc_auc",
        )
        _, info_diverse = select_from_pareto(
            pareto_F, pareto_X,
            oof_list, test_list, y,
            n_models=3,
            diversity_weight=1.0,
            metric="roc_auc",
        )
        # With dw=0, should pick solution 0 (best metric, weight ~0.8 on model 0)
        # With dw=1, should pick solution 4 (best diversity, equal weights)
        assert info_metric["selected_models"] != info_diverse["selected_models"] or \
               info_metric["weights"] != info_diverse["weights"]

    def test_knee_selection(self, diverse_preds, synthetic_pareto):
        """use_knee=True should select a compromise solution."""
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]
        pareto_F, pareto_X = synthetic_pareto

        test_preds, info = select_from_pareto(
            pareto_F, pareto_X,
            oof_list, test_list, y,
            n_models=3,
            diversity_weight=0.3,
            metric="roc_auc",
            use_knee=True,
        )
        assert test_preds.shape == (50,)
        assert len(info["selected_models"]) >= 1
        np.testing.assert_almost_equal(sum(info["weights"]), 1.0, decimal=5)

    def test_single_solution_pareto(self, diverse_preds):
        """Edge case: single solution on Pareto front."""
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]
        pareto_F = np.array([[-0.85, -2.0]])
        pareto_X = np.array([[0.4, 0.35, 0.25]])

        test_preds, info = select_from_pareto(
            pareto_F, pareto_X,
            oof_list, test_list, y,
            n_models=3,
            diversity_weight=0.3,
            metric="roc_auc",
        )
        assert test_preds.shape == (50,)
        assert len(info["selected_models"]) == 3


# ---------------------------------------------------------------------------
# run_nsga2_ensemble (integration)
# ---------------------------------------------------------------------------

class TestRunNsga2Ensemble:
    def test_returns_predictions_and_info(self, diverse_preds):
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]

        test_preds, info = run_nsga2_ensemble(
            oof_list, test_list, y,
            n_trials=50,
            metric="roc_auc",
            diversity_weight=0.3,
            seed=42,
            pop_size=10,
        )
        assert test_preds.shape == (50,)
        assert "selected_models" in info
        assert "weights" in info
        assert "metric_score" in info
        assert "effective_size" in info
        assert "error_effective_size" in info

    def test_weights_sum_to_one(self, diverse_preds):
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]
        _, info = run_nsga2_ensemble(
            oof_list, test_list, y,
            n_trials=50, seed=42, pop_size=10,
        )
        np.testing.assert_almost_equal(sum(info["weights"]), 1.0, decimal=5)

    def test_selects_at_least_one_model(self, diverse_preds):
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]
        _, info = run_nsga2_ensemble(
            oof_list, test_list, y,
            n_trials=50, seed=42, pop_size=10,
        )
        assert len(info["selected_models"]) >= 1

    def test_pareto_data_structure(self, diverse_preds):
        """pareto_trials should be a dict with 'F' and 'X' numpy arrays."""
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]
        _, info = run_nsga2_ensemble(
            oof_list, test_list, y,
            n_trials=50, seed=42, pop_size=10,
        )
        pareto_data = info["pareto_trials"]
        assert isinstance(pareto_data, dict)
        assert "F" in pareto_data
        assert "X" in pareto_data
        assert isinstance(pareto_data["F"], np.ndarray)
        assert isinstance(pareto_data["X"], np.ndarray)
        assert pareto_data["F"].shape[1] == 2
        assert pareto_data["X"].shape[1] == 3  # 3 models

    def test_pareto_reuse_with_different_dw(self, diverse_preds):
        """Pareto data can be reused with select_from_pareto for different dw."""
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]
        _, info = run_nsga2_ensemble(
            oof_list, test_list, y,
            n_trials=50, seed=42, pop_size=10,
        )
        pareto_data = info["pareto_trials"]

        # Re-select with different diversity_weight
        test_preds_2, info_2 = select_from_pareto(
            pareto_data["F"], pareto_data["X"],
            oof_list, test_list, y,
            n_models=3,
            diversity_weight=0.8,
            metric="roc_auc",
        )
        assert test_preds_2.shape == (50,)
        np.testing.assert_almost_equal(sum(info_2["weights"]), 1.0, decimal=5)

    def test_deterministic_with_seed(self, diverse_preds):
        """Same seed should produce same results."""
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]

        _, info1 = run_nsga2_ensemble(
            oof_list, test_list, y,
            n_trials=50, seed=99, pop_size=10,
        )
        _, info2 = run_nsga2_ensemble(
            oof_list, test_list, y,
            n_trials=50, seed=99, pop_size=10,
        )
        assert info1["selected_models"] == info2["selected_models"]
        np.testing.assert_array_almost_equal(info1["weights"], info2["weights"])


# ---------------------------------------------------------------------------
# compute_spearman_correlation_matrix
# ---------------------------------------------------------------------------

class TestComputeSpearmanCorrelationMatrix:
    def test_shape(self, diverse_preds):
        oof_list, _ = diverse_preds
        corr = compute_spearman_correlation_matrix(oof_list)
        assert corr.shape == (3, 3)

    def test_diagonal_is_one(self, diverse_preds):
        oof_list, _ = diverse_preds
        corr = compute_spearman_correlation_matrix(oof_list)
        np.testing.assert_array_almost_equal(np.diag(corr), [1.0, 1.0, 1.0])

    def test_symmetric(self, diverse_preds):
        oof_list, _ = diverse_preds
        corr = compute_spearman_correlation_matrix(oof_list)
        np.testing.assert_array_almost_equal(corr, corr.T)

    def test_identical_predictions_corr_one(self):
        preds = np.random.default_rng(0).random(100)
        corr = compute_spearman_correlation_matrix([preds, preds])
        np.testing.assert_almost_equal(corr[0, 1], 1.0)

    def test_monotonic_transform_high_spearman(self):
        """Spearman should detect monotonic (nonlinear) relationships.

        If model B = exp(model A), Pearson drops but Spearman stays ~1.0
        because Spearman only cares about rank ordering.
        """
        rng = np.random.default_rng(42)
        a = rng.random(200)
        b = np.exp(a * 3)  # monotonic transform
        spearman_corr = compute_spearman_correlation_matrix([a, b])
        pearson_corr = compute_correlation_matrix([a, b])
        # Spearman should be higher than Pearson for nonlinear monotonic
        assert spearman_corr[0, 1] > pearson_corr[0, 1] - 0.01

    def test_constant_oof_produces_zero_not_nan(self):
        constant = np.full(100, 0.5)
        varying = np.random.default_rng(0).random(100)
        corr = compute_spearman_correlation_matrix([constant, varying])
        assert not np.any(np.isnan(corr))
        np.testing.assert_array_almost_equal(np.diag(corr), [1.0, 1.0])


class TestComputeSpearmanErrorCorrelationMatrix:
    def test_shape(self, diverse_preds):
        oof_list, y = diverse_preds
        corr = compute_spearman_error_correlation_matrix(oof_list, y)
        assert corr.shape == (3, 3)

    def test_diagonal_is_one(self, diverse_preds):
        oof_list, y = diverse_preds
        corr = compute_spearman_error_correlation_matrix(oof_list, y)
        np.testing.assert_array_almost_equal(np.diag(corr), [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# compute_ambiguity
# ---------------------------------------------------------------------------

class TestComputeAmbiguity:
    def test_identical_models_zero_ambiguity(self):
        """Identical models disagree on nothing → ambiguity = 0."""
        preds = np.random.default_rng(0).random(100)
        amb = compute_ambiguity([preds, preds], np.array([0.5, 0.5]))
        np.testing.assert_almost_equal(amb, 0.0)

    def test_diverse_models_positive_ambiguity(self, diverse_preds):
        """Diverse models should have positive ambiguity."""
        oof_list, _ = diverse_preds
        weights = np.array([1.0 / 3] * 3)
        amb = compute_ambiguity(oof_list, weights)
        assert amb > 0.0

    def test_more_diverse_higher_ambiguity(self):
        """Adding a diverse model should increase ambiguity."""
        rng = np.random.default_rng(42)
        a = rng.random(200)
        b = a + rng.normal(0, 0.01, 200)  # near-identical
        c = rng.random(200)                 # independent

        amb_similar = compute_ambiguity([a, b], np.array([0.5, 0.5]))
        amb_diverse = compute_ambiguity([a, c], np.array([0.5, 0.5]))
        assert amb_diverse > amb_similar

    def test_single_model_zero_ambiguity(self):
        """Single model has no disagreement."""
        preds = np.random.default_rng(0).random(100)
        amb = compute_ambiguity([preds], np.array([1.0]))
        np.testing.assert_almost_equal(amb, 0.0)


# ---------------------------------------------------------------------------
# _compute_diversity (dispatcher)
# ---------------------------------------------------------------------------

class TestComputeDiversity:
    def test_pearson_neff(self, diverse_preds):
        oof_list, y = diverse_preds
        weights = np.array([1.0 / 3] * 3)
        val = _compute_diversity(oof_list, y, weights, "pearson_neff")
        assert val > 1.0  # 3 models, some diversity

    def test_spearman_neff(self, diverse_preds):
        oof_list, y = diverse_preds
        weights = np.array([1.0 / 3] * 3)
        val = _compute_diversity(oof_list, y, weights, "spearman_neff")
        assert val > 1.0

    def test_ambiguity(self, diverse_preds):
        oof_list, y = diverse_preds
        weights = np.array([1.0 / 3] * 3)
        val = _compute_diversity(oof_list, y, weights, "ambiguity")
        assert val > 0.0

    def test_single_model_neff_returns_one(self):
        preds = [np.random.default_rng(0).random(50)]
        y = np.random.default_rng(1).choice([0, 1], 50).astype(float)
        val = _compute_diversity(preds, y, np.array([1.0]), "pearson_neff")
        assert val == 1.0

    def test_single_model_ambiguity_returns_zero(self):
        preds = [np.random.default_rng(0).random(50)]
        y = np.random.default_rng(1).choice([0, 1], 50).astype(float)
        val = _compute_diversity(preds, y, np.array([1.0]), "ambiguity")
        assert val == 0.0

    def test_unknown_metric_raises(self, diverse_preds):
        oof_list, y = diverse_preds
        weights = np.array([1.0 / 3] * 3)
        with pytest.raises(ValueError, match="Unknown diversity_metric"):
            _compute_diversity(oof_list, y, weights, "invalid_metric")


# ---------------------------------------------------------------------------
# run_nsga2_ensemble with different diversity_metrics
# ---------------------------------------------------------------------------

class TestNSGA2DiversityMetrics:
    """NSGA-II works with all three diversity metrics."""

    @pytest.mark.parametrize("div_metric", DIVERSITY_METRICS)
    def test_nsga2_completes_with_metric(self, diverse_preds, div_metric):
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]

        test_preds, info = run_nsga2_ensemble(
            oof_list, test_list, y,
            n_trials=50, seed=42, pop_size=10,
            diversity_metric=div_metric,
        )
        assert test_preds.shape == (50,)
        assert info["diversity_metric"] == div_metric
        assert len(info["selected_models"]) >= 1
        np.testing.assert_almost_equal(sum(info["weights"]), 1.0, decimal=5)

    def test_spearman_and_pearson_differ_on_nonlinear(self):
        """Spearman and Pearson should produce different N_eff when
        predictions have nonlinear monotonic relationships."""
        rng = np.random.default_rng(42)
        n = 300
        y = rng.choice([0, 1], n).astype(float)
        a = np.clip(y * 0.7 + rng.normal(0, 0.2, n), 0.01, 0.99)
        b = np.clip(a ** 3, 0.01, 0.99)  # nonlinear monotonic

        # Pearson error correlation
        pearson_corr = compute_error_correlation_matrix([a, b], y)
        pearson_neff = effective_ensemble_size(pearson_corr)

        # Spearman error correlation
        spearman_corr = compute_spearman_error_correlation_matrix([a, b], y)
        spearman_neff = effective_ensemble_size(spearman_corr)

        # They should give different values (Spearman detects rank-order)
        assert pearson_neff != pytest.approx(spearman_neff, abs=0.001)


# ---------------------------------------------------------------------------
# log_fold_diversity
# ---------------------------------------------------------------------------

from src.ensemble.diversity import log_fold_diversity


class TestLogFoldDiversity:
    def test_returns_expected_keys(self, diverse_preds):
        oof_list, y = diverse_preds
        # Simulate 5-fold CV indices
        n = len(y)
        fold_size = n // 5
        fold_indices = [
            np.arange(i * fold_size, (i + 1) * fold_size) for i in range(5)
        ]
        weights = [1.0 / 3] * 3

        result = log_fold_diversity(
            oof_list, y, fold_indices, weights, metric="roc_auc"
        )
        assert "fold_neffs" in result
        assert "fold_blend_scores" in result
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        assert "worst_fold" in result

    def test_fold_neffs_length_matches_folds(self, diverse_preds):
        oof_list, y = diverse_preds
        n = len(y)
        fold_indices = [np.arange(i * 40, (i + 1) * 40) for i in range(5)]
        weights = [1.0 / 3] * 3

        result = log_fold_diversity(
            oof_list, y, fold_indices, weights, metric="roc_auc"
        )
        assert len(result["fold_neffs"]) == 5
        assert len(result["fold_blend_scores"]) == 5

    def test_neff_values_positive(self, diverse_preds):
        oof_list, y = diverse_preds
        fold_indices = [np.arange(i * 40, (i + 1) * 40) for i in range(5)]
        weights = [1.0 / 3] * 3

        result = log_fold_diversity(
            oof_list, y, fold_indices, weights, metric="roc_auc"
        )
        for neff in result["fold_neffs"]:
            assert neff >= 1.0

    def test_single_model_returns_empty(self):
        """Single model → early return with empty fold_neffs."""
        rng = np.random.default_rng(42)
        preds = [rng.random(100)]
        y = rng.choice([0, 1], 100).astype(float)
        fold_indices = [np.arange(i * 20, (i + 1) * 20) for i in range(5)]

        result = log_fold_diversity(preds, y, fold_indices, [1.0])
        assert result["fold_neffs"] == []
        assert result["mean"] == 0.0

    def test_mean_std_consistency(self, diverse_preds):
        oof_list, y = diverse_preds
        fold_indices = [np.arange(i * 40, (i + 1) * 40) for i in range(5)]
        weights = [1.0 / 3] * 3

        result = log_fold_diversity(
            oof_list, y, fold_indices, weights, metric="roc_auc"
        )
        np.testing.assert_almost_equal(
            result["mean"], np.mean(result["fold_neffs"])
        )
        np.testing.assert_almost_equal(
            result["std"], np.std(result["fold_neffs"])
        )

    def test_worst_fold_is_min_neff(self, diverse_preds):
        oof_list, y = diverse_preds
        fold_indices = [np.arange(i * 40, (i + 1) * 40) for i in range(5)]
        weights = [1.0 / 3] * 3

        result = log_fold_diversity(
            oof_list, y, fold_indices, weights, metric="roc_auc"
        )
        assert result["worst_fold"] == int(np.argmin(result["fold_neffs"]))

    def test_works_with_neg_rmse(self):
        """log_fold_diversity works for regression metric."""
        rng = np.random.default_rng(42)
        n = 100
        y = rng.random(n) * 10
        a = y + rng.normal(0, 1, n)
        b = y + rng.normal(0, 2, n)
        fold_indices = [np.arange(i * 20, (i + 1) * 20) for i in range(5)]

        result = log_fold_diversity(
            [a, b], y, fold_indices, [0.6, 0.4], metric="neg_rmse"
        )
        assert len(result["fold_neffs"]) == 5
        for score in result["fold_blend_scores"]:
            assert score <= 0  # neg_rmse is always <= 0


# ---------------------------------------------------------------------------
# print_diversity_report
# ---------------------------------------------------------------------------

from src.ensemble.diversity import print_diversity_report


class TestPrintDiversityReport:
    def test_prints_output(self, capsys):
        corr = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ])
        labels = ["CatBoost", "XGBoost", "LightGBM"]
        print_diversity_report(corr, labels)

        captured = capsys.readouterr()
        assert "ENSEMBLE DIVERSITY REPORT" in captured.out
        assert "N_eff" in captured.out
        assert "CatBoost" in captured.out
        assert "XGBoost" in captured.out

    def test_redundant_ensemble_warning(self, capsys):
        """All-ones corr matrix → N_eff ≈ 1 → redundancy warning."""
        corr = np.ones((3, 3))
        labels = ["A", "B", "C"]
        print_diversity_report(corr, labels)

        captured = capsys.readouterr()
        assert "redundant" in captured.out.lower()

    def test_good_diversity_message(self, capsys):
        """Identity corr matrix → N_eff = N → good diversity."""
        corr = np.eye(4)
        labels = ["A", "B", "C", "D"]
        print_diversity_report(corr, labels)

        captured = capsys.readouterr()
        assert "Good diversity" in captured.out

    def test_most_least_correlated_pairs(self, capsys):
        corr = np.array([
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.3],
            [0.1, 0.3, 1.0],
        ])
        labels = ["A", "B", "C"]
        print_diversity_report(corr, labels)

        captured = capsys.readouterr()
        assert "Most correlated pair" in captured.out
        assert "Least correlated pair" in captured.out
        # Most correlated should be A & B (r=0.9)
        assert "A" in captured.out and "B" in captured.out

    def test_single_model(self, capsys):
        """Single model → no pair info, just N_eff."""
        corr = np.array([[1.0]])
        labels = ["Solo"]
        print_diversity_report(corr, labels)

        captured = capsys.readouterr()
        assert "N_eff" in captured.out
        assert "Most correlated pair" not in captured.out

    def test_output_to_file(self, tmp_path, capsys):
        """When output_path is given, full report goes to file, console is concise."""
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        labels = ["A", "B"]
        report_path = tmp_path / "diversity_report.txt"
        print_diversity_report(corr, labels, output_path=report_path)

        # File has full correlation matrix
        content = report_path.read_text()
        assert "ENSEMBLE DIVERSITY REPORT" in content
        assert "N_eff" in content
        assert "0.800" in content  # correlation value

        # Console has concise summary
        captured = capsys.readouterr()
        assert "N_eff" in captured.out
        assert "Full report:" in captured.out
        # Full matrix NOT on console
        assert "ENSEMBLE DIVERSITY REPORT" not in captured.out


# ---------------------------------------------------------------------------
# greedy_diverse_select edge cases
# ---------------------------------------------------------------------------

class TestGreedyDiverseSelectEdgeCases:
    def test_negative_scores_min_score_ratio(self):
        """REGRESSION BUG: min_score_ratio breaks for negative metrics.

        When best score is negative (e.g., neg_rmse = -0.5),
        min_score = -0.5 * 0.95 = -0.475 > -0.5, filtering out the
        best model itself. Falls back to all models, ignoring the ratio.
        """
        rng = np.random.default_rng(42)
        a = rng.random(100)
        b = rng.random(100)
        c = rng.random(100)
        oof_list = [a, b, c]
        # Negative scores (neg_rmse-style): -0.5 is best, -2.0 is worst
        scores = [-0.5, -0.8, -2.0]

        selected = greedy_diverse_select(
            oof_list, scores, n_select=2, min_score_ratio=0.95
        )
        # With correct implementation: model 2 (score=-2.0) should be excluded
        # because it's far worse than -0.5.
        # BUG: min_score = -0.5 * 0.95 = -0.475, which is HIGHER than -0.5,
        # so ALL models fail the threshold and fallback includes all.
        # This test documents the bug — model 2 SHOULD be excluded.
        assert 2 not in selected, (
            "Model with score=-2.0 should be excluded at 95% ratio of best=-0.5"
        )


# ---------------------------------------------------------------------------
# _score_metric
# ---------------------------------------------------------------------------

from src.ensemble.diversity import _score_metric


class TestScoreMetric:
    def test_roc_auc(self):
        y = np.array([0, 0, 1, 1])
        preds = np.array([0.1, 0.2, 0.8, 0.9])
        score = _score_metric(y, preds, "roc_auc")
        assert score == pytest.approx(1.0)

    def test_neg_rmse(self):
        y = np.array([1.0, 2.0, 3.0])
        preds = np.array([1.0, 2.0, 3.0])
        score = _score_metric(y, preds, "neg_rmse")
        assert score == pytest.approx(0.0)

    def test_neg_rmse_nonzero(self):
        y = np.array([1.0, 2.0, 3.0])
        preds = np.array([1.5, 2.5, 3.5])
        score = _score_metric(y, preds, "neg_rmse")
        assert score < 0.0

    def test_unknown_metric_fallback_classification(self):
        """Unknown metric falls back to roc_auc for binary data."""
        y = np.array([0, 0, 1, 1])
        preds = np.array([0.1, 0.2, 0.8, 0.9])
        score = _score_metric(y, preds, "unknown_metric")
        assert score == pytest.approx(1.0)

    def test_unknown_metric_fallback_regression(self):
        """Unknown metric falls back to neg_rmse when roc_auc fails."""
        y = np.array([1.0, 2.5, 3.7, 4.2])
        preds = np.array([1.1, 2.4, 3.8, 4.1])
        score = _score_metric(y, preds, "unknown_metric")
        # y values not binary, roc_auc fails → neg_rmse
        assert score <= 0.0


# ---------------------------------------------------------------------------
# compute_spearman_correlation_matrix — single model edge case
# ---------------------------------------------------------------------------

class TestSpearmanSingleModel:
    def test_single_model_returns_1x1(self):
        """Single model returns 1x1 matrix with 1.0 on diagonal."""
        preds = np.random.default_rng(0).random(100)
        corr = compute_spearman_correlation_matrix([preds])
        assert corr.shape == (1, 1)
        assert corr[0, 0] == 1.0


# ---------------------------------------------------------------------------
# select_from_pareto — two-solution edge case (< 3 → no knee)
# ---------------------------------------------------------------------------

class TestSelectFromParetoTwoSolutions:
    def test_two_solutions_no_knee(self, diverse_preds):
        """With < 3 solutions, knee selection is skipped → linear weighting."""
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]
        pareto_F = np.array([[-0.90, -1.0], [-0.80, -3.0]])
        pareto_X = np.array([[0.8, 0.1, 0.1], [0.33, 0.33, 0.34]])

        test_preds, info = select_from_pareto(
            pareto_F, pareto_X,
            oof_list, test_list, y,
            n_models=3,
            diversity_weight=0.5,
            metric="roc_auc",
            use_knee=True,  # should be ignored since < 3 solutions
        )
        assert test_preds.shape == (50,)
        assert len(info["selected_models"]) >= 1


# ---------------------------------------------------------------------------
# _EnsembleProblem — direct evaluation
# ---------------------------------------------------------------------------

from src.ensemble.diversity import _EnsembleProblem, _WEIGHT_THRESHOLD


class TestEnsembleProblem:
    def test_evaluate_normal(self):
        """Normal evaluation produces two negative objectives."""
        rng = np.random.default_rng(42)
        y = rng.choice([0, 1], 100).astype(float)
        a = np.clip(y * 0.7 + rng.normal(0, 0.2, 100), 0.01, 0.99)
        b = np.clip(y * 0.6 + rng.normal(0, 0.25, 100), 0.01, 0.99)

        problem = _EnsembleProblem([a, b], y, "roc_auc")
        out = {}
        problem._evaluate(np.array([0.6, 0.4]), out)
        assert "F" in out
        assert out["F"].shape == (2,)
        assert out["F"][0] < 0  # negated AUC
        assert out["F"][1] < 0  # negated N_eff

    def test_evaluate_all_below_threshold(self):
        """All weights below threshold → penalty output."""
        rng = np.random.default_rng(42)
        y = rng.choice([0, 1], 50).astype(float)
        a = rng.random(50)

        problem = _EnsembleProblem([a], y, "roc_auc")
        out = {}
        problem._evaluate(np.array([_WEIGHT_THRESHOLD / 2]), out)
        assert "F" in out
        np.testing.assert_array_equal(out["F"], [1e6, 1e6])


# ---------------------------------------------------------------------------
# greedy_diverse_select — request more than available
# ---------------------------------------------------------------------------

class TestGreedyDiverseSelectRequestMore:
    def test_n_select_exceeds_candidates(self, diverse_preds):
        """When n_select > number of candidates, returns all available."""
        oof_list, y = diverse_preds
        scores = [0.8, 0.78, 0.79]
        selected = greedy_diverse_select(
            oof_list, scores, n_select=10, y_true=y,
        )
        assert len(selected) == 3  # only 3 models available


# ---------------------------------------------------------------------------
# Pre-ranked Spearman optimization in _EnsembleProblem
# ---------------------------------------------------------------------------

class TestPrerankedSpearman:
    """Verify that pre-ranking errors in _EnsembleProblem produces correct
    Spearman-based diversity (Pearson on ranks == Spearman)."""

    def test_preranked_errors_created_for_spearman(self):
        """_EnsembleProblem should pre-rank errors when diversity_metric=spearman_neff."""
        rng = np.random.default_rng(42)
        y = rng.choice([0, 1], 100).astype(float)
        a = np.clip(y * 0.7 + rng.normal(0, 0.2, 100), 0.01, 0.99)
        b = np.clip(y * 0.6 + rng.normal(0, 0.25, 100), 0.01, 0.99)

        problem = _EnsembleProblem([a, b], y, "roc_auc", diversity_metric="spearman_neff")
        assert problem._preranked_errors is not None
        assert len(problem._preranked_errors) == 2
        assert problem._preranked_errors[0].shape == (100,)

    def test_no_preranking_for_pearson(self):
        """_EnsembleProblem should NOT pre-rank for pearson_neff."""
        rng = np.random.default_rng(42)
        y = rng.choice([0, 1], 100).astype(float)
        a = rng.random(100)

        problem = _EnsembleProblem([a], y, "roc_auc", diversity_metric="pearson_neff")
        assert problem._preranked_errors is None

    def test_no_preranking_for_ambiguity(self):
        """_EnsembleProblem should NOT pre-rank for ambiguity."""
        rng = np.random.default_rng(42)
        y = rng.choice([0, 1], 100).astype(float)
        a = rng.random(100)

        problem = _EnsembleProblem([a], y, "roc_auc", diversity_metric="ambiguity")
        assert problem._preranked_errors is None

    def test_preranked_diversity_matches_direct_spearman(self):
        """Pre-ranked Pearson should give the same N_eff as direct Spearman."""
        rng = np.random.default_rng(42)
        n = 200
        y = rng.choice([0, 1], n).astype(float)
        a = np.clip(y * 0.7 + rng.normal(0, 0.2, n), 0.01, 0.99)
        b = np.clip(y * 0.6 + rng.normal(0, 0.25, n), 0.01, 0.99)
        c = np.clip(rng.random(n), 0.01, 0.99)

        # Direct Spearman N_eff via _compute_diversity
        direct_neff = _compute_diversity([a, b, c], y, np.array([1/3]*3), "spearman_neff")

        # Pre-ranked approach (what _EnsembleProblem does internally)
        from scipy.stats import rankdata
        errors = [oof - y for oof in [a, b, c]]
        ranked = [rankdata(err) for err in errors]
        corr = compute_correlation_matrix(ranked)
        preranked_neff = effective_ensemble_size(corr)

        np.testing.assert_almost_equal(preranked_neff, direct_neff, decimal=4)

    def test_nsga2_evaluate_spearman_uses_preranked(self):
        """Evaluate with spearman_neff should use pre-ranked path and return valid F."""
        rng = np.random.default_rng(42)
        y = rng.choice([0, 1], 100).astype(float)
        a = np.clip(y * 0.7 + rng.normal(0, 0.2, 100), 0.01, 0.99)
        b = np.clip(y * 0.6 + rng.normal(0, 0.25, 100), 0.01, 0.99)

        problem = _EnsembleProblem([a, b], y, "roc_auc", diversity_metric="spearman_neff")
        out = {}
        problem._evaluate(np.array([0.6, 0.4]), out)

        assert "F" in out
        assert out["F"].shape == (2,)
        assert out["F"][0] < 0  # negated AUC
        assert out["F"][1] < 0  # negated N_eff

    def test_nsga2_spearman_single_model_returns_neff_one(self):
        """Single model in spearman_neff should return N_eff=1.0."""
        rng = np.random.default_rng(42)
        y = rng.choice([0, 1], 100).astype(float)
        a = np.clip(y * 0.7 + rng.normal(0, 0.2, 100), 0.01, 0.99)
        b = np.clip(y * 0.6 + rng.normal(0, 0.25, 100), 0.01, 0.99)

        problem = _EnsembleProblem([a, b], y, "roc_auc", diversity_metric="spearman_neff")
        out = {}
        # Only model 0 above threshold
        problem._evaluate(np.array([0.5, 0.005]), out)

        assert out["F"][1] == -1.0  # negated N_eff for single model
