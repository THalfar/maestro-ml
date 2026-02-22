"""Tests for src/ensemble/diversity.py — NSGA-II ensemble diversity."""
from __future__ import annotations

import numpy as np
import pytest

from src.ensemble.diversity import (
    compute_correlation_matrix,
    effective_ensemble_size,
    greedy_diverse_select,
    run_nsga2_ensemble,
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
# effective_ensemble_size
# ---------------------------------------------------------------------------

class TestEffectiveEnsembleSize:
    def test_identical_models(self):
        """N identical models → N_eff = 1."""
        corr = np.ones((3, 3))
        n_eff = effective_ensemble_size(corr)
        np.testing.assert_almost_equal(n_eff, 1.0, decimal=3)

    def test_uncorrelated_models(self):
        """N uncorrelated models → N_eff = N."""
        corr = np.eye(4)
        n_eff = effective_ensemble_size(corr)
        np.testing.assert_almost_equal(n_eff, 4.0, decimal=3)

    def test_partial_correlation(self):
        """Partially correlated → 1 < N_eff < N."""
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
        oof_list, _ = diverse_preds
        # B score must be >= max * min_score_ratio (default 0.95) to be eligible
        scores = [0.8, 0.78, 0.79]  # all above 0.8*0.95=0.76
        selected = greedy_diverse_select(oof_list, scores, n_select=2)
        assert len(selected) == 2
        assert 0 in selected  # best score
        # Should prefer B (diverse from A) over C (near-duplicate of A)
        assert 1 in selected

    def test_respects_min_score_ratio(self, diverse_preds):
        oof_list, _ = diverse_preds
        scores = [0.9, 0.5, 0.88]
        selected = greedy_diverse_select(
            oof_list, scores, n_select=3, min_score_ratio=0.95
        )
        # Model B (score=0.5) is below 95% of 0.9 → excluded
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
# run_nsga2_ensemble
# ---------------------------------------------------------------------------

class TestRunNsga2Ensemble:
    def test_returns_predictions_and_info(self, diverse_preds):
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]

        test_preds, info = run_nsga2_ensemble(
            oof_list, test_list, y,
            n_trials=20,
            metric="roc_auc",
            diversity_weight=0.3,
            seed=42,
        )
        assert test_preds.shape == (50,)
        assert "selected_models" in info
        assert "weights" in info
        assert "metric_score" in info
        assert "effective_size" in info

    def test_weights_sum_to_one(self, diverse_preds):
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]
        _, info = run_nsga2_ensemble(
            oof_list, test_list, y,
            n_trials=20, seed=42,
        )
        np.testing.assert_almost_equal(sum(info["weights"]), 1.0, decimal=5)

    def test_selects_at_least_one_model(self, diverse_preds):
        oof_list, y = diverse_preds
        test_list = [np.random.default_rng(i).random(50) for i in range(3)]
        _, info = run_nsga2_ensemble(
            oof_list, test_list, y,
            n_trials=20, seed=42,
        )
        assert len(info["selected_models"]) >= 1
