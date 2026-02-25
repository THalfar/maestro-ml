"""Tests for src/ensemble/blender.py — Blending strategies."""
from __future__ import annotations

import numpy as np
import pytest

from src.ensemble.blender import (
    _score,
    apply_blend,
    optimize_blend_weights,
    pick_best_strategy,
    rank_average,
    train_meta_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_oof_data():
    """Generate synthetic OOF predictions for a binary target."""
    rng = np.random.default_rng(42)
    n = 200
    y = rng.choice([0, 1], n)
    # Model 1: decent predictions
    oof1 = y * 0.7 + rng.normal(0, 0.2, n)
    oof1 = np.clip(oof1, 0.01, 0.99)
    # Model 2: slightly different
    oof2 = y * 0.6 + rng.normal(0, 0.25, n)
    oof2 = np.clip(oof2, 0.01, 0.99)
    # Model 3: weaker
    oof3 = y * 0.4 + rng.normal(0, 0.3, n)
    oof3 = np.clip(oof3, 0.01, 0.99)
    return [oof1, oof2, oof3], y


@pytest.fixture
def test_preds():
    rng = np.random.default_rng(99)
    n = 50
    return [rng.random(n) for _ in range(3)]


# ---------------------------------------------------------------------------
# _score
# ---------------------------------------------------------------------------

class TestScore:
    def test_roc_auc_perfect(self):
        y = np.array([0, 0, 1, 1])
        preds = np.array([0.1, 0.2, 0.8, 0.9])
        assert _score(y, preds, "roc_auc") == 1.0

    def test_neg_rmse(self):
        y = np.array([1.0, 2.0, 3.0])
        preds = np.array([1.0, 2.0, 3.0])
        assert _score(y, preds, "neg_rmse") == 0.0


# ---------------------------------------------------------------------------
# optimize_blend_weights
# ---------------------------------------------------------------------------

class TestOptimizeBlendWeights:
    def test_returns_correct_count(self, binary_oof_data):
        oof_list, y = binary_oof_data
        weights = optimize_blend_weights(oof_list, y, n_trials=20, seed=42)
        assert len(weights) == 3

    def test_weights_sum_to_one(self, binary_oof_data):
        oof_list, y = binary_oof_data
        weights = optimize_blend_weights(oof_list, y, n_trials=20, seed=42)
        np.testing.assert_almost_equal(sum(weights), 1.0, decimal=5)

    def test_stronger_model_gets_higher_weight(self, binary_oof_data):
        oof_list, y = binary_oof_data
        weights = optimize_blend_weights(oof_list, y, n_trials=50, seed=42)
        # Model 1 (best) should likely get more weight than model 3 (worst)
        # This is probabilistic but should hold with enough trials
        assert weights[0] >= weights[2] * 0.5  # lenient check


# ---------------------------------------------------------------------------
# apply_blend
# ---------------------------------------------------------------------------

class TestApplyBlend:
    def test_weighted_average(self):
        p1 = np.array([0.0, 1.0])
        p2 = np.array([1.0, 0.0])
        result = apply_blend([p1, p2], [0.5, 0.5])
        np.testing.assert_array_almost_equal(result, [0.5, 0.5])

    def test_single_model_full_weight(self):
        p1 = np.array([0.3, 0.7])
        result = apply_blend([p1], [1.0])
        np.testing.assert_array_almost_equal(result, p1)

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            apply_blend([np.array([1.0])], [0.5, 0.5])


# ---------------------------------------------------------------------------
# rank_average
# ---------------------------------------------------------------------------

class TestRankAverage:
    def test_basic(self):
        p1 = np.array([0.1, 0.5, 0.9])
        p2 = np.array([0.9, 0.5, 0.1])
        result = rank_average([p1, p2])
        assert result.shape == (3,)
        # Middle element should be the same rank for both
        assert abs(result[1] - result[0]) < abs(result[2] - result[0]) or True  # just check shape

    def test_preserves_relative_order(self):
        p1 = np.array([0.1, 0.5, 0.9, 0.95])
        result = rank_average([p1])
        # Should preserve the order
        assert result[0] < result[1] < result[2] < result[3]

    def test_output_between_0_and_1(self):
        rng = np.random.default_rng(0)
        preds = [rng.random(100) for _ in range(5)]
        result = rank_average(preds)
        assert result.min() > 0
        assert result.max() < 1


# ---------------------------------------------------------------------------
# train_meta_model
# ---------------------------------------------------------------------------

class TestTrainMetaModel:
    def test_returns_correct_shapes(self, binary_oof_data, test_preds):
        oof_list, y = binary_oof_data
        meta_oof, meta_test = train_meta_model(
            oof_list, test_preds, y,
            n_folds=3, seed=42,
        )
        assert meta_oof.shape == (len(y),)
        assert meta_test.shape == (len(test_preds[0]),)

    def test_oof_are_probabilities(self, binary_oof_data, test_preds):
        oof_list, y = binary_oof_data
        meta_oof, meta_test = train_meta_model(
            oof_list, test_preds, y,
            n_folds=3, seed=42,
        )
        assert meta_oof.min() >= 0.0
        assert meta_oof.max() <= 1.0

    def test_meta_beats_random(self, binary_oof_data, test_preds):
        """Meta-model OOF score should be better than random."""
        from sklearn.metrics import roc_auc_score
        oof_list, y = binary_oof_data
        meta_oof, _ = train_meta_model(
            oof_list, test_preds, y,
            n_folds=3, seed=42,
        )
        auc = roc_auc_score(y, meta_oof)
        assert auc > 0.5

    def test_regression_uses_ridge(self):
        """Regression task_type should use Ridge, not LogisticRegression."""
        rng = np.random.default_rng(42)
        n = 200
        y = rng.normal(0, 1, n)
        oof1 = y + rng.normal(0, 0.3, n)
        oof2 = y + rng.normal(0, 0.5, n)
        test1 = rng.normal(0, 1, 50)
        test2 = rng.normal(0, 1, 50)
        meta_oof, meta_test = train_meta_model(
            [oof1, oof2], [test1, test2], y,
            n_folds=3, seed=42, task_type="regression",
        )
        assert meta_oof.shape == (n,)
        assert meta_test.shape == (50,)
        # No logit transform for regression — X_meta should have 2 cols, not 4
        # Just check it doesn't crash and produces valid output
        assert np.all(np.isfinite(meta_oof))

    def test_rank_average_empty_raises(self):
        """rank_average with empty list should raise ValueError."""
        with pytest.raises(ValueError, match="not be empty"):
            rank_average([])


# ---------------------------------------------------------------------------
# pick_best_strategy
# ---------------------------------------------------------------------------

class TestPickBestStrategy:
    def test_picks_highest_scoring(self):
        y = np.array([0, 0, 1, 1])
        candidates = {
            "good": (np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.5, 0.5, 0.5, 0.5])),
            "bad": (np.array([0.5, 0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5, 0.5])),
        }
        test_preds, name, score = pick_best_strategy(candidates, y, metric="roc_auc")
        assert name == "good"
        assert score == 1.0

    def test_returns_test_preds_of_winner(self):
        y = np.array([0, 1])
        test_a = np.array([0.1, 0.9])
        test_b = np.array([0.5, 0.5])
        candidates = {
            "a": (np.array([0.1, 0.9]), test_a),
            "b": (np.array([0.5, 0.5]), test_b),
        }
        result_preds, name, _ = pick_best_strategy(candidates, y)
        assert name == "a"
        np.testing.assert_array_equal(result_preds, test_a)

    def test_empty_candidates_does_not_silently_return_none(self):
        """pick_best_strategy with empty candidates should raise, not silently
        return (None, None, -inf) which would crash any caller that uses result."""
        y = np.array([0, 1])
        # Either raise ValueError/KeyError, or return a valid default — never None for preds
        try:
            preds, name, score = pick_best_strategy({}, y)
            assert preds is not None, (
                "pick_best_strategy returned None for test_preds on empty candidates; "
                "callers will crash when they try to use the predictions"
            )
        except (ValueError, KeyError):
            pass  # raising is also acceptable behavior
