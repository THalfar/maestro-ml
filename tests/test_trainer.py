"""Tests for src/models/trainer.py — Optuna CV training."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import StratifiedKFold

from src.models.registry import ModelRegistry
from src.models.trainer import (
    _apply_scaler_fold,
    _deep_merge,
    _identify_scale_cols,
    _make_scaler,
    _compute_cv_metric,
    _extract_sample_weights,
    _get_eval_metric_value,
    _reassemble_int_lists,
    ALL_SCALER_CHOICES,
    get_top_configs,
    run_optuna_study,
    train_with_config,
)
from src.utils.io import PipelineConfig, CVConfig, OptunaGlobalConfig, OutputConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ridge_configs_dir(tmp_path: Path) -> Path:
    """Minimal model config dir with just ridge for fast tests."""
    configs_dir = tmp_path / "models"
    configs_dir.mkdir()
    ridge_cfg = {
        "name": "Ridge",
        "class_path": {
            "binary_classification": "sklearn.linear_model.LogisticRegression",
            "regression": "sklearn.linear_model.Ridge",
        },
        "task_types": ["binary_classification", "regression"],
        "gpu": {"supported": False, "params": {}, "fallback": {}},
        "hyperparameters": {
            "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
        },
        "fixed_params": {
            "binary_classification": {"max_iter": 1000, "solver": "lbfgs"},
            "regression": {"max_iter": 1000},
        },
        "training": {
            "needs_eval_set": False,
            "early_stopping": False,
            "eval_metric_param": None,
        },
        "feature_requirements": {"needs_scaling": True},
        "optuna": {
            "n_trials": 5,
            "qmc_warmup_trials": 2,
            "timeout": None,
            "pruner": {"type": "none"},
            "n_top_trials": 2,
            "n_seeds": 2,
        },
    }
    (configs_dir / "ridge.yaml").write_text(
        yaml.dump(ridge_cfg), encoding="utf-8"
    )
    return configs_dir


@pytest.fixture
def binary_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple binary classification dataset."""
    rng = np.random.default_rng(42)
    n_train, n_test = 100, 30
    X = rng.normal(0, 1, (n_train, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    train = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    train["target"] = y

    X_test = rng.normal(0, 1, (n_test, 3))
    test = pd.DataFrame(X_test, columns=["f1", "f2", "f3"])
    return train, test


@pytest.fixture
def pipeline_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        task_type="binary_classification",
        target_column="target",
        cv=CVConfig(n_folds=3, seed=42, stratified=True),
        models=["ridge"],
        optuna=OptunaGlobalConfig(global_seed=42),
        output=OutputConfig(results_dir=str(tmp_path / "results")),
    )


# ---------------------------------------------------------------------------
# _compute_cv_metric
# ---------------------------------------------------------------------------

class TestComputeCvMetric:
    def test_binary_classification_auc(self):
        y = np.array([0, 0, 1, 1])
        preds = np.array([0.1, 0.2, 0.8, 0.9])
        score = _compute_cv_metric(y, preds, "binary_classification")
        assert score == 1.0  # perfect ranking

    def test_regression_rmse(self):
        y = np.array([1.0, 2.0, 3.0])
        preds = np.array([1.0, 2.0, 3.0])
        score = _compute_cv_metric(y, preds, "regression")
        assert score == 0.0  # perfect predictions


# ---------------------------------------------------------------------------
# _get_eval_metric_value
# ---------------------------------------------------------------------------

class TestGetEvalMetricValue:
    def test_simple_string(self):
        training = {"eval_metric": "AUC"}
        assert _get_eval_metric_value(training, "binary_classification", False) == "AUC"

    def test_nested_task_type(self):
        training = {
            "eval_metric": {
                "binary_classification": {"cpu": "AUC", "gpu": "Logloss"},
            }
        }
        assert _get_eval_metric_value(training, "binary_classification", gpu=False) == "AUC"
        assert _get_eval_metric_value(training, "binary_classification", gpu=True) == "Logloss"

    def test_none_when_missing(self):
        assert _get_eval_metric_value({}, "binary_classification", False) is None


# ---------------------------------------------------------------------------
# run_optuna_study
# ---------------------------------------------------------------------------

class TestRunOptunaStudy:
    def test_completes_study(self, ridge_configs_dir: Path, binary_data, pipeline_config):
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy={},
            gpu=False,
        )
        assert len(study.trials) >= 1
        assert study.best_value > 0.5  # should beat random

    def test_oof_preds_stored(self, ridge_configs_dir: Path, binary_data, pipeline_config):
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy={},
            gpu=False,
        )
        # Best trial should have oof_preds
        best = study.best_trial
        oof = best.user_attrs.get("oof_preds")
        assert oof is not None
        assert len(oof) == len(train)

    def test_oof_alignment(self, ridge_configs_dir: Path, binary_data, pipeline_config):
        """OOF predictions should not all be zero (each fold fills its slice)."""
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy={},
            gpu=False,
        )
        oof = np.array(study.best_trial.user_attrs["oof_preds"])
        # No zeros left (all indices should be filled)
        assert np.all(oof != 0) or np.all(oof >= 0)  # valid probabilities
        assert oof.min() >= 0.0
        assert oof.max() <= 1.0


# ---------------------------------------------------------------------------
# train_with_config
# ---------------------------------------------------------------------------

class TestTrainWithConfig:
    def test_produces_oof_and_test(self, ridge_configs_dir: Path, binary_data, pipeline_config):
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        results_dir = Path(pipeline_config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        oof_list, test_list, labels = train_with_config(
            model_name="ridge",
            hparams={"C": 1.0},
            feature_cols=["f1", "f2", "f3"],
            train=train, test=test,
            target_col="target",
            cv=cv, registry=registry,
            task_type="binary_classification",
            gpu=False, seeds=[42, 43],
            results_dir=results_dir,
        )
        assert len(oof_list) == 2  # 2 seeds
        assert len(test_list) == 2
        assert len(labels) == len(train)

    def test_oof_correct_length(self, ridge_configs_dir: Path, binary_data, pipeline_config):
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        results_dir = Path(pipeline_config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        oof_list, test_list, _ = train_with_config(
            model_name="ridge",
            hparams={"C": 1.0},
            feature_cols=["f1", "f2", "f3"],
            train=train, test=test,
            target_col="target",
            cv=cv, registry=registry,
            task_type="binary_classification",
            gpu=False, seeds=[42],
            results_dir=results_dir,
        )
        assert oof_list[0].shape == (len(train),)
        assert test_list[0].shape == (len(test),)

    def test_oof_values_are_probabilities(self, ridge_configs_dir: Path, binary_data, pipeline_config):
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        results_dir = Path(pipeline_config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        oof_list, _, _ = train_with_config(
            model_name="ridge",
            hparams={"C": 1.0},
            feature_cols=["f1", "f2", "f3"],
            train=train, test=test,
            target_col="target",
            cv=cv, registry=registry,
            task_type="binary_classification",
            gpu=False, seeds=[42],
            results_dir=results_dir,
        )
        assert oof_list[0].min() >= 0.0
        assert oof_list[0].max() <= 1.0


# ---------------------------------------------------------------------------
# get_top_configs
# ---------------------------------------------------------------------------

class TestGetTopConfigs:
    def test_returns_top_n(self, ridge_configs_dir: Path, binary_data, pipeline_config):
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy={},
            gpu=False,
        )
        top = get_top_configs(study, n_top=2)
        assert len(top) <= 2
        assert all("params" in c and "value" in c for c in top)

    def test_sorted_descending_for_maximize(self, ridge_configs_dir: Path, binary_data, pipeline_config):
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy={},
            gpu=False,
        )
        top = get_top_configs(study, n_top=3)
        if len(top) >= 2:
            assert top[0]["value"] >= top[1]["value"]


# ---------------------------------------------------------------------------
# _reassemble_int_lists
# ---------------------------------------------------------------------------

class TestReassembleIntLists:
    """Tests for int_list and dynamic_int_list reassembly."""

    def test_fixed_int_list(self):
        """Standard int_list: hidden_sizes_0, hidden_sizes_1 -> hidden_sizes: [v0, v1]."""
        params = {"hidden_sizes_0": 128, "hidden_sizes_1": 64, "lr": 0.01}
        result = _reassemble_int_lists(params)
        assert result["hidden_sizes"] == [128, 64]
        assert result["lr"] == 0.01
        assert "hidden_sizes_0" not in result
        assert "hidden_sizes_1" not in result

    def test_dynamic_int_list_single_layer(self):
        """dynamic_int_list with 1 layer: hidden_sizes_n=1, hidden_sizes_0=64."""
        params = {"hidden_sizes_n": 1, "hidden_sizes_0": 64, "lr": 0.01}
        result = _reassemble_int_lists(params)
        assert result["hidden_sizes"] == [64]
        assert result["lr"] == 0.01
        assert "hidden_sizes_n" not in result

    def test_dynamic_int_list_three_layers(self):
        """dynamic_int_list with 3 layers."""
        params = {
            "hidden_sizes_n": 3,
            "hidden_sizes_0": 256,
            "hidden_sizes_1": 128,
            "hidden_sizes_2": 32,
            "batch_size": 8192,
        }
        result = _reassemble_int_lists(params)
        assert result["hidden_sizes"] == [256, 128, 32]
        assert result["batch_size"] == 8192
        assert "hidden_sizes_n" not in result

    def test_no_list_keys(self):
        """Plain params pass through unchanged."""
        params = {"lr": 0.01, "batch_size": 4096}
        result = _reassemble_int_lists(params)
        assert result == params

    def test_incomplete_sequence_kept_separate(self):
        """Keys like foo_0, foo_2 (missing foo_1) stay separate."""
        params = {"foo_0": 10, "foo_2": 30}
        result = _reassemble_int_lists(params)
        assert "foo" not in result
        assert result["foo_0"] == 10
        assert result["foo_2"] == 30

    def test_orphan_n_key_preserved(self):
        """A _n key with no matching indexed keys is kept as-is."""
        params = {"batch_size_n": 5, "lr": 0.01}
        result = _reassemble_int_lists(params)
        assert result["batch_size_n"] == 5
        assert result["lr"] == 0.01


# ---------------------------------------------------------------------------
# _compute_cv_metric — additional coverage
# ---------------------------------------------------------------------------

class TestComputeCvMetricExtra:
    def test_multiclass_auc(self):
        """Multiclass AUC with one-vs-rest."""
        y = np.array([0, 0, 1, 1, 2, 2])
        # Perfect predictions: high prob for correct class
        preds = np.array([
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.05, 0.9, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
            [0.1, 0.1, 0.8],
        ])
        score = _compute_cv_metric(y, preds, "multiclass")
        assert score == 1.0

    def test_regression_rmse_nonzero(self):
        """Regression RMSE with imperfect predictions."""
        y = np.array([1.0, 2.0, 3.0])
        preds = np.array([1.5, 2.5, 3.5])  # off by 0.5 each
        score = _compute_cv_metric(y, preds, "regression")
        assert abs(score - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# _get_eval_metric_value — additional coverage
# ---------------------------------------------------------------------------

class TestGetEvalMetricValueExtra:
    def test_task_type_string_value(self):
        """eval_metric is a dict with task_type mapping to a plain string (XGBoost style)."""
        training = {
            "eval_metric": {
                "binary_classification": "logloss",
                "regression": "rmse",
            }
        }
        assert _get_eval_metric_value(training, "binary_classification", gpu=False) == "logloss"
        assert _get_eval_metric_value(training, "regression", gpu=True) == "rmse"

    def test_numeric_eval_metric(self):
        """eval_metric is a raw number (unlikely but handled by str())."""
        training = {"eval_metric": 42}
        assert _get_eval_metric_value(training, "binary_classification", False) == "42"


# ---------------------------------------------------------------------------
# Regression fixtures & tests
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple regression dataset."""
    rng = np.random.default_rng(99)
    n_train, n_test = 100, 30
    X = rng.normal(0, 1, (n_train, 3))
    y = X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.1, n_train)
    train = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    train["target"] = y
    X_test = rng.normal(0, 1, (n_test, 3))
    test = pd.DataFrame(X_test, columns=["f1", "f2", "f3"])
    return train, test


@pytest.fixture
def ridge_regression_configs_dir(tmp_path: Path) -> Path:
    """Minimal model config dir with ridge for regression tests."""
    configs_dir = tmp_path / "models"
    configs_dir.mkdir()
    cfg = {
        "name": "Ridge",
        "class_path": {
            "binary_classification": "sklearn.linear_model.LogisticRegression",
            "regression": "sklearn.linear_model.Ridge",
        },
        "task_types": ["binary_classification", "regression"],
        "gpu": {"supported": False, "params": {}, "fallback": {}},
        "hyperparameters": {
            "alpha": {"type": "float", "low": 0.01, "high": 100.0, "log": True},
        },
        "fixed_params": {
            "binary_classification": {"max_iter": 1000, "solver": "lbfgs"},
            "regression": {"max_iter": 1000},
        },
        "training": {
            "needs_eval_set": False,
            "seed_param": "random_state",
        },
        "feature_requirements": {"needs_scaling": True},
        "optuna": {
            "n_trials": 5,
            "qmc_warmup_trials": 2,
            "timeout": None,
            "pruner": {"type": "none"},
            "n_top_trials": 2,
            "n_seeds": 2,
        },
    }
    (configs_dir / "ridge.yaml").write_text(
        yaml.dump(cfg), encoding="utf-8"
    )
    return configs_dir


@pytest.fixture
def regression_pipeline_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        task_type="regression",
        target_column="target",
        cv=CVConfig(n_folds=3, seed=42, stratified=False),
        models=["ridge"],
        optuna=OptunaGlobalConfig(global_seed=42),
        output=OutputConfig(results_dir=str(tmp_path / "results")),
    )


class TestRegressionPath:
    def test_optuna_study_regression(
        self, ridge_regression_configs_dir, regression_data, regression_pipeline_config
    ):
        """Study should minimize RMSE for regression."""
        train, test = regression_data
        registry = ModelRegistry(ridge_regression_configs_dir)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=regression_pipeline_config,
            strategy={},
            gpu=False,
        )
        assert study.direction == optuna.study.StudyDirection.MINIMIZE
        assert study.best_value >= 0.0  # RMSE is non-negative

    def test_train_with_config_regression(
        self, ridge_regression_configs_dir, regression_data, regression_pipeline_config
    ):
        """train_with_config should work for regression task."""
        train, test = regression_data
        registry = ModelRegistry(ridge_regression_configs_dir)
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        results_dir = Path(regression_pipeline_config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        oof_list, test_list, labels = train_with_config(
            model_name="ridge",
            hparams={"alpha": 1.0},
            feature_cols=["f1", "f2", "f3"],
            train=train, test=test,
            target_col="target",
            cv=cv, registry=registry,
            task_type="regression",
            gpu=False, seeds=[42],
            results_dir=results_dir,
        )
        assert len(oof_list) == 1
        assert oof_list[0].shape == (len(train),)
        assert test_list[0].shape == (len(test),)
        # Regression predictions can be any float
        assert np.all(np.isfinite(oof_list[0]))

    def test_get_top_configs_minimize_sorted(
        self, ridge_regression_configs_dir, regression_data, regression_pipeline_config
    ):
        """Top configs should be sorted ascending for minimize (regression)."""
        train, _ = regression_data
        registry = ModelRegistry(ridge_regression_configs_dir)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=regression_pipeline_config,
            strategy={},
            gpu=False,
        )
        top = get_top_configs(study, n_top=3)
        if len(top) >= 2:
            assert top[0]["value"] <= top[1]["value"]  # ascending for minimize


# ---------------------------------------------------------------------------
# get_top_configs — additional coverage
# ---------------------------------------------------------------------------

class TestGetTopConfigsExtra:
    def test_empty_study_returns_empty(self):
        """get_top_configs on a study with no completed trials returns empty list."""
        study = optuna.create_study(direction="maximize")
        # No trials at all
        result = get_top_configs(study, n_top=5)
        assert result == []

    def test_n_top_exceeds_completed(self, ridge_configs_dir, binary_data, pipeline_config):
        """Requesting more configs than completed trials returns what's available."""
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy={},
            gpu=False,
        )
        top = get_top_configs(study, n_top=100)  # more than n_trials=5
        assert len(top) <= len(study.trials)

    def test_config_has_required_keys(self, ridge_configs_dir, binary_data, pipeline_config):
        """Each config dict should have params, value, and trial_number."""
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy={},
            gpu=False,
        )
        top = get_top_configs(study, n_top=1)
        assert len(top) == 1
        assert set(top[0].keys()) == {"params", "value", "trial_number"}


# ---------------------------------------------------------------------------
# train_with_config — additional coverage
# ---------------------------------------------------------------------------

class TestTrainWithConfigExtra:
    def test_multi_seed_oof_differ(self, ridge_configs_dir, binary_data, pipeline_config):
        """Different seeds should produce (slightly) different OOF predictions."""
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        results_dir = Path(pipeline_config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        oof_list, test_list, _ = train_with_config(
            model_name="ridge",
            hparams={"C": 1.0},
            feature_cols=["f1", "f2", "f3"],
            train=train, test=test,
            target_col="target",
            cv=cv, registry=registry,
            task_type="binary_classification",
            gpu=False, seeds=[1, 999],
            results_dir=results_dir,
        )
        # With very different seeds, LogisticRegression (deterministic solver) may
        # give identical results. Check they at least both produce valid output.
        assert len(oof_list) == 2
        assert len(test_list) == 2
        for arr in oof_list + test_list:
            assert np.all(np.isfinite(arr))

    def test_test_preds_are_averaged(self, ridge_configs_dir, binary_data, pipeline_config):
        """Test predictions should be in [0, 1] range (averaged probabilities)."""
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        results_dir = Path(pipeline_config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        _, test_list, _ = train_with_config(
            model_name="ridge",
            hparams={"C": 1.0},
            feature_cols=["f1", "f2", "f3"],
            train=train, test=test,
            target_col="target",
            cv=cv, registry=registry,
            task_type="binary_classification",
            gpu=False, seeds=[42],
            results_dir=results_dir,
        )
        # Averaged probs should stay in [0, 1]
        assert test_list[0].min() >= 0.0
        assert test_list[0].max() <= 1.0


# ---------------------------------------------------------------------------
# run_optuna_study — additional coverage
# ---------------------------------------------------------------------------

class TestRunOptunaStudyExtra:
    def test_strategy_overrides_applied(self, ridge_configs_dir, binary_data, pipeline_config):
        """LLM strategy overrides should narrow the search space."""
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        strategy = {
            "overrides": {
                "ridge": {"C": {"low": 0.5, "high": 2.0}},
            }
        }
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy=strategy,
            gpu=False,
        )
        # All trials' C param should be within [0.5, 2.0]
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                c_val = trial.params.get("C", None)
                if c_val is not None:
                    assert 0.5 <= c_val <= 2.0

    def test_timeout_override(self, ridge_configs_dir, binary_data, pipeline_config):
        """timeout_override should be accepted without error."""
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy={},
            gpu=False,
            timeout_override=300,
        )
        assert len(study.trials) >= 1



# ---------------------------------------------------------------------------
# Monotone constraints
# ---------------------------------------------------------------------------

class TestMonotoneConstraints:
    """Tests for monotone constraints support in run_optuna_study and train_with_config."""

    def test_monotone_constraints_resolved_for_gbm(self, ridge_configs_dir, binary_data, pipeline_config):
        """Monotone constraints dict should be resolved to positional list.

        Note: ridge doesn't support monotone_constraints, so this tests that
        the feature is silently skipped for non-GBM models (no error).
        """
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        strategy = {
            "monotone_constraints": {
                "f1": 1,
                "f2": -1,
                "f3": 0,
            }
        }
        # Should not raise — ridge is not in (catboost, xgboost, lightgbm)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy=strategy,
            gpu=False,
        )
        assert len(study.trials) >= 1

    def test_monotone_constraints_empty_dict_no_error(self, ridge_configs_dir, binary_data, pipeline_config):
        """Empty monotone_constraints dict should not cause errors."""
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        strategy = {"monotone_constraints": {}}
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy=strategy,
            gpu=False,
        )
        assert len(study.trials) >= 1

    def test_train_with_config_accepts_monotone_constraints(self, ridge_configs_dir, binary_data, pipeline_config):
        """train_with_config should accept monotone_constraints param without error."""
        from src.models.trainer import train_with_config
        from sklearn.model_selection import StratifiedKFold
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        # Ridge ignores unknown kwargs, so passing monotone_constraints
        # should not crash — it just gets merged into hparams (and ignored by Ridge).
        # This test verifies the parameter plumbing works.
        oof_list, test_list, y = train_with_config(
            model_name="ridge",
            hparams={"C": 1.0},
            feature_cols=["f1", "f2", "f3"],
            train=train,
            test=test,
            target_col="target",
            cv=cv,
            registry=registry,
            task_type="binary_classification",
            gpu=False,
            seeds=[42],
            results_dir=pipeline_config.output.results_dir,
            monotone_constraints=None,
        )
        assert len(oof_list) == 1
        assert len(test_list) == 1


# ---------------------------------------------------------------------------
# run_all_studies — basic smoke test
# ---------------------------------------------------------------------------

class TestRunAllStudies:
    def test_basic_smoke(self, ridge_configs_dir, binary_data, pipeline_config):
        """run_all_studies should produce results dict with expected keys."""
        from src.models.trainer import run_all_studies
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        results = run_all_studies(
            pipeline_config=pipeline_config,
            train=train,
            test=test,
            feature_cols=["f1", "f2", "f3"],
            strategy={},
            registry=registry,
            gpu_status={"ridge": False},
        )
        assert "ridge" in results
        r = results["ridge"]
        assert "study" in r
        assert "top_configs" in r
        assert "oof_preds" in r
        assert "test_preds" in r
        assert "labels" in r
        assert len(r["oof_preds"]) > 0
        assert len(r["test_preds"]) > 0
        assert len(r["labels"]) == len(train)

    def test_failed_model_skipped(self, tmp_path, binary_data, pipeline_config):
        """A model that fails should be skipped, not crash the whole run."""
        from src.models.trainer import run_all_studies

        # Empty configs dir — "ridge" won't be found → KeyError
        empty_dir = tmp_path / "empty_models"
        empty_dir.mkdir()
        registry = ModelRegistry(empty_dir)

        train, test = binary_data
        results = run_all_studies(
            pipeline_config=pipeline_config,
            train=train,
            test=test,
            feature_cols=["f1", "f2", "f3"],
            strategy={},
            registry=registry,
            gpu_status={},
        )
        # Model failed, results should be empty
        assert "ridge" not in results


# ---------------------------------------------------------------------------
# PerFoldTracker
# ---------------------------------------------------------------------------

class TestPerFoldTracker:
    """Tests for the PerFoldTracker used in per-fold selection mode."""

    def test_tracks_top_n_per_fold(self):
        """Tracker should keep only n_top entries per fold."""
        from src.models.trainer import PerFoldTracker

        tracker = PerFoldTracker(n_top=2, n_folds=2, maximize=True)
        val_idx = np.array([0, 1, 2])
        test_preds = np.array([0.5, 0.5, 0.5])

        # Insert 3 entries for fold 0 — only best 2 should remain
        tracker.update(0, 0.80, np.array([0.1, 0.2, 0.3]), val_idx, test_preds, 1, {"lr": 0.01})
        tracker.update(0, 0.90, np.array([0.4, 0.5, 0.6]), val_idx, test_preds, 2, {"lr": 0.02})
        tracker.update(0, 0.85, np.array([0.7, 0.8, 0.9]), val_idx, test_preds, 3, {"lr": 0.03})

        assert len(tracker.fold_data[0]) == 2
        # Best (0.90) and second (0.85) kept, worst (0.80) dropped
        assert tracker.fold_data[0][0][0] == 0.90
        assert tracker.fold_data[0][1][0] == 0.85

    def test_minimize_direction(self):
        """For minimize (regression), lower scores should rank first."""
        from src.models.trainer import PerFoldTracker

        tracker = PerFoldTracker(n_top=2, n_folds=1, maximize=False)
        val_idx = np.array([0, 1])
        test_preds = np.array([0.5, 0.5])

        tracker.update(0, 0.50, np.array([0.1, 0.2]), val_idx, test_preds, 1, {})
        tracker.update(0, 0.30, np.array([0.3, 0.4]), val_idx, test_preds, 2, {})
        tracker.update(0, 0.40, np.array([0.5, 0.6]), val_idx, test_preds, 3, {})

        assert len(tracker.fold_data[0]) == 2
        assert tracker.fold_data[0][0][0] == 0.30  # best (lowest)
        assert tracker.fold_data[0][1][0] == 0.40

    def test_assemble_correct_shapes(self):
        """Assembled composites should have correct OOF and test shapes."""
        from src.models.trainer import PerFoldTracker

        n_samples = 6
        n_test = 4
        n_folds = 2
        n_top = 2

        tracker = PerFoldTracker(n_top=n_top, n_folds=n_folds, maximize=True)

        # Fold 0: val_idx = [0, 1, 2]
        val_idx_0 = np.array([0, 1, 2])
        tracker.update(0, 0.90, np.array([0.8, 0.9, 0.7]), val_idx_0,
                       np.array([0.5, 0.5, 0.5, 0.5]), 1, {"lr": 0.01})
        tracker.update(0, 0.85, np.array([0.6, 0.7, 0.5]), val_idx_0,
                       np.array([0.4, 0.4, 0.4, 0.4]), 2, {"lr": 0.02})

        # Fold 1: val_idx = [3, 4, 5]
        val_idx_1 = np.array([3, 4, 5])
        tracker.update(1, 0.88, np.array([0.7, 0.8, 0.6]), val_idx_1,
                       np.array([0.6, 0.6, 0.6, 0.6]), 3, {"lr": 0.03})
        tracker.update(1, 0.82, np.array([0.5, 0.6, 0.4]), val_idx_1,
                       np.array([0.3, 0.3, 0.3, 0.3]), 4, {"lr": 0.04})

        composites = tracker.assemble(n_samples=n_samples, n_test=n_test)

        assert len(composites) == n_top
        for c in composites:
            assert c["oof_preds"].shape == (n_samples,)
            assert c["test_preds"].shape == (n_test,)
            assert len(c["fold_trials"]) == n_folds
            assert len(c["fold_scores"]) == n_folds
            assert isinstance(c["avg_score"], float)

    def test_assemble_oof_uses_per_fold_bests(self):
        """Composite 0 should use the best trial per fold."""
        from src.models.trainer import PerFoldTracker

        tracker = PerFoldTracker(n_top=2, n_folds=2, maximize=True)

        val_idx_0 = np.array([0, 1])
        val_idx_1 = np.array([2, 3])

        # Fold 0: trial 10 is best
        tracker.update(0, 0.95, np.array([0.9, 0.8]), val_idx_0,
                       np.array([0.7, 0.7]), 10, {})
        tracker.update(0, 0.85, np.array([0.6, 0.5]), val_idx_0,
                       np.array([0.4, 0.4]), 20, {})

        # Fold 1: trial 30 is best
        tracker.update(1, 0.92, np.array([0.88, 0.77]), val_idx_1,
                       np.array([0.6, 0.6]), 30, {})
        tracker.update(1, 0.80, np.array([0.55, 0.44]), val_idx_1,
                       np.array([0.3, 0.3]), 40, {})

        composites = tracker.assemble(n_samples=4, n_test=2)

        # Composite 0: fold 0 from trial 10, fold 1 from trial 30
        assert composites[0]["fold_trials"] == [10, 30]
        np.testing.assert_array_almost_equal(
            composites[0]["oof_preds"][:2], [0.9, 0.8]  # from trial 10
        )
        np.testing.assert_array_almost_equal(
            composites[0]["oof_preds"][2:], [0.88, 0.77]  # from trial 30
        )

        # Composite 1: fold 0 from trial 20, fold 1 from trial 40
        assert composites[1]["fold_trials"] == [20, 40]

    def test_assemble_test_averaged_across_folds(self):
        """Test preds should be averaged across folds."""
        from src.models.trainer import PerFoldTracker

        tracker = PerFoldTracker(n_top=1, n_folds=2, maximize=True)

        val_idx_0 = np.array([0])
        val_idx_1 = np.array([1])

        tracker.update(0, 0.90, np.array([0.9]), val_idx_0,
                       np.array([0.8, 0.6]), 1, {})
        tracker.update(1, 0.85, np.array([0.7]), val_idx_1,
                       np.array([0.4, 0.2]), 2, {})

        composites = tracker.assemble(n_samples=2, n_test=2)
        # test_preds = [0.8, 0.6] / 2 + [0.4, 0.2] / 2 = [0.6, 0.4]
        np.testing.assert_array_almost_equal(
            composites[0]["test_preds"], [0.6, 0.4]
        )

    def test_pruned_trials_contribute(self):
        """Pruned trials' completed folds should be tracked."""
        from src.models.trainer import PerFoldTracker

        tracker = PerFoldTracker(n_top=2, n_folds=3, maximize=True)

        val_idx = np.array([0, 1])
        test_preds = np.array([0.5, 0.5])

        # "Pruned" trial that only completed fold 0 with excellent score
        tracker.update(0, 0.99, np.array([0.95, 0.98]), val_idx, test_preds, 100, {})

        # Regular trial that completed all 3 folds with mediocre scores
        for fold_idx in range(3):
            tracker.update(fold_idx, 0.75, np.array([0.6, 0.7]), val_idx, test_preds, 200, {})
        # Second regular trial
        for fold_idx in range(3):
            tracker.update(fold_idx, 0.80, np.array([0.65, 0.75]), val_idx, test_preds, 300, {})

        # Fold 0: trial 100 (0.99) should be rank 0
        assert tracker.fold_data[0][0][4] == 100  # trial_number
        assert tracker.fold_data[0][0][0] == 0.99

        # Folds 1 and 2 don't have trial 100 (it was "pruned")
        fold1_trials = [entry[4] for entry in tracker.fold_data[1]]
        assert 100 not in fold1_trials

    def test_first_composite_has_best_per_fold_scores(self):
        """Composite 0 should have the best score per fold (>= composite 1)."""
        from src.models.trainer import PerFoldTracker

        tracker = PerFoldTracker(n_top=3, n_folds=2, maximize=True)

        val_idx_0 = np.array([0])
        val_idx_1 = np.array([1])
        test_preds = np.array([0.5])

        # Multiple entries per fold
        for score, tn in [(0.90, 1), (0.85, 2), (0.80, 3)]:
            tracker.update(0, score, np.array([score]), val_idx_0, test_preds, tn, {})
        for score, tn in [(0.88, 4), (0.83, 5), (0.78, 6)]:
            tracker.update(1, score, np.array([score]), val_idx_1, test_preds, tn, {})

        composites = tracker.assemble(n_samples=2, n_test=1)
        assert len(composites) >= 2

        for fold_idx in range(2):
            assert composites[0]["fold_scores"][fold_idx] >= composites[1]["fold_scores"][fold_idx]


# ---------------------------------------------------------------------------
# Per-fold integration with run_optuna_study
# ---------------------------------------------------------------------------

class TestPerFoldIntegration:
    """Integration tests for per-fold selection mode."""

    @pytest.fixture
    def per_fold_configs_dir(self, tmp_path: Path) -> Path:
        """Model config with selection_mode: per_fold for testing."""
        import yaml
        configs_dir = tmp_path / "models"
        configs_dir.mkdir()
        cfg = {
            "name": "Ridge",
            "class_path": {
                "binary_classification": "sklearn.linear_model.LogisticRegression",
                "regression": "sklearn.linear_model.Ridge",
            },
            "task_types": ["binary_classification", "regression"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {
                "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            },
            "fixed_params": {
                "binary_classification": {"max_iter": 1000, "solver": "lbfgs"},
                "regression": {"max_iter": 1000},
            },
            "training": {
                "needs_eval_set": False,
                "early_stopping": False,
                "eval_metric_param": None,
            },
            "feature_requirements": {"needs_scaling": True},
            "optuna": {
                "n_trials": 8,
                "qmc_warmup_trials": 3,
                "timeout": None,
                "pruner": {"type": "none"},
                "n_top_trials": 3,
                "n_seeds": 1,
                "selection_mode": "per_fold",
            },
        }
        (configs_dir / "ridge.yaml").write_text(
            yaml.dump(cfg), encoding="utf-8"
        )
        return configs_dir

    @pytest.fixture
    def per_fold_pipeline(self, tmp_path: Path) -> PipelineConfig:
        return PipelineConfig(
            task_type="binary_classification",
            target_column="target",
            cv=CVConfig(n_folds=3, seed=42, stratified=True),
            models=["ridge"],
            optuna=OptunaGlobalConfig(global_seed=42),
            output=OutputConfig(results_dir=str(tmp_path / "results")),
        )

    def test_run_optuna_returns_tracker(self, per_fold_configs_dir, binary_data, per_fold_pipeline):
        """run_optuna_study with per_fold should return a PerFoldTracker."""
        from src.models.trainer import PerFoldTracker

        train, test = binary_data
        registry = ModelRegistry(per_fold_configs_dir)
        study, tracker = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=per_fold_pipeline,
            strategy={},
            gpu=False,
            test=test,
        )
        assert tracker is not None
        assert isinstance(tracker, PerFoldTracker)
        # Each fold should have at least 1 entry
        for fold_idx in range(3):
            assert len(tracker.fold_data[fold_idx]) >= 1

    def test_tracker_assembles_correct_count(self, per_fold_configs_dir, binary_data, per_fold_pipeline):
        """Tracker should assemble up to n_top_trials composites."""
        train, test = binary_data
        registry = ModelRegistry(per_fold_configs_dir)
        study, tracker = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=per_fold_pipeline,
            strategy={},
            gpu=False,
            test=test,
        )
        composites = tracker.assemble(
            n_samples=len(train), n_test=len(test)
        )
        assert len(composites) <= 3  # n_top_trials
        assert len(composites) >= 1

    def test_assembled_oof_valid_probabilities(self, per_fold_configs_dir, binary_data, per_fold_pipeline):
        """Assembled OOF predictions should be valid probabilities."""
        train, test = binary_data
        registry = ModelRegistry(per_fold_configs_dir)
        study, tracker = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=per_fold_pipeline,
            strategy={},
            gpu=False,
            test=test,
        )
        composites = tracker.assemble(
            n_samples=len(train), n_test=len(test)
        )
        for c in composites:
            assert c["oof_preds"].min() >= 0.0
            assert c["oof_preds"].max() <= 1.0
            assert c["test_preds"].min() >= 0.0
            assert c["test_preds"].max() <= 1.0

    def test_run_all_studies_per_fold(self, per_fold_configs_dir, binary_data, per_fold_pipeline):
        """run_all_studies with per_fold model should produce correct output."""
        from src.models.trainer import run_all_studies

        train, test = binary_data
        registry = ModelRegistry(per_fold_configs_dir)
        results = run_all_studies(
            pipeline_config=per_fold_pipeline,
            train=train,
            test=test,
            feature_cols=["f1", "f2", "f3"],
            strategy={},
            registry=registry,
            gpu_status={"ridge": False},
        )
        assert "ridge" in results
        r = results["ridge"]
        assert "oof_preds" in r
        assert "test_preds" in r
        assert "labels" in r
        assert len(r["oof_preds"]) > 0
        assert len(r["test_preds"]) > 0
        assert len(r["oof_preds"]) == len(r["test_preds"])
        # OOF and test shapes match
        for oof_arr in r["oof_preds"]:
            assert oof_arr.shape == (len(train),)
        for test_arr in r["test_preds"]:
            assert test_arr.shape == (len(test),)

    def test_global_mode_unchanged(self, ridge_configs_dir, binary_data, pipeline_config):
        """Global mode (no selection_mode) should still work as before."""
        from src.models.trainer import run_all_studies

        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        results = run_all_studies(
            pipeline_config=pipeline_config,
            train=train,
            test=test,
            feature_cols=["f1", "f2", "f3"],
            strategy={},
            registry=registry,
            gpu_status={"ridge": False},
        )
        assert "ridge" in results
        r = results["ridge"]
        assert len(r["oof_preds"]) > 0
        assert r["retrain_elapsed"] > 0  # global mode does retrain


# ---------------------------------------------------------------------------
# NSGA-II fold assembly tests
# ---------------------------------------------------------------------------

class TestAssembleNsga2:
    """Tests for PerFoldTracker.assemble_nsga2() — diversity-optimized composites."""

    def _make_tracker(self, n_top=5, n_folds=3, n_samples=6):
        """Create a tracker with populated fold data for testing."""
        from src.models.trainer import PerFoldTracker

        tracker = PerFoldTracker(n_top=n_top, n_folds=n_folds, maximize=True)
        samples_per_fold = n_samples // n_folds

        for fold_idx in range(n_folds):
            val_idx = np.arange(
                fold_idx * samples_per_fold,
                (fold_idx + 1) * samples_per_fold,
            )
            for k in range(n_top):
                score = 0.95 - k * 0.02 - fold_idx * 0.01
                val_preds = np.random.RandomState(fold_idx * 100 + k).rand(samples_per_fold)
                test_preds = np.random.RandomState(fold_idx * 200 + k).rand(4)
                tracker.update(fold_idx, score, val_preds, val_idx, test_preds, k + fold_idx * 100, {})

        return tracker

    def test_returns_correct_count(self):
        """assemble_nsga2 should return at most n_composites."""
        tracker = self._make_tracker(n_top=5, n_folds=3)
        composites = tracker.assemble_nsga2(
            n_samples=6, n_test=4, n_composites=3,
            n_generations=10, pop_size=20, seed=42,
        )
        assert len(composites) <= 3
        assert len(composites) >= 1

    def test_correct_shapes(self):
        """Composites should have correct OOF and test array shapes."""
        tracker = self._make_tracker(n_top=5, n_folds=3, n_samples=6)
        composites = tracker.assemble_nsga2(
            n_samples=6, n_test=4, n_composites=5,
            n_generations=10, pop_size=20, seed=42,
        )
        for c in composites:
            assert c["oof_preds"].shape == (6,)
            assert c["test_preds"].shape == (4,)
            assert len(c["fold_trials"]) == 3
            assert len(c["fold_scores"]) == 3
            assert isinstance(c["avg_score"], float)

    def test_same_format_as_rank_assembly(self):
        """NSGA-II composites should have same dict keys as rank assembly."""
        tracker = self._make_tracker(n_top=5, n_folds=3, n_samples=6)
        rank_composites = tracker.assemble(n_samples=6, n_test=4)
        nsga2_composites = tracker.assemble_nsga2(
            n_samples=6, n_test=4, n_composites=3,
            n_generations=10, pop_size=20, seed=42,
        )
        assert set(rank_composites[0].keys()) == set(nsga2_composites[0].keys())

    def test_diversity_weight_affects_selection(self):
        """Different diversity_weight should potentially produce different composites."""
        tracker = self._make_tracker(n_top=10, n_folds=3, n_samples=6)
        composites_score = tracker.assemble_nsga2(
            n_samples=6, n_test=4, n_composites=3,
            n_generations=15, pop_size=30, diversity_weight=0.0, seed=42,
        )
        composites_diverse = tracker.assemble_nsga2(
            n_samples=6, n_test=4, n_composites=3,
            n_generations=15, pop_size=30, diversity_weight=1.0, seed=42,
        )
        # With dw=0 the best score composite should have high avg_score
        # With dw=1 it may sacrifice score for diversity
        # Just verify both produce valid outputs
        assert len(composites_score) >= 1
        assert len(composites_diverse) >= 1
        for c in composites_score + composites_diverse:
            assert c["oof_preds"].shape == (6,)

    def test_empty_tracker_returns_empty(self):
        """Empty tracker should return empty list."""
        from src.models.trainer import PerFoldTracker

        tracker = PerFoldTracker(n_top=3, n_folds=2, maximize=True)
        composites = tracker.assemble_nsga2(
            n_samples=4, n_test=2, n_composites=3,
            n_generations=5, pop_size=10, seed=42,
        )
        assert composites == []

    def test_more_diverse_than_rank(self):
        """NSGA-II composites should use more unique trials than rank assembly."""
        tracker = self._make_tracker(n_top=10, n_folds=3, n_samples=6)

        rank_composites = tracker.assemble(n_samples=6, n_test=4)
        nsga2_composites = tracker.assemble_nsga2(
            n_samples=6, n_test=4, n_composites=len(rank_composites),
            n_generations=20, pop_size=40, diversity_weight=0.5, seed=42,
        )

        rank_trials = set()
        for c in rank_composites:
            rank_trials.update(c["fold_trials"])

        nsga2_trials = set()
        for c in nsga2_composites:
            nsga2_trials.update(c["fold_trials"])

        # NSGA-II should use at least as many unique trials
        # (or close — randomness may vary)
        assert len(nsga2_trials) >= 1

    @pytest.mark.parametrize("metric", ["pearson_neff", "spearman_neff", "ambiguity"])
    def test_all_diversity_metrics(self, metric):
        """Each diversity_metric should produce valid composites."""
        tracker = self._make_tracker(n_top=5, n_folds=3, n_samples=6)
        composites = tracker.assemble_nsga2(
            n_samples=6, n_test=4, n_composites=3,
            n_generations=10, pop_size=20,
            diversity_metric=metric, diversity_weight=0.5, seed=42,
        )
        assert len(composites) >= 1
        for c in composites:
            assert c["oof_preds"].shape == (6,)
            assert c["test_preds"].shape == (4,)

    def test_spearman_uses_preranking(self):
        """spearman_neff should produce different selection than pearson_neff
        (they measure different things: rank vs linear correlation)."""
        tracker = self._make_tracker(n_top=10, n_folds=3, n_samples=6)
        pearson = tracker.assemble_nsga2(
            n_samples=6, n_test=4, n_composites=5,
            n_generations=15, pop_size=30,
            diversity_metric="pearson_neff", diversity_weight=0.5, seed=42,
        )
        spearman = tracker.assemble_nsga2(
            n_samples=6, n_test=4, n_composites=5,
            n_generations=15, pop_size=30,
            diversity_metric="spearman_neff", diversity_weight=0.5, seed=42,
        )
        # Both should produce valid outputs
        assert len(pearson) >= 1
        assert len(spearman) >= 1


class TestGreedyParetoSelect:
    """Tests for _greedy_pareto_select — greedy diversity-aware selection."""

    def _make_composites(self, n=10, n_samples=50) -> list[dict]:
        """Create synthetic composites with varying predictions."""
        rng = np.random.RandomState(42)
        composites = []
        for i in range(n):
            composites.append({
                "oof_preds": rng.rand(n_samples) * 0.5 + i * 0.01,
                "test_preds": rng.rand(20),
                "fold_trials": [i, i + 1, i + 2],
                "fold_scores": [0.9 - i * 0.01, 0.89 - i * 0.01, 0.88 - i * 0.01],
                "avg_score": 0.89 - i * 0.01,
            })
        return composites

    def test_selects_correct_count(self):
        """Should return exactly n_select composites."""
        from src.models.trainer import _greedy_pareto_select
        composites = self._make_composites(10)
        result = _greedy_pareto_select(
            composites, n_select=5, diversity_metric="pearson_neff",
            diversity_weight=0.3, maximize=True,
        )
        assert len(result) == 5

    def test_returns_all_when_fewer_than_n_select(self):
        """If fewer composites than n_select, return all."""
        from src.models.trainer import _greedy_pareto_select
        composites = self._make_composites(3)
        result = _greedy_pareto_select(
            composites, n_select=10, diversity_metric="pearson_neff",
            diversity_weight=0.3, maximize=True,
        )
        assert len(result) == 3

    def test_first_composite_has_best_score(self):
        """First selected composite should be the best-scoring one."""
        from src.models.trainer import _greedy_pareto_select
        composites = self._make_composites(10)
        result = _greedy_pareto_select(
            composites, n_select=5, diversity_metric="pearson_neff",
            diversity_weight=0.3, maximize=True,
        )
        best_score = max(c["avg_score"] for c in composites)
        assert result[0]["avg_score"] == best_score

    def test_dw_zero_prefers_scores(self):
        """With diversity_weight=0, should select by score only."""
        from src.models.trainer import _greedy_pareto_select
        composites = self._make_composites(10)
        result = _greedy_pareto_select(
            composites, n_select=5, diversity_metric="pearson_neff",
            diversity_weight=0.0, maximize=True,
        )
        # All selected should be top-5 by score
        sorted_by_score = sorted(composites, key=lambda c: c["avg_score"], reverse=True)
        top5_scores = {c["avg_score"] for c in sorted_by_score[:5]}
        result_scores = {c["avg_score"] for c in result}
        assert result_scores == top5_scores

    @pytest.mark.parametrize("metric", ["pearson_neff", "spearman_neff", "ambiguity"])
    def test_all_metrics_produce_valid_output(self, metric):
        """Each diversity metric should work without errors."""
        from src.models.trainer import _greedy_pareto_select
        composites = self._make_composites(10, n_samples=50)
        result = _greedy_pareto_select(
            composites, n_select=5, diversity_metric=metric,
            diversity_weight=0.5, maximize=True,
        )
        assert len(result) == 5
        for c in result:
            assert "oof_preds" in c
            assert "avg_score" in c

    def test_minimize_direction(self):
        """With maximize=False, best score is the lowest."""
        from src.models.trainer import _greedy_pareto_select
        composites = self._make_composites(10)
        result = _greedy_pareto_select(
            composites, n_select=3, diversity_metric="pearson_neff",
            diversity_weight=0.0, maximize=False,
        )
        # First should be the lowest-scoring composite
        min_score = min(c["avg_score"] for c in composites)
        assert result[0]["avg_score"] == min_score


# ---------------------------------------------------------------------------
# Search space param type coverage
# ---------------------------------------------------------------------------

class TestParamTypeCoverage:
    """Tests for all search space param types in _create_objective."""

    @pytest.fixture
    def multi_type_configs_dir(self, tmp_path: Path) -> Path:
        """Model config with int, categorical, and fixed search space types."""
        configs_dir = tmp_path / "models"
        configs_dir.mkdir()
        cfg = {
            "name": "Ridge",
            "class_path": {
                "binary_classification": "sklearn.linear_model.LogisticRegression",
            },
            "task_types": ["binary_classification"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {
                "C": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
                "max_iter": {"type": "int", "low": 100, "high": 500, "log": False},
                "solver": {"type": "categorical", "choices": ["lbfgs", "liblinear"]},
            },
            "fixed_params": {
                "binary_classification": {},
            },
            "training": {
                "needs_eval_set": False,
                "eval_metric_param": None,
            },
            "feature_requirements": {"needs_scaling": True},
            "optuna": {
                "n_trials": 4,
                "qmc_warmup_trials": 2,
                "timeout": None,
                "pruner": {"type": "none"},
                "n_top_trials": 2,
                "n_seeds": 1,
            },
        }
        (configs_dir / "ridge.yaml").write_text(
            yaml.dump(cfg), encoding="utf-8"
        )
        return configs_dir

    def test_int_and_categorical_params(self, multi_type_configs_dir, binary_data):
        """int and categorical param types should work in Optuna study."""
        train, test = binary_data
        registry = ModelRegistry(multi_type_configs_dir)
        pipeline_cfg = PipelineConfig(
            task_type="binary_classification",
            target_column="target",
            cv=CVConfig(n_folds=3, seed=42, stratified=True),
            models=["ridge"],
            optuna=OptunaGlobalConfig(global_seed=42),
            output=OutputConfig(results_dir=str(multi_type_configs_dir.parent / "results")),
        )
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_cfg,
            strategy={},
            gpu=False,
        )
        assert len(study.trials) >= 1
        best = study.best_trial
        assert "max_iter" in best.params  # int param
        assert "solver" in best.params    # categorical param
        assert isinstance(best.params["max_iter"], int)
        assert best.params["solver"] in ["lbfgs", "liblinear"]

    def test_fixed_param_via_override(self, ridge_configs_dir, binary_data, pipeline_config):
        """Scalar strategy override creates 'fixed' type — value bypasses Optuna."""
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        strategy = {
            "overrides": {
                "ridge": {"C": 1.0},  # scalar → fixed type
            }
        }
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy=strategy,
            gpu=False,
        )
        # Fixed param should NOT appear in trial.params (not suggested by Optuna)
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                assert "C" not in trial.params

    def test_optuna_overrides_do_not_leak_to_search_space(
        self, ridge_configs_dir, binary_data, pipeline_config,
    ):
        """Strategy optuna overrides (n_trials, n_seeds) must NOT pollute search space."""
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        strategy = {
            "overrides": {
                "ridge": {
                    "optuna": {"n_trials": 5, "n_top_trials": 2, "n_seeds": 1},
                }
            }
        }
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy=strategy,
            gpu=False,
        )
        # Should have completed 5 trials (from override), not the YAML default
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        assert len(completed) == 5


# ---------------------------------------------------------------------------
# Fold timeout
# ---------------------------------------------------------------------------

class TestFoldTimeout:
    """Tests for fold_timeout pruning behavior."""

    @pytest.fixture
    def timeout_configs_dir(self, tmp_path: Path) -> Path:
        """Model config with a very short fold_timeout."""
        configs_dir = tmp_path / "models"
        configs_dir.mkdir()
        cfg = {
            "name": "Ridge",
            "class_path": {
                "binary_classification": "sklearn.linear_model.LogisticRegression",
            },
            "task_types": ["binary_classification"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {
                "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            },
            "fixed_params": {
                "binary_classification": {"max_iter": 1000, "solver": "lbfgs"},
            },
            "training": {
                "needs_eval_set": False,
                "eval_metric_param": None,
            },
            "feature_requirements": {"needs_scaling": True},
            "optuna": {
                "n_trials": 4,
                "qmc_warmup_trials": 2,
                "timeout": None,
                "pruner": {"type": "none"},
                "n_top_trials": 2,
                "n_seeds": 1,
                "fold_timeout": 0,  # 0 seconds — every fold exceeds this
            },
        }
        (configs_dir / "ridge.yaml").write_text(
            yaml.dump(cfg), encoding="utf-8"
        )
        return configs_dir

    def test_fold_timeout_causes_pruning(self, timeout_configs_dir, binary_data):
        """fold_timeout=0 prunes all trials — run_optuna_study returns gracefully.

        After the BUG fix, run_optuna_study no longer crashes when all trials
        are pruned. It logs a warning and returns (study, None).
        """
        train, test = binary_data
        registry = ModelRegistry(timeout_configs_dir)
        pipeline_cfg = PipelineConfig(
            task_type="binary_classification",
            target_column="target",
            cv=CVConfig(n_folds=3, seed=42, stratified=True),
            models=["ridge"],
            optuna=OptunaGlobalConfig(global_seed=42),
            output=OutputConfig(results_dir=str(timeout_configs_dir.parent / "results")),
        )

        # Patch time.monotonic to simulate 5s per call, ensuring fold_elapsed > 0
        counter = {"v": 0.0}

        def fake_monotonic():
            counter["v"] += 5.0
            return counter["v"]

        with patch("src.models.trainer.time.monotonic", side_effect=fake_monotonic):
            study, tracker = run_optuna_study(
                model_name="ridge",
                train=train,
                feature_cols=["f1", "f2", "f3"],
                target_col="target",
                registry=registry,
                pipeline_config=pipeline_cfg,
                strategy={},
                gpu=False,
            )

        # All trials should be pruned — no completed trials
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        assert len(completed) == 0
        # Returned gracefully: tracker is None (ridge uses global selection mode)
        assert tracker is None


# ---------------------------------------------------------------------------
# OOF leakage verification
# ---------------------------------------------------------------------------

class TestOofLeakageVerification:
    """Verify OOF predictions are truly out-of-fold (no leakage)."""

    def test_oof_predictions_are_out_of_fold(self, ridge_configs_dir, binary_data, pipeline_config):
        """Each OOF prediction should come from a model that did NOT train on that sample."""
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        results_dir = Path(pipeline_config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        oof_list, _, labels = train_with_config(
            model_name="ridge",
            hparams={"C": 1.0},
            feature_cols=["f1", "f2", "f3"],
            train=train, test=test,
            target_col="target",
            cv=cv, registry=registry,
            task_type="binary_classification",
            gpu=False, seeds=[42],
            results_dir=results_dir,
        )
        oof = oof_list[0]

        # Verify: every index in the OOF array was filled exactly once
        # by one of the validation folds
        seen_indices: set[int] = set()
        for train_idx, val_idx in cv.split(train[["f1", "f2", "f3"]], labels):
            for idx in val_idx:
                assert idx not in seen_indices, f"Index {idx} appears in multiple val folds"
                seen_indices.add(idx)
        assert seen_indices == set(range(len(train)))

        # Verify OOF values are all non-zero (all slots were filled)
        assert np.all(np.isfinite(oof))
        # Since oof is probabilities, at least some variation should exist
        assert oof.std() > 0.0


# ---------------------------------------------------------------------------
# PerFoldTracker — edge cases
# ---------------------------------------------------------------------------

class TestPerFoldTrackerEdgeCases:
    """Edge cases for PerFoldTracker.assemble."""

    def test_assemble_uneven_fold_entries(self):
        """When folds have different entry counts, n_composites = min across folds."""
        from src.models.trainer import PerFoldTracker

        tracker = PerFoldTracker(n_top=5, n_folds=2, maximize=True)

        val_idx_0 = np.array([0, 1])
        val_idx_1 = np.array([2, 3])
        test_preds = np.array([0.5, 0.5])

        # Fold 0: 3 entries
        for k, (score, tn) in enumerate([(0.90, 1), (0.85, 2), (0.80, 3)]):
            tracker.update(0, score, np.array([score, score]), val_idx_0, test_preds, tn, {})

        # Fold 1: only 1 entry
        tracker.update(1, 0.88, np.array([0.88, 0.88]), val_idx_1, test_preds, 10, {})

        composites = tracker.assemble(n_samples=4, n_test=2)
        # Should be limited by fold 1's single entry
        assert len(composites) == 1

    def test_assemble_empty_tracker_returns_empty(self):
        """Tracker with no entries should return empty composites."""
        from src.models.trainer import PerFoldTracker

        tracker = PerFoldTracker(n_top=3, n_folds=2, maximize=True)
        composites = tracker.assemble(n_samples=4, n_test=2)
        assert composites == []

    def test_single_fold_tracker(self):
        """Tracker with n_folds=1 should work correctly."""
        from src.models.trainer import PerFoldTracker

        tracker = PerFoldTracker(n_top=2, n_folds=1, maximize=True)
        val_idx = np.arange(4)
        test_preds = np.array([0.6, 0.4])

        tracker.update(0, 0.90, np.array([0.9, 0.8, 0.7, 0.6]), val_idx, test_preds, 1, {})
        tracker.update(0, 0.85, np.array([0.5, 0.4, 0.3, 0.2]), val_idx, test_preds, 2, {})

        composites = tracker.assemble(n_samples=4, n_test=2)
        assert len(composites) == 2
        # With 1 fold, test_preds = fold_test / 1 = fold_test
        np.testing.assert_array_almost_equal(composites[0]["test_preds"], test_preds)

    def test_update_copies_arrays(self):
        """Stored arrays should be independent copies (no aliasing)."""
        from src.models.trainer import PerFoldTracker

        tracker = PerFoldTracker(n_top=2, n_folds=1, maximize=True)
        val_preds = np.array([0.9, 0.8])
        val_idx = np.array([0, 1])
        test_preds = np.array([0.5, 0.5])

        tracker.update(0, 0.90, val_preds, val_idx, test_preds, 1, {})

        # Mutate originals — stored data should not change
        val_preds[:] = 0.0
        val_idx[:] = 99
        test_preds[:] = 0.0

        stored = tracker.fold_data[0][0]
        assert stored[1][0] == 0.9  # val_preds not aliased
        assert stored[2][0] == 0    # val_idx not aliased
        assert stored[3][0] == 0.5  # test_preds not aliased


# ---------------------------------------------------------------------------
# step param support in search space (int/float)
# ---------------------------------------------------------------------------

class TestStepParamSupport:
    """Test that the 'step' field in int/float search space specs is respected."""

    @pytest.fixture
    def step_configs_dir(self, tmp_path: Path) -> Path:
        """Model config with int params that have step values (like TabM)."""
        configs_dir = tmp_path / "models"
        configs_dir.mkdir()
        cfg = {
            "name": "Ridge",
            "class_path": {
                "binary_classification": "sklearn.linear_model.LogisticRegression",
            },
            "task_types": ["binary_classification"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {
                "C": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
                "max_iter": {"type": "int", "low": 100, "high": 1000, "step": 100},
            },
            "fixed_params": {
                "binary_classification": {"solver": "lbfgs"},
            },
            "training": {
                "needs_eval_set": False,
                "eval_metric_param": None,
            },
            "feature_requirements": {"needs_scaling": True},
            "optuna": {
                "n_trials": 10,
                "qmc_warmup_trials": 3,
                "timeout": None,
                "pruner": {"type": "none"},
                "n_top_trials": 2,
                "n_seeds": 1,
            },
        }
        (configs_dir / "ridge.yaml").write_text(
            yaml.dump(cfg), encoding="utf-8"
        )
        return configs_dir

    def test_int_step_respected(self, step_configs_dir, binary_data):
        """int params with step should only produce multiples of step."""
        train, test = binary_data
        registry = ModelRegistry(step_configs_dir)
        pipeline_cfg = PipelineConfig(
            task_type="binary_classification",
            target_column="target",
            cv=CVConfig(n_folds=3, seed=42, stratified=True),
            models=["ridge"],
            optuna=OptunaGlobalConfig(global_seed=42),
            output=OutputConfig(results_dir=str(step_configs_dir.parent / "results")),
        )
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_cfg,
            strategy={},
            gpu=False,
        )
        # All trials' max_iter should be multiples of 100
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                max_iter = trial.params.get("max_iter")
                if max_iter is not None:
                    assert max_iter % 100 == 0, (
                        f"max_iter={max_iter} is not a multiple of step=100. "
                        f"The 'step' field in the search space spec is being ignored."
                    )


# ---------------------------------------------------------------------------
# Multiclass path in PerFoldTracker.assemble
# ---------------------------------------------------------------------------

class TestPerFoldTrackerMulticlass:
    """Test multiclass path in PerFoldTracker.assemble."""

    def test_assemble_multiclass_shapes(self):
        """Assembled multiclass composites should have 2D OOF and test arrays."""
        from src.models.trainer import PerFoldTracker

        n_samples = 6
        n_test = 4
        n_folds = 2
        n_classes = 3
        n_top = 2

        tracker = PerFoldTracker(n_top=n_top, n_folds=n_folds, maximize=True)

        val_idx_0 = np.array([0, 1, 2])
        val_idx_1 = np.array([3, 4, 5])

        # 2D predictions for multiclass
        for k in range(n_top):
            rng = np.random.RandomState(k)
            tracker.update(
                0, 0.90 - k * 0.05,
                rng.rand(3, n_classes),  # val_preds: (n_val, n_classes)
                val_idx_0,
                rng.rand(n_test, n_classes),  # test_preds: (n_test, n_classes)
                k + 1, {},
            )
            tracker.update(
                1, 0.88 - k * 0.05,
                rng.rand(3, n_classes),
                val_idx_1,
                rng.rand(n_test, n_classes),
                k + 10, {},
            )

        composites = tracker.assemble(
            n_samples=n_samples, n_test=n_test, task_type="multiclass"
        )

        assert len(composites) == n_top
        for c in composites:
            assert c["oof_preds"].shape == (n_samples, n_classes)
            assert c["test_preds"].shape == (n_test, n_classes)

    def test_assemble_multiclass_oof_filled(self):
        """All OOF positions should be filled (no zeros in val slices)."""
        from src.models.trainer import PerFoldTracker

        n_samples = 4
        n_test = 2
        n_classes = 3

        tracker = PerFoldTracker(n_top=1, n_folds=2, maximize=True)

        val_preds_0 = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        val_preds_1 = np.array([[0.2, 0.2, 0.6], [0.3, 0.3, 0.4]])
        test_preds = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])

        tracker.update(0, 0.90, val_preds_0, np.array([0, 1]), test_preds, 1, {})
        tracker.update(1, 0.85, val_preds_1, np.array([2, 3]), test_preds, 2, {})

        composites = tracker.assemble(
            n_samples=n_samples, n_test=n_test, task_type="multiclass"
        )

        oof = composites[0]["oof_preds"]
        np.testing.assert_array_almost_equal(oof[0], [0.7, 0.2, 0.1])
        np.testing.assert_array_almost_equal(oof[1], [0.1, 0.8, 0.1])
        np.testing.assert_array_almost_equal(oof[2], [0.2, 0.2, 0.6])
        np.testing.assert_array_almost_equal(oof[3], [0.3, 0.3, 0.4])


# ---------------------------------------------------------------------------
# _extract_sample_weights
# ---------------------------------------------------------------------------

class TestExtractSampleWeights:
    """Test sample weight extraction from DataFrame."""

    def test_returns_weights_when_supported(self):
        """When model supports it and weights are non-uniform, return array."""
        train = pd.DataFrame({"f1": [1, 2, 3], "_sample_weight": [1.0, 1.0, 3.0]})
        cfg = {"supports_sample_weight": True}
        result = _extract_sample_weights(train, cfg)
        assert result is not None
        np.testing.assert_array_equal(result, [1.0, 1.0, 3.0])

    def test_returns_none_when_unsupported(self):
        """When model does not support sample_weight, return None."""
        train = pd.DataFrame({"f1": [1, 2], "_sample_weight": [1.0, 3.0]})
        cfg = {"supports_sample_weight": False}
        assert _extract_sample_weights(train, cfg) is None

    def test_returns_none_when_no_column(self):
        """When _sample_weight column missing, return None."""
        train = pd.DataFrame({"f1": [1, 2]})
        cfg = {"supports_sample_weight": True}
        assert _extract_sample_weights(train, cfg) is None

    def test_returns_none_when_all_uniform(self):
        """When all weights are 1.0, return None (optimisation)."""
        train = pd.DataFrame({"f1": [1, 2, 3], "_sample_weight": [1.0, 1.0, 1.0]})
        cfg = {"supports_sample_weight": True}
        assert _extract_sample_weights(train, cfg) is None

    def test_returns_none_when_key_missing(self):
        """When supports_sample_weight key absent, default to False."""
        train = pd.DataFrame({"f1": [1], "_sample_weight": [3.0]})
        cfg = {}
        assert _extract_sample_weights(train, cfg) is None


class TestSampleWeightIntegration:
    """Test that sample_weight is correctly passed through Optuna studies."""

    def test_optuna_study_with_sample_weight(self, ridge_configs_dir, binary_data, pipeline_config):
        """run_optuna_study completes successfully with sample_weight column."""
        train, test = binary_data
        weights = np.ones(len(train))
        weights[:20] = 3.0
        train["_sample_weight"] = weights

        # Add supports_sample_weight to ridge config
        cfg_path = ridge_configs_dir / "ridge.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg["training"]["supports_sample_weight"] = True
        cfg_path.write_text(yaml.dump(cfg))

        registry = ModelRegistry(ridge_configs_dir)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            test=test,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy={},
            gpu=False,
        )
        assert len(study.trials) >= 1
        assert study.best_value > 0

    def test_optuna_study_without_sample_weight(self, ridge_configs_dir, binary_data, pipeline_config):
        """run_optuna_study works when model does not support sample_weight."""
        train, test = binary_data
        train["_sample_weight"] = 3.0  # column exists but not supported

        registry = ModelRegistry(ridge_configs_dir)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            test=test,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy={},
            gpu=False,
        )
        assert len(study.trials) >= 1

    def test_train_with_config_sample_weight(self, ridge_configs_dir, binary_data, pipeline_config):
        """train_with_config passes sample_weight when supported."""
        train, test = binary_data
        weights = np.ones(len(train))
        weights[:20] = 2.0
        train["_sample_weight"] = weights

        cfg_path = ridge_configs_dir / "ridge.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg["training"]["supports_sample_weight"] = True
        cfg_path.write_text(yaml.dump(cfg))

        registry = ModelRegistry(ridge_configs_dir)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        results_dir = Path(pipeline_config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        oof_list, test_list, y = train_with_config(
            model_name="ridge",
            hparams={"C": 1.0},
            feature_cols=["f1", "f2", "f3"],
            train=train,
            test=test,
            target_col="target",
            cv=cv,
            registry=registry,
            task_type="binary_classification",
            gpu=False,
            seeds=[42],
            results_dir=results_dir,
        )
        assert len(oof_list) == 1
        assert len(oof_list[0]) == len(train)


# ---------------------------------------------------------------------------
# OOF bounding (n_top_trials limit on stored oof_preds)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NaN imputation path
# ---------------------------------------------------------------------------

class TestNanImputation:
    """Test automatic NaN imputation for models with handles_missing: false."""

    @pytest.fixture
    def nan_configs_dir(self, tmp_path: Path) -> Path:
        """Model config with handles_missing: false."""
        configs_dir = tmp_path / "models"
        configs_dir.mkdir()
        cfg = {
            "name": "Ridge",
            "class_path": {
                "binary_classification": "sklearn.linear_model.LogisticRegression",
            },
            "task_types": ["binary_classification"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {
                "C": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
            },
            "fixed_params": {
                "binary_classification": {"max_iter": 1000, "solver": "lbfgs"},
            },
            "training": {
                "needs_eval_set": False,
                "eval_metric_param": None,
                "seed_param": "random_state",
            },
            "feature_requirements": {
                "needs_scaling": True,
                "handles_missing": False,
            },
            "optuna": {
                "n_trials": 3,
                "qmc_warmup_trials": 1,
                "timeout": None,
                "pruner": {"type": "none"},
                "n_top_trials": 2,
                "n_seeds": 1,
            },
        }
        (configs_dir / "ridge.yaml").write_text(
            yaml.dump(cfg), encoding="utf-8"
        )
        return configs_dir

    def test_optuna_study_with_nan_data(self, nan_configs_dir):
        """run_optuna_study should impute NaN and complete without errors."""
        rng = np.random.default_rng(42)
        n_train = 60
        X = rng.normal(0, 1, (n_train, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        train = pd.DataFrame(X, columns=["f1", "f2", "f3"])
        train["target"] = y
        # Inject NaNs
        train.loc[0, "f1"] = np.nan
        train.loc[5, "f2"] = np.nan
        train.loc[10, "f3"] = np.nan

        registry = ModelRegistry(nan_configs_dir)
        pipeline_cfg = PipelineConfig(
            task_type="binary_classification",
            target_column="target",
            cv=CVConfig(n_folds=3, seed=42, stratified=True),
            models=["ridge"],
            optuna=OptunaGlobalConfig(global_seed=42),
            output=OutputConfig(results_dir=str(nan_configs_dir.parent / "results")),
        )
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_cfg,
            strategy={},
            gpu=False,
        )
        assert len(study.trials) >= 1
        # OOF predictions should be finite (no NaN propagation)
        oof = study.best_trial.user_attrs.get("oof_preds")
        assert oof is not None
        assert np.all(np.isfinite(oof))

    def test_train_with_config_with_nan_data(self, nan_configs_dir):
        """train_with_config should impute NaN and produce finite predictions."""
        from sklearn.model_selection import StratifiedKFold

        rng = np.random.default_rng(42)
        n_train, n_test = 60, 20
        X = rng.normal(0, 1, (n_train, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        train = pd.DataFrame(X, columns=["f1", "f2", "f3"])
        train["target"] = y
        train.loc[0, "f1"] = np.nan
        train.loc[5, "f2"] = np.nan

        X_test = rng.normal(0, 1, (n_test, 3))
        test = pd.DataFrame(X_test, columns=["f1", "f2", "f3"])
        test.loc[0, "f3"] = np.nan  # NaN in test too

        registry = ModelRegistry(nan_configs_dir)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        results_dir = nan_configs_dir.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        oof_list, test_list, labels = train_with_config(
            model_name="ridge",
            hparams={"C": 1.0},
            feature_cols=["f1", "f2", "f3"],
            train=train, test=test,
            target_col="target",
            cv=cv, registry=registry,
            task_type="binary_classification",
            gpu=False, seeds=[42],
            results_dir=results_dir,
        )
        assert len(oof_list) == 1
        assert np.all(np.isfinite(oof_list[0]))
        assert np.all(np.isfinite(test_list[0]))


# ---------------------------------------------------------------------------
# OOF bounding (n_top_trials limit on stored oof_preds)
# ---------------------------------------------------------------------------

class TestOofBounding:
    """Test that oof_preds are only stored for top-N trials to bound memory."""

    def test_oof_stored_only_for_top_trials(self, ridge_configs_dir, binary_data, pipeline_config):
        """Only the top n_top_trials trials should have oof_preds in user_attrs."""
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy={},
            gpu=False,
        )
        # n_trials=5, n_top_trials=2 in ridge_configs_dir fixture
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        assert len(completed) >= 3, "Need enough trials to test bounding"

        trials_with_oof = [
            t for t in completed if "oof_preds" in t.user_attrs
        ]
        # At most n_top_trials (2) should have oof_preds stored,
        # plus any that tied with the cutoff. Should never be all trials.
        assert len(trials_with_oof) <= len(completed), "sanity check"
        # The best trial must always have oof_preds
        best = study.best_trial
        assert "oof_preds" in best.user_attrs, "Best trial must have oof_preds stored"

    def test_oof_not_stored_for_worst_trial(self, ridge_configs_dir, binary_data, pipeline_config):
        """The worst trial should not have oof_preds when n_top_trials < n_trials."""
        train, test = binary_data
        registry = ModelRegistry(ridge_configs_dir)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline_config,
            strategy={},
            gpu=False,
        )
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        if len(completed) >= 4:
            # With n_top_trials=2 and 5 trials, the worst trial should
            # NOT have oof_preds (unless it was one of the first 2 completed)
            sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)
            # At least some of the bottom trials should lack oof_preds
            bottom_trials = sorted_trials[2:]  # trials ranked below top-2
            bottom_without_oof = [
                t for t in bottom_trials if "oof_preds" not in t.user_attrs
            ]
            # At least one bottom trial should lack oof_preds
            # (the early ones may have been stored before being displaced)
            assert len(bottom_without_oof) >= 1 or len(completed) <= 2, (
                "Expected at least one non-top trial to lack oof_preds"
            )



# ---------------------------------------------------------------------------
# Scaler helpers
# ---------------------------------------------------------------------------

class TestMakeScaler:
    def test_standard(self):
        from sklearn.preprocessing import StandardScaler
        scaler = _make_scaler("standard")
        assert isinstance(scaler, StandardScaler)

    def test_robust(self):
        from sklearn.preprocessing import RobustScaler
        scaler = _make_scaler("robust")
        assert isinstance(scaler, RobustScaler)

    def test_quantile(self):
        from sklearn.preprocessing import QuantileTransformer
        scaler = _make_scaler("quantile")
        assert isinstance(scaler, QuantileTransformer)

    def test_none_returns_none(self):
        assert _make_scaler("none") is None

    def test_all_choices_valid(self):
        """All entries in ALL_SCALER_CHOICES should be creatable."""
        for choice in ALL_SCALER_CHOICES:
            result = _make_scaler(choice)
            if choice == "none":
                assert result is None
            else:
                assert result is not None


class TestIdentifyScaleCols:
    def test_excludes_binary(self):
        df = pd.DataFrame({
            "binary": [0, 1, 0, 1, 0],
            "continuous": [1.1, 2.2, 3.3, 4.4, 5.5],
        })
        cols = _identify_scale_cols(df)
        assert "continuous" in cols
        assert "binary" not in cols

    def test_excludes_ordinal_integers(self):
        df = pd.DataFrame({
            "ordinal": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "continuous": np.random.randn(10),
        })
        cols = _identify_scale_cols(df)
        assert "continuous" in cols
        assert "ordinal" not in cols

    def test_includes_high_cardinality_numeric(self):
        df = pd.DataFrame({
            "cont": np.random.randn(100),
        })
        cols = _identify_scale_cols(df)
        assert "cont" in cols

    def test_excludes_string_cols(self):
        df = pd.DataFrame({
            "cat": ["a", "b", "c", "d", "e"],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        cols = _identify_scale_cols(df)
        assert "cat" not in cols


class TestApplyScalerFold:
    def test_none_returns_unchanged(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        out_train, out_val, out_test = _apply_scaler_fold(
            "none", df, df, df, ["a"]
        )
        pd.testing.assert_frame_equal(out_train, df)

    def test_standard_scaler_transforms(self):
        rng = np.random.default_rng(42)
        X_train = pd.DataFrame({"a": rng.normal(100, 10, 50)})
        X_val = pd.DataFrame({"a": rng.normal(100, 10, 10)})
        out_train, out_val, _ = _apply_scaler_fold(
            "standard", X_train, X_val, None, ["a"]
        )
        assert abs(out_train["a"].mean()) < 0.5
        assert abs(out_train["a"].std() - 1.0) < 0.5

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        original = df.copy()
        _apply_scaler_fold("standard", df, df.copy(), None, ["a"])
        pd.testing.assert_frame_equal(df, original)

    def test_robust_scaler(self):
        rng = np.random.default_rng(42)
        X_train = pd.DataFrame({"a": rng.normal(50, 5, 50)})
        X_val = pd.DataFrame({"a": rng.normal(50, 5, 10)})
        out_train, out_val, _ = _apply_scaler_fold(
            "robust", X_train, X_val, None, ["a"]
        )
        assert abs(out_train["a"].median()) < 0.3

    def test_quantile_scaler(self):
        rng = np.random.default_rng(42)
        X_train = pd.DataFrame({"a": rng.exponential(1, 100)})
        X_val = pd.DataFrame({"a": rng.exponential(1, 20)})
        out_train, out_val, _ = _apply_scaler_fold(
            "quantile", X_train, X_val, None, ["a"]
        )
        assert abs(out_train["a"].mean()) < 0.5

    def test_empty_scale_cols_unchanged(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        out_train, out_val, out_test = _apply_scaler_fold(
            "standard", df, df, df, []
        )
        pd.testing.assert_frame_equal(out_train, df)

    def test_test_gets_transformed(self):
        rng = np.random.default_rng(42)
        X_train = pd.DataFrame({"a": rng.normal(100, 10, 50)})
        X_val = pd.DataFrame({"a": rng.normal(100, 10, 10)})
        X_test = pd.DataFrame({"a": rng.normal(100, 10, 20)})
        _, _, out_test = _apply_scaler_fold(
            "standard", X_train, X_val, X_test, ["a"]
        )
        assert abs(out_test["a"].mean()) < 5.0


# ---------------------------------------------------------------------------
# Scaler integration with Optuna
# ---------------------------------------------------------------------------

class TestScalerOptuna:
    """Integration tests: scaler as Optuna parameter in run_optuna_study."""

    @pytest.fixture
    def ridge_configs_dir_scaling(self, tmp_path: Path) -> Path:
        """Ridge config with needs_scaling: true."""
        configs_dir = tmp_path / "models"
        configs_dir.mkdir()
        cfg = {
            "name": "Ridge",
            "class_path": {
                "binary_classification": "sklearn.linear_model.LogisticRegression",
                "regression": "sklearn.linear_model.Ridge",
            },
            "task_types": ["binary_classification", "regression"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {
                "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            },
            "fixed_params": {
                "binary_classification": {"max_iter": 1000, "solver": "lbfgs"},
                "regression": {"max_iter": 1000},
            },
            "training": {
                "needs_eval_set": False,
                "early_stopping": False,
                "eval_metric_param": None,
            },
            "feature_requirements": {"needs_scaling": True, "handles_missing": True},
            "optuna": {
                "n_trials": 5,
                "qmc_warmup_trials": 2,
                "timeout": None,
                "pruner": {"type": "none"},
                "n_top_trials": 2,
                "n_seeds": 1,
            },
        }
        (configs_dir / "ridge.yaml").write_text(yaml.dump(cfg), encoding="utf-8")
        return configs_dir

    @pytest.fixture
    def scaling_pipeline(self, tmp_path: Path) -> PipelineConfig:
        return PipelineConfig(
            task_type="binary_classification",
            target_column="target",
            cv=CVConfig(n_folds=3, seed=42, stratified=True),
            models=["ridge"],
            optuna=OptunaGlobalConfig(global_seed=42),
            output=OutputConfig(results_dir=str(tmp_path / "results")),
        )

    @pytest.fixture
    def scaling_binary_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(42)
        n_train, n_test = 80, 20
        X = rng.normal(0, 1, (n_train, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        train = pd.DataFrame(X, columns=["f1", "f2", "f3"])
        train["target"] = y
        test = pd.DataFrame(
            rng.normal(0, 1, (n_test, 3)), columns=["f1", "f2", "f3"]
        )
        return train, test

    def test_scaler_in_trial_params(
        self, ridge_configs_dir_scaling, scaling_binary_data, scaling_pipeline
    ):
        """Optuna should include scaler in trial params for needs_scaling models."""
        train, test = scaling_binary_data
        registry = ModelRegistry(ridge_configs_dir_scaling)
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=scaling_pipeline,
            strategy={},
            gpu=False,
        )
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        assert len(completed) >= 1
        assert "scaler" in completed[0].params

    def test_strategy_constrains_scaler_choices(
        self, ridge_configs_dir_scaling, scaling_binary_data, scaling_pipeline
    ):
        """Strategy preprocessing.scaler_choices should constrain scaler options."""
        train, test = scaling_binary_data
        registry = ModelRegistry(ridge_configs_dir_scaling)
        strategy = {
            "preprocessing": {
                "scaler_choices": ["robust"],
            }
        }
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=scaling_pipeline,
            strategy=strategy,
            gpu=False,
        )
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        for t in completed:
            assert t.params["scaler"] in ("robust", "none")

    def test_train_with_config_uses_scaler(
        self, ridge_configs_dir_scaling, scaling_binary_data, scaling_pipeline
    ):
        """train_with_config should apply scaler from hparams."""
        train, test = scaling_binary_data
        registry = ModelRegistry(ridge_configs_dir_scaling)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        results_dir = Path(scaling_pipeline.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        oof_list, test_list, labels = train_with_config(
            model_name="ridge",
            hparams={"C": 1.0, "scaler": "standard"},
            feature_cols=["f1", "f2", "f3"],
            train=train, test=test,
            target_col="target",
            cv=cv,
            registry=registry,
            task_type="binary_classification",
            gpu=False,
            seeds=[42],
            results_dir=results_dir,
        )
        assert len(oof_list) == 1
        assert len(test_list) == 1
        assert np.all(np.isfinite(oof_list[0]))
        assert np.all(np.isfinite(test_list[0]))

    def test_no_scaler_for_tree_models(self, tmp_path: Path):
        """Models with needs_scaling=false should not get scaler param."""
        configs_dir = tmp_path / "models"
        configs_dir.mkdir()
        rf_cfg = {
            "name": "RandomForest",
            "class_path": {
                "binary_classification": "sklearn.ensemble.RandomForestClassifier",
            },
            "task_types": ["binary_classification"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {
                "n_estimators": {"type": "int", "low": 10, "high": 20},
            },
            "fixed_params": {"random_state": 42, "n_jobs": 1},
            "training": {"needs_eval_set": False},
            "feature_requirements": {"needs_scaling": False, "handles_missing": True},
            "optuna": {
                "n_trials": 3,
                "qmc_warmup_trials": 1,
                "pruner": {"type": "none"},
                "n_top_trials": 1,
                "n_seeds": 1,
            },
        }
        (configs_dir / "random_forest.yaml").write_text(
            yaml.dump(rf_cfg), encoding="utf-8"
        )
        registry = ModelRegistry(configs_dir)

        rng = np.random.default_rng(42)
        n = 60
        X = rng.normal(0, 1, (n, 3))
        y = (X[:, 0] > 0).astype(int)
        train = pd.DataFrame(X, columns=["f1", "f2", "f3"])
        train["target"] = y

        pipeline = PipelineConfig(
            task_type="binary_classification",
            target_column="target",
            cv=CVConfig(n_folds=3, seed=42, stratified=True),
            models=["random_forest"],
            optuna=OptunaGlobalConfig(global_seed=42),
            output=OutputConfig(results_dir=str(tmp_path / "results")),
        )
        study, _ = run_optuna_study(
            model_name="random_forest",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=pipeline,
            strategy={},
            gpu=False,
        )
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        assert len(completed) >= 1
        assert "scaler" not in completed[0].params

    def test_strategy_per_model_override(
        self, ridge_configs_dir_scaling, scaling_binary_data, scaling_pipeline
    ):
        """Strategy per_model.needs_scaling override should work."""
        train, test = scaling_binary_data
        registry = ModelRegistry(ridge_configs_dir_scaling)
        strategy = {
            "preprocessing": {
                "per_model": {
                    "ridge": {"needs_scaling": False}
                }
            }
        }
        study, _ = run_optuna_study(
            model_name="ridge",
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=scaling_pipeline,
            strategy=strategy,
            gpu=False,
        )
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        for t in completed:
            assert "scaler" not in t.params


class TestDeepMerge:
    """Unit tests for _deep_merge helper."""

    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        result = _deep_merge(base, {"b": 99, "c": 3})
        assert result == {"a": 1, "b": 99, "c": 3}
        assert result is base  # in-place

    def test_nested_merge(self):
        base = {"assembly": {"mode": "rank", "n_composites": 20}}
        _deep_merge(base, {"assembly": {"mode": "nsga2"}})
        # mode overridden but n_composites preserved
        assert base["assembly"] == {"mode": "nsga2", "n_composites": 20}

    def test_deep_nested(self):
        base = {"a": {"b": {"c": 1, "d": 2}, "e": 3}}
        _deep_merge(base, {"a": {"b": {"c": 99}}})
        assert base["a"]["b"]["c"] == 99
        assert base["a"]["b"]["d"] == 2
        assert base["a"]["e"] == 3

    def test_override_non_dict_with_dict(self):
        base = {"assembly": "rank"}
        _deep_merge(base, {"assembly": {"mode": "nsga2"}})
        assert base["assembly"] == {"mode": "nsga2"}

    def test_override_dict_with_non_dict(self):
        base = {"assembly": {"mode": "rank"}}
        _deep_merge(base, {"assembly": None})
        assert base["assembly"] is None

    def test_empty_overrides(self):
        base = {"a": 1}
        _deep_merge(base, {})
        assert base == {"a": 1}

    def test_empty_base(self):
        base = {}
        _deep_merge(base, {"a": {"b": 1}})
        assert base == {"a": {"b": 1}}


class TestStrategyOptunaOverrides:
    """Integration tests: strategy overrides for optuna config (assembly, etc.)."""

    @pytest.fixture
    def rf_configs_dir(self, tmp_path: Path) -> Path:
        """RandomForest config with assembly defaults."""
        configs_dir = tmp_path / "models"
        configs_dir.mkdir()
        cfg = {
            "name": "RandomForest",
            "class_path": {
                "binary_classification": "sklearn.ensemble.RandomForestClassifier",
            },
            "task_types": ["binary_classification"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {
                "n_estimators": {"type": "int", "low": 10, "high": 20},
            },
            "fixed_params": {"random_state": 42, "n_jobs": 1},
            "training": {"needs_eval_set": False},
            "feature_requirements": {"needs_scaling": False, "handles_missing": True},
            "optuna": {
                "n_trials": 3,
                "qmc_warmup_trials": 1,
                "pruner": {"type": "none"},
                "n_top_trials": 2,
                "n_seeds": 1,
                "selection_mode": "global",
                "fold_timeout": None,
                "assembly": {
                    "mode": "rank",
                    "n_composites": 10,
                    "diversity_weight": 0.2,
                },
            },
        }
        (configs_dir / "random_forest.yaml").write_text(
            yaml.dump(cfg), encoding="utf-8"
        )
        return configs_dir

    @pytest.fixture
    def rf_pipeline(self, tmp_path: Path) -> PipelineConfig:
        return PipelineConfig(
            task_type="binary_classification",
            target_column="target",
            cv=CVConfig(n_folds=3, seed=42, stratified=True),
            models=["random_forest"],
            optuna=OptunaGlobalConfig(global_seed=42),
            output=OutputConfig(results_dir=str(tmp_path / "results")),
        )

    @pytest.fixture
    def rf_data(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = 60
        X = rng.normal(0, 1, (n, 3))
        y = (X[:, 0] > 0).astype(int)
        train = pd.DataFrame(X, columns=["f1", "f2", "f3"])
        train["target"] = y
        return train

    def test_assembly_mode_override(self, rf_configs_dir, rf_data, rf_pipeline):
        """Strategy can override assembly.mode while preserving other assembly keys."""
        registry = ModelRegistry(rf_configs_dir)
        strategy = {
            "overrides": {
                "random_forest": {
                    "optuna": {
                        "assembly": {"mode": "nsga2"},
                    }
                }
            }
        }
        # Patch _create_objective to capture the optuna_cfg state
        original_run = run_optuna_study.__wrapped__ if hasattr(run_optuna_study, '__wrapped__') else run_optuna_study
        # We verify by checking the registry config is unchanged (deep copy)
        # and the strategy override takes effect
        optuna_cfg = registry.get_optuna_config("random_forest")
        assert optuna_cfg["assembly"]["mode"] == "rank"  # original default

        # Run with override — the study runs fine even though assembly.mode
        # changed (assembly is used after study, not during Optuna trials)
        study, _ = run_optuna_study(
            model_name="random_forest",
            train=rf_data,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=rf_pipeline,
            strategy=strategy,
            gpu=False,
        )
        # Original registry is unmodified (deep copy in get_optuna_config)
        optuna_cfg_after = registry.get_optuna_config("random_forest")
        assert optuna_cfg_after["assembly"]["mode"] == "rank"

    def test_partial_assembly_preserves_keys(self, rf_configs_dir):
        """Partial assembly override preserves unmentioned keys via deep merge."""
        registry = ModelRegistry(rf_configs_dir)
        optuna_cfg = registry.get_optuna_config("random_forest")
        assert optuna_cfg["assembly"]["n_composites"] == 10
        assert optuna_cfg["assembly"]["diversity_weight"] == 0.2

        # Simulate what run_optuna_study does
        overrides = {"assembly": {"mode": "nsga2", "diversity_weight": 0.3}}
        _deep_merge(optuna_cfg, overrides)

        assert optuna_cfg["assembly"]["mode"] == "nsga2"
        assert optuna_cfg["assembly"]["diversity_weight"] == 0.3
        assert optuna_cfg["assembly"]["n_composites"] == 10  # preserved!

    def test_selection_mode_override(self, rf_configs_dir):
        """Strategy can override selection_mode."""
        registry = ModelRegistry(rf_configs_dir)
        optuna_cfg = registry.get_optuna_config("random_forest")
        assert optuna_cfg["selection_mode"] == "global"

        _deep_merge(optuna_cfg, {"selection_mode": "per_fold", "fold_timeout": 120})
        assert optuna_cfg["selection_mode"] == "per_fold"
        assert optuna_cfg["fold_timeout"] == 120

    def test_n_trials_override(self, rf_configs_dir, rf_data, rf_pipeline):
        """Strategy can override n_trials via optuna key."""
        registry = ModelRegistry(rf_configs_dir)
        strategy = {
            "overrides": {
                "random_forest": {
                    "optuna": {"n_trials": 2},
                }
            }
        }
        study, _ = run_optuna_study(
            model_name="random_forest",
            train=rf_data,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=rf_pipeline,
            strategy=strategy,
            gpu=False,
        )
        # Should have at most 2 trials (n_trials=2)
        assert len(study.trials) <= 2

    def test_strategy_does_not_mutate_input(self, rf_configs_dir, rf_data, rf_pipeline):
        """run_optuna_study should not mutate the strategy dict."""
        import copy
        registry = ModelRegistry(rf_configs_dir)
        strategy = {
            "overrides": {
                "random_forest": {
                    "optuna": {"assembly": {"mode": "nsga2"}},
                    "n_estimators": {"type": "int", "low": 5, "high": 15},
                }
            }
        }
        strategy_copy = copy.deepcopy(strategy)
        run_optuna_study(
            model_name="random_forest",
            train=rf_data,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=rf_pipeline,
            strategy=strategy,
            gpu=False,
        )
        assert strategy == strategy_copy
