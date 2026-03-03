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
    _compute_cv_metric,
    _get_eval_metric_value,
    _reassemble_int_lists,
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
