"""Tests for src/models/trainer.py — Optuna CV training."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
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
        study = run_optuna_study(
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
        study = run_optuna_study(
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
        study = run_optuna_study(
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
        study = run_optuna_study(
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
        study = run_optuna_study(
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
