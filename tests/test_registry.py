"""Tests for src/models/registry.py — Model factory and search space."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.models.registry import ModelRegistry
from src.utils.io import load_model_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_configs_dir(tmp_path: Path) -> Path:
    """Create a temp directory with model YAML configs."""
    configs_dir = tmp_path / "models"
    configs_dir.mkdir()

    # Ridge-like config (simple, no GPU)
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
        "training": {"needs_eval_set": False},
        "feature_requirements": {
            "needs_scaling": True,
            "handles_categorical": False,
            "handles_missing": False,
        },
        "optuna": {
            "n_trials": 20,
            "qmc_warmup_ratio": 0.3,
            "timeout": None,
            "pruner": {"type": "none"},
            "n_top_trials": 1,
            "n_seeds": 1,
        },
    }
    (configs_dir / "ridge.yaml").write_text(
        yaml.dump(ridge_cfg), encoding="utf-8"
    )

    # Random Forest config
    rf_cfg = {
        "name": "RandomForest",
        "class_path": {
            "binary_classification": "sklearn.ensemble.RandomForestClassifier",
            "regression": "sklearn.ensemble.RandomForestRegressor",
        },
        "task_types": ["binary_classification", "regression"],
        "gpu": {"supported": False, "params": {}, "fallback": {}},
        "hyperparameters": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 15},
        },
        "fixed_params": {"random_state": 42},
        "training": {"needs_eval_set": False},
        "feature_requirements": {
            "needs_scaling": False,
            "handles_categorical": False,
            "handles_missing": False,
        },
        "optuna": {
            "n_trials": 30,
            "qmc_warmup_ratio": 0.2,
            "n_top_trials": 3,
            "n_seeds": 2,
        },
    }
    (configs_dir / "random_forest.yaml").write_text(
        yaml.dump(rf_cfg), encoding="utf-8"
    )

    return configs_dir


@pytest.fixture
def registry(model_configs_dir: Path) -> ModelRegistry:
    return ModelRegistry(model_configs_dir)


# ---------------------------------------------------------------------------
# Init and registration
# ---------------------------------------------------------------------------

class TestRegistryInit:
    def test_auto_loads_yamls(self, registry: ModelRegistry):
        models = registry.list_models()
        assert "ridge" in models
        assert "random_forest" in models

    def test_empty_dir(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        reg = ModelRegistry(empty)
        assert reg.list_models() == []

    def test_nonexistent_dir(self, tmp_path: Path):
        reg = ModelRegistry(tmp_path / "nope")
        assert reg.list_models() == []


class TestRegister:
    def test_manual_register(self, model_configs_dir: Path):
        reg = ModelRegistry(tmp_path := model_configs_dir.parent / "empty_reg")
        tmp_path.mkdir(exist_ok=True)
        reg.register("ridge", model_configs_dir / "ridge.yaml")
        assert "ridge" in reg.list_models()


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------

class TestGetModel:
    def test_creates_logistic_regression(self, registry: ModelRegistry):
        model = registry.get_model(
            "ridge", hparams={"C": 1.0},
            task_type="binary_classification",
        )
        from sklearn.linear_model import LogisticRegression
        assert isinstance(model, LogisticRegression)
        assert model.C == 1.0

    def test_creates_ridge_regression(self, registry: ModelRegistry):
        model = registry.get_model(
            "ridge", hparams={},
            task_type="regression",
        )
        from sklearn.linear_model import Ridge
        assert isinstance(model, Ridge)

    def test_hparams_override_fixed(self, registry: ModelRegistry):
        model = registry.get_model(
            "ridge", hparams={"C": 5.0, "max_iter": 500},
            task_type="binary_classification",
        )
        assert model.C == 5.0
        assert model.max_iter == 500

    def test_task_type_specific_fixed_params(self, registry: ModelRegistry):
        model = registry.get_model(
            "ridge", hparams={"C": 1.0},
            task_type="binary_classification",
        )
        assert model.solver == "lbfgs"

    def test_unknown_model_raises(self, registry: ModelRegistry):
        with pytest.raises(KeyError, match="not registered"):
            registry.get_model("nonexistent", hparams={})

    def test_random_forest_with_hparams(self, registry: ModelRegistry):
        model = registry.get_model(
            "random_forest", hparams={"n_estimators": 100, "max_depth": 5},
            task_type="binary_classification",
        )
        assert model.n_estimators == 100
        assert model.max_depth == 5


# ---------------------------------------------------------------------------
# get_search_space
# ---------------------------------------------------------------------------

class TestGetSearchSpace:
    def test_returns_hyperparams(self, registry: ModelRegistry):
        space = registry.get_search_space("ridge")
        assert "C" in space
        assert space["C"]["type"] == "float"

    def test_overrides_applied(self, registry: ModelRegistry):
        space = registry.get_search_space(
            "random_forest",
            overrides={"max_depth": {"low": 5, "high": 8}},
        )
        assert space["max_depth"]["low"] == 5
        assert space["max_depth"]["high"] == 8
        # n_estimators unchanged
        assert space["n_estimators"]["low"] == 50

    def test_deep_copy_no_mutation(self, registry: ModelRegistry):
        space1 = registry.get_search_space("random_forest")
        space1["max_depth"]["low"] = 999
        space2 = registry.get_search_space("random_forest")
        assert space2["max_depth"]["low"] != 999


# ---------------------------------------------------------------------------
# get_optuna_config
# ---------------------------------------------------------------------------

class TestGetOptunaConfig:
    def test_returns_all_fields(self, registry: ModelRegistry):
        cfg = registry.get_optuna_config("ridge")
        assert cfg["n_trials"] == 20
        assert cfg["n_top_trials"] == 1
        assert cfg["n_seeds"] == 1

    def test_unknown_model_raises(self, registry: ModelRegistry):
        with pytest.raises(KeyError):
            registry.get_optuna_config("nope")


# ---------------------------------------------------------------------------
# get_feature_requirements
# ---------------------------------------------------------------------------

class TestGetFeatureRequirements:
    def test_ridge_needs_scaling(self, registry: ModelRegistry):
        req = registry.get_feature_requirements("ridge")
        assert req["needs_scaling"] is True
        assert req["handles_categorical"] is False

    def test_rf_no_scaling(self, registry: ModelRegistry):
        req = registry.get_feature_requirements("random_forest")
        assert req["needs_scaling"] is False


# ---------------------------------------------------------------------------
# check_gpu
# ---------------------------------------------------------------------------

class TestCheckGpu:
    def test_unsupported_returns_false(self, registry: ModelRegistry):
        result = registry.check_gpu("ridge")
        assert result is False

    def test_caches_result(self, registry: ModelRegistry):
        registry.check_gpu("ridge")
        # Second call should use cache
        assert registry._gpu_status.get("ridge_binary_classification") is False

    def test_unknown_model_returns_false(self, registry: ModelRegistry):
        assert registry.check_gpu("nonexistent") is False
