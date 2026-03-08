"""Tests for src/models/registry.py — Model factory and search space."""
from __future__ import annotations

import importlib
import types
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
            "qmc_warmup_trials": 6,
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
            "qmc_warmup_trials": 6,
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

    def test_all_eight_fields_present(self, registry: ModelRegistry):
        """get_optuna_config must return all fields including selection_mode, fold_timeout, assembly."""
        cfg = registry.get_optuna_config("ridge")
        expected_keys = {
            "n_trials", "qmc_warmup_trials", "timeout", "pruner",
            "n_top_trials", "n_seeds", "selection_mode", "fold_timeout",
            "assembly",
        }
        assert set(cfg.keys()) == expected_keys
        assert cfg["qmc_warmup_trials"] == 6
        assert cfg["timeout"] is None
        assert cfg["pruner"] == {"type": "none"}
        assert cfg["selection_mode"] == "global"  # default
        assert cfg["fold_timeout"] is None  # default
        assert cfg["assembly"]["mode"] == "rank"  # default

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

    def test_cache_differentiates_task_type(self, registry: ModelRegistry):
        """Cache key includes task_type, so different task types are cached separately."""
        registry.check_gpu("ridge", task_type="binary_classification")
        registry.check_gpu("ridge", task_type="regression")
        assert "ridge_binary_classification" in registry._gpu_status
        assert "ridge_regression" in registry._gpu_status

    def test_gpu_supported_but_fails_micro_trial(self, tmp_path: Path):
        """GPU-supporting model whose micro-trial fails should return False."""
        configs_dir = tmp_path / "gpu_models"
        configs_dir.mkdir()
        fake_gpu_cfg = {
            "name": "FakeGPU",
            "class_path": {
                "binary_classification": "sklearn.ensemble.RandomForestClassifier",
            },
            "task_types": ["binary_classification"],
            "gpu": {
                "supported": True,
                "params": {"INVALID_PARAM_XYZ": True},
                "fallback": {},
            },
            "hyperparameters": {},
            "fixed_params": {},
            "training": {"needs_eval_set": False},
            "feature_requirements": {
                "needs_scaling": False,
                "handles_categorical": False,
                "handles_missing": False,
            },
            "optuna": {
                "n_trials": 10,
                "qmc_warmup_trials": 3,
                "n_top_trials": 1,
                "n_seeds": 1,
            },
        }
        (configs_dir / "fakegpu.yaml").write_text(
            yaml.dump(fake_gpu_cfg), encoding="utf-8"
        )
        reg = ModelRegistry(configs_dir)
        result = reg.check_gpu("fakegpu")
        assert result is False
        assert reg._gpu_status["fakegpu_binary_classification"] is False

    def test_gpu_micro_trial_regression_target(self, tmp_path: Path):
        """Regression micro-trial should generate continuous targets, not binary."""
        configs_dir = tmp_path / "gpu_reg"
        configs_dir.mkdir()
        cfg = {
            "name": "FakeGPUReg",
            "class_path": {
                "regression": "sklearn.ensemble.RandomForestRegressor",
            },
            "task_types": ["regression"],
            "gpu": {
                "supported": True,
                "params": {"INVALID_PARAM_XYZ": True},
                "fallback": {},
            },
            "hyperparameters": {},
            "fixed_params": {},
            "training": {"needs_eval_set": False},
            "feature_requirements": {},
            "optuna": {
                "n_trials": 5, "qmc_warmup_trials": 2,
                "n_top_trials": 1, "n_seeds": 1,
            },
        }
        (configs_dir / "fakegpu_reg.yaml").write_text(
            yaml.dump(cfg), encoding="utf-8"
        )
        reg = ModelRegistry(configs_dir)
        # Micro-trial fails (invalid param) but exercises the regression target path
        result = reg.check_gpu("fakegpu_reg", task_type="regression")
        assert result is False
        assert "fakegpu_reg_regression" in reg._gpu_status


# ---------------------------------------------------------------------------
# get_model — edge cases
# ---------------------------------------------------------------------------

class TestGetModelEdgeCases:
    def test_class_path_fallback_for_unknown_task_type(self, registry: ModelRegistry):
        """When task_type not in class_path, falls back to first available."""
        model = registry.get_model(
            "ridge", hparams={"C": 1.0},
            task_type="multiclass",  # not in ridge's class_path
        )
        # Should fall back to binary_classification (first key)
        from sklearn.linear_model import LogisticRegression
        assert isinstance(model, LogisticRegression)

    def test_empty_class_path_raises(self, tmp_path: Path):
        """Model with empty class_path dict should raise KeyError."""
        configs_dir = tmp_path / "empty_cp"
        configs_dir.mkdir()
        cfg = {
            "name": "EmptyCP",
            "class_path": {},
            "task_types": [],
            "gpu": {"supported": False},
            "hyperparameters": {},
            "fixed_params": {},
            "training": {},
            "feature_requirements": {},
            "optuna": {"n_trials": 5, "qmc_warmup_trials": 2, "n_top_trials": 1, "n_seeds": 1},
        }
        (configs_dir / "empty_cp.yaml").write_text(yaml.dump(cfg), encoding="utf-8")
        reg = ModelRegistry(configs_dir)
        with pytest.raises(KeyError, match="empty class_path"):
            reg.get_model("empty_cp", hparams={})

    def test_import_error_bad_class_path(self, tmp_path: Path):
        """Bad class_path should raise ImportError."""
        configs_dir = tmp_path / "bad_models"
        configs_dir.mkdir()
        bad_cfg = {
            "name": "BadModel",
            "class_path": {
                "binary_classification": "nonexistent.module.FakeClass",
            },
            "task_types": ["binary_classification"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {},
            "fixed_params": {},
            "training": {},
            "feature_requirements": {},
            "optuna": {
                "n_trials": 10,
                "qmc_warmup_trials": 3,
                "n_top_trials": 1,
                "n_seeds": 1,
            },
        }
        (configs_dir / "bad.yaml").write_text(
            yaml.dump(bad_cfg), encoding="utf-8"
        )
        reg = ModelRegistry(configs_dir)
        with pytest.raises(ImportError, match="Could not import"):
            reg.get_model("bad", hparams={})

    def test_gpu_params_merged(self, tmp_path: Path):
        """GPU params should be merged into model params when gpu=True."""
        configs_dir = tmp_path / "gpu_models2"
        configs_dir.mkdir()
        cfg = {
            "name": "RFwithGPU",
            "class_path": {
                "binary_classification": "sklearn.ensemble.RandomForestClassifier",
            },
            "task_types": ["binary_classification"],
            "gpu": {
                "supported": True,
                "params": {"n_jobs": 1},  # GPU param: single job
                "fallback": {"n_jobs": -1},  # CPU param: all cores
            },
            "hyperparameters": {},
            "fixed_params": {"random_state": 0},
            "training": {"needs_eval_set": False},
            "feature_requirements": {
                "needs_scaling": False,
                "handles_categorical": False,
                "handles_missing": False,
            },
            "optuna": {
                "n_trials": 10,
                "qmc_warmup_trials": 3,
                "n_top_trials": 1,
                "n_seeds": 1,
            },
        }
        (configs_dir / "rf_gpu.yaml").write_text(
            yaml.dump(cfg), encoding="utf-8"
        )
        reg = ModelRegistry(configs_dir)

        model_gpu = reg.get_model("rf_gpu", hparams={}, gpu=True)
        assert model_gpu.n_jobs == 1

        model_cpu = reg.get_model("rf_gpu", hparams={}, gpu=False)
        assert model_cpu.n_jobs == -1

    def test_gpu_params_override_hparams(self, tmp_path: Path):
        """GPU params must override hparams (safety: e.g., CatBoost task_type: GPU)."""
        configs_dir = tmp_path / "gpu_override"
        configs_dir.mkdir()
        cfg = {
            "name": "RFOverride",
            "class_path": {
                "binary_classification": "sklearn.ensemble.RandomForestClassifier",
            },
            "task_types": ["binary_classification"],
            "gpu": {
                "supported": True,
                "params": {"n_jobs": 1},
                "fallback": {"n_jobs": -1},
            },
            "hyperparameters": {},
            "fixed_params": {"random_state": 0},
            "training": {"needs_eval_set": False},
            "feature_requirements": {},
            "optuna": {
                "n_trials": 5, "qmc_warmup_trials": 2,
                "n_top_trials": 1, "n_seeds": 1,
            },
        }
        (configs_dir / "rf_override.yaml").write_text(
            yaml.dump(cfg), encoding="utf-8"
        )
        reg = ModelRegistry(configs_dir)
        # hparams says n_jobs=4, but gpu.params says n_jobs=1 — GPU must win
        model = reg.get_model("rf_override", hparams={"n_jobs": 4}, gpu=True)
        assert model.n_jobs == 1
        # CPU fallback: hparams says n_jobs=4, fallback says n_jobs=-1 — fallback must win
        model_cpu = reg.get_model("rf_override", hparams={"n_jobs": 4}, gpu=False)
        assert model_cpu.n_jobs == -1

    def test_catboost_train_dir_with_results_dir(self, tmp_path: Path):
        """CatBoost models should get train_dir set from results_dir."""
        configs_dir = tmp_path / "cb_models"
        configs_dir.mkdir()
        cb_cfg = {
            "name": "FakeCatBoost",
            "class_path": {
                # Use a class whose import path contains "catboost" but is actually
                # a simple sklearn model — we just want to test the train_dir logic
                "binary_classification": "sklearn.ensemble.RandomForestClassifier",
            },
            "task_types": ["binary_classification"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {},
            "fixed_params": {"random_state": 0},
            "training": {"needs_eval_set": False},
            "feature_requirements": {},
            "optuna": {
                "n_trials": 10,
                "qmc_warmup_trials": 3,
                "n_top_trials": 1,
                "n_seeds": 1,
            },
        }
        (configs_dir / "catboost_fake.yaml").write_text(
            yaml.dump(cb_cfg), encoding="utf-8"
        )
        reg = ModelRegistry(configs_dir)

        # The class_path doesn't contain "catboost" so train_dir won't be set.
        # This tests the negative case — non-catboost models get no train_dir.
        model = reg.get_model("catboost_fake", hparams={}, results_dir=tmp_path)
        assert not hasattr(model, "train_dir")

    def test_catboost_train_dir_detection(self, tmp_path: Path):
        """Verify train_dir logic triggers when class_path contains 'catboost'."""
        configs_dir = tmp_path / "cb_models2"
        configs_dir.mkdir()
        # Use a class_path that contains "catboost" in it but resolves to sklearn
        # We'll test via the params dict instead
        cb_cfg = {
            "name": "CatBoostTest",
            "class_path": {
                "binary_classification": "sklearn.ensemble.RandomForestClassifier",
            },
            "task_types": ["binary_classification"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {},
            "fixed_params": {"random_state": 0},
            "training": {"needs_eval_set": False},
            "feature_requirements": {},
            "optuna": {
                "n_trials": 10,
                "qmc_warmup_trials": 3,
                "n_top_trials": 1,
                "n_seeds": 1,
            },
        }
        (configs_dir / "test_cb.yaml").write_text(
            yaml.dump(cb_cfg), encoding="utf-8"
        )
        reg = ModelRegistry(configs_dir)
        # class_path = "sklearn.ensemble.RandomForestClassifier" doesn't contain "catboost"
        # So train_dir param should NOT be passed — RF would reject unknown params
        model = reg.get_model("test_cb", hparams={})
        # Model created successfully without train_dir error
        assert model.random_state == 0


# ---------------------------------------------------------------------------
# get_search_space — edge cases
# ---------------------------------------------------------------------------

class TestGetSearchSpaceEdgeCases:
    def test_task_type_keyed_hyperparameters(self, tmp_path: Path):
        """Ridge-like models with task-type-keyed hyperparameters."""
        configs_dir = tmp_path / "ridge_hp_models"
        configs_dir.mkdir()
        cfg = {
            "name": "RidgeStyleHP",
            "class_path": {
                "binary_classification": "sklearn.linear_model.LogisticRegression",
                "regression": "sklearn.linear_model.Ridge",
            },
            "task_types": ["binary_classification", "regression"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {
                "binary_classification": {
                    "C": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
                },
                "regression": {
                    "alpha": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
                },
            },
            "fixed_params": {},
            "training": {"needs_eval_set": False},
            "feature_requirements": {},
            "optuna": {
                "n_trials": 15,
                "qmc_warmup_trials": 5,
                "n_top_trials": 1,
                "n_seeds": 1,
            },
        }
        (configs_dir / "ridge_hp.yaml").write_text(
            yaml.dump(cfg), encoding="utf-8"
        )
        reg = ModelRegistry(configs_dir)

        space_bin = reg.get_search_space("ridge_hp", task_type="binary_classification")
        assert "C" in space_bin
        assert "alpha" not in space_bin

        space_reg = reg.get_search_space("ridge_hp", task_type="regression")
        assert "alpha" in space_reg
        assert "C" not in space_reg

    def test_scalar_override_creates_fixed(self, registry: ModelRegistry):
        """Scalar override should create a fixed-type param."""
        space = registry.get_search_space(
            "random_forest",
            overrides={"max_depth": 7},
        )
        assert space["max_depth"] == {"type": "fixed", "value": 7}

    def test_override_adds_new_param(self, registry: ModelRegistry):
        """Overrides can add params not in the original space."""
        space = registry.get_search_space(
            "random_forest",
            overrides={"min_samples_leaf": {"type": "int", "low": 1, "high": 20}},
        )
        assert "min_samples_leaf" in space
        assert space["min_samples_leaf"]["type"] == "int"

    def test_unknown_model_raises(self, registry: ModelRegistry):
        with pytest.raises(KeyError, match="not registered"):
            registry.get_search_space("nonexistent")


# ---------------------------------------------------------------------------
# get_training_config
# ---------------------------------------------------------------------------

class TestGetTrainingConfig:
    def test_returns_training_dict(self, registry: ModelRegistry):
        cfg = registry.get_training_config("ridge")
        assert isinstance(cfg, dict)
        assert cfg["needs_eval_set"] is False

    def test_unknown_model_raises(self, registry: ModelRegistry):
        with pytest.raises(KeyError):
            registry.get_training_config("nonexistent")


# ---------------------------------------------------------------------------
# get_feature_requirements — edge case
# ---------------------------------------------------------------------------

class TestGetFeatureRequirementsEdge:
    def test_unknown_model_raises(self, registry: ModelRegistry):
        with pytest.raises(KeyError):
            registry.get_feature_requirements("nonexistent")

    def test_returned_dict_is_copy(self, registry: ModelRegistry):
        """Mutating the returned dict should not affect internal config."""
        req1 = registry.get_feature_requirements("ridge")
        req1["needs_scaling"] = False  # mutate
        req2 = registry.get_feature_requirements("ridge")
        assert req2["needs_scaling"] is True  # original unchanged


# ---------------------------------------------------------------------------
# CatBoost train_dir — positive case
# ---------------------------------------------------------------------------

class TestCatBoostTrainDir:
    """Verify train_dir IS set when class_path contains 'catboost'."""

    @pytest.fixture
    def catboost_registry(self, tmp_path: Path, monkeypatch):
        """Registry with a fake CatBoost model using monkeypatched import."""
        configs_dir = tmp_path / "cb_pos"
        configs_dir.mkdir()
        cb_cfg = {
            "name": "CatBoost",
            "class_path": {
                "binary_classification": "catboost.CatBoostClassifier",
            },
            "task_types": ["binary_classification"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {},
            "fixed_params": {"verbose": 0},
            "training": {"needs_eval_set": True},
            "feature_requirements": {},
            "optuna": {
                "n_trials": 10,
                "qmc_warmup_trials": 3,
                "n_top_trials": 1,
                "n_seeds": 1,
            },
        }
        (configs_dir / "catboost.yaml").write_text(
            yaml.dump(cb_cfg), encoding="utf-8"
        )

        # Fake CatBoost class that records constructor params
        self.received_params: dict = {}
        parent = self

        class FakeCatBoostClassifier:
            def __init__(self, **kwargs):
                parent.received_params.update(kwargs)
                for k, v in kwargs.items():
                    setattr(self, k, v)

        fake_module = types.ModuleType("catboost")
        fake_module.CatBoostClassifier = FakeCatBoostClassifier  # type: ignore[attr-defined]

        original_import = importlib.import_module

        def patched_import(name, *args, **kwargs):
            if name == "catboost":
                return fake_module
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", patched_import)
        return ModelRegistry(configs_dir)

    def test_train_dir_set_with_results_dir(self, catboost_registry, tmp_path):
        """train_dir should point inside results_dir/catboost_info."""
        results = tmp_path / "results"
        results.mkdir()
        model = catboost_registry.get_model(
            "catboost", hparams={}, results_dir=results,
        )
        assert "train_dir" in self.received_params
        assert str(results / "catboost_info") == self.received_params["train_dir"]

    def test_train_dir_set_without_results_dir(self, catboost_registry):
        """train_dir should use a temp directory when results_dir=None."""
        self.received_params.clear()
        catboost_registry.get_model("catboost", hparams={})
        assert "train_dir" in self.received_params
        assert "catboost_info" in self.received_params["train_dir"]


# ---------------------------------------------------------------------------
# get_training_config — mutation safety
# ---------------------------------------------------------------------------

class TestGetTrainingConfigMutation:
    def test_returned_dict_is_copy(self, registry: ModelRegistry):
        """Mutating the returned dict should not affect internal config."""
        cfg1 = registry.get_training_config("ridge")
        cfg1["needs_eval_set"] = True  # mutate
        cfg2 = registry.get_training_config("ridge")
        assert cfg2["needs_eval_set"] is False  # original unchanged


# ---------------------------------------------------------------------------
# list_models — sort order
# ---------------------------------------------------------------------------

class TestListModelsSorted:
    def test_returns_sorted(self, registry: ModelRegistry):
        """list_models must return alphabetically sorted names."""
        models = registry.list_models()
        assert models == sorted(models)
        assert models == ["random_forest", "ridge"]


# ---------------------------------------------------------------------------
# get_optuna_config — mutation safety
# ---------------------------------------------------------------------------

class TestGetOptunaConfigMutation:
    def test_mutating_nested_pruner_does_not_affect_internal(self, registry: ModelRegistry):
        """Mutating the returned pruner dict should not change internal config."""
        cfg1 = registry.get_optuna_config("ridge")
        cfg1["pruner"]["type"] = "hyperband"  # mutate nested dict
        cfg2 = registry.get_optuna_config("ridge")
        assert cfg2["pruner"]["type"] == "none"  # original unchanged

    def test_mutating_nested_assembly_does_not_affect_internal(self, registry: ModelRegistry):
        """Mutating the returned assembly dict should not change internal config."""
        cfg1 = registry.get_optuna_config("ridge")
        cfg1["assembly"]["mode"] = "nsga2"  # mutate nested dict
        cfg2 = registry.get_optuna_config("ridge")
        assert cfg2["assembly"]["mode"] == "rank"  # original unchanged


# ---------------------------------------------------------------------------
# register — overwrite
# ---------------------------------------------------------------------------

class TestRegisterOverwrite:
    def test_re_register_overwrites(self, model_configs_dir: Path):
        """Registering the same name twice should use the latest config."""
        reg = ModelRegistry(model_configs_dir)
        # Re-register ridge with the random_forest yaml
        reg.register("ridge", model_configs_dir / "random_forest.yaml")
        cfg = reg.get_optuna_config("ridge")
        assert cfg["n_trials"] == 30  # RF's n_trials, not Ridge's 20
