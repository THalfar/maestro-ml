"""Tests for new models: SVM, Elastic Net, Gaussian NB, AdaBoost, RealMLP.

Verifies that all new model YAML configs load correctly, produce valid
model instances, and work end-to-end through Optuna studies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import StratifiedKFold

from src.models.registry import ModelRegistry
from src.models.trainer import run_optuna_study, train_with_config
from src.utils.io import PipelineConfig, CVConfig, OptunaGlobalConfig, OutputConfig

# ---------------------------------------------------------------------------
# RealMLP availability guard
# ---------------------------------------------------------------------------
# conftest.py pre-imports torch to load shm.dll before pytabkit.
# If that pre-import failed (OSError caught silently), torch is absent from
# sys.modules. Re-importing torch at this point would cause a Windows fatal
# exception (0xc0000139 STATUS_ENTRYPOINT_NOT_FOUND) — uncatchable by Python.
# We check sys.modules instead of importing, and skip all RealMLP tests when
# torch is not already loaded.
_TORCH_OK = "torch" in sys.modules
_SKIP_REALMLP = pytest.mark.skipif(
    not _TORCH_OK,
    reason="torch/pytabkit unavailable: shm.dll failed to load (Windows). "
           "Run check_realmlp_gpu.py to diagnose.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path("configs/models")


@pytest.fixture
def registry() -> ModelRegistry:
    """Registry loaded from actual config files."""
    return ModelRegistry(CONFIGS_DIR)


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
def regression_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple regression dataset."""
    rng = np.random.default_rng(42)
    n_train, n_test = 100, 30
    X = rng.normal(0, 1, (n_train, 3))
    y = X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.1, n_train)
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
        models=["svm", "elastic_net", "gaussian_nb", "adaboost", "realmlp"],
        optuna=OptunaGlobalConfig(global_seed=42),
        output=OutputConfig(results_dir=str(tmp_path / "results")),
    )


# ---------------------------------------------------------------------------
# Registration — all new models load from YAML
# ---------------------------------------------------------------------------

class TestNewModelsRegistered:
    @pytest.mark.parametrize("model_name", [
        "svm", "elastic_net", "gaussian_nb", "adaboost", "realmlp",
    ])
    def test_model_registered(self, registry: ModelRegistry, model_name: str):
        assert model_name in registry.list_models()

    @pytest.mark.parametrize("model_name", [
        "svm", "elastic_net", "gaussian_nb", "adaboost", "realmlp",
    ])
    def test_search_space_not_empty(self, registry: ModelRegistry, model_name: str):
        space = registry.get_search_space(model_name)
        assert len(space) > 0

    @pytest.mark.parametrize("model_name", [
        "svm", "elastic_net", "gaussian_nb", "adaboost",
    ])
    def test_gpu_not_supported(self, registry: ModelRegistry, model_name: str):
        assert registry.check_gpu(model_name) is False


# ---------------------------------------------------------------------------
# Model instantiation — binary classification
# ---------------------------------------------------------------------------

class TestModelInstantiation:
    def test_svm_classifier(self, registry: ModelRegistry):
        model = registry.get_model(
            "svm", hparams={"C": 1.0, "kernel": "rbf", "gamma": "scale"},
            task_type="binary_classification",
        )
        from sklearn.svm import SVC
        assert isinstance(model, SVC)
        assert model.probability is True
        assert model.C == 1.0

    def test_svm_regressor(self, registry: ModelRegistry):
        model = registry.get_model(
            "svm", hparams={"C": 1.0, "kernel": "rbf", "gamma": "scale"},
            task_type="regression",
        )
        from sklearn.svm import SVR
        assert isinstance(model, SVR)

    def test_elastic_net_classifier(self, registry: ModelRegistry):
        model = registry.get_model(
            "elastic_net", hparams={"alpha": 0.01, "l1_ratio": 0.5},
            task_type="binary_classification",
        )
        from sklearn.linear_model import SGDClassifier
        assert isinstance(model, SGDClassifier)
        assert model.loss == "log_loss"
        assert model.penalty == "elasticnet"

    def test_elastic_net_regressor(self, registry: ModelRegistry):
        model = registry.get_model(
            "elastic_net", hparams={"alpha": 0.01, "l1_ratio": 0.5},
            task_type="regression",
        )
        from sklearn.linear_model import ElasticNet
        assert isinstance(model, ElasticNet)

    def test_gaussian_nb_classifier(self, registry: ModelRegistry):
        model = registry.get_model(
            "gaussian_nb", hparams={"var_smoothing": 1e-9},
            task_type="binary_classification",
        )
        from sklearn.naive_bayes import GaussianNB
        assert isinstance(model, GaussianNB)

    def test_adaboost_classifier(self, registry: ModelRegistry):
        model = registry.get_model(
            "adaboost", hparams={"n_estimators": 100, "learning_rate": 0.1},
            task_type="binary_classification",
        )
        from sklearn.ensemble import AdaBoostClassifier
        assert isinstance(model, AdaBoostClassifier)
        assert model.n_estimators == 100

    def test_adaboost_regressor(self, registry: ModelRegistry):
        model = registry.get_model(
            "adaboost", hparams={"n_estimators": 100, "learning_rate": 0.1},
            task_type="regression",
        )
        from sklearn.ensemble import AdaBoostRegressor
        assert isinstance(model, AdaBoostRegressor)

    @_SKIP_REALMLP
    def test_realmlp_classifier(self, registry: ModelRegistry):
        model = registry.get_model(
            "realmlp",
            hparams={"hidden_sizes": [128, 128], "lr": 0.01,
                      "batch_size": 128, "use_ls": False},
            task_type="binary_classification",
        )
        from pytabkit import RealMLP_TD_Classifier
        assert isinstance(model, RealMLP_TD_Classifier)

    @_SKIP_REALMLP
    def test_realmlp_regressor(self, registry: ModelRegistry):
        model = registry.get_model(
            "realmlp",
            hparams={"hidden_sizes": [128, 128], "lr": 0.01,
                      "batch_size": 128, "use_ls": False},
            task_type="regression",
        )
        from pytabkit import RealMLP_TD_Regressor
        assert isinstance(model, RealMLP_TD_Regressor)


# ---------------------------------------------------------------------------
# predict_proba — all classifiers must support it
# ---------------------------------------------------------------------------

class TestPredictProba:
    """Ensure all new classifiers produce valid probabilities."""

    @pytest.mark.parametrize("model_name,hparams", [
        ("svm", {"C": 1.0, "kernel": "rbf", "gamma": "scale"}),
        ("elastic_net", {"alpha": 0.01, "l1_ratio": 0.5}),
        ("gaussian_nb", {"var_smoothing": 1e-9}),
        ("adaboost", {"n_estimators": 50, "learning_rate": 0.1}),
        pytest.param(
            "realmlp", {"hidden_sizes": [128, 128], "lr": 0.01,
                        "batch_size": 64, "use_ls": False},
            marks=_SKIP_REALMLP,
        ),
    ])
    def test_predict_proba_works(self, registry: ModelRegistry,
                                  binary_data, model_name, hparams):
        train, _ = binary_data
        X = train[["f1", "f2", "f3"]].values
        y = train["target"].values

        model = registry.get_model(
            model_name, hparams=hparams,
            task_type="binary_classification",
        )
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (len(X), 2)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Feature requirements
# ---------------------------------------------------------------------------

class TestFeatureRequirements:
    @pytest.mark.parametrize("model_name,needs_scaling", [
        ("svm", True),
        ("elastic_net", True),
        ("gaussian_nb", True),
        ("adaboost", False),
        ("realmlp", False),
    ])
    def test_scaling_requirement(self, registry: ModelRegistry,
                                  model_name, needs_scaling):
        req = registry.get_feature_requirements(model_name)
        assert req["needs_scaling"] is needs_scaling


# ---------------------------------------------------------------------------
# Optuna study — quick smoke test per model
# ---------------------------------------------------------------------------

class TestOptunaStudy:
    """Run a minimal Optuna study for each new model."""

    @pytest.fixture
    def quick_pipeline(self, tmp_path: Path) -> PipelineConfig:
        return PipelineConfig(
            task_type="binary_classification",
            target_column="target",
            cv=CVConfig(n_folds=2, seed=42, stratified=True),
            models=[],
            optuna=OptunaGlobalConfig(global_seed=42),
            output=OutputConfig(results_dir=str(tmp_path / "results")),
        )

    @pytest.fixture
    def quick_configs_dir(self, tmp_path: Path) -> Path:
        """Create minimal configs for fast Optuna tests."""
        configs_dir = tmp_path / "models"
        configs_dir.mkdir()

        # SVM
        svm_cfg = {
            "name": "SVM",
            "class_path": {
                "binary_classification": "sklearn.svm.SVC",
                "regression": "sklearn.svm.SVR",
            },
            "task_types": ["binary_classification", "regression"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {
                "C": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
                "kernel": {"type": "categorical", "choices": ["rbf"]},
                "gamma": {"type": "categorical", "choices": ["scale"]},
            },
            "fixed_params": {
                "binary_classification": {"probability": True, "cache_size": 200},
                "regression": {"cache_size": 200},
            },
            "training": {"needs_eval_set": False, "early_stopping": False,
                         "eval_metric_param": None, "seed_param": None},
            "feature_requirements": {"needs_scaling": True},
            "optuna": {"n_trials": 3, "qmc_warmup_trials": 1,
                       "timeout": None, "pruner": {"type": "none"},
                       "n_top_trials": 1, "n_seeds": 1},
        }
        (configs_dir / "svm.yaml").write_text(yaml.dump(svm_cfg), encoding="utf-8")

        # Elastic Net
        enet_cfg = {
            "name": "Elastic Net",
            "class_path": {
                "binary_classification": "sklearn.linear_model.SGDClassifier",
                "regression": "sklearn.linear_model.ElasticNet",
            },
            "task_types": ["binary_classification", "regression"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {
                "alpha": {"type": "float", "low": 0.001, "high": 1.0, "log": True},
                "l1_ratio": {"type": "float", "low": 0.1, "high": 0.9},
            },
            "fixed_params": {
                "binary_classification": {"loss": "log_loss", "penalty": "elasticnet",
                                          "max_iter": 2000, "random_state": 42},
                "regression": {"max_iter": 1000, "random_state": 42},
            },
            "training": {"needs_eval_set": False, "early_stopping": False,
                         "eval_metric_param": None, "seed_param": "random_state"},
            "feature_requirements": {"needs_scaling": True},
            "optuna": {"n_trials": 3, "qmc_warmup_trials": 1,
                       "timeout": None, "pruner": {"type": "none"},
                       "n_top_trials": 1, "n_seeds": 1},
        }
        (configs_dir / "elastic_net.yaml").write_text(yaml.dump(enet_cfg), encoding="utf-8")

        # Gaussian NB
        gnb_cfg = {
            "name": "Gaussian NB",
            "class_path": {
                "binary_classification": "sklearn.naive_bayes.GaussianNB",
            },
            "task_types": ["binary_classification"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {
                "var_smoothing": {"type": "float", "low": 1e-12, "high": 1e-6, "log": True},
            },
            "fixed_params": {},
            "training": {"needs_eval_set": False, "early_stopping": False,
                         "eval_metric_param": None, "seed_param": None},
            "feature_requirements": {"needs_scaling": True},
            "optuna": {"n_trials": 3, "qmc_warmup_trials": 1,
                       "timeout": None, "pruner": {"type": "none"},
                       "n_top_trials": 1, "n_seeds": 1},
        }
        (configs_dir / "gaussian_nb.yaml").write_text(yaml.dump(gnb_cfg), encoding="utf-8")

        # AdaBoost
        ada_cfg = {
            "name": "AdaBoost",
            "class_path": {
                "binary_classification": "sklearn.ensemble.AdaBoostClassifier",
                "regression": "sklearn.ensemble.AdaBoostRegressor",
            },
            "task_types": ["binary_classification", "regression"],
            "gpu": {"supported": False, "params": {}, "fallback": {}},
            "hyperparameters": {
                "n_estimators": {"type": "int", "low": 20, "high": 100},
                "learning_rate": {"type": "float", "low": 0.05, "high": 0.5, "log": True},
            },
            "fixed_params": {"random_state": 42},
            "training": {"needs_eval_set": False, "early_stopping": False,
                         "eval_metric_param": None, "seed_param": "random_state"},
            "feature_requirements": {"needs_scaling": False},
            "optuna": {"n_trials": 3, "qmc_warmup_trials": 1,
                       "timeout": None, "pruner": {"type": "none"},
                       "n_top_trials": 1, "n_seeds": 1},
        }
        (configs_dir / "adaboost.yaml").write_text(yaml.dump(ada_cfg), encoding="utf-8")

        # RealMLP
        realmlp_cfg = {
            "name": "RealMLP",
            "class_path": {
                "binary_classification": "pytabkit.RealMLP_TD_Classifier",
                "regression": "pytabkit.RealMLP_TD_Regressor",
            },
            "task_types": ["binary_classification", "regression"],
            "gpu": {"supported": True, "params": {"device": "cuda"},
                    "fallback": {"device": "cpu"}},
            "hyperparameters": {
                "hidden_sizes": {"type": "int_list", "n": 2,
                                 "low": 8, "high": 32},
                "lr": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
            },
            "fixed_params": {"n_epochs": 10, "n_cv": 1, "n_refit": 0,
                             "verbosity": 0, "random_state": 42},
            "training": {"needs_eval_set": False, "early_stopping": False,
                         "eval_metric_param": None, "seed_param": "random_state"},
            "feature_requirements": {"needs_scaling": False,
                                     "handles_categorical": True,
                                     "handles_missing": False},
            "optuna": {"n_trials": 3, "qmc_warmup_trials": 1,
                       "timeout": None, "pruner": {"type": "none"},
                       "n_top_trials": 1, "n_seeds": 1},
        }
        (configs_dir / "realmlp.yaml").write_text(yaml.dump(realmlp_cfg), encoding="utf-8")

        return configs_dir

    @pytest.mark.parametrize("model_name", [
        "svm", "elastic_net", "gaussian_nb", "adaboost",
        pytest.param("realmlp", marks=_SKIP_REALMLP),
    ])
    def test_optuna_study_completes(self, quick_configs_dir, binary_data,
                                     quick_pipeline, model_name):
        train, _ = binary_data
        registry = ModelRegistry(quick_configs_dir)

        if model_name not in registry.list_models():
            pytest.skip(f"{model_name} not in minimal configs")

        study, _ = run_optuna_study(
            model_name=model_name,
            train=train,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            registry=registry,
            pipeline_config=quick_pipeline,
            strategy={},
            gpu=False,
        )
        assert len(study.trials) >= 1
        assert study.best_value > 0.0



# ---------------------------------------------------------------------------
# train_with_config — end-to-end retraining
# ---------------------------------------------------------------------------

class TestTrainWithConfig:
    @pytest.mark.parametrize("model_name,hparams", [
        ("svm", {"C": 1.0, "kernel": "rbf", "gamma": "scale"}),
        ("elastic_net", {"alpha": 0.01, "l1_ratio": 0.5}),
        ("gaussian_nb", {"var_smoothing": 1e-9}),
        ("adaboost", {"n_estimators": 50, "learning_rate": 0.1}),
        pytest.param(
            "realmlp", {"hidden_sizes": [128, 128], "lr": 0.01,
                        "batch_size": 64, "use_ls": False},
            marks=_SKIP_REALMLP,
        ),
    ])
    def test_train_produces_oof(self, registry, binary_data, tmp_path,
                                 model_name, hparams):
        train, test = binary_data
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        oof_list, test_list, labels = train_with_config(
            model_name=model_name,
            hparams=hparams,
            feature_cols=["f1", "f2", "f3"],
            train=train, test=test,
            target_col="target",
            cv=cv, registry=registry,
            task_type="binary_classification",
            gpu=False, seeds=[42],
            results_dir=results_dir,
        )
        assert len(oof_list) == 1
        assert len(test_list) == 1
        assert oof_list[0].shape == (len(train),)
        assert test_list[0].shape == (len(test),)
        # Valid probabilities
        assert oof_list[0].min() >= 0.0
        assert oof_list[0].max() <= 1.0
