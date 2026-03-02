"""Tests for run.py — Full pipeline orchestrator."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from run import main, _parse_args


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_env(tmp_path: Path) -> dict:
    """Create a full minimal pipeline environment on disk.

    Returns dict with paths: train_csv, test_csv, pipeline_yaml,
    strategy_yaml, results_dir.
    """
    rng = np.random.default_rng(42)
    n_train, n_test = 80, 20

    # Create train/test CSVs
    X_train = rng.normal(0, 1, (n_train, 3))
    y = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    train = pd.DataFrame(X_train, columns=["f1", "f2", "f3"])
    train["id"] = range(n_train)
    train["target"] = y

    X_test = rng.normal(0, 1, (n_test, 3))
    test = pd.DataFrame(X_test, columns=["f1", "f2", "f3"])
    test["id"] = range(n_train, n_train + n_test)

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    # Create minimal model config dir
    models_dir = tmp_path / "configs" / "models"
    models_dir.mkdir(parents=True)

    ridge_cfg = {
        "name": "Ridge",
        "class_path": {
            "binary_classification": "sklearn.linear_model.LogisticRegression",
        },
        "task_types": ["binary_classification"],
        "gpu": {"supported": False, "params": {}, "fallback": {}},
        "hyperparameters": {
            "C": {"type": "float", "low": 0.1, "high": 5.0, "log": True},
        },
        "fixed_params": {
            "binary_classification": {"max_iter": 200, "solver": "lbfgs"},
        },
        "training": {
            "needs_eval_set": False,
            "eval_metric_param": None,
            "seed_param": "random_state",
        },
        "feature_requirements": {
            "needs_scaling": True,
            "handles_categorical": False,
            "handles_missing": False,
        },
        "optuna": {
            "n_trials": 3,
            "qmc_warmup_trials": 1,
            "timeout": None,
            "pruner": {"type": "none"},
            "n_top_trials": 1,
            "n_seeds": 1,
        },
    }
    (models_dir / "ridge.yaml").write_text(
        yaml.dump(ridge_cfg), encoding="utf-8"
    )

    # Strategy YAML (pre-made for manual mode)
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    strategy = {
        "features": {
            "interactions": [["f1", "f2"]],
            "ratios": [],
            "target_encoding": {"columns": [], "pairs": [], "alpha": 15},
            "custom": [],
        },
        "models": ["ridge"],
        "overrides": {},
        "reasoning": "Minimal test strategy.",
    }
    strategy_path = results_dir / "strategy.yaml"
    strategy_path.write_text(
        yaml.dump(strategy, default_flow_style=False), encoding="utf-8"
    )

    # Pipeline YAML
    pipeline_cfg = {
        "data": {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "target_column": "target",
            "id_column": "id",
            "task_type": "binary_classification",
        },
        "cv": {"n_folds": 3, "seed": 42, "stratified": True},
        "strategy": {
            "mode": "manual",
            "manual": {"strategy_input_path": str(strategy_path)},
        },
        "models": ["ridge"],
        "features": {},
        "ensemble": {
            "strategy": "blend",
            "blend_trials": 10,
            "meta_trials": 5,
            "nsga2_trials": 10,
            "diversity_weight": 0.3,
        },
        "optuna": {"global_seed": 42},
        "runtime": {
            "gpu_check": False,
            "gpu_fallback": False,
            "n_jobs": 1,
            "verbose": 0,
        },
        "output": {
            "submission_path": str(results_dir / "submission.csv"),
            "results_dir": str(results_dir),
            "save_oof": True,
        },
    }
    pipeline_yaml = tmp_path / "pipeline.yaml"
    pipeline_yaml.write_text(
        yaml.dump(pipeline_cfg, default_flow_style=False), encoding="utf-8"
    )

    return {
        "train_csv": train_path,
        "test_csv": test_path,
        "pipeline_yaml": pipeline_yaml,
        "strategy_yaml": strategy_path,
        "results_dir": results_dir,
        "models_dir": models_dir,
    }


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_config_required(self):
        with pytest.raises(SystemExit):
            _parse_args()

    def test_parses_config(self):
        with patch("sys.argv", ["run.py", "--config", "test.yaml"]):
            args = _parse_args()
            assert args.config == "test.yaml"
            assert args.strategy is None

    def test_parses_strategy_override(self):
        with patch("sys.argv", ["run.py", "--config", "t.yaml", "--strategy", "manual"]):
            args = _parse_args()
            assert args.strategy == "manual"


# ---------------------------------------------------------------------------
# main (integration)
# ---------------------------------------------------------------------------

class TestMain:
    def test_end_to_end_manual_mode(self, pipeline_env: dict):
        """Run the full pipeline in manual mode with pre-saved strategy."""
        pipeline_yaml = pipeline_env["pipeline_yaml"]
        results_dir = pipeline_env["results_dir"]

        from src.models.registry import ModelRegistry
        real_reg = ModelRegistry(pipeline_env["models_dir"])

        with patch("builtins.input", return_value=""), \
             patch("run.ModelRegistry", return_value=real_reg):
            main(pipeline_yaml)

        # Verify outputs
        submission_path = results_dir / "submission.csv"
        assert submission_path.exists(), "Submission CSV should be created"
        sub_df = pd.read_csv(submission_path)
        assert len(sub_df) == 20  # n_test rows
        assert "target" in sub_df.columns

        # Verify OOF predictions saved
        oof_path = results_dir / "oof_predictions.npy"
        assert oof_path.exists(), "OOF predictions should be saved"
        oof = np.load(str(oof_path))
        assert len(oof) == 80  # n_train rows

        # EDA report saved
        eda_path = results_dir / "eda_report.json"
        assert eda_path.exists(), "EDA report should be saved"

        # Strategy saved
        strategy_out = results_dir / "strategy.yaml"
        assert strategy_out.exists(), "Strategy should be saved"

    def test_submission_has_correct_id_column(self, pipeline_env: dict):
        """Submission should use the id_column from pipeline config."""
        pipeline_yaml = pipeline_env["pipeline_yaml"]
        results_dir = pipeline_env["results_dir"]

        from src.models.registry import ModelRegistry
        real_reg = ModelRegistry(pipeline_env["models_dir"])

        with patch("builtins.input", return_value=""), \
             patch("run.ModelRegistry", return_value=real_reg):
            main(pipeline_yaml)

        sub_df = pd.read_csv(results_dir / "submission.csv")
        assert "id" in sub_df.columns
        # IDs should be 80-99 (the test IDs)
        assert sub_df["id"].min() == 80
        assert sub_df["id"].max() == 99

    def test_predictions_are_probabilities(self, pipeline_env: dict):
        """For binary classification, predictions should be in [0, 1]."""
        pipeline_yaml = pipeline_env["pipeline_yaml"]
        results_dir = pipeline_env["results_dir"]

        from src.models.registry import ModelRegistry
        real_reg = ModelRegistry(pipeline_env["models_dir"])

        with patch("builtins.input", return_value=""), \
             patch("run.ModelRegistry", return_value=real_reg):
            main(pipeline_yaml)

        sub_df = pd.read_csv(results_dir / "submission.csv")
        assert sub_df["target"].min() >= 0.0
        assert sub_df["target"].max() <= 1.0


# ---------------------------------------------------------------------------
# Regression + log_transform_target integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_env(tmp_path: Path) -> dict:
    """Minimal regression pipeline env with log_transform_target=true.

    Target values are house-price-like (50k–300k) so log scale vs original
    scale is clearly distinguishable: log1p(150000) ≈ 11.9, original ≈ 150000.
    """
    rng = np.random.default_rng(0)
    n_train, n_test = 80, 20

    # House-price-like targets: 50k–300k
    price = rng.uniform(50_000, 300_000, n_train)
    f1 = rng.normal(0, 1, n_train)
    f2 = rng.normal(0, 1, n_train)
    cat = rng.choice(["A", "B", "C"], n_train)  # string column → ordinal encode

    train = pd.DataFrame({"id": range(n_train), "f1": f1, "f2": f2,
                           "neighborhood": cat, "SalePrice": price})
    test = pd.DataFrame({
        "id": range(n_train, n_train + n_test),
        "f1": rng.normal(0, 1, n_test),
        "f2": rng.normal(0, 1, n_test),
        "neighborhood": rng.choice(["A", "B", "C"], n_test),
    })
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    # Minimal Ridge regression model config
    models_dir = tmp_path / "configs" / "models"
    models_dir.mkdir(parents=True)
    ridge_cfg = {
        "name": "Ridge",
        "class_path": {"regression": "sklearn.linear_model.Ridge"},
        "task_types": ["regression"],
        "gpu": {"supported": False, "params": {}, "fallback": {}},
        "hyperparameters": {
            "regression": {
                "alpha": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            }
        },
        "fixed_params": {},
        "training": {
            "needs_eval_set": False,
            "eval_metric_param": None,
            "seed_param": "random_state",
            "early_stopping": False,
            "early_stopping_param": None,
            "early_stopping_rounds": None,
            "uses_callbacks_for_early_stopping": False,
        },
        "feature_requirements": {
            "needs_scaling": True,
            "handles_categorical": False,
            "handles_missing": False,
        },
        "optuna": {
            "n_trials": 4,
            "qmc_warmup_trials": 1,
            "timeout": None,
            "pruner": {"type": "none"},
            "n_top_trials": 2,   # 2 arrays → nsga2 diversity works
            "n_seeds": 1,
        },
    }
    (models_dir / "ridge.yaml").write_text(yaml.dump(ridge_cfg), encoding="utf-8")

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    strategy = {
        "features": {"interactions": [], "ratios": [], "custom": [],
                     "target_encoding": {"columns": [], "pairs": [], "alpha": 15}},
        "models": ["ridge"],
        "overrides": {},
        "reasoning": "Minimal regression test.",
    }
    strategy_path = results_dir / "strategy.yaml"
    strategy_path.write_text(yaml.dump(strategy), encoding="utf-8")

    return {
        "train_path": train_path,
        "test_path": test_path,
        "models_dir": models_dir,
        "results_dir": results_dir,
        "strategy_path": strategy_path,
        "n_train": n_train,
        "n_test": n_test,
    }


def _make_regression_pipeline_yaml(env: dict, tmp_path: Path,
                                    ensemble_strategy: str = "blend",
                                    diversity_weight=0.3) -> Path:
    """Helper: write a regression pipeline.yaml into tmp_path."""
    cfg = {
        "data": {
            "train_path": str(env["train_path"]),
            "test_path": str(env["test_path"]),
            "target_column": "SalePrice",
            "id_column": "id",
            "task_type": "regression",
            "log_transform_target": True,
        },
        "cv": {"n_folds": 3, "seed": 42, "stratified": False},
        "strategy": {
            "mode": "manual",
            "manual": {"strategy_input_path": str(env["strategy_path"])},
        },
        "models": ["ridge"],
        "features": {},
        "ensemble": {
            "strategy": ensemble_strategy,
            "blend_trials": 10,
            "meta_trials": 5,
            "nsga2_trials": 50,
            "diversity_weight": diversity_weight,
        },
        "optuna": {"global_seed": 42},
        "runtime": {"gpu_check": False, "gpu_fallback": False,
                    "n_jobs": 1, "verbose": 0},
        "output": {
            "submission_path": str(env["results_dir"] / "submission.csv"),
            "results_dir": str(env["results_dir"]),
            "save_oof": False,
        },
    }
    yaml_path = tmp_path / "pipeline_regression.yaml"
    yaml_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    return yaml_path


class TestLogTransform:
    """Verify that log_transform_target=true produces original-scale submission."""

    def test_submission_in_original_scale_blend(self, regression_env: dict, tmp_path: Path):
        """Submission must contain original-scale prices, not log1p values.

        If expm1 is missing, Ridge predicts ~11–12 (log1p scale) which would
        appear in the CSV. Original-scale prices must be >> 100.
        """
        from src.models.registry import ModelRegistry
        real_reg = ModelRegistry(regression_env["models_dir"])
        yaml_path = _make_regression_pipeline_yaml(regression_env, tmp_path, "blend")

        with patch("builtins.input", return_value=""), \
             patch("run.ModelRegistry", return_value=real_reg):
            main(yaml_path)

        sub = pd.read_csv(regression_env["results_dir"] / "submission.csv")
        assert len(sub) == regression_env["n_test"]
        assert "SalePrice" in sub.columns
        # Log-scale values would be ~11-12; original-scale must be >> 100
        assert sub["SalePrice"].min() > 100, (
            f"Predictions look like log scale: min={sub['SalePrice'].min():.2f}. "
            "expm1 inverse transform was not applied."
        )

    def test_nsga2_multi_dw_submissions_in_original_scale(
        self, regression_env: dict, tmp_path: Path
    ):
        """nsga2 with multiple diversity_weights must apply expm1 to ALL files.

        This is the specific bug that produced log-scale CSVs: the multi-dw
        save loop used raw dw_test from nsga2_submissions dict (log scale)
        instead of expm1-transformed values.
        """
        from src.models.registry import ModelRegistry
        real_reg = ModelRegistry(regression_env["models_dir"])
        yaml_path = _make_regression_pipeline_yaml(
            regression_env, tmp_path,
            ensemble_strategy="nsga2",
            diversity_weight=[0.3, 0.5],
        )

        with patch("builtins.input", return_value=""), \
             patch("run.ModelRegistry", return_value=real_reg):
            main(yaml_path)

        results_dir = regression_env["results_dir"]
        dw_files = list(results_dir.glob("submission_dw*.csv"))
        assert len(dw_files) == 2, f"Expected 2 dw submission files, got {len(dw_files)}"

        for f in dw_files:
            sub = pd.read_csv(f)
            assert sub["SalePrice"].min() > 100, (
                f"{f.name}: predictions look like log scale "
                f"(min={sub['SalePrice'].min():.2f}). "
                "expm1 was not applied in the nsga2 multi-dw save loop."
            )


# ---------------------------------------------------------------------------
# NSGA-II → meta-model stacking chain integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def nsga2_env(tmp_path: Path) -> dict:
    """Create a pipeline env with 2 diverse models for NSGA-II testing.

    Uses LogisticRegression + GaussianNB (very different algorithms)
    with n_top_trials=2 each → 4 prediction arrays for NSGA-II.
    """
    rng = np.random.default_rng(99)
    n_train, n_test = 120, 30

    X = rng.normal(0, 1, (n_train, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] - X[:, 2] > 0).astype(int)
    train = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    train["id"] = range(n_train)
    train["target"] = y

    X_test = rng.normal(0, 1, (n_test, 4))
    test = pd.DataFrame(X_test, columns=["a", "b", "c", "d"])
    test["id"] = range(n_train, n_train + n_test)

    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)

    models_dir = tmp_path / "configs" / "models"
    models_dir.mkdir(parents=True)

    # Model 1: LogisticRegression (linear)
    ridge_cfg = {
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
            "binary_classification": {"max_iter": 200, "solver": "lbfgs"},
        },
        "training": {
            "needs_eval_set": False,
            "eval_metric_param": None,
            "seed_param": "random_state",
        },
        "feature_requirements": {
            "needs_scaling": True,
            "handles_categorical": False,
            "handles_missing": False,
        },
        "optuna": {
            "n_trials": 4,
            "qmc_warmup_trials": 1,
            "timeout": None,
            "pruner": {"type": "none"},
            "n_top_trials": 2,
            "n_seeds": 1,
        },
    }
    (models_dir / "ridge.yaml").write_text(
        yaml.dump(ridge_cfg), encoding="utf-8"
    )

    # Model 2: GaussianNB (probabilistic, very different from linear)
    gnb_cfg = {
        "name": "Gaussian NB",
        "class_path": {
            "binary_classification": "sklearn.naive_bayes.GaussianNB",
        },
        "task_types": ["binary_classification"],
        "gpu": {"supported": False, "params": {}, "fallback": {}},
        "hyperparameters": {
            "var_smoothing": {
                "type": "float", "low": 1e-12, "high": 1e-6, "log": True,
            },
        },
        "fixed_params": {},
        "training": {
            "needs_eval_set": False,
            "eval_metric_param": None,
            "seed_param": None,
        },
        "feature_requirements": {
            "needs_scaling": True,
            "handles_categorical": False,
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
    (models_dir / "gaussian_nb.yaml").write_text(
        yaml.dump(gnb_cfg), encoding="utf-8"
    )

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    strategy = {
        "features": {
            "interactions": [],
            "ratios": [],
            "target_encoding": {"columns": [], "pairs": [], "alpha": 15},
            "custom": [],
        },
        "models": ["ridge", "gaussian_nb"],
        "overrides": {},
        "reasoning": "NSGA-II chain test with 2 diverse models.",
    }
    strategy_path = results_dir / "strategy.yaml"
    strategy_path.write_text(
        yaml.dump(strategy, default_flow_style=False), encoding="utf-8"
    )

    pipeline_cfg = {
        "data": {
            "train_path": str(tmp_path / "train.csv"),
            "test_path": str(tmp_path / "test.csv"),
            "target_column": "target",
            "id_column": "id",
            "task_type": "binary_classification",
        },
        "cv": {"n_folds": 3, "seed": 42, "stratified": True},
        "strategy": {
            "mode": "manual",
            "manual": {"strategy_input_path": str(strategy_path)},
        },
        "models": ["ridge", "gaussian_nb"],
        "features": {},
        "ensemble": {
            "strategy": "nsga2",
            "blend_trials": 10,
            "meta_trials": 5,
            "nsga2_trials": 50,
            "diversity_weight": 0.3,
        },
        "optuna": {"global_seed": 42},
        "runtime": {
            "gpu_check": False,
            "gpu_fallback": False,
            "n_jobs": 1,
            "verbose": 1,
        },
        "output": {
            "submission_path": str(results_dir / "submission.csv"),
            "results_dir": str(results_dir),
            "save_oof": True,
        },
    }
    pipeline_yaml = tmp_path / "pipeline.yaml"
    pipeline_yaml.write_text(
        yaml.dump(pipeline_cfg, default_flow_style=False), encoding="utf-8"
    )

    return {
        "pipeline_yaml": pipeline_yaml,
        "results_dir": results_dir,
        "models_dir": models_dir,
        "n_train": n_train,
        "n_test": n_test,
    }


class TestNSGA2MetaChain:
    """End-to-end test for NSGA-II → meta-model stacking chain.

    Verifies the complete code path: NSGA-II selects diverse models,
    then compares linear blend vs meta-model stacking, picks the
    winner, and produces valid submission + OOF predictions.
    """

    def test_nsga2_meta_chain_completes(self, nsga2_env: dict):
        """Full pipeline with nsga2 strategy produces valid output."""
        from src.models.registry import ModelRegistry
        real_reg = ModelRegistry(nsga2_env["models_dir"])

        with patch("builtins.input", return_value=""), \
             patch("run.ModelRegistry", return_value=real_reg):
            main(nsga2_env["pipeline_yaml"])

        results_dir = nsga2_env["results_dir"]

        # Submission exists and has correct shape
        sub = pd.read_csv(results_dir / "submission.csv")
        assert len(sub) == nsga2_env["n_test"]
        assert "target" in sub.columns
        assert sub["target"].min() >= 0.0
        assert sub["target"].max() <= 1.0

        # OOF predictions saved
        oof_path = results_dir / "oof_predictions.npy"
        assert oof_path.exists()
        oof = np.load(str(oof_path))
        assert len(oof) == nsga2_env["n_train"]
        assert oof.min() >= 0.0
        assert oof.max() <= 1.0

    def test_nsga2_meta_chain_with_auto_fallback(self, nsga2_env: dict):
        """Even if meta-model stacking fails, pipeline falls back gracefully."""
        from src.models.registry import ModelRegistry
        real_reg = ModelRegistry(nsga2_env["models_dir"])

        # Force meta-model to fail → should fall back to nsga2+blend
        with patch("builtins.input", return_value=""), \
             patch("run.ModelRegistry", return_value=real_reg), \
             patch("run.train_meta_model", side_effect=RuntimeError("forced fail")):
            main(nsga2_env["pipeline_yaml"])

        sub = pd.read_csv(nsga2_env["results_dir"] / "submission.csv")
        assert len(sub) == nsga2_env["n_test"]
        assert sub["target"].min() >= 0.0
        assert sub["target"].max() <= 1.0
