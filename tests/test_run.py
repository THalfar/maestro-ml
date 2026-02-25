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
            "qmc_warmup_ratio": 0.3,
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
