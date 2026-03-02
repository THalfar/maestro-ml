"""Tests for src/utils/io.py — YAML loading, dataclasses, and file I/O."""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.utils.io import (
    CVConfig,
    EnsembleConfig,
    FeatureConfig,
    ModelConfig,
    OptunaGlobalConfig,
    OptunaModelConfig,
    OutputConfig,
    PipelineConfig,
    RuntimeConfig,
    StrategyConfig,
    load_model_config,
    load_pipeline_config,
    load_yaml,
    save_eda_report,
    save_submission,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_pipeline_yaml(tmp_dir: Path) -> Path:
    content = {
        "data": {
            "train_path": "data/train.csv",
            "test_path": "data/test.csv",
            "target_column": "target",
            "id_column": "id",
            "task_type": "binary_classification",
        },
        "cv": {"n_folds": 5, "seed": 123, "stratified": True},
        "strategy": {
            "mode": "manual",
            "api": {"provider": "anthropic"},
            "manual": {"strategy_input_path": "strat.yaml"},
        },
        "models": ["catboost", "ridge"],
        "features": {
            "interactions": [["a", "b"]],
            "ratios": [["x", "y"]],
            "target_encoding": {"columns": ["cat1"], "alpha": 20},
            "custom": [{"name": "f1", "formula": "a + b"}],
        },
        "ensemble": {
            "strategy": "blend",
            "blend_trials": 100,
            "meta_trials": 50,
            "nsga2_trials": 150,
            "diversity_weight": 0.5,
        },
        "optuna": {"global_seed": 99, "global_timeout": 600},
        "runtime": {"gpu_check": False, "gpu_fallback": False, "n_jobs": 4, "verbose": 2},
        "output": {
            "submission_path": "out/sub.csv",
            "results_dir": "out/",
            "save_oof": False,
        },
    }
    path = tmp_dir / "pipeline.yaml"
    path.write_text(yaml.dump(content), encoding="utf-8")
    return path


@pytest.fixture
def sample_model_yaml(tmp_dir: Path) -> Path:
    content = {
        "name": "TestModel",
        "class_path": {"binary_classification": "sklearn.linear_model.LogisticRegression"},
        "task_types": ["binary_classification"],
        "gpu": {"supported": False},
        "hyperparameters": {
            "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
        },
        "fixed_params": {"max_iter": 1000},
        "training": {"needs_eval_set": False},
        "feature_requirements": {"needs_scaling": True},
        "optuna": {
            "n_trials": 30,
            "qmc_warmup_trials": 6,
            "timeout": 120,
            "pruner": {"type": "none"},
            "n_top_trials": 2,
            "n_seeds": 2,
        },
    }
    path = tmp_dir / "testmodel.yaml"
    path.write_text(yaml.dump(content), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_yaml
# ---------------------------------------------------------------------------

class TestLoadYaml:
    def test_valid_yaml(self, tmp_dir: Path):
        path = tmp_dir / "valid.yaml"
        path.write_text("key: value\nlist:\n  - 1\n  - 2\n", encoding="utf-8")
        result = load_yaml(path)
        assert result == {"key": "value", "list": [1, 2]}

    def test_file_not_found(self, tmp_dir: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_yaml(tmp_dir / "nonexistent.yaml")

    def test_empty_yaml(self, tmp_dir: Path):
        path = tmp_dir / "empty.yaml"
        path.write_text("", encoding="utf-8")
        result = load_yaml(path)
        assert result == {}

    def test_accepts_string_path(self, tmp_dir: Path):
        path = tmp_dir / "str.yaml"
        path.write_text("a: 1\n", encoding="utf-8")
        result = load_yaml(str(path))
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# load_pipeline_config
# ---------------------------------------------------------------------------

class TestLoadPipelineConfig:
    def test_full_parse(self, sample_pipeline_yaml: Path):
        cfg = load_pipeline_config(sample_pipeline_yaml)
        assert isinstance(cfg, PipelineConfig)
        assert cfg.train_path == "data/train.csv"
        assert cfg.target_column == "target"
        assert cfg.task_type == "binary_classification"

    def test_cv_section(self, sample_pipeline_yaml: Path):
        cfg = load_pipeline_config(sample_pipeline_yaml)
        assert cfg.cv.n_folds == 5
        assert cfg.cv.seed == 123
        assert cfg.cv.stratified is True

    def test_strategy_section(self, sample_pipeline_yaml: Path):
        cfg = load_pipeline_config(sample_pipeline_yaml)
        assert cfg.strategy.mode == "manual"
        assert cfg.strategy.api["provider"] == "anthropic"

    def test_models_list(self, sample_pipeline_yaml: Path):
        cfg = load_pipeline_config(sample_pipeline_yaml)
        assert cfg.models == ["catboost", "ridge"]

    def test_features_section(self, sample_pipeline_yaml: Path):
        cfg = load_pipeline_config(sample_pipeline_yaml)
        assert cfg.features.interactions == [["a", "b"]]
        assert cfg.features.ratios == [["x", "y"]]
        assert cfg.features.target_encoding["columns"] == ["cat1"]

    def test_ensemble_section(self, sample_pipeline_yaml: Path):
        cfg = load_pipeline_config(sample_pipeline_yaml)
        assert cfg.ensemble.strategy == "blend"
        assert cfg.ensemble.blend_trials == 100
        assert cfg.ensemble.diversity_weight == 0.5

    def test_optuna_section(self, sample_pipeline_yaml: Path):
        cfg = load_pipeline_config(sample_pipeline_yaml)
        assert cfg.optuna.global_seed == 99
        assert cfg.optuna.global_timeout == 600

    def test_runtime_section(self, sample_pipeline_yaml: Path):
        cfg = load_pipeline_config(sample_pipeline_yaml)
        assert cfg.runtime.gpu_check is False
        assert cfg.runtime.n_jobs == 4
        assert cfg.runtime.verbose == 2

    def test_output_section(self, sample_pipeline_yaml: Path):
        cfg = load_pipeline_config(sample_pipeline_yaml)
        assert cfg.output.submission_path == "out/sub.csv"
        assert cfg.output.save_oof is False

    def test_defaults_when_sections_missing(self, tmp_dir: Path):
        path = tmp_dir / "minimal.yaml"
        path.write_text("data:\n  train_path: train.csv\n", encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.cv.n_folds == 10
        assert cfg.cv.seed == 42
        assert cfg.ensemble.strategy == "auto"
        assert cfg.runtime.gpu_check is True

    def test_null_features_handled(self, tmp_dir: Path):
        """features: null should not crash."""
        path = tmp_dir / "null_feat.yaml"
        path.write_text("data:\n  train_path: t.csv\nfeatures: null\n", encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.features.interactions == []


# ---------------------------------------------------------------------------
# load_model_config
# ---------------------------------------------------------------------------

class TestLoadModelConfig:
    def test_full_parse(self, sample_model_yaml: Path):
        cfg = load_model_config(sample_model_yaml)
        assert isinstance(cfg, ModelConfig)
        assert cfg.name == "TestModel"
        assert "binary_classification" in cfg.class_path

    def test_optuna_section(self, sample_model_yaml: Path):
        cfg = load_model_config(sample_model_yaml)
        assert cfg.optuna.n_trials == 30
        assert cfg.optuna.qmc_warmup_trials == 6
        assert cfg.optuna.timeout == 120
        assert cfg.optuna.n_top_trials == 2

    def test_hyperparameters(self, sample_model_yaml: Path):
        cfg = load_model_config(sample_model_yaml)
        assert "C" in cfg.hyperparameters
        assert cfg.hyperparameters["C"]["type"] == "float"
        assert cfg.hyperparameters["C"]["log"] is True


# ---------------------------------------------------------------------------
# save_submission
# ---------------------------------------------------------------------------

class TestSaveSubmission:
    def test_creates_csv(self, tmp_dir: Path):
        ids = pd.Series([100, 200, 300], name="row_id")
        preds = np.array([0.1, 0.5, 0.9])
        path = tmp_dir / "sub" / "submission.csv"
        save_submission(ids, preds, "target", path)
        assert path.exists()
        df = pd.read_csv(path)
        assert list(df.columns) == ["row_id", "target"]
        assert len(df) == 3

    def test_creates_parent_dirs(self, tmp_dir: Path):
        path = tmp_dir / "deep" / "nested" / "sub.csv"
        ids = np.array([1, 2])
        preds = np.array([0.5, 0.6])
        save_submission(ids, preds, "t", path)
        assert path.exists()

    def test_id_column_name_from_series(self, tmp_dir: Path):
        ids = pd.Series([1, 2], name="patient_id")
        preds = np.array([0.1, 0.2])
        path = tmp_dir / "sub.csv"
        save_submission(ids, preds, "outcome", path)
        df = pd.read_csv(path)
        assert "patient_id" in df.columns

    def test_unnamed_ids_default_to_id(self, tmp_dir: Path):
        ids = np.array([10, 20])
        preds = np.array([0.3, 0.7])
        path = tmp_dir / "sub.csv"
        save_submission(ids, preds, "target", path)
        df = pd.read_csv(path)
        assert "id" in df.columns


# ---------------------------------------------------------------------------
# save_eda_report
# ---------------------------------------------------------------------------

class TestSaveEdaReport:
    def test_creates_json(self, tmp_dir: Path):
        report = {"key": "value", "nested": {"a": 1}}
        path = tmp_dir / "report.json"
        save_eda_report(report, path)
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == report

    def test_numpy_type_conversion(self, tmp_dir: Path):
        report = {
            "int_val": np.int64(42),
            "float_val": np.float64(3.14),
            "array": np.array([1, 2, 3]),
            "nested": {"x": np.float32(1.5)},
        }
        path = tmp_dir / "np_report.json"
        save_eda_report(report, path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["int_val"] == 42
        assert isinstance(loaded["int_val"], int)
        assert loaded["array"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_returns_logger(self):
        log = setup_logging(1)
        assert isinstance(log, logging.Logger)
        assert log.name == "maestro"

    def test_verbose_levels(self):
        log0 = setup_logging(0)
        assert log0.level == logging.WARNING
        log1 = setup_logging(1)
        assert log1.level == logging.INFO
        log2 = setup_logging(2)
        assert log2.level == logging.DEBUG

    def test_no_duplicate_handlers(self):
        # Reset
        logger = logging.getLogger("maestro")
        logger.handlers.clear()

        setup_logging(1)
        n1 = len(logger.handlers)
        setup_logging(1)
        n2 = len(logger.handlers)
        assert n1 == n2 == 1


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------

class TestDataclassDefaults:
    def test_cv_config_defaults(self):
        cv = CVConfig()
        assert cv.n_folds == 10
        assert cv.seed == 42
        assert cv.stratified is True

    def test_pipeline_config_defaults(self):
        cfg = PipelineConfig()
        assert cfg.task_type == "binary_classification"
        assert isinstance(cfg.cv, CVConfig)
        assert isinstance(cfg.features, FeatureConfig)
