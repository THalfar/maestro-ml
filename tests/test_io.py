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
    parse_timeout,
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

    def test_invalid_yaml_raises(self, tmp_dir: Path):
        """Malformed YAML should raise yaml.YAMLError."""
        path = tmp_dir / "bad.yaml"
        path.write_text("key: [unclosed\n", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            load_yaml(path)


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

    def test_meta_models_default(self, tmp_dir: Path):
        """Default meta_models should be ['logreg'] when not specified."""
        path = tmp_dir / "meta_default.yaml"
        path.write_text("data:\n  train_path: t.csv\n", encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.ensemble.meta_models == ["logreg"]

    def test_meta_models_list(self, tmp_dir: Path):
        """meta_models should be parsed as a list from YAML."""
        path = tmp_dir / "meta_list.yaml"
        path.write_text(
            "data:\n  train_path: t.csv\n"
            "ensemble:\n  meta_models:\n    - xgboost\n    - logreg\n",
            encoding="utf-8",
        )
        cfg = load_pipeline_config(path)
        assert cfg.ensemble.meta_models == ["xgboost", "logreg"]

    def test_meta_trials_int(self, tmp_dir: Path):
        """meta_trials as int → same for all meta-models."""
        path = tmp_dir / "meta_int.yaml"
        path.write_text(
            "data:\n  train_path: t.csv\n"
            "ensemble:\n  meta_trials: 200\n",
            encoding="utf-8",
        )
        cfg = load_pipeline_config(path)
        assert cfg.ensemble.get_meta_trials("logreg") == 200
        assert cfg.ensemble.get_meta_trials("xgboost") == 200

    def test_meta_trials_dict(self, tmp_dir: Path):
        """meta_trials as dict → per-model trials with fallback."""
        path = tmp_dir / "meta_dict.yaml"
        path.write_text(
            "data:\n  train_path: t.csv\n"
            "ensemble:\n  meta_trials:\n    logreg: 150\n    xgboost: 80\n",
            encoding="utf-8",
        )
        cfg = load_pipeline_config(path)
        assert cfg.ensemble.get_meta_trials("logreg") == 150
        assert cfg.ensemble.get_meta_trials("xgboost") == 80
        assert cfg.ensemble.get_meta_trials("unknown") == 100  # fallback

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


# ---------------------------------------------------------------------------
# parse_timeout
# ---------------------------------------------------------------------------

class TestParseTimeout:
    def test_none_returns_none(self):
        assert parse_timeout(None) is None

    def test_int_passthrough(self):
        assert parse_timeout(7200) == 7200

    def test_float_truncated_to_int(self):
        assert parse_timeout(90.7) == 90

    def test_digit_string(self):
        assert parse_timeout("3600") == 3600

    def test_hours(self):
        assert parse_timeout("2h") == 7200

    def test_minutes(self):
        assert parse_timeout("30m") == 1800

    def test_seconds(self):
        assert parse_timeout("90s") == 90

    def test_combined_hours_minutes(self):
        assert parse_timeout("1h30m") == 5400

    def test_combined_all_units(self):
        assert parse_timeout("1h15m30s") == 4530

    def test_empty_string_returns_none(self):
        assert parse_timeout("") is None

    def test_invalid_string_returns_none(self):
        assert parse_timeout("abc") is None

    def test_case_insensitive(self):
        assert parse_timeout("2H") == 7200
        assert parse_timeout("30M") == 1800

    def test_whitespace_stripped(self):
        assert parse_timeout("  45m  ") == 2700

    def test_fractional_units(self):
        assert parse_timeout("1.5h") == 5400


# ---------------------------------------------------------------------------
# Pipeline config: target_mapping, log_transform_target, diversity_metric,
# model_timeouts with human strings, diversity_weight as list
# ---------------------------------------------------------------------------

class TestPipelineConfigExtended:
    def test_target_mapping(self, tmp_dir: Path):
        content = {
            "data": {
                "train_path": "t.csv",
                "target_column": "churn",
                "task_type": "binary_classification",
                "target_mapping": {"Yes": 1, "No": 0},
            },
        }
        path = tmp_dir / "tm.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.target_mapping == {"Yes": 1, "No": 0}

    def test_target_mapping_none_when_absent(self, tmp_dir: Path):
        content = {"data": {"train_path": "t.csv"}}
        path = tmp_dir / "no_tm.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.target_mapping is None

    def test_log_transform_target(self, tmp_dir: Path):
        content = {
            "data": {
                "train_path": "t.csv",
                "task_type": "regression",
                "log_transform_target": True,
            },
        }
        path = tmp_dir / "log.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.log_transform_target is True

    def test_log_transform_target_default_false(self, tmp_dir: Path):
        content = {"data": {"train_path": "t.csv"}}
        path = tmp_dir / "nolog.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.log_transform_target is False

    def test_diversity_metric(self, tmp_dir: Path):
        content = {
            "data": {"train_path": "t.csv"},
            "ensemble": {"diversity_metric": "spearman_neff"},
        }
        path = tmp_dir / "dm.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.ensemble.diversity_metric == "spearman_neff"

    def test_diversity_metric_default(self, tmp_dir: Path):
        content = {"data": {"train_path": "t.csv"}}
        path = tmp_dir / "dm_def.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.ensemble.diversity_metric == "pearson_neff"

    def test_model_timeouts_human_strings(self, tmp_dir: Path):
        content = {
            "data": {"train_path": "t.csv"},
            "optuna": {
                "model_timeouts": {
                    "catboost": "1h30m",
                    "xgboost": "45m",
                    "ridge": "5m",
                },
            },
        }
        path = tmp_dir / "timeouts.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.optuna.model_timeouts["catboost"] == 5400
        assert cfg.optuna.model_timeouts["xgboost"] == 2700
        assert cfg.optuna.model_timeouts["ridge"] == 300

    def test_diversity_weight_as_list(self, tmp_dir: Path):
        content = {
            "data": {"train_path": "t.csv"},
            "ensemble": {"diversity_weight": [0.3, 0.4, 0.5]},
        }
        path = tmp_dir / "dw_list.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.ensemble.diversity_weight == [0.3, 0.4, 0.5]

    def test_meta_cv_folds_explicit(self, tmp_dir: Path):
        """meta_cv_folds set explicitly should be parsed as int."""
        content = {
            "data": {"train_path": "t.csv"},
            "ensemble": {"meta_cv_folds": 15},
        }
        path = tmp_dir / "mcf.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.ensemble.meta_cv_folds == 15

    def test_meta_cv_folds_default_none(self, tmp_dir: Path):
        """Missing meta_cv_folds should default to None."""
        content = {"data": {"train_path": "t.csv"}}
        path = tmp_dir / "mcf_def.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.ensemble.meta_cv_folds is None

    def test_null_models_handled(self, tmp_dir: Path):
        """models: null should produce empty list, not None."""
        path = tmp_dir / "null_models.yaml"
        path.write_text("data:\n  train_path: t.csv\nmodels: null\n", encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.models == []

    def test_null_meta_models_defaults_to_logreg(self, tmp_dir: Path):
        """meta_models: null should default to ['logreg'], not None."""
        content = {
            "data": {"train_path": "t.csv"},
            "ensemble": {"meta_models": None},
        }
        path = tmp_dir / "null_meta.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        # BUG: currently returns None instead of ["logreg"]
        # because .get() returns None when key exists with null value
        assert cfg.ensemble.meta_models == ["logreg"]

    def test_extra_data_list_of_dicts(self, tmp_dir: Path):
        """extra_data as list of dicts should be parsed correctly."""
        content = {
            "data": {
                "train_path": "t.csv",
                "extra_data": [
                    {"path": "orig.csv", "drop_columns": ["customerID"]},
                    {"path": "extra2.csv", "target_column": "label"},
                ],
            },
        }
        path = tmp_dir / "ed.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert len(cfg.extra_data) == 2
        assert cfg.extra_data[0]["path"] == "orig.csv"
        assert cfg.extra_data[0]["drop_columns"] == ["customerID"]
        assert cfg.extra_data[1]["target_column"] == "label"

    def test_extra_data_single_string(self, tmp_dir: Path):
        """extra_data as a single string shorthand."""
        content = {
            "data": {
                "train_path": "t.csv",
                "extra_data": "orig.csv",
            },
        }
        path = tmp_dir / "ed_str.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert len(cfg.extra_data) == 1
        assert cfg.extra_data[0]["path"] == "orig.csv"

    def test_extra_data_list_of_strings(self, tmp_dir: Path):
        """extra_data as list of strings (normalized to dicts)."""
        content = {
            "data": {
                "train_path": "t.csv",
                "extra_data": ["a.csv", "b.csv"],
            },
        }
        path = tmp_dir / "ed_strs.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert len(cfg.extra_data) == 2
        assert cfg.extra_data[0] == {"path": "a.csv"}
        assert cfg.extra_data[1] == {"path": "b.csv"}

    def test_extra_data_default_empty(self, tmp_dir: Path):
        """Missing extra_data defaults to empty list."""
        content = {"data": {"train_path": "t.csv"}}
        path = tmp_dir / "no_ed.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_pipeline_config(path)
        assert cfg.extra_data == []


# ---------------------------------------------------------------------------
# load_model_config — edge cases
# ---------------------------------------------------------------------------

class TestLoadModelConfigDefaults:
    def test_minimal_model_yaml(self, tmp_dir: Path):
        """Model YAML with only name should use defaults for all other fields."""
        content = {"name": "Minimal"}
        path = tmp_dir / "minimal.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.name == "Minimal"
        assert cfg.class_path == {}
        assert cfg.task_types == []
        assert cfg.hyperparameters == {}
        assert cfg.optuna.n_trials == 150
        assert cfg.optuna.pruner["type"] == "median"


# ---------------------------------------------------------------------------
# save_submission — edge cases
# ---------------------------------------------------------------------------

class TestSaveSubmissionEdgeCases:
    def test_series_with_none_name(self, tmp_dir: Path):
        """pd.Series with name=None should fall back to 'id'."""
        ids = pd.Series([1, 2, 3])  # name defaults to None
        preds = np.array([0.1, 0.2, 0.3])
        path = tmp_dir / "sub.csv"
        save_submission(ids, preds, "target", path)
        df = pd.read_csv(path)
        assert "id" in df.columns

    def test_empty_arrays(self, tmp_dir: Path):
        """Empty arrays should produce a CSV with headers only."""
        ids = np.array([])
        preds = np.array([])
        path = tmp_dir / "empty_sub.csv"
        save_submission(ids, preds, "target", path)
        df = pd.read_csv(path)
        assert len(df) == 0
        assert "target" in df.columns


# ---------------------------------------------------------------------------
# save_eda_report — edge cases
# ---------------------------------------------------------------------------

class TestSaveEdaReportEdgeCases:
    def test_creates_parent_dirs(self, tmp_dir: Path):
        report = {"key": "value"}
        path = tmp_dir / "deep" / "nested" / "report.json"
        save_eda_report(report, path)
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == report


# ---------------------------------------------------------------------------
# setup_logging — edge cases
# ---------------------------------------------------------------------------

class TestSetupLoggingEdgeCases:
    def test_unknown_verbose_defaults_to_info(self):
        """verbose=99 should fall back to INFO level."""
        log = setup_logging(99)
        assert log.level == logging.INFO

    def test_negative_verbose_defaults_to_info(self):
        log = setup_logging(-1)
        assert log.level == logging.INFO


# ---------------------------------------------------------------------------
# parse_timeout — edge cases
# ---------------------------------------------------------------------------

class TestParseTimeoutEdgeCases:
    def test_zero_string_returns_zero(self):
        """'0' is a valid digit string, should return 0."""
        assert parse_timeout("0") == 0

    def test_zero_unit_returns_none(self):
        """'0s' matches regex but total is 0, returns None."""
        assert parse_timeout("0s") is None


# ---------------------------------------------------------------------------
# load_model_config — per-fold selection fields
# ---------------------------------------------------------------------------

class TestLoadModelConfigPerFold:
    def test_selection_mode_parsed(self, tmp_dir: Path):
        """selection_mode: per_fold should be parsed from optuna section."""
        content = {
            "name": "PerFoldModel",
            "optuna": {
                "selection_mode": "per_fold",
                "fold_timeout": 180,
            },
        }
        path = tmp_dir / "perfold.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.optuna.selection_mode == "per_fold"

    def test_fold_timeout_parsed(self, tmp_dir: Path):
        content = {
            "name": "FoldTimeout",
            "optuna": {"fold_timeout": 300},
        }
        path = tmp_dir / "ft.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.optuna.fold_timeout == 300

    def test_assembly_nsga2_parsed(self, tmp_dir: Path):
        """Full assembly config with nsga2 mode should be parsed."""
        content = {
            "name": "AssemblyModel",
            "optuna": {
                "selection_mode": "per_fold",
                "assembly": {
                    "mode": "nsga2",
                    "n_composites": 20,
                    "n_generations": 50,
                    "pop_size": 100,
                    "diversity_metric": "spearman_neff",
                    "diversity_weight": 0.3,
                },
            },
        }
        path = tmp_dir / "asm.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.optuna.assembly["mode"] == "nsga2"
        assert cfg.optuna.assembly["n_composites"] == 20
        assert cfg.optuna.assembly["diversity_metric"] == "spearman_neff"

    def test_selection_mode_default_global(self, tmp_dir: Path):
        """Missing selection_mode should default to 'global'."""
        content = {"name": "DefaultModel"}
        path = tmp_dir / "default.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.optuna.selection_mode == "global"
        assert cfg.optuna.fold_timeout is None
        assert cfg.optuna.assembly == {"mode": "rank"}


# ---------------------------------------------------------------------------
# save_eda_report — np.bool_ handling (NumPy 2.x)
# ---------------------------------------------------------------------------

class TestLoadModelConfigNewFields:
    """Tests for newer OptunaModelConfig fields: tracker, diversity_pruning,
    substudy, enqueue_trials, tpe."""

    def test_tracker_config_parsed(self, tmp_dir: Path):
        content = {
            "name": "TrackerModel",
            "optuna": {
                "tracker": {
                    "diversity_mode": "tiered",
                    "tier1_size": 5,
                    "tier2_corr_threshold": 0.99,
                },
            },
        }
        path = tmp_dir / "tracker.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.optuna.tracker["diversity_mode"] == "tiered"
        assert cfg.optuna.tracker["tier1_size"] == 5
        assert cfg.optuna.tracker["tier2_corr_threshold"] == 0.99

    def test_tracker_default_empty(self, tmp_dir: Path):
        content = {"name": "NoTracker"}
        path = tmp_dir / "no_tracker.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.optuna.tracker == {}

    def test_diversity_pruning_parsed(self, tmp_dir: Path):
        content = {
            "name": "DivPrune",
            "optuna": {
                "diversity_pruning": {
                    "corr_threshold": 0.995,
                    "warmup_entries": 5,
                    "n_consecutive": 2,
                    "score_tolerance": 0.001,
                },
            },
        }
        path = tmp_dir / "divprune.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.optuna.diversity_pruning is not None
        assert cfg.optuna.diversity_pruning["corr_threshold"] == 0.995
        assert cfg.optuna.diversity_pruning["n_consecutive"] == 2

    def test_diversity_pruning_default_none(self, tmp_dir: Path):
        content = {"name": "NoDivPrune"}
        path = tmp_dir / "no_divprune.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.optuna.diversity_pruning is None

    def test_substudy_parsed(self, tmp_dir: Path):
        content = {
            "name": "SubstudyModel",
            "optuna": {
                "substudy": {
                    "enabled": True,
                    "sample_fraction": 0.10,
                    "n_folds": 3,
                    "timeout": "15m",
                    "n_trials": 100,
                    "n_enqueue": 20,
                    "temperature": 0.3,
                    "lock_scaler": True,
                },
            },
        }
        path = tmp_dir / "substudy.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.optuna.substudy is not None
        assert cfg.optuna.substudy["enabled"] is True
        assert cfg.optuna.substudy["sample_fraction"] == 0.10
        assert cfg.optuna.substudy["timeout"] == "15m"
        assert cfg.optuna.substudy["lock_scaler"] is True

    def test_substudy_default_none(self, tmp_dir: Path):
        content = {"name": "NoSubstudy"}
        path = tmp_dir / "no_sub.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.optuna.substudy is None

    def test_enqueue_trials_parsed(self, tmp_dir: Path):
        content = {
            "name": "EnqueueModel",
            "optuna": {
                "enqueue_trials": [
                    {"max_depth": 6, "learning_rate": 0.03},
                    {"max_depth": 8},
                ],
            },
        }
        path = tmp_dir / "enqueue.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.optuna.enqueue_trials is not None
        assert len(cfg.optuna.enqueue_trials) == 2
        assert cfg.optuna.enqueue_trials[0]["max_depth"] == 6
        assert cfg.optuna.enqueue_trials[1] == {"max_depth": 8}

    def test_enqueue_trials_default_none(self, tmp_dir: Path):
        content = {"name": "NoEnqueue"}
        path = tmp_dir / "no_enqueue.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.optuna.enqueue_trials is None

    def test_tpe_config_parsed(self, tmp_dir: Path):
        content = {
            "name": "TpeModel",
            "optuna": {
                "tpe": {
                    "gamma_ratio": 0.15,
                    "gamma_min": 5,
                    "n_startup_trials": 0,
                },
            },
        }
        path = tmp_dir / "tpe.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.optuna.tpe is not None
        assert cfg.optuna.tpe["gamma_ratio"] == 0.15
        assert cfg.optuna.tpe["gamma_min"] == 5

    def test_tpe_default_none(self, tmp_dir: Path):
        content = {"name": "NoTpe"}
        path = tmp_dir / "no_tpe.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        cfg = load_model_config(path)
        assert cfg.optuna.tpe is None


class TestSaveEdaReportNpBool:
    def test_numpy_bool_in_report(self, tmp_dir: Path):
        """np.bool_ must be converted to native bool for JSON serialization.

        With NumPy 2.x, np.bool_ is no longer a subclass of int, so
        json.dump raises TypeError if _convert doesn't handle it.
        """
        report = {
            "flag": np.bool_(True),
            "nested": {"active": np.bool_(False)},
        }
        path = tmp_dir / "bool_report.json"
        save_eda_report(report, path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["flag"] is True
        assert loaded["nested"]["active"] is False
