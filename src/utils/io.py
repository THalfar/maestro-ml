"""
Utility functions for YAML loading, configuration dataclasses, and file I/O.

This module is the foundation — it must be implemented first because
every other module depends on it for loading YAML configs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CVConfig:
    """Cross-validation configuration."""

    n_folds: int = 10
    seed: int = 42
    stratified: bool = True


@dataclass
class StrategyConfig:
    """LLM strategy configuration."""

    mode: str = "manual"  # "api" | "manual"
    api: dict[str, Any] = field(default_factory=dict)
    manual: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    interactions: list[list[str]] = field(default_factory=list)
    ratios: list[list[str]] = field(default_factory=list)
    target_encoding: dict[str, Any] = field(default_factory=dict)
    custom: list[dict[str, str]] = field(default_factory=list)


@dataclass
class EnsembleConfig:
    """Ensemble strategy configuration."""

    strategy: str = "auto"
    blend_trials: int = 500
    meta_trials: int = 100
    nsga2_trials: int = 300
    diversity_weight: float | list[float] = 0.3
    diversity_metric: str = "pearson_neff"


@dataclass
class OptunaGlobalConfig:
    """Global Optuna configuration (per-model settings are in model YAMLs)."""

    global_seed: int = 42
    global_timeout: Optional[int] = None
    model_timeouts: dict[str, int] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    """Runtime environment configuration."""

    gpu_check: bool = True
    gpu_fallback: bool = True
    n_jobs: int = -1
    verbose: int = 1


@dataclass
class OutputConfig:
    """Output paths and settings."""

    submission_path: str = "results/submission.csv"
    results_dir: str = "results/"
    save_oof: bool = True


@dataclass
class PipelineConfig:
    """Complete pipeline configuration parsed from pipeline.yaml.

    Attributes:
        train_path: Path to training CSV file.
        test_path: Path to test CSV file.
        target_column: Name of the target variable column.
        id_column: Name of the row identifier column.
        task_type: One of 'binary_classification', 'multiclass', 'regression'.
        cv: Cross-validation settings.
        strategy: LLM strategy mode and API settings.
        models: List of model names referencing configs/models/{name}.yaml.
        features: Feature engineering plan (interactions, ratios, etc.).
        ensemble: Ensemble strategy and trial counts.
        optuna: Global Optuna overrides.
        runtime: GPU, parallelism, and verbosity settings.
        output: Output paths for submissions and results.
    """

    train_path: str = ""
    test_path: str = ""
    target_column: str = ""
    id_column: str = ""
    task_type: str = "binary_classification"
    target_mapping: dict[str, int] | None = None
    log_transform_target: bool = False
    cv: CVConfig = field(default_factory=CVConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    models: list[str] = field(default_factory=list)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    optuna: OptunaGlobalConfig = field(default_factory=OptunaGlobalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class OptunaModelConfig:
    """Per-model Optuna study configuration."""

    n_trials: int = 150
    qmc_warmup_trials: int = 50
    timeout: Optional[int] = None
    pruner: dict[str, Any] = field(default_factory=lambda: {
        "type": "median",
        "n_warmup_steps": 3,
        "n_startup_trials": 10,
    })
    n_top_trials: int = 5
    n_seeds: int = 3
    selection_mode: str = "global"  # "global" | "per_fold"
    fold_timeout: Optional[int] = None  # per-fold training timeout in seconds
    assembly: dict[str, Any] = field(default_factory=lambda: {
        "mode": "rank",  # "rank" | "nsga2"
    })


@dataclass
class ModelConfig:
    """Complete model configuration parsed from configs/models/{name}.yaml.

    Attributes:
        name: Display name (e.g., 'CatBoost').
        class_path: Python import path(s) keyed by task_type.
        task_types: Supported task types.
        gpu: GPU configuration (supported, params, fallback).
        hyperparameters: Optuna search space definition.
        fixed_params: Parameters always passed to constructor.
        training: Early stopping and eval metric settings.
        feature_requirements: Scaling, categorical, missing value handling.
        optuna: Per-model Optuna study settings.
    """

    name: str = ""
    class_path: dict[str, str] = field(default_factory=dict)
    task_types: list[str] = field(default_factory=list)
    gpu: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    fixed_params: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    feature_requirements: dict[str, bool] = field(default_factory=dict)
    optuna: OptunaModelConfig = field(default_factory=OptunaModelConfig)


# ---------------------------------------------------------------------------
# Time Parsing
# ---------------------------------------------------------------------------

def parse_timeout(value: int | str | None) -> int | None:
    """Parse a timeout value that can be seconds (int) or a human string.

    Supported formats: 7200, "7200", "2h", "30m", "1h30m", "90s".
    Returns seconds as int, or None if value is None/empty.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    import re
    s = str(value).strip().lower()
    if not s:
        return None
    # Pure number string
    if s.isdigit():
        return int(s)
    total = 0
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*([hms])")
    for match in pattern.finditer(s):
        amount = float(match.group(1))
        unit = match.group(2)
        if unit == "h":
            total += amount * 3600
        elif unit == "m":
            total += amount * 60
        else:
            total += amount
    return int(total) if total > 0 else None


# ---------------------------------------------------------------------------
# YAML Loading
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        path: Path to the YAML file (string or Path object).

    Returns:
        Dictionary containing the parsed YAML contents.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.

    Steps:
        1. Convert path to Path object if needed.
        2. Verify the file exists.
        3. Read and parse the YAML content using yaml.safe_load.
        4. Return the resulting dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load a pipeline YAML file and return a typed PipelineConfig dataclass.

    Args:
        path: Path to the pipeline.yaml file.

    Returns:
        PipelineConfig dataclass with all fields populated from the YAML.

    Raises:
        FileNotFoundError: If the pipeline YAML does not exist.
        KeyError: If required fields are missing from the YAML.

    Steps:
        1. Call load_yaml(path) to get the raw dictionary.
        2. Extract the 'data' section and set train_path, test_path,
           target_column, id_column, task_type.
        3. Parse the 'cv' section into a CVConfig dataclass.
        4. Parse the 'strategy' section into a StrategyConfig dataclass.
        5. Extract the 'models' list.
        6. Parse the 'features' section into a FeatureConfig dataclass.
        7. Parse the 'ensemble' section into an EnsembleConfig dataclass.
        8. Parse the 'optuna' section into an OptunaGlobalConfig dataclass.
        9. Parse the 'runtime' section into a RuntimeConfig dataclass.
        10. Parse the 'output' section into an OutputConfig dataclass.
        11. Return the assembled PipelineConfig.
    """
    raw = load_yaml(path)

    data = raw.get("data", {})
    cv_raw = raw.get("cv", {})
    strategy_raw = raw.get("strategy", {})
    features_raw = raw.get("features", {}) or {}
    ensemble_raw = raw.get("ensemble", {})
    optuna_raw = raw.get("optuna", {})
    runtime_raw = raw.get("runtime", {})
    output_raw = raw.get("output", {})

    cv = CVConfig(
        n_folds=cv_raw.get("n_folds", 10),
        seed=cv_raw.get("seed", 42),
        stratified=cv_raw.get("stratified", True),
    )

    strategy = StrategyConfig(
        mode=strategy_raw.get("mode", "manual"),
        api=strategy_raw.get("api", {}) or {},
        manual=strategy_raw.get("manual", {}) or {},
    )

    features = FeatureConfig(
        interactions=features_raw.get("interactions", []) or [],
        ratios=features_raw.get("ratios", []) or [],
        target_encoding=features_raw.get("target_encoding", {}) or {},
        custom=features_raw.get("custom", []) or [],
    )

    ensemble = EnsembleConfig(
        strategy=ensemble_raw.get("strategy", "auto"),
        blend_trials=ensemble_raw.get("blend_trials", 500),
        meta_trials=ensemble_raw.get("meta_trials", 100),
        nsga2_trials=ensemble_raw.get("nsga2_trials", 300),
        diversity_weight=ensemble_raw.get("diversity_weight", 0.3),
        diversity_metric=ensemble_raw.get("diversity_metric", "pearson_neff"),
    )

    raw_timeouts = optuna_raw.get("model_timeouts", {}) or {}
    model_timeouts = {
        name: parsed
        for name, val in raw_timeouts.items()
        if (parsed := parse_timeout(val)) is not None
    }

    optuna = OptunaGlobalConfig(
        global_seed=optuna_raw.get("global_seed", 42),
        global_timeout=parse_timeout(optuna_raw.get("global_timeout", None)),
        model_timeouts=model_timeouts,
    )

    runtime = RuntimeConfig(
        gpu_check=runtime_raw.get("gpu_check", True),
        gpu_fallback=runtime_raw.get("gpu_fallback", True),
        n_jobs=runtime_raw.get("n_jobs", -1),
        verbose=runtime_raw.get("verbose", 1),
    )

    output = OutputConfig(
        submission_path=output_raw.get("submission_path", "results/submission.csv"),
        results_dir=output_raw.get("results_dir", "results/"),
        save_oof=output_raw.get("save_oof", True),
    )

    target_mapping_raw = data.get("target_mapping")
    target_mapping = (
        {str(k): int(v) for k, v in target_mapping_raw.items()}
        if target_mapping_raw
        else None
    )

    return PipelineConfig(
        train_path=data.get("train_path", ""),
        test_path=data.get("test_path", ""),
        target_column=data.get("target_column", ""),
        id_column=data.get("id_column", ""),
        task_type=data.get("task_type", "binary_classification"),
        target_mapping=target_mapping,
        log_transform_target=bool(data.get("log_transform_target", False)),
        cv=cv,
        strategy=strategy,
        models=raw.get("models", []) or [],
        features=features,
        ensemble=ensemble,
        optuna=optuna,
        runtime=runtime,
        output=output,
    )


def load_model_config(path: str | Path) -> ModelConfig:
    """Load a model YAML file and return a typed ModelConfig dataclass.

    Args:
        path: Path to a model YAML file (e.g., configs/models/catboost.yaml).

    Returns:
        ModelConfig dataclass with all fields populated from the YAML.

    Raises:
        FileNotFoundError: If the model YAML does not exist.
        KeyError: If required fields are missing.

    Steps:
        1. Call load_yaml(path) to get the raw dictionary.
        2. Extract name, class_path, task_types, gpu, hyperparameters,
           fixed_params, training, feature_requirements.
        3. Parse the 'optuna' section into an OptunaModelConfig dataclass.
        4. Return the assembled ModelConfig.
    """
    raw = load_yaml(path)

    optuna_raw = raw.get("optuna", {})
    optuna = OptunaModelConfig(
        n_trials=optuna_raw.get("n_trials", 150),
        qmc_warmup_trials=optuna_raw.get("qmc_warmup_trials", 50),
        timeout=optuna_raw.get("timeout", None),
        pruner=optuna_raw.get("pruner", {"type": "median", "n_warmup_steps": 3, "n_startup_trials": 10}),
        n_top_trials=optuna_raw.get("n_top_trials", 5),
        n_seeds=optuna_raw.get("n_seeds", 3),
        selection_mode=optuna_raw.get("selection_mode", "global"),
        fold_timeout=optuna_raw.get("fold_timeout", None),
        assembly=optuna_raw.get("assembly", {"mode": "rank"}),
    )

    return ModelConfig(
        name=raw.get("name", ""),
        class_path=raw.get("class_path", {}),
        task_types=raw.get("task_types", []),
        gpu=raw.get("gpu", {}),
        hyperparameters=raw.get("hyperparameters", {}),
        fixed_params=raw.get("fixed_params", {}),
        training=raw.get("training", {}),
        feature_requirements=raw.get("feature_requirements", {}),
        optuna=optuna,
    )


# ---------------------------------------------------------------------------
# Output Helpers
# ---------------------------------------------------------------------------

def save_submission(
    ids: np.ndarray | pd.Series,
    preds: np.ndarray,
    target_col: str,
    path: str | Path,
) -> None:
    """Save a competition submission CSV file.

    Args:
        ids: Array of row identifiers (from the id_column).
        preds: Array of predictions (probabilities for classification,
               values for regression).
        target_col: Name of the target column for the submission header.
        path: Output path for the CSV file.

    Steps:
        1. Create a DataFrame with the id column and target_col column.
        2. Ensure the output directory exists (create if needed).
        3. Write to CSV with index=False.
        4. Log the submission path and shape.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    id_col_name = ids.name if hasattr(ids, "name") and ids.name is not None else "id"
    df = pd.DataFrame({id_col_name: ids, target_col: preds})
    df.to_csv(path, index=False)

    logger = logging.getLogger("maestro")
    logger.info(f"Submission saved: {path} | shape={df.shape}")


def save_eda_report(report: dict, path: str | Path) -> None:
    """Save the EDA report dictionary as a JSON file.

    Args:
        report: EDA report dictionary matching eda_schema.yaml structure.
        path: Output path for the JSON file.

    Steps:
        1. Ensure the output directory exists.
        2. Convert any numpy types to Python native types for JSON
           serialization (np.int64 -> int, np.float64 -> float, etc.).
        3. Write the report as formatted JSON with indent=2.
        4. Log the save path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert(x) for x in obj]
        return obj

    clean_report = _convert(report)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_report, f, indent=2)

    logger = logging.getLogger("maestro")
    logger.info(f"EDA report saved: {path}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: int = 1) -> logging.Logger:
    """Configure and return the maestro-ml logger.

    Args:
        verbose: Logging level. 0=WARNING, 1=INFO, 2=DEBUG.

    Returns:
        Configured Logger instance named 'maestro'.

    Steps:
        1. Map verbose int to logging level (0->WARNING, 1->INFO, 2->DEBUG).
        2. Create or get the 'maestro' logger.
        3. Set the logging level.
        4. Add a StreamHandler with a formatted output
           (timestamp, level, message).
        5. Return the logger.
    """
    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    level = level_map.get(verbose, logging.INFO)

    logger = logging.getLogger("maestro")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for h in logger.handlers:
            h.setLevel(level)

    return logger
