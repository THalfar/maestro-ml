"""
Utility functions for YAML loading, configuration dataclasses, and file I/O.

This module is the foundation — it must be implemented first because
every other module depends on it for loading YAML configs.
"""

from __future__ import annotations

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
    diversity_weight: float = 0.3


@dataclass
class OptunaGlobalConfig:
    """Global Optuna configuration (per-model settings are in model YAMLs)."""

    global_seed: int = 42
    global_timeout: Optional[int] = None


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
    qmc_warmup_ratio: float = 0.3
    timeout: Optional[int] = None
    pruner: dict[str, Any] = field(default_factory=lambda: {
        "type": "median",
        "n_warmup_steps": 3,
        "n_startup_trials": 10,
    })
    n_top_trials: int = 5
    n_seeds: int = 3


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
