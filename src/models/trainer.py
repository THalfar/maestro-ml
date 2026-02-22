"""
Model Trainer — Optuna-based hyperparameter optimization with CV.

Runs per-model Optuna studies with QMC warmup followed by TPE sampling.
Stores OOF predictions for every completed trial, which are later used
by the ensemble layer.

Each model gets its own independent study with its own trial budget,
pruning configuration, and search space (from model YAML + LLM overrides).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

from src.models.registry import ModelRegistry
from src.utils.io import PipelineConfig


def run_optuna_study(
    model_name: str,
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    registry: ModelRegistry,
    pipeline_config: PipelineConfig,
    strategy: dict,
    gpu: bool = False,
) -> optuna.Study:
    """Run a complete Optuna study for a single model.

    Creates an Optuna study with the model's configured pruner, runs
    QMC warmup trials followed by TPE trials, and returns the study
    with all trial results.

    Args:
        model_name: Name of the model (must be registered in registry).
        train: Training DataFrame with features and target.
        feature_cols: List of feature column names to use.
        target_col: Name of the target column.
        registry: ModelRegistry instance with loaded configs.
        pipeline_config: Pipeline configuration for CV settings, etc.
        strategy: Strategy dict (may contain per-model overrides).
        gpu: Whether to use GPU for this model.

    Returns:
        Completed optuna.Study object with all trials.

    Steps:
        1. Get the model's Optuna config from registry.get_optuna_config.
        2. Get the search space from registry.get_search_space, applying
           any LLM overrides from strategy['overrides'].get(model_name).
        3. Create the CV splitter (StratifiedKFold or KFold) using
           pipeline_config.cv settings.
        4. Create the objective function via _create_objective.
        5. Configure the pruner based on model's optuna.pruner settings:
           - 'median' → optuna.pruners.MedianPruner
           - 'hyperband' → optuna.pruners.HyperbandPruner
           - 'none' → optuna.pruners.NopPruner
        6. Create the study with direction='maximize' (for AUC-ROC)
           or 'minimize' (for RMSE) based on task_type.
        7. Call _run_two_phase_study to execute QMC + TPE trials.
        8. Log the best trial value and parameters.
        9. Return the study.
    """
    raise NotImplementedError


def _create_objective(
    model_name: str,
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cv: StratifiedKFold | KFold,
    search_space: dict[str, dict[str, Any]],
    registry: ModelRegistry,
    task_type: str,
    gpu: bool,
    results_dir: Path,
) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective function for a model.

    The returned callable samples hyperparameters from the search space,
    runs cross-validation, optionally prunes early, and stores OOF
    predictions as a trial user attribute.

    Args:
        model_name: Name of the model.
        train: Training DataFrame.
        feature_cols: Feature column names.
        target_col: Target column name.
        cv: Cross-validation splitter.
        search_space: Optuna search space definition.
        registry: ModelRegistry for creating model instances.
        task_type: 'binary_classification', 'multiclass', or 'regression'.
        gpu: Whether to use GPU.
        results_dir: Directory for saving model artifacts.

    Returns:
        Callable that takes an optuna.Trial and returns the CV score.

    Steps:
        1. Define the inner objective(trial) function:
           a. For each param in search_space, call the appropriate
              trial.suggest_* method based on param['type']:
              - 'int' → trial.suggest_int(name, low, high, log=log)
              - 'float' → trial.suggest_float(name, low, high, log=log)
              - 'categorical' → trial.suggest_categorical(name, choices)
           b. Create the model via registry.get_model with sampled hparams.
           c. Initialize OOF prediction array (len(train),).
           d. For each fold_idx, (train_idx, val_idx) in enumerate(cv.split):
              - Split features and target.
              - If model needs eval_set (early stopping): fit with
                eval_set=[(X_val, y_val)] and early_stopping_rounds.
              - Else: fit normally.
              - Predict on validation set. For classification, use
                predict_proba[:, 1].
              - Store predictions in oof[val_idx].
              - Compute fold metric. Report to trial for pruning:
                trial.report(fold_metric, fold_idx).
              - Check if trial should be pruned:
                if trial.should_prune(): raise TrialPruned.
           e. Compute overall CV metric from OOF predictions.
           f. Store OOF predictions as trial.set_user_attr('oof_preds', oof).
           g. Return the overall CV metric.
        2. Return the objective function.
    """
    raise NotImplementedError


def _run_two_phase_study(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    qmc_warmup_ratio: float,
    timeout: int | None = None,
    global_seed: int = 42,
) -> None:
    """Run a two-phase Optuna study: QMC warmup then TPE.

    Phase 1 (QMC): Uses Quasi-Monte Carlo sampling for space-filling
    exploration of the search space. This provides better initial coverage
    than random sampling.

    Phase 2 (TPE): Switches to Tree-structured Parzen Estimator for
    Bayesian optimization guided by Phase 1 results.

    Args:
        study: Optuna study object (already created with pruner).
        objective: Objective function to optimize.
        n_trials: Total number of trials (QMC + TPE combined).
        qmc_warmup_ratio: Fraction of n_trials to use for QMC warmup
                          (e.g., 0.3 means 30% QMC, 70% TPE).
        timeout: Optional timeout in seconds for the entire study.
        global_seed: Seed for reproducibility.

    Steps:
        1. Calculate n_qmc = int(n_trials * qmc_warmup_ratio).
        2. Calculate n_tpe = n_trials - n_qmc.
        3. Phase 1: Set study.sampler to QMCSampler(seed=global_seed).
           Run study.optimize(objective, n_trials=n_qmc, timeout=timeout).
        4. Phase 2: Set study.sampler to TPESampler(seed=global_seed,
           n_startup_trials=0) (startup=0 because QMC already explored).
           Run study.optimize(objective, n_trials=n_tpe, timeout=remaining).
        5. Log phase completion summaries.
    """
    raise NotImplementedError


def train_with_config(
    model_name: str,
    hparams: dict[str, Any],
    feature_cols: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    cv: StratifiedKFold | KFold,
    registry: ModelRegistry,
    task_type: str,
    gpu: bool,
    seeds: list[int],
    results_dir: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Train a model with fixed hyperparameters across multiple seeds.

    Used after Optuna selects top configs. Trains with multiple seeds
    for stability and produces OOF + test predictions.

    Args:
        model_name: Name of the model.
        hparams: Fixed hyperparameters (from a top Optuna trial).
        feature_cols: Feature column names.
        train: Training DataFrame.
        test: Test DataFrame.
        target_col: Target column name.
        cv: Cross-validation splitter.
        registry: ModelRegistry instance.
        task_type: Task type string.
        gpu: Whether to use GPU.
        seeds: List of random seeds for stability.
        results_dir: Directory for model artifacts.

    Returns:
        Tuple of:
        - oof_preds_list: List of OOF prediction arrays (one per seed).
        - test_preds_list: List of test prediction arrays (one per seed).
        - labels: The true target values (y_train).

    Steps:
        1. Extract X_train, y_train, X_test from DataFrames.
        2. For each seed in seeds:
           a. Set the random state in hparams to the current seed.
           b. Create the model via registry.get_model.
           c. Initialize oof_preds (len(train),) and test_preds (len(test),).
           d. For each fold (train_idx, val_idx):
              - Fit model on training fold (with eval_set if applicable).
              - Predict on validation set → oof_preds[val_idx].
              - Predict on test set → test_preds += preds / n_folds.
           e. Append oof_preds to oof_preds_list.
           f. Append test_preds to test_preds_list.
        3. Return (oof_preds_list, test_preds_list, y_train).
    """
    raise NotImplementedError


def get_top_configs(
    study: optuna.Study,
    n_top: int = 5,
) -> list[dict[str, Any]]:
    """Extract the top N trial configurations from a completed study.

    Args:
        study: Completed Optuna study.
        n_top: Number of top configurations to extract.

    Returns:
        List of dicts, each containing:
        - params: dict of hyperparameter values
        - value: CV score achieved
        - trial_number: original trial number

    Steps:
        1. Get all completed (non-pruned) trials from the study.
        2. Sort by value (descending for maximize, ascending for minimize).
        3. Take the top n_top trials.
        4. For each, extract params, value, and number into a dict.
        5. Return the list.
    """
    raise NotImplementedError


def run_all_studies(
    pipeline_config: PipelineConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    strategy: dict,
    registry: ModelRegistry,
    gpu_status: dict[str, bool],
) -> dict[str, dict[str, Any]]:
    """Run Optuna studies for all models in the pipeline, then retrain
    top configs with multiple seeds.

    This is the main orchestrator for Layer 3 model training.

    Args:
        pipeline_config: Pipeline configuration.
        train: Training DataFrame with engineered features.
        test: Test DataFrame with engineered features.
        feature_cols: Feature column names to use.
        strategy: Strategy dict with optional per-model overrides.
        registry: ModelRegistry with loaded configs.
        gpu_status: Dict of {model_name: gpu_available_bool}.

    Returns:
        Dictionary keyed by model name, each value containing:
        - study: the Optuna Study object
        - top_configs: list of top config dicts
        - oof_preds: list of OOF prediction arrays (all seeds, all configs)
        - test_preds: list of test prediction arrays
        - labels: true target values

    Steps:
        1. Create the CV splitter from pipeline_config.cv.
        2. For each model_name in pipeline_config.models:
           a. Log the model name and start.
           b. Run run_optuna_study to get the study.
           c. Call get_top_configs to get the best N configs.
           d. For each top config, call train_with_config with
              the configured seeds.
           e. Collect all OOF and test prediction arrays.
           f. Store results in the output dict.
        3. Return the complete results dict.
    """
    raise NotImplementedError
