"""
Model Trainer — Optuna-based hyperparameter optimization with CV.

Runs per-model Optuna studies with QMC warmup followed by TPE sampling.
Stores OOF predictions for every completed trial, which are later used
by the ensemble layer.

Each model gets its own independent study with its own trial budget,
pruning configuration, and search space (from model YAML + LLM overrides).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold

from src.models.registry import ModelRegistry
from src.utils.io import PipelineConfig

logger = logging.getLogger("maestro")

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _compute_cv_metric(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> float:
    """Compute the cross-validation metric for a given task type."""
    if task_type == "binary_classification":
        return float(roc_auc_score(y_true, y_pred))
    elif task_type == "multiclass":
        return float(roc_auc_score(y_true, y_pred, multi_class="ovr"))
    else:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _get_eval_metric_value(training: dict, task_type: str, gpu: bool) -> str | None:
    """Extract the eval metric value from training config for a given task/device."""
    em = training.get("eval_metric")
    if not em:
        return None
    # Level 1: by task_type
    if isinstance(em, dict):
        task_em = em.get(task_type, em)
        if isinstance(task_em, str):
            return task_em
        if isinstance(task_em, dict):
            # Has gpu/cpu variants
            return task_em.get("gpu" if gpu else "cpu", next(iter(task_em.values()), None))
    return str(em)


def run_optuna_study(
    model_name: str,
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    registry: ModelRegistry,
    pipeline_config: PipelineConfig,
    strategy: dict,
    gpu: bool = False,
    timeout_override: int | None = None,
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
    optuna_cfg = registry.get_optuna_config(model_name)

    # Search space with optional LLM overrides
    overrides_all = strategy.get("overrides", {}) or {}
    model_overrides = overrides_all.get(model_name, None)
    search_space = registry.get_search_space(model_name, overrides=model_overrides)

    # CV splitter
    cv_cfg = pipeline_config.cv
    task_type = pipeline_config.task_type
    if cv_cfg.stratified and task_type != "regression":
        cv = StratifiedKFold(n_splits=cv_cfg.n_folds, shuffle=True, random_state=cv_cfg.seed)
    else:
        cv = KFold(n_splits=cv_cfg.n_folds, shuffle=True, random_state=cv_cfg.seed)

    results_dir = Path(pipeline_config.output.results_dir)

    objective = _create_objective(
        model_name=model_name,
        train=train,
        feature_cols=feature_cols,
        target_col=target_col,
        cv=cv,
        search_space=search_space,
        registry=registry,
        task_type=task_type,
        gpu=gpu,
        results_dir=results_dir,
    )

    # Configure pruner
    pruner_cfg = optuna_cfg.get("pruner", {}) or {}
    pruner_type = pruner_cfg.get("type", "median")
    if pruner_type == "median":
        pruner = optuna.pruners.MedianPruner(
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 3),
            n_startup_trials=pruner_cfg.get("n_startup_trials", 10),
        )
    elif pruner_type == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    # Study direction
    direction = "minimize" if task_type == "regression" else "maximize"

    study = optuna.create_study(direction=direction, pruner=pruner)

    # Pipeline-level timeout override takes precedence over model YAML timeout
    effective_timeout = timeout_override if timeout_override is not None else optuna_cfg.get("timeout")
    if effective_timeout:
        logger.info(f"[{model_name}] Timeout: {effective_timeout}s ({effective_timeout / 3600:.1f}h)")

    _run_two_phase_study(
        study=study,
        objective=objective,
        n_trials=optuna_cfg["n_trials"],
        qmc_warmup_ratio=optuna_cfg["qmc_warmup_ratio"],
        timeout=effective_timeout,
        global_seed=pipeline_config.optuna.global_seed,
        verbose=pipeline_config.runtime.verbose,
    )

    best = study.best_trial
    display_params = _reassemble_int_lists(dict(best.params))
    logger.info(
        f"[{model_name}] Best trial #{best.number}: "
        f"value={best.value:.6f}, params={display_params}"
    )

    return study


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
    X = train[feature_cols]
    y = train[target_col].values

    training_cfg = registry.get_training_config(model_name)

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        hparams: dict[str, Any] = {}
        for param_name, spec in search_space.items():
            param_type = spec.get("type", "float")
            use_log = spec.get("log", False)
            if param_type == "fixed":
                hparams[param_name] = spec["value"]
            elif param_type == "int":
                hparams[param_name] = trial.suggest_int(
                    param_name, int(spec["low"]), int(spec["high"]), log=use_log
                )
            elif param_type == "float":
                hparams[param_name] = trial.suggest_float(
                    param_name, float(spec["low"]), float(spec["high"]), log=use_log
                )
            elif param_type == "categorical":
                raw_choices = spec["choices"]
                # Optuna categorical only supports None/bool/int/float/str.
                # For list/tuple choices (e.g. RealMLP hidden_sizes),
                # convert to JSON strings for Optuna, then decode back.
                has_complex = any(isinstance(c, (list, tuple)) for c in raw_choices)
                if has_complex:
                    import json
                    str_choices = [json.dumps(c) for c in raw_choices]
                    picked = trial.suggest_categorical(param_name, str_choices)
                    hparams[param_name] = json.loads(picked)
                else:
                    hparams[param_name] = trial.suggest_categorical(
                        param_name, raw_choices
                    )
            elif param_type == "int_list":
                # Suggest N independent int params, combine into a list.
                # YAML: hidden_sizes: {type: int_list, n: 2, low: 8, high: 128}
                n_elements = int(spec["n"])
                low = int(spec["low"])
                high = int(spec["high"])
                hparams[param_name] = [
                    trial.suggest_int(
                        f"{param_name}_{i}", low, high, log=use_log
                    )
                    for i in range(n_elements)
                ]

        # Add eval_metric to constructor params where needed
        eval_metric_val = _get_eval_metric_value(training_cfg, task_type, gpu)
        eval_metric_param = training_cfg.get("eval_metric_param")
        if eval_metric_val and eval_metric_param:
            hparams[eval_metric_param] = eval_metric_val

        needs_eval_set = training_cfg.get("needs_eval_set", False)
        early_stopping_rounds = training_cfg.get("early_stopping_rounds")
        is_lgbm = training_cfg.get("uses_callbacks_for_early_stopping", False)
        es_in_constructor = training_cfg.get("early_stopping_in_constructor", False)

        if task_type == "multiclass":
            n_classes = len(np.unique(y))
            oof = np.zeros((len(X), n_classes))
        else:
            oof = np.zeros(len(X))

        try:
            splits = list(cv.split(X, y))
        except Exception:
            splits = list(cv.split(X))

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Early stopping via constructor (XGBoost v2.0+)
            model_hparams = dict(hparams)
            if es_in_constructor and early_stopping_rounds:
                model_hparams["early_stopping_rounds"] = early_stopping_rounds

            # Fresh model per fold — each fold gets an independent RandomState,
            # matching the behaviour of train_with_config for reproducibility.
            model = registry.get_model(
                model_name, hparams=model_hparams, task_type=task_type,
                gpu=gpu, results_dir=results_dir
            )

            if needs_eval_set:
                fit_params: dict[str, Any] = {
                    "eval_set": [(X_val, y_val)],
                    "verbose": False,  # suppress per-iteration eval output
                }
                if is_lgbm:
                    import lightgbm as lgb
                    del fit_params["verbose"]  # LightGBM uses callbacks instead
                    fit_params["callbacks"] = [
                        lgb.early_stopping(
                            stopping_rounds=early_stopping_rounds or 50,
                            verbose=False,
                        ),
                        lgb.log_evaluation(-1),
                    ]
                elif early_stopping_rounds and not es_in_constructor:
                    fit_params["early_stopping_rounds"] = early_stopping_rounds
                model.fit(X_train, y_train, **fit_params)
            else:
                model.fit(X_train, y_train)

            if task_type == "binary_classification":
                fold_preds = model.predict_proba(X_val)[:, 1]
            elif task_type == "multiclass":
                fold_preds = model.predict_proba(X_val)  # full (n_val, n_classes) matrix
            else:
                fold_preds = model.predict(X_val)

            oof[val_idx] = fold_preds

            fold_metric = _compute_cv_metric(y_val, fold_preds, task_type)
            trial.report(fold_metric, fold_idx)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        overall_metric = _compute_cv_metric(y, oof, task_type)
        trial.set_user_attr("oof_preds", oof.tolist())
        return overall_metric

    return objective


def _run_two_phase_study(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    qmc_warmup_ratio: float,
    timeout: int | None = None,
    global_seed: int = 42,
    verbose: int = 1,
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
    n_qmc = max(1, int(n_trials * qmc_warmup_ratio))
    n_tpe = max(1, n_trials - n_qmc)

    start_time = time.time()

    # Progress callback
    log_interval = max(1, n_trials // 8)  # ~8 progress updates per study (verbose=1)

    def _progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        n_done = len(study.trials)
        best_val = study.best_value if study.best_trial else float("nan")
        elapsed = time.time() - start_time
        if verbose >= 2:
            # Show every trial with its own score
            if trial.value is not None:
                trial_val = f"{trial.value:.6f}"
            else:
                trial_val = "pruned"
            logger.info(
                f"  Trial {n_done}/{n_trials} | "
                f"score={trial_val} | "
                f"best={best_val:.6f} | "
                f"{elapsed:.0f}s"
            )
        elif n_done % log_interval == 0 or n_done == n_qmc:
            logger.info(
                f"  Trial {n_done}/{n_trials} | "
                f"best={best_val:.6f} | "
                f"elapsed={elapsed:.0f}s"
            )

    # Phase 1: QMC — categorical params fall back to RandomSampler automatically
    logger.info(f"Phase 1 (QMC): {n_qmc} trials")
    study.sampler = optuna.samplers.QMCSampler(
        seed=global_seed,
        independent_sampler=optuna.samplers.RandomSampler(seed=global_seed),
        warn_independent_sampling=False,
    )
    study.optimize(
        objective,
        n_trials=n_qmc,
        timeout=timeout,
        callbacks=[_progress_callback],
        show_progress_bar=False,
    )
    qmc_elapsed = time.time() - start_time
    logger.info(
        f"Phase 1 done: {len(study.trials)} trials completed in {qmc_elapsed:.1f}s"
    )

    # Phase 2: TPE
    remaining_timeout = None
    if timeout is not None:
        remaining_timeout = max(1, timeout - int(qmc_elapsed))

    logger.info(f"Phase 2 (TPE): {n_tpe} trials")
    study.sampler = optuna.samplers.TPESampler(
        seed=global_seed, n_startup_trials=0
    )
    study.optimize(
        objective,
        n_trials=n_tpe,
        timeout=remaining_timeout,
        callbacks=[_progress_callback],
        show_progress_bar=False,
    )
    total_elapsed = time.time() - start_time
    logger.info(
        f"Phase 2 done: {len(study.trials)} total trials in {total_elapsed:.1f}s"
    )


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
    X_train = train[feature_cols]
    y_train = train[target_col].values
    X_test = test[feature_cols]

    training_cfg = registry.get_training_config(model_name)
    needs_eval_set = training_cfg.get("needs_eval_set", False)
    early_stopping_rounds = training_cfg.get("early_stopping_rounds")
    is_lgbm = training_cfg.get("uses_callbacks_for_early_stopping", False)
    es_in_constructor = training_cfg.get("early_stopping_in_constructor", False)

    # Eval metric for constructor
    eval_metric_val = _get_eval_metric_value(training_cfg, task_type, gpu)
    eval_metric_param = training_cfg.get("eval_metric_param")

    # seed_param is declared per-model in the YAML training section.
    seed_param = training_cfg.get("seed_param", "random_state")

    oof_preds_list: list[np.ndarray] = []
    test_preds_list: list[np.ndarray] = []

    try:
        splits = list(cv.split(X_train, y_train))
    except Exception:
        splits = list(cv.split(X_train))

    n_folds = len(splits)

    if task_type == "multiclass":
        n_classes = len(np.unique(y_train))

    for seed_idx, seed in enumerate(seeds):
        hparams_seeded = {**hparams, seed_param: seed} if seed_param else dict(hparams)
        if eval_metric_val and eval_metric_param:
            hparams_seeded[eval_metric_param] = eval_metric_val
        if es_in_constructor and early_stopping_rounds:
            hparams_seeded["early_stopping_rounds"] = early_stopping_rounds

        if task_type == "multiclass":
            oof_preds = np.zeros((len(X_train), n_classes))
            test_preds = np.zeros((len(X_test), n_classes))
        else:
            oof_preds = np.zeros(len(X_train))
            test_preds = np.zeros(len(X_test))

        seed_start = time.time()
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_start = time.time()
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            model = registry.get_model(
                model_name, hparams=hparams_seeded,
                task_type=task_type, gpu=gpu, results_dir=results_dir
            )

            if needs_eval_set:
                fit_params: dict[str, Any] = {
                    "eval_set": [(X_fold_val, y_fold_val)],
                    "verbose": False,  # suppress per-iteration eval output
                }
                if is_lgbm:
                    import lightgbm as lgb
                    del fit_params["verbose"]  # LightGBM uses callbacks instead
                    fit_params["callbacks"] = [
                        lgb.early_stopping(
                            stopping_rounds=early_stopping_rounds or 50,
                            verbose=False,
                        ),
                        lgb.log_evaluation(-1),
                    ]
                elif early_stopping_rounds and not es_in_constructor:
                    fit_params["early_stopping_rounds"] = early_stopping_rounds
                model.fit(X_fold_train, y_fold_train, **fit_params)
            else:
                model.fit(X_fold_train, y_fold_train)

            if task_type == "binary_classification":
                val_preds = model.predict_proba(X_fold_val)[:, 1]
                tst_preds = model.predict_proba(X_test)[:, 1]
            elif task_type == "multiclass":
                val_preds = model.predict_proba(X_fold_val)  # (n_val, n_classes)
                tst_preds = model.predict_proba(X_test)       # (n_test, n_classes)
            else:
                val_preds = model.predict(X_fold_val)
                tst_preds = model.predict(X_test)

            oof_preds[val_idx] = val_preds
            test_preds += tst_preds / n_folds

            fold_metric = _compute_cv_metric(y_fold_val, val_preds, task_type)
            logger.debug(
                f"    Fold {fold_idx + 1}/{n_folds}: "
                f"score={fold_metric:.6f} ({time.time() - fold_start:.1f}s)"
            )

        seed_elapsed = time.time() - seed_start
        oof_metric = _compute_cv_metric(y_train, oof_preds, task_type)
        logger.debug(
            f"  Seed {seed} (#{seed_idx + 1}/{len(seeds)}): "
            f"CV={oof_metric:.6f} in {seed_elapsed:.1f}s"
        )
        oof_preds_list.append(oof_preds)
        test_preds_list.append(test_preds)

    return oof_preds_list, test_preds_list, y_train


def _reassemble_int_lists(params: dict[str, Any]) -> dict[str, Any]:
    """Reassemble int_list params (e.g. hidden_sizes_0, hidden_sizes_1) back into lists.

    Optuna stores them as separate keys. This function detects the pattern
    ``<name>_0, <name>_1, ...`` and combines them into ``<name>: [v0, v1, ...]``.
    """
    import re
    # Find all keys matching the pattern <name>_<digit>
    list_keys: dict[str, dict[int, Any]] = {}
    plain_keys: dict[str, Any] = {}
    for k, v in params.items():
        m = re.match(r"^(.+)_(\d+)$", k)
        if m:
            name, idx = m.group(1), int(m.group(2))
            list_keys.setdefault(name, {})[idx] = v
        else:
            plain_keys[k] = v

    # Only reassemble if ALL indices 0..N-1 are present (avoid false positives)
    result = dict(plain_keys)
    for name, idx_map in list_keys.items():
        n = len(idx_map)
        if all(i in idx_map for i in range(n)):
            result[name] = [idx_map[i] for i in range(n)]
        else:
            # Not a complete sequence, keep as separate keys
            for idx, val in idx_map.items():
                result[f"{name}_{idx}"] = val
    return result


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
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]

    descending = (study.direction == optuna.study.StudyDirection.MAXIMIZE)
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=descending)
    top = sorted_trials[:n_top]

    return [
        {
            "params": _reassemble_int_lists(dict(t.params)),
            "value": t.value,
            "trial_number": t.number,
        }
        for t in top
    ]


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
    cv_cfg = pipeline_config.cv
    task_type = pipeline_config.task_type

    if cv_cfg.stratified and task_type != "regression":
        cv = StratifiedKFold(n_splits=cv_cfg.n_folds, shuffle=True, random_state=cv_cfg.seed)
    else:
        cv = KFold(n_splits=cv_cfg.n_folds, shuffle=True, random_state=cv_cfg.seed)

    results: dict[str, dict[str, Any]] = {}
    global_seed = pipeline_config.optuna.global_seed
    results_dir = Path(pipeline_config.output.results_dir)

    n_models = len(pipeline_config.models)
    all_models_start = time.time()

    for model_idx, model_name in enumerate(pipeline_config.models, 1):
        model_start = time.time()
        logger.info(
            f"[{model_idx}/{n_models}] Starting model: {model_name}"
        )
        gpu = gpu_status.get(model_name, False)

        # Resolve per-model timeout from pipeline config
        model_timeout = pipeline_config.optuna.model_timeouts.get(model_name)

        try:
            study = run_optuna_study(
                model_name=model_name,
                train=train,
                feature_cols=feature_cols,
                target_col=pipeline_config.target_column,
                registry=registry,
                pipeline_config=pipeline_config,
                strategy=strategy,
                gpu=gpu,
                timeout_override=model_timeout,
            )
            optuna_elapsed = time.time() - model_start

            # Study statistics
            n_completed = sum(
                1 for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            )
            n_pruned = sum(
                1 for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            )
            n_failed = len(study.trials) - n_completed - n_pruned
            avg_trial_time = optuna_elapsed / max(len(study.trials), 1)
            logger.info(
                f"[{model_name}] Optuna done in {optuna_elapsed:.0f}s "
                f"({optuna_elapsed / 60:.1f}min) | "
                f"{len(study.trials)} trials "
                f"({n_completed} ok, {n_pruned} pruned, {n_failed} failed) | "
                f"avg {avg_trial_time:.1f}s/trial | "
                f"best={study.best_value:.6f}"
            )

            optuna_cfg = registry.get_optuna_config(model_name)
            n_top = optuna_cfg["n_top_trials"]
            n_seeds = optuna_cfg["n_seeds"]
            seeds = [global_seed + i for i in range(n_seeds)]

            top_configs = get_top_configs(study, n_top=n_top)

            all_oof: list[np.ndarray] = []
            all_test: list[np.ndarray] = []
            labels: np.ndarray | None = None

            retrain_start = time.time()
            total_retrain_fits = len(top_configs) * n_seeds * cv_cfg.n_folds
            logger.info(
                f"[{model_name}] Retraining top {len(top_configs)} configs "
                f"x {n_seeds} seeds x {cv_cfg.n_folds} folds = "
                f"{total_retrain_fits} fits"
            )

            for cfg_idx, cfg in enumerate(top_configs, 1):
                logger.debug(
                    f"[{model_name}] Retrain config {cfg_idx}/{len(top_configs)} "
                    f"(score={cfg['value']:.6f})"
                )
                oof_list, test_list, y = train_with_config(
                    model_name=model_name,
                    hparams=cfg["params"],
                    feature_cols=feature_cols,
                    train=train,
                    test=test,
                    target_col=pipeline_config.target_column,
                    cv=cv,
                    registry=registry,
                    task_type=task_type,
                    gpu=gpu,
                    seeds=seeds,
                    results_dir=results_dir,
                )
                all_oof.extend(oof_list)
                all_test.extend(test_list)
                if labels is None:
                    labels = y

            retrain_elapsed = time.time() - retrain_start
            model_elapsed = time.time() - model_start

            results[model_name] = {
                "study": study,
                "top_configs": top_configs,
                "oof_preds": all_oof,
                "test_preds": all_test,
                "labels": labels,
                "elapsed": model_elapsed,
                "optuna_elapsed": optuna_elapsed,
                "retrain_elapsed": retrain_elapsed,
                "n_trials": len(study.trials),
                "avg_trial_time": avg_trial_time,
            }
            logger.info(
                f"[{model_idx}/{n_models}] {model_name} done: "
                f"{len(all_oof)} arrays | "
                f"optuna={optuna_elapsed:.0f}s retrain={retrain_elapsed:.0f}s "
                f"total={model_elapsed:.0f}s ({model_elapsed / 60:.1f}min)"
            )

        except Exception as exc:
            logger.error(f"Model '{model_name}' failed: {exc}", exc_info=True)

    total_training = time.time() - all_models_start
    total_m, total_s = divmod(int(total_training), 60)
    total_h, total_m = divmod(total_m, 60)
    if total_h > 0:
        fmt = f"{total_h}h {total_m:02d}m {total_s:02d}s"
    elif total_m > 0:
        fmt = f"{total_m}m {total_s:02d}s"
    else:
        fmt = f"{total_training:.1f}s"
    logger.info(
        f"All model training complete: {len(results)}/{n_models} models "
        f"in {fmt}"
    )

    return results
