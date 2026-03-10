"""
Maestro ML — Main pipeline orchestrator.

Entry point for the entire Maestro pipeline. Executes the three layers
in sequence:
  1. EDA: Profile the dataset → JSON report
  2. Strategy: LLM generates feature/model plan → strategy YAML
  3. Engine: Feature engineering → Optuna studies → Ensemble → Submission

Usage:
    python run.py --config pipeline.yaml
    python run.py --config pipeline.yaml --strategy manual
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

# Fix OpenMP DLL conflict (libomp vs libiomp5md) on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Pre-import torch so shm.dll loads correctly before pytabkit
# (pytabkit uses importlib to import torch, which fails on Windows
# unless torch is already loaded in the process)
try:
    import torch  # noqa: F401

    # Enable TF32 for all PyTorch matmuls/convolutions on Ampere+ GPUs
    # (RTX 30xx, 40xx, 50xx). ~2x faster with negligible precision loss.
    # Use ONLY the legacy API — pytabkit internally toggles allow_tf32
    # (sets False, then restores). If we use set_float32_matmul_precision(),
    # pytabkit's legacy write creates a mixed-API state → PyTorch 2.10+
    # throws RuntimeError in get_float32_matmul_precision().
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except (ImportError, OSError):
    pass

# Suppress ALL pytorch_lightning / lightning output (GPU/TPU info, tips, LOCAL_RANK)
import logging as _logging
import warnings
for _name in ("pytorch_lightning", "lightning.pytorch", "lightning",
              "lightning.fabric", "pytorch_lightning.utilities.rank_zero"):
    _pl = _logging.getLogger(_name)
    _pl.setLevel(_logging.CRITICAL)
    _pl.propagate = False
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.jit.script.*is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
# PL Trainer reconfigures loggers at runtime — patch rank_zero functions directly
try:
    import pytorch_lightning.utilities.rank_zero as _rz
    _noop = lambda *a, **kw: None
    _rz.rank_zero_info = _noop
    _rz.rank_zero_warn = _noop
    _rz.rank_zero_deprecation = _noop
except (ImportError, OSError, AttributeError):
    pass


def _fmt_time(seconds: float) -> str:
    """Format seconds into human-readable string like '5h 19m 19s'."""
    s = int(seconds)
    if s < 60:
        return f"{seconds:.1f}s"
    h, remainder = divmod(s, 3600)
    m, sec = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold

from src.utils.io import (
    load_pipeline_config,
    save_submission,
    save_eda_report,
    setup_logging,
)
from src.eda.profiler import run_eda
from src.features.engineer import build_features, get_feature_columns
from src.models.registry import ModelRegistry
from src.models.trainer import run_all_studies
from src.ensemble.blender import (
    optimize_blend_weights,
    apply_blend,
    rank_average,
    train_meta_model,
    optimize_meta_C,
    optimize_meta_xgb,
    pick_best_strategy,
)
from src.ensemble.diversity import (
    compute_correlation_matrix,
    run_nsga2_ensemble,
    select_from_pareto,
    log_fold_diversity,
    print_diversity_report,
)
from src.strategy.llm_strategist import generate_strategy


def _concat_extra_data(
    train: pd.DataFrame,
    pipeline_config: "PipelineConfig",
    logger: "logging.Logger",
) -> pd.DataFrame:
    """Concat extra datasets (e.g., original competition data) into train.

    Each entry in pipeline_config.extra_data is a dict with:
      - path (str): CSV file path (required)
      - target_column (str): target column name in this file (default: same as pipeline)
      - column_mapping (dict): rename columns {original_name: pipeline_name}
      - drop_columns (list): columns to drop before concat
      - sample_weight (float): weight for these rows in model training (default: 1.0)

    Handles:
      - Column name differences via column_mapping
      - Target mapping (same as pipeline's target_mapping)
      - Missing id_column (original data usually doesn't have it)
      - Numeric conversion for columns that are strings in original (e.g. TotalCharges)
      - Only keeps columns that exist in train (drops extras, ignores missing)
      - Adds _is_original (bool) and _sample_weight (float) metadata columns

    Returns:
        New DataFrame with extra data appended and metadata columns.
    """
    target_col = pipeline_config.target_column
    id_col = pipeline_config.id_column
    target_mapping = pipeline_config.target_mapping

    original_len = len(train)
    result = train.copy()
    result["_is_original"] = False
    result["_sample_weight"] = 1.0

    for entry in pipeline_config.extra_data:
        path = entry.get("path", "")
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            logger.warning(f"Extra data file not found: {path}, skipping")
            continue

        extra = pd.read_csv(p)
        extra_target = entry.get("target_column", target_col)
        col_mapping = entry.get("column_mapping", {}) or {}
        drop_cols = entry.get("drop_columns", []) or []

        logger.info(f"Loading extra data: {path} ({len(extra)} rows, {len(extra.columns)} cols)")

        # Drop specified columns
        if drop_cols:
            extra = extra.drop(columns=[c for c in drop_cols if c in extra.columns])

        # Rename columns
        if col_mapping:
            extra = extra.rename(columns=col_mapping)

        # Rename target column if different
        if extra_target != target_col and extra_target in extra.columns:
            extra = extra.rename(columns={extra_target: target_col})

        # Apply target mapping
        if target_mapping and target_col in extra.columns:
            extra[target_col] = extra[target_col].map(
                lambda v: target_mapping.get(str(v), v)
            )

        # Add id column if missing (use negative IDs to distinguish)
        if id_col and id_col not in extra.columns:
            extra[id_col] = range(-len(extra), 0)

        # Keep only columns that exist in train
        common_cols = [c for c in result.columns if c in extra.columns]
        missing_cols = [c for c in result.columns if c not in extra.columns]
        if missing_cols:
            logger.debug(f"  Columns missing in extra data (filled NaN): {missing_cols}")
        extra = extra[common_cols]

        # Coerce dtypes to match train (e.g., TotalCharges: str → float)
        for col in common_cols:
            if col == id_col or col == target_col:
                continue
            if result[col].dtype != extra[col].dtype:
                try:
                    extra[col] = pd.to_numeric(extra[col], errors="coerce")
                except Exception:
                    pass

        old_len = len(result)
        result = pd.concat([result, extra], ignore_index=True)
        sw = float(entry.get("sample_weight", 1.0))
        result.loc[old_len:, "_is_original"] = True
        result.loc[old_len:, "_sample_weight"] = sw
        logger.info(
            f"  +{len(extra)} rows from {Path(path).name} "
            f"(weight={sw}) → train: {original_len} → {len(result)} rows"
        )

    n_original = int(result["_is_original"].sum())
    if n_original > 0:
        logger.info(
            f"  Sample weights: {n_original} original rows, "
            f"{len(result) - n_original} synthetic rows (weight=1.0)"
        )

    return result


_META_XGB_MIN_SAMPLES = 5000


def _score_fn(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute score (higher is better) for ensemble comparison."""
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_pred))
    return -float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main(pipeline_yaml_path: str | Path) -> None:
    """Run the complete Maestro pipeline from config to submission.

    This is the top-level orchestrator that coordinates all three layers.

    Args:
        pipeline_yaml_path: Path to the pipeline.yaml configuration file.

    Steps:
        1. Load pipeline configuration:
           a. Call load_pipeline_config(pipeline_yaml_path).
           b. Set up logging with configured verbosity.
           c. Log the pipeline configuration summary.

        2. GPU detection:
           a. If runtime.gpu_check is True:
              - Create ModelRegistry and load model configs.
              - For each model in pipeline_config.models, call
                registry.check_gpu(model_name).
              - Store GPU status dict: {model_name: bool}.
              - Log GPU availability per model.
           b. If gpu_check is False, assume CPU for all models.

        3. Layer 1 — EDA:
           a. Call run_eda(train_path, test_path, target_column).
           b. Save the EDA report to results_dir/eda_report.json.
           c. Log EDA summary (dataset shape, top correlations).

        4. Layer 2 — Strategy:
           a. Call generate_strategy(eda_report, pipeline_config).
           b. Save the strategy to results_dir/strategy.yaml.
           c. Log the strategy summary (selected features, models).
           d. Merge strategy features into pipeline_config.features
              (strategy overrides template defaults).

        5. Data loading and feature engineering:
           a. Read train.csv and test.csv into DataFrames.
           b. Create the CV splitter (StratifiedKFold or KFold).
           c. Call build_features(train, test, strategy, cv_folds).
           d. Determine the final feature column list.
           e. Log feature count (original + engineered).

        6. Layer 3 — Model training:
           a. Call run_all_studies(pipeline_config, train, test,
              feature_cols, strategy, registry, gpu_status).
           b. Collect all OOF and test prediction arrays.
           c. Log per-model best scores.

        7. Ensemble:
           a. Prepare the list of OOF and test predictions.
           b. Based on ensemble.strategy:
              - 'blend': optimize_blend_weights → apply_blend
              - 'rank': rank_average
              - 'meta': train_meta_model
              - 'nsga2': run_nsga2_ensemble
              - 'auto': try all strategies, pick_best_strategy
           c. Log the ensemble score and strategy used.
           d. Print the diversity report.

        8. Output:
           a. Save submission CSV to output.submission_path.
           b. If save_oof, save OOF predictions to results_dir.
           c. Log final summary: best score, ensemble strategy,
              number of models, submission path.
    """
    # -------------------------------------------------------------------------
    # Step 1: Load configuration
    # -------------------------------------------------------------------------
    pipeline_start = time.time()
    step_times: dict[str, float] = {}

    pipeline_config = load_pipeline_config(pipeline_yaml_path)
    logger = setup_logging(pipeline_config.runtime.verbose)

    logger.info(f"Pipeline config loaded: {pipeline_yaml_path}")
    logger.info(
        f"Task: {pipeline_config.task_type} | "
        f"Models: {pipeline_config.models} | "
        f"CV: {pipeline_config.cv.n_folds}-fold"
    )
    if pipeline_config.target_mapping:
        logger.info(f"Target mapping: {pipeline_config.target_mapping}")

    results_dir = Path(pipeline_config.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 2: GPU detection
    # -------------------------------------------------------------------------
    gpu_start = time.time()
    registry = ModelRegistry("configs/models")
    gpu_status: dict[str, bool] = {}

    if pipeline_config.runtime.gpu_check:
        logger.info("Checking GPU availability per model...")
        for model_name in pipeline_config.models:
            if model_name not in registry.list_models():
                logger.warning(f"Model '{model_name}' not in registry, skipping GPU check.")
                gpu_status[model_name] = False
                continue
            gpu_ok = registry.check_gpu(model_name, pipeline_config.task_type)
            gpu_status[model_name] = gpu_ok
            logger.info(f"  {model_name}: GPU={'YES' if gpu_ok else 'NO (CPU fallback)'}")
    else:
        gpu_status = {m: False for m in pipeline_config.models}
        logger.info("GPU check disabled — all models using CPU.")
    step_times["gpu_check"] = time.time() - gpu_start

    # -------------------------------------------------------------------------
    # Step 3 & 4: EDA + Strategy (skip EDA if strategy already exists)
    # -------------------------------------------------------------------------
    # When strategy_input_path already exists (previous run), skip EDA entirely
    # — it's expensive and the strategy is already generated.
    eda_start = time.time()
    strategy_mode = pipeline_config.strategy.mode
    manual_cfg = pipeline_config.strategy.manual or {}
    strategy_input_path = manual_cfg.get("strategy_input_path")
    skip_eda = (
        strategy_mode == "manual"
        and strategy_input_path
        and Path(strategy_input_path).exists()
    )

    if skip_eda:
        logger.info(
            f"Strategy file exists: {strategy_input_path} — skipping EDA"
        )
        # Load data directly (same as run_eda but without analysis)
        train = pd.read_csv(pipeline_config.train_path)
        test = pd.read_csv(pipeline_config.test_path)
        if pipeline_config.target_mapping:
            target_col = pipeline_config.target_column
            train[target_col] = train[target_col].map(
                pipeline_config.target_mapping
            )
        logger.info(
            f"Data loaded: train={train.shape}, test={test.shape}"
        )
        eda_report = {}
    else:
        logger.info("Layer 1: Running EDA...")
        eda_report, train, test = run_eda(
            train_path=pipeline_config.train_path,
            test_path=pipeline_config.test_path,
            target_col=pipeline_config.target_column,
            id_col=pipeline_config.id_column or None,
            target_mapping=pipeline_config.target_mapping,
            task_type=pipeline_config.task_type,
        )
        eda_path = results_dir / "eda_report.json"
        save_eda_report(eda_report, eda_path)

        dataset_info = eda_report.get("dataset_info", {})
        logger.info(
            f"EDA complete: train={dataset_info.get('train_shape')}, "
            f"test={dataset_info.get('test_shape')}"
        )

    # Apply log1p to target (e.g. for RMSLE competitions like House Prices)
    if pipeline_config.log_transform_target:
        target_col = pipeline_config.target_column
        logger.info(f"Applying log1p transform to target '{target_col}'")
        train[target_col] = np.log1p(train[target_col])

    # -------------------------------------------------------------------------
    # Step 3b: Load and concat extra data (original datasets)
    # -------------------------------------------------------------------------
    if pipeline_config.extra_data:
        train = _concat_extra_data(train, pipeline_config, logger)

    # Log top 5 correlated features (only if EDA was run)
    if eda_report:
        sorted_cols = sorted(
            eda_report.get("columns", {}).items(),
            key=lambda kv: abs(kv[1].get("target_correlation", 0)),
            reverse=True,
        )
        for col, info in sorted_cols[:5]:
            logger.info(f"  {col}: target_corr={info.get('target_correlation', 0):.4f}")
    step_times["eda"] = time.time() - eda_start

    # -------------------------------------------------------------------------
    # Step 4: Layer 2 — Strategy
    # -------------------------------------------------------------------------
    strategy_start = time.time()
    logger.info("Layer 2: Generating strategy...")
    strategy = generate_strategy(eda_report, pipeline_config)

    # Save strategy
    strategy_path = results_dir / "strategy.yaml"
    with open(strategy_path, "w", encoding="utf-8") as f:
        yaml.dump(strategy, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Strategy saved: {strategy_path}")

    # Log summary
    strategy_features = strategy.get("features", {}) or {}
    strategy_models = strategy.get("models") or pipeline_config.models
    logger.info(
        f"Strategy: {len(strategy_models)} models, "
        f"{len(strategy_features.get('interactions', []) or [])} interactions, "
        f"{len(strategy_features.get('ratios', []) or [])} ratios"
    )
    logger.info(f"Strategy reasoning: {strategy.get('reasoning', '')[:200]}")

    # Merge strategy's model list into pipeline config if provided
    valid_models = [m for m in strategy_models if m in registry.list_models()]
    if valid_models:
        pipeline_config.models = valid_models
    step_times["strategy"] = time.time() - strategy_start

    # -------------------------------------------------------------------------
    # Step 5: Feature engineering
    # -------------------------------------------------------------------------
    feat_start = time.time()
    logger.info("Building features...")

    # Create CV splitter (same as model training)
    cv_cfg = pipeline_config.cv
    task_type = pipeline_config.task_type
    if cv_cfg.stratified and task_type != "regression":
        cv_folds = StratifiedKFold(
            n_splits=cv_cfg.n_folds, shuffle=True, random_state=cv_cfg.seed
        )
    else:
        cv_folds = KFold(
            n_splits=cv_cfg.n_folds, shuffle=True, random_state=cv_cfg.seed
        )

    # Drop columns specified in strategy (e.g., noise features like ps_calc_*)
    strategy_drop = strategy.get("drop_columns", []) or []
    if strategy_drop:
        existing_drop = [c for c in strategy_drop if c in train.columns]
        if existing_drop:
            train = train.drop(columns=existing_drop)
            test = test.drop(columns=[c for c in existing_drop if c in test.columns])
            logger.info(f"Dropped {len(existing_drop)} features from strategy: {existing_drop[:5]}{'...' if len(existing_drop) > 5 else ''}")

    # Strip metadata columns before feature engineering (they are not
    # real features and _is_original can become object dtype after concat,
    # which confuses the ordinal encoder in build_features).
    meta_cols = ["_is_original", "_sample_weight"]
    meta_backup = {c: train[c].copy() for c in meta_cols if c in train.columns}
    train_for_feat = train.drop(columns=[c for c in meta_cols if c in train.columns])

    train_feat, test_feat = build_features(
        train=train_for_feat,
        test=test,
        strategy=strategy,
        cv_folds=cv_folds,
        target_col=pipeline_config.target_column,
    )

    # Re-attach metadata columns to engineered train
    for c, series in meta_backup.items():
        train_feat[c] = series.values

    # Determine feature columns
    original_cols = list(train.columns)
    exclude_cols = [c for c in [pipeline_config.target_column, pipeline_config.id_column] if c]
    exclude_cols.extend(meta_cols)
    feature_cols = get_feature_columns(strategy, original_cols, exclude=exclude_cols)
    # Ensure all feature cols exist in the engineered dataframe
    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    n_original = len([c for c in original_cols if c not in exclude_cols])
    logger.info(
        f"Features: {n_original} original + "
        f"{len(feature_cols) - n_original} engineered "
        f"= {len(feature_cols)} total"
    )
    step_times["features"] = time.time() - feat_start

    # -------------------------------------------------------------------------
    # Step 6: Layer 3 — Model training
    # -------------------------------------------------------------------------
    training_start = time.time()
    logger.info("Layer 3: Running Optuna studies for all models...")
    model_results = run_all_studies(
        pipeline_config=pipeline_config,
        train=train_feat,
        test=test_feat,
        feature_cols=feature_cols,
        strategy=strategy,
        registry=registry,
        gpu_status=gpu_status,
    )

    # Collect all OOF and test prediction arrays
    all_oof: list[np.ndarray] = []
    all_test: list[np.ndarray] = []
    y_true: np.ndarray | None = None
    model_labels: list[str] = []

    for model_name, res in model_results.items():
        n_preds = len(res["oof_preds"])
        all_oof.extend(res["oof_preds"])
        all_test.extend(res["test_preds"])
        model_labels.extend([f"{model_name}_{i}" for i in range(n_preds)])
        if y_true is None:
            y_true = res["labels"]
        best_val = res["top_configs"][0]["value"] if res["top_configs"] else float("nan")
        logger.info(
            f"  {model_name}: {n_preds} prediction arrays, "
            f"best_trial_score={best_val:.6f}"
        )
    step_times["training"] = time.time() - training_start

    if not all_oof or y_true is None:
        raise RuntimeError("No model predictions collected. Check model training logs.")

    # -------------------------------------------------------------------------
    # Step 7: Ensemble
    # -------------------------------------------------------------------------
    ensemble_start = time.time()
    logger.info("Building ensemble...")
    ensemble_cfg = pipeline_config.ensemble
    metric = "roc_auc" if task_type != "regression" else "neg_rmse"
    seed = pipeline_config.optuna.global_seed
    ensemble_strategy = ensemble_cfg.strategy

    chosen_strategy = ensemble_strategy
    final_oof: np.ndarray
    final_test_preds: np.ndarray

    if ensemble_strategy == "blend":
        weights = optimize_blend_weights(
            all_oof, y_true, n_trials=ensemble_cfg.blend_trials,
            metric=metric, seed=seed
        )
        final_oof = apply_blend(all_oof, weights)
        final_test_preds = apply_blend(all_test, weights)

    elif ensemble_strategy == "rank":
        final_oof = rank_average(all_oof)
        final_test_preds = rank_average(all_test)

    elif ensemble_strategy == "meta":
        meta_n_folds = ensemble_cfg.meta_cv_folds or 2 * cv_cfg.n_folds
        best_meta_score = -np.inf
        final_oof = None
        final_test_preds = None
        meta_models_meta = ensemble_cfg.meta_models
        if len(y_true) < _META_XGB_MIN_SAMPLES and "xgboost" in meta_models_meta:
            logger.info(
                f"  Auto-disabling meta-xgboost: {len(y_true)} samples "
                f"< {_META_XGB_MIN_SAMPLES} minimum (overfitting risk)"
            )
            meta_models_meta = [m for m in meta_models_meta if m != "xgboost"]
        for meta_name in meta_models_meta:
            n_trials_for = ensemble_cfg.get_meta_trials(meta_name)
            try:
                if meta_name == "logreg":
                    m_oof, m_test, _ = optimize_meta_C(
                        all_oof, all_test, y_true,
                        n_folds=meta_n_folds, seed=seed, task_type=task_type,
                        metric=metric, n_trials=n_trials_for,
                    )
                elif meta_name == "xgboost":
                    m_oof, m_test, _ = optimize_meta_xgb(
                        all_oof, all_test, y_true,
                        n_folds=meta_n_folds, seed=seed, task_type=task_type,
                        metric=metric, n_trials=n_trials_for,
                        gpu=gpu_status.get("xgboost", False),
                    )
                else:
                    continue
                m_score = _score_fn(y_true, m_oof, metric)
                if m_score > best_meta_score:
                    best_meta_score = m_score
                    final_oof = m_oof
                    final_test_preds = m_test
                    chosen_strategy = f"meta+{meta_name}"
            except Exception as exc:
                logger.warning(f"Meta-model '{meta_name}' failed: {exc}")
        if final_oof is None:
            raise RuntimeError("All meta-models failed.")

    elif ensemble_strategy == "nsga2":
        # Normalize diversity_weight to list
        dw_values = ensemble_cfg.diversity_weight
        if isinstance(dw_values, (int, float)):
            dw_values = [float(dw_values)]

        # Run NSGA-II once (study is weight-agnostic; weight only affects
        # Pareto front selection). Then select per diversity_weight.
        first_test, first_info = run_nsga2_ensemble(
            all_oof, all_test, y_true,
            n_trials=ensemble_cfg.nsga2_trials,
            metric=metric,
            diversity_weight=dw_values[0],
            seed=seed,
            labels=model_labels,
            diversity_metric=ensemble_cfg.diversity_metric,
        )
        sel = first_info["selected_models"]
        wts = first_info["weights"]
        nsga2_blend_oof = apply_blend([all_oof[i] for i in sel], wts)
        nsga2_blend_test = first_test

        # Log selected models compactly
        logger.info(
            f"NSGA-II selected {len(sel)}/{len(all_oof)} arrays: "
            + ", ".join(
                f"{model_labels[sel[j]]}({wts[j]:.3f})"
                for j in range(len(sel))
            )
        )

        # Chain: train meta-models on NSGA-II selected models
        meta_n_folds = ensemble_cfg.meta_cv_folds or 2 * cv_cfg.n_folds
        sel_oof = [all_oof[i] for i in sel]
        sel_test = [all_test[i] for i in sel]
        sel_labels = [model_labels[i] for i in sel]

        # Candidates: start with linear blend
        blend_score = _score_fn(y_true, nsga2_blend_oof, metric)
        best_meta_score = blend_score
        best_meta_oof = nsga2_blend_oof
        best_meta_test = nsga2_blend_test
        best_meta_name = "blend"
        logger.info(f"  NSGA-II linear blend: {metric}={blend_score:.6f}")

        # Try each configured meta-model (auto-disable xgboost on small data)
        meta_models = ensemble_cfg.meta_models
        if len(y_true) < _META_XGB_MIN_SAMPLES and "xgboost" in meta_models:
            logger.info(
                f"  Auto-disabling meta-xgboost: {len(y_true)} samples "
                f"< {_META_XGB_MIN_SAMPLES} minimum (overfitting risk)"
            )
            meta_models = [m for m in meta_models if m != "xgboost"]
        n_meta_features = len(sel_oof)
        logger.info(
            f"Meta-model stage: {len(meta_models)} meta-model(s) "
            f"[{', '.join(meta_models)}], "
            f"{n_meta_features} base predictions, "
            f"{meta_n_folds}-fold meta-CV, "
            f"{len(y_true)} samples"
        )
        for meta_idx, meta_name in enumerate(meta_models, 1):
            try:
                n_trials_for = ensemble_cfg.get_meta_trials(meta_name)
                logger.info(
                    f"  [{meta_idx}/{len(meta_models)}] Training meta-{meta_name} "
                    f"({n_trials_for} Optuna trials)..."
                )
                meta_start = time.time()
                if meta_name == "logreg":
                    m_oof, m_test, best_C = optimize_meta_C(
                        sel_oof, sel_test, y_true,
                        n_folds=meta_n_folds, seed=seed, task_type=task_type,
                        metric=metric, n_trials=n_trials_for,
                    )
                    m_score = _score_fn(y_true, m_oof, metric)
                    meta_elapsed = time.time() - meta_start
                    logger.info(
                        f"  Meta-logreg done: C={best_C:.6f}, "
                        f"{metric}={m_score:.6f} ({meta_elapsed:.1f}s)"
                    )
                elif meta_name == "xgboost":
                    m_oof, m_test, best_params = optimize_meta_xgb(
                        sel_oof, sel_test, y_true,
                        n_folds=meta_n_folds, seed=seed, task_type=task_type,
                        metric=metric, n_trials=n_trials_for,
                        gpu=gpu_status.get("xgboost", False),
                    )
                    m_score = _score_fn(y_true, m_oof, metric)
                    meta_elapsed = time.time() - meta_start
                    logger.info(
                        f"  Meta-xgboost done: {metric}={m_score:.6f} ({meta_elapsed:.1f}s)"
                    )
                else:
                    logger.warning(f"  Unknown meta-model '{meta_name}', skipping")
                    continue

                if m_score > best_meta_score:
                    best_meta_score = m_score
                    best_meta_oof = m_oof
                    best_meta_test = m_test
                    best_meta_name = meta_name
            except Exception as exc:
                logger.warning(f"  Meta-model '{meta_name}' failed: {exc}")

        final_oof = best_meta_oof
        final_test_preds = best_meta_test
        if best_meta_name == "blend":
            chosen_strategy = "nsga2+blend"
            logger.info("  -> Linear blend wins!")
        else:
            chosen_strategy = f"nsga2+{best_meta_name}"
            logger.info(
                f"  -> {best_meta_name} meta-stacking wins! "
                f"({metric}={best_meta_score:.6f})"
            )

        # Re-select from same Pareto front for additional weights
        nsga2_submissions: dict[float, tuple[np.ndarray, dict]] = {
            dw_values[0]: (first_test, first_info),
        }
        if len(dw_values) > 1:
            pareto_data = first_info["pareto_trials"]
            for dw in dw_values[1:]:
                extra_test, extra_info = select_from_pareto(
                    pareto_data["F"], pareto_data["X"],
                    all_oof, all_test, y_true,
                    n_models=len(all_oof),
                    diversity_weight=dw,
                    metric=metric,
                    labels=model_labels,
                )
                nsga2_submissions[dw] = (extra_test, extra_info)

        # Fold-level diversity diagnostics for primary selection
        if len(sel) > 1:
            fold_val_indices = [
                val_idx for _, val_idx in cv_folds.split(
                    train_feat[feature_cols], y_true
                )
            ]
            sel_oofs = [all_oof[i] for i in sel]
            sel_labels = [model_labels[i] for i in sel]
            log_fold_diversity(
                sel_oofs, y_true, fold_val_indices,
                weights=wts, metric=metric, labels=sel_labels,
            )

    else:  # 'auto' — try all, pick best
        candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        weights = optimize_blend_weights(
            all_oof, y_true, n_trials=ensemble_cfg.blend_trials,
            metric=metric, seed=seed
        )
        candidates["blend"] = (apply_blend(all_oof, weights), apply_blend(all_test, weights))
        candidates["rank"] = (rank_average(all_oof), rank_average(all_test))

        auto_meta_folds = ensemble_cfg.meta_cv_folds or 2 * cv_cfg.n_folds
        auto_meta_models = ensemble_cfg.meta_models
        if len(y_true) < _META_XGB_MIN_SAMPLES and "xgboost" in auto_meta_models:
            logger.info(
                f"  Auto-disabling meta-xgboost: {len(y_true)} samples "
                f"< {_META_XGB_MIN_SAMPLES} minimum (overfitting risk)"
            )
            auto_meta_models = [m for m in auto_meta_models if m != "xgboost"]
        for meta_name in auto_meta_models:
            n_trials_for = ensemble_cfg.get_meta_trials(meta_name)
            try:
                if meta_name == "logreg":
                    m_oof, m_test, _ = optimize_meta_C(
                        all_oof, all_test, y_true,
                        n_folds=auto_meta_folds, seed=seed, task_type=task_type,
                        metric=metric, n_trials=n_trials_for,
                    )
                elif meta_name == "xgboost":
                    m_oof, m_test, _ = optimize_meta_xgb(
                        all_oof, all_test, y_true,
                        n_folds=auto_meta_folds, seed=seed, task_type=task_type,
                        metric=metric, n_trials=n_trials_for,
                        gpu=gpu_status.get("xgboost", False),
                    )
                else:
                    continue
                candidates[f"meta_{meta_name}"] = (m_oof, m_test)
            except Exception as exc:
                logger.warning(f"Meta-model '{meta_name}' failed: {exc}")

        try:
            _auto_dw = (
                ensemble_cfg.diversity_weight
                if isinstance(ensemble_cfg.diversity_weight, (int, float))
                else ensemble_cfg.diversity_weight[0]
            )
            nsga2_test, nsga2_info = run_nsga2_ensemble(
                all_oof, all_test, y_true,
                n_trials=ensemble_cfg.nsga2_trials,
                metric=metric,
                diversity_weight=_auto_dw,
                seed=seed,
                labels=model_labels,
                diversity_metric=ensemble_cfg.diversity_metric,
            )
            sel = nsga2_info["selected_models"]
            wts = nsga2_info["weights"]
            nsga2_oof = apply_blend([all_oof[i] for i in sel], wts)
            candidates["nsga2"] = (nsga2_oof, nsga2_test)
        except Exception as exc:
            logger.warning(f"NSGA-II ensemble failed: {exc}")

        final_test_preds, chosen_strategy, _ = pick_best_strategy(
            candidates, y_true, metric=metric
        )
        final_oof = candidates[chosen_strategy][0]

    # Compute ensemble score
    if task_type != "regression":
        ensemble_score = roc_auc_score(y_true, final_oof)
        display_metric = metric
    else:
        ensemble_score = float(np.sqrt(mean_squared_error(y_true, final_oof)))
        display_metric = "rmse"

    step_times["ensemble"] = time.time() - ensemble_start
    logger.info(
        f"Ensemble: strategy='{chosen_strategy}', "
        f"{display_metric}={ensemble_score:.6f}, "
        f"n_predictions={len(all_oof)}"
    )

    # Diversity report (show only top-level per-model averages to keep it readable)
    if len(all_oof) > 1:
        # Average per model for a cleaner diversity report
        model_avg_oof: list[np.ndarray] = []
        model_avg_labels: list[str] = []
        for model_name, res in model_results.items():
            if res["oof_preds"]:
                avg = np.mean(res["oof_preds"], axis=0)
                model_avg_oof.append(avg)
                model_avg_labels.append(model_name[:15])
        if len(model_avg_oof) > 1:
            corr_mat = compute_correlation_matrix(model_avg_oof)
            diversity_report_path = Path(pipeline_config.output.results_dir) / "diversity_report.txt"
            print_diversity_report(corr_mat, model_avg_labels, output_path=diversity_report_path)

    # -------------------------------------------------------------------------
    # Step 8: Output
    # -------------------------------------------------------------------------
    if pipeline_config.id_column:
        test_ids = test[pipeline_config.id_column]
    else:
        test_ids = test.index.to_series()

    # Reverse log1p → expm1 before saving predictions
    if pipeline_config.log_transform_target:
        logger.info("Applying expm1 (inverse log1p) to final predictions")
        final_test_preds = np.expm1(final_test_preds)

    base_sub_path = Path(pipeline_config.output.submission_path)
    submission_paths: list[str] = []

    # Multiple submissions when nsga2 with list of diversity_weights
    if (
        ensemble_strategy == "nsga2"
        and isinstance(ensemble_cfg.diversity_weight, list)
        and len(ensemble_cfg.diversity_weight) > 1
    ):
        for dw in sorted(nsga2_submissions.keys()):
            dw_test, dw_info = nsga2_submissions[dw]
            if pipeline_config.log_transform_target:
                dw_test = np.expm1(dw_test)
            dw_path = base_sub_path.parent / f"{base_sub_path.stem}_dw{dw:.2f}{base_sub_path.suffix}"
            save_submission(
                ids=test_ids,
                preds=dw_test,
                target_col=pipeline_config.target_column,
                path=dw_path,
            )
            submission_paths.append(str(dw_path))
            logger.info(
                f"  dw={dw:.2f}: {metric}={dw_info['metric_score']:.6f}, "
                f"N_eff={dw_info['effective_size']:.2f}, "
                f"{len(dw_info['selected_models'])} models → {dw_path.name}"
            )
    else:
        save_submission(
            ids=test_ids,
            preds=final_test_preds,
            target_col=pipeline_config.target_column,
            path=pipeline_config.output.submission_path,
        )
        submission_paths.append(pipeline_config.output.submission_path)

    if pipeline_config.output.save_oof:
        oof_path = results_dir / "oof_predictions.npy"
        oof_to_save = np.expm1(final_oof) if pipeline_config.log_transform_target else final_oof
        np.save(str(oof_path), oof_to_save)
        logger.info(f"OOF predictions saved: {oof_path}")

    total_elapsed = time.time() - pipeline_start
    step_times["total"] = total_elapsed

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Ensemble strategy : {chosen_strategy}")
    logger.info(f"  {metric:<18}: {ensemble_score:.6f}")
    logger.info(f"  Models used        : {len(model_results)}")
    logger.info(f"  Prediction arrays  : {len(all_oof)}")
    if len(submission_paths) == 1:
        logger.info(f"  Submission path    : {submission_paths[0]}")
    else:
        logger.info(f"  Submissions        : {len(submission_paths)} files")
        for sp in submission_paths:
            logger.info(f"    {sp}")
    logger.info("-" * 60)
    logger.info("TIMING BREAKDOWN")
    for step_name, elapsed in step_times.items():
        if step_name == "total":
            continue
        pct = (elapsed / total_elapsed * 100) if total_elapsed > 0 else 0
        logger.info(
            f"  {step_name:<18}: {_fmt_time(elapsed):>12s}  ({pct:>5.1f}%)"
        )
    logger.info(f"  {'total':<18}: {_fmt_time(total_elapsed):>12s}")
    logger.info("")
    logger.info("  MODEL DETAIL")
    # Sort models by elapsed time descending (slowest first)
    model_timings = [
        (
            name,
            res.get("elapsed", 0),
            res.get("optuna_elapsed", 0),
            res.get("retrain_elapsed", 0),
            res.get("n_trials", 0),
            res.get("avg_trial_time", 0),
        )
        for name, res in model_results.items()
        if "elapsed" in res
    ]
    model_timings.sort(key=lambda x: x[1], reverse=True)
    for name, total, optuna_t, retrain_t, n_trials, avg_t in model_timings:
        pct = (total / total_elapsed * 100) if total_elapsed > 0 else 0
        logger.info(
            f"    {name:<16}: {_fmt_time(total):>12s}  ({pct:>5.1f}%)  "
            f"optuna={_fmt_time(optuna_t)}  retrain={_fmt_time(retrain_t)}  "
            f"{n_trials} trials @ {avg_t:.1f}s/trial"
        )
    logger.info("=" * 60)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace with:
        - config: path to pipeline.yaml (required)
        - strategy: optional override for strategy mode ('api' or 'manual')

    Steps:
        1. Create ArgumentParser with description.
        2. Add --config argument (required, type=str).
        3. Add --strategy argument (optional, choices=['api', 'manual']).
        4. Parse and return args.
    """
    parser = argparse.ArgumentParser(
        description="Maestro ML — LLM-orchestrated AutoML pipeline for tabular data."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the pipeline.yaml configuration file.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["api", "manual"],
        default=None,
        help="Override the strategy mode from the pipeline config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.strategy:
        with open(args.config, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        raw.setdefault("strategy", {})["mode"] = args.strategy
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(raw, tmp, default_flow_style=False)
            tmp_path = tmp.name
        try:
            main(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        main(args.config)
