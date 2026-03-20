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
from datetime import datetime
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


def _generate_round_report(
    model_results: dict,
    pipeline_config: "PipelineConfig",
    all_oof: "list[np.ndarray]",
    model_labels: "list[str]",
    y_true: "np.ndarray | None",
    metric: str,
    ensemble_score: "float | None",
    chosen_strategy: "str | None",
    nsga2_info: "dict | None" = None,
    output_path: "str | Path | None" = None,
) -> str:
    """Generate a comprehensive round report from Optuna DBs and model results.

    Covers 11 sections: run history, score trajectory, pruning analysis,
    HP convergence, fold-level score matrix, assembly diagnostics,
    substudy→main transfer, cross-model diversity, ensemble result,
    GBM config similarity, and OOF-LB gap.

    Args:
        model_results: Dict from run_all_studies() — pass {} for standalone mode.
        pipeline_config: Loaded pipeline configuration.
        all_oof: Per-model averaged OOF arrays — pass [] for standalone mode.
        model_labels: Labels corresponding to all_oof entries.
        y_true: Ground-truth target array — None in standalone mode.
        metric: Scoring metric string ("roc_auc" or "neg_rmse").
        ensemble_score: Final ensemble OOF score — None in standalone mode.
        chosen_strategy: Ensemble strategy name — None in standalone mode.
        nsga2_info: Dict from run_nsga2_ensemble() — None when not NSGA-II.
        output_path: If provided, write the report to this file.

    Returns:
        Full report as a string.
    """
    import optuna as _optuna
    from math import ceil

    lines: list[str] = []
    storage_dir = pipeline_config.optuna.storage_dir
    run_name = pipeline_config.run_name
    models_order = list(model_results.keys()) if model_results else list(pipeline_config.models)
    maximize = (metric == "roc_auc")

    def _sep(char: str = "=", width: int = 78) -> str:
        return char * width

    def _h(title: str) -> str:
        return f"\n{_sep()}\n{title}\n{_sep('-')}"

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(_sep())
    lines.append(f"ROUND REPORT — {run_name} — {now_str}")
    lines.append(_sep())

    # ------------------------------------------------------------------
    # Load Optuna studies from DB (used by multiple sections)
    # ------------------------------------------------------------------
    _studies: dict[str, "_optuna.Study"] = {}
    _sub_studies: dict[str, "_optuna.Study"] = {}
    if storage_dir:
        for mn in models_order:
            db = Path(storage_dir) / f"{run_name}__{mn}.db"
            if db.exists():
                try:
                    _studies[mn] = _optuna.load_study(
                        study_name=f"{run_name}__{mn}",
                        storage=f"sqlite:///{db}",
                    )
                except Exception:
                    pass
            sub_db = Path(storage_dir) / f"{run_name}__{mn}__sub.db"
            if sub_db.exists():
                try:
                    _sub_studies[mn] = _optuna.load_study(
                        study_name=f"{run_name}__{mn}__sub",
                        storage=f"sqlite:///{sub_db}",
                    )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Section 1: Run History
    # ------------------------------------------------------------------
    lines.append(_h("1. RUN HISTORY"))
    hdr = f"  {'model':<14} {'best':>8}  {'complete':>8}  {'pruned':>7}  {'failed':>6}  {'sec/trial':>9}  {'elapsed':>10}  sub_trials  sub_best"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for mn in models_order:
        study = _studies.get(mn)
        res = model_results.get(mn, {})
        if study is None and not res:
            lines.append(f"  {mn:<14}  (no data)")
            continue
        if study is not None:
            trials = study.trials
            n_complete = sum(1 for t in trials if t.state == _optuna.trial.TrialState.COMPLETE)
            n_pruned = sum(1 for t in trials if t.state == _optuna.trial.TrialState.PRUNED)
            n_failed = len(trials) - n_complete - n_pruned
            try:
                best_val = study.best_value
            except ValueError:
                best_val = float("nan")
        else:
            n_complete = res.get("n_trials", 0)
            n_pruned = 0
            n_failed = 0
            best_val = res.get("top_configs", [{}])[0].get("value", float("nan")) if res.get("top_configs") else float("nan")

        # sec/trial: prefer in-memory result, fall back to DB timestamps
        avg_t = res.get("avg_trial_time", 0.0)
        if avg_t == 0.0 and study is not None:
            completed_trials = [
                t for t in study.trials
                if t.state == _optuna.trial.TrialState.COMPLETE
                and t.datetime_start is not None
                and t.datetime_complete is not None
            ]
            if completed_trials:
                durations = [
                    (t.datetime_complete - t.datetime_start).total_seconds()
                    for t in completed_trials
                ]
                avg_t = sum(durations) / len(durations)

        # elapsed: prefer in-memory result, fall back to DB first/last timestamp
        elapsed = res.get("optuna_elapsed", 0.0)
        if elapsed == 0.0 and study is not None:
            timed = [
                t for t in study.trials
                if t.datetime_start is not None and t.datetime_complete is not None
            ]
            if timed:
                wall_start = min(t.datetime_start for t in timed)
                wall_end = max(t.datetime_complete for t in timed)
                elapsed = (wall_end - wall_start).total_seconds()
        elapsed_str = _fmt_time(elapsed) if elapsed > 0 else "n/a"

        sub_study = _sub_studies.get(mn)
        sub_n = len(sub_study.trials) if sub_study else 0
        sub_str = str(sub_n) if sub_n > 0 else "-"

        # sub_best: best value from sub study
        sub_best_str = "-"
        if sub_study:
            try:
                sub_best_str = f"{sub_study.best_value:.6f}"
            except ValueError:
                pass

        lines.append(
            f"  {mn:<14} {best_val:>8.6f}  {n_complete:>8}  {n_pruned:>7}  {n_failed:>6}"
            f"  {avg_t:>8.1f}s  {elapsed_str:>10}  {sub_str:>10}  {sub_best_str}"
        )

    # ------------------------------------------------------------------
    # Section 2: Score Trajectory
    # ------------------------------------------------------------------
    # _synth_models accumulates per-model data for Section 12 strategy synthesis
    _synth_models: dict[str, dict] = {}

    lines.append(_h("2. SCORE TRAJECTORY (best-so-far at trial checkpoints)"))
    checkpoints = [10, 25, 50, 100, 150, 200, 300]
    for mn in models_order:
        study = _studies.get(mn)
        if study is None:
            continue
        completed = sorted(
            [t for t in study.trials if t.state == _optuna.trial.TrialState.COMPLETE],
            key=lambda t: t.number,
        )
        if not completed:
            lines.append(f"  {mn}: no completed trials")
            continue
        values = [t.value for t in completed]
        best_so_far: list[float] = []
        cur_best = -float("inf") if maximize else float("inf")
        for v in values:
            if (maximize and v > cur_best) or (not maximize and v < cur_best):
                cur_best = v
            best_so_far.append(cur_best)

        trajectory_parts: list[str] = []
        for cp in checkpoints:
            if cp <= len(best_so_far):
                trajectory_parts.append(f"t{cp}:{best_so_far[cp - 1]:.6f}")
        if not trajectory_parts:
            trajectory_parts.append(f"t{len(best_so_far)}:{best_so_far[-1]:.6f}")

        # Marginal gain: adaptive window = min(50, n_trials // 2), min 10
        n_comp = len(completed)
        gain_window = max(10, min(50, n_comp // 2))
        if n_comp > gain_window:
            best_excl = best_so_far[-(gain_window + 1)]
            marginal = best_so_far[-1] - best_excl
            marginal_str = f"  marginal_gain_last{gain_window}={marginal:+.6f}"
        else:
            marginal_str = f"  (<{gain_window * 2} trials, no marginal gain)"

        # "At current rate" estimate: if avg_t known, estimate trials per budget hour
        rate_str = ""
        avg_t_trial = res.get("avg_trial_time", 0.0)
        timed_completed = []
        if study is not None:
            timed_completed = [
                t for t in study.trials
                if t.state == _optuna.trial.TrialState.COMPLETE
                and t.datetime_start is not None and t.datetime_complete is not None
            ]
        if avg_t_trial == 0.0 and timed_completed:
            avg_t_trial = sum(
                (t.datetime_complete - t.datetime_start).total_seconds()
                for t in timed_completed
            ) / len(timed_completed)
        if avg_t_trial > 0:
            per_hour = 3600.0 / avg_t_trial
            rate_str = f"  [{per_hour:.0f} trials/h]"

        # Score distribution: Q25 / median / Q75 / worst
        import numpy as _np_rr
        dist_str = ""
        if values:
            q25, med, q75 = _np_rr.percentile(values, [25, 50, 75])
            worst = min(values) if maximize else max(values)
            dist_str = f"\n    score_dist: Q25={q25:.6f}  median={med:.6f}  Q75={q75:.6f}  worst={worst:.6f}"

        # Time-to-best: how long until the best trial was found
        ttb_str = ""
        if timed_completed:
            study_start = min(t.datetime_start for t in timed_completed)
            completed_sorted_by_num = sorted(
                timed_completed, key=lambda t: t.number
            )
            cur = -float("inf") if maximize else float("inf")
            best_t_obj = None
            for t in completed_sorted_by_num:
                if (maximize and t.value > cur) or (not maximize and t.value < cur):
                    cur = t.value
                    best_t_obj = t
            if best_t_obj and best_t_obj.datetime_complete:
                secs_to_best = (best_t_obj.datetime_complete - study_start).total_seconds()
                total_elapsed_ttb = (
                    max(t.datetime_complete for t in timed_completed) - study_start
                ).total_seconds()
                ttb_str = (
                    f"\n    time_to_best: {_fmt_time(secs_to_best)}"
                    f" of {_fmt_time(total_elapsed_ttb)} total"
                    f" (trial#{best_t_obj.number})"
                )

        # Estimated ceiling: gain_rate = last_window_gain / prev_window_gain
        ceiling_str = ""
        if n_comp > gain_window * 2:
            prev_excl = best_so_far[-(gain_window * 2 + 1)]
            prev_gain = best_so_far[-(gain_window + 1)] - prev_excl
            last_gain = best_so_far[-1] - best_so_far[-(gain_window + 1)]
            if prev_gain > 1e-9:
                gain_rate = last_gain / prev_gain
                if gain_rate < 0.15:
                    status = "SATURATED"
                elif gain_rate < 0.40:
                    status = "slowing"
                else:
                    status = "still improving"
                ceiling_str = f"\n    ceiling: gain_rate={gain_rate:.2f} ({status})"

        lines.append(
            f"  {mn}: {', '.join(trajectory_parts)}{marginal_str}{rate_str}"
            f"{dist_str}{ttb_str}{ceiling_str}"
        )

        # Accumulate for Section 12 synthesis
        _synth_gain_rate: float | None = None
        _synth_status: str = "unknown"
        if n_comp > gain_window * 2:
            _prev_excl = best_so_far[-(gain_window * 2 + 1)]
            _prev_g = best_so_far[-(gain_window + 1)] - _prev_excl
            _last_g = best_so_far[-1] - best_so_far[-(gain_window + 1)]
            if _prev_g > 1e-9:
                _synth_gain_rate = _last_g / _prev_g
                if _synth_gain_rate < 0.15:
                    _synth_status = "SATURATED"
                elif _synth_gain_rate < 0.40:
                    _synth_status = "slowing"
                else:
                    _synth_status = "improving"
        elif n_comp < 30:
            _synth_status = "few_trials"
        _synth_models[mn] = {
            "best": best_so_far[-1] if best_so_far else None,
            "n_trials": n_comp,
            "gain_rate": _synth_gain_rate,
            "status": _synth_status,
            "marginal": marginal if n_comp > gain_window else None,
        }

    # ------------------------------------------------------------------
    # Section 3: Pruning Analysis
    # ------------------------------------------------------------------
    lines.append(_h("3. PRUNING ANALYSIS"))
    for mn in models_order:
        study = _studies.get(mn)
        if study is None:
            continue
        pruned = [t for t in study.trials if t.state == _optuna.trial.TrialState.PRUNED]
        n_total = len(study.trials)
        n_pruned = len(pruned)
        if n_total == 0:
            continue
        pct = 100.0 * n_pruned / n_total

        # Infer cause: has intermediate_values → got through ≥1 fold → MedianPruner
        #              no intermediate_values → pruned on fold 0 (fold_timeout or step 0)
        n_median = sum(1 for t in pruned if t.intermediate_values)
        n_timeout = n_pruned - n_median

        # Sub study
        sub_study = _sub_studies.get(mn)
        sub_info = ""
        if sub_study:
            sub_pruned = sum(1 for t in sub_study.trials if t.state == _optuna.trial.TrialState.PRUNED)
            sub_total = len(sub_study.trials)
            sub_info = f"  sub: {sub_pruned}/{sub_total} pruned"

        lines.append(
            f"  {mn:<14}: {n_pruned}/{n_total} pruned ({pct:.0f}%)"
            f"  [median_pruner={n_median}, fold_timeout/fold0={n_timeout}]{sub_info}"
        )

    # ------------------------------------------------------------------
    # Section 4: HP Convergence
    # ------------------------------------------------------------------
    lines.append(_h("4. HYPERPARAMETER CONVERGENCE (top-20% vs full explored range)"))
    for mn in models_order:
        study = _studies.get(mn)
        if study is None:
            continue
        completed = [t for t in study.trials if t.state == _optuna.trial.TrialState.COMPLETE]
        if len(completed) < 3:
            lines.append(f"  {mn}: insufficient trials for convergence analysis")
            continue
        completed_sorted = sorted(
            completed, key=lambda t: t.value, reverse=maximize
        )
        n_top = max(5, len(completed) // 5)
        top_trials = completed_sorted[:n_top]
        best_trial = completed_sorted[0]

        hp_imp = model_results.get(mn, {}).get("hp_importance", {})

        lines.append(f"\n  {mn} ({len(completed)} trials, top-20%={n_top}):")
        all_param_names = sorted(
            {k for t in completed for k in t.params},
            key=lambda k: -hp_imp.get(k, 0),
        )
        for param in all_param_names:
            all_vals = [t.params[param] for t in completed if param in t.params]
            top_vals = [t.params[param] for t in top_trials if param in t.params]
            if not all_vals or not top_vals:
                continue
            best_val = best_trial.params.get(param, "n/a")
            imp_str = f" imp={hp_imp[param]:.3f}" if param in hp_imp else ""
            if isinstance(all_vals[0], (int, float)):
                lo, hi = min(all_vals), max(all_vals)
                t_lo, t_hi = min(top_vals), max(top_vals)
                full_range = hi - lo
                if full_range > 1e-10:
                    top_range = t_hi - t_lo
                    frac = top_range / full_range
                    conf = "HIGH" if frac < 0.30 else ("MED" if frac < 0.60 else "LOW")
                    # Search space utilization: what % of trials explored upper 25% of range
                    upper_thresh = lo + 0.75 * full_range
                    n_upper = sum(1 for v in all_vals if v >= upper_thresh)
                    util_pct = 100.0 * n_upper / len(all_vals)
                    util_str = f"  util_upper25%={util_pct:.0f}%"
                else:
                    conf = "HIGH"  # all same value
                    util_str = ""
                lines.append(
                    f"    {param:<30} all=[{lo:.4g},{hi:.4g}]  top=[{t_lo:.4g},{t_hi:.4g}]"
                    f"  {conf}  best={best_val:.4g}{imp_str}{util_str}"
                )
            else:
                # Categorical — show utilization as fraction of choices tried
                from collections import Counter
                all_counts = Counter(all_vals)
                top_counts = Counter(top_vals)
                most_common = top_counts.most_common(2)
                mc_str = ", ".join(f"{v}:{c}" for v, c in most_common)
                all_str = ", ".join(f"{v}:{c}" for v, c in all_counts.most_common())
                lines.append(
                    f"    {param:<30} top-20%: [{mc_str}]  all: [{all_str}]  best={best_val}{imp_str}"
                )

        # Top-5 HP values table (AI can infer interactions)
        top5 = completed_sorted[:5]
        if top5:
            # Select params that actually vary across top-5
            varying = [
                p for p in all_param_names
                if len({t.params.get(p) for t in top5 if p in t.params}) > 1
                   or len(top5) == 1
            ][:8]  # at most 8 cols to keep readable
            if varying:
                lines.append(f"\n    top-5 trials (AI: infer HP interactions):")
                # header
                header = f"    {'score':>9}" + "".join(f"  {p[:10]:>10}" for p in varying)
                lines.append(header)
                for t in top5:
                    vals = "".join(
                        f"  {str(t.params.get(p, '?'))[:10]:>10}" for p in varying
                    )
                    lines.append(f"    {t.value:>9.6f}{vals}")

            # HP Landscape Roughness: std of top-5 scores / best_score → smooth vs rough
            top5_scores = [t.value for t in top5]
            if len(top5_scores) >= 3:
                roughness = float(np.std(top5_scores))
                if roughness < 0.0001:
                    r_label = "smooth  (converged — hard to improve further)"
                elif roughness < 0.001:
                    r_label = "moderate (some unexplored regions possible)"
                else:
                    r_label = "rough   (high variance — promising regions likely remain)"
                lines.append(f"\n    landscape_roughness: top5_score_std={roughness:.6f}  {r_label}")

            # Score-Weighted HP Importance: mean separation top-5% vs top-20%
            # High separation = this HP distinguishes champions from the pack
            n5pct = max(3, n_comp // 20)
            n20pct = max(5, n_comp // 5)
            if n_comp >= 20 and n5pct < n20pct:
                top5pct = completed_sorted[:n5pct]
                top20pct = completed_sorted[:n20pct]
                sep_items: list[tuple[float, str, float, float]] = []
                for param in all_param_names:
                    v5 = [t.params[param] for t in top5pct if param in t.params
                          and isinstance(t.params[param], (int, float))]
                    v20 = [t.params[param] for t in top20pct if param in t.params
                           and isinstance(t.params[param], (int, float))]
                    if len(v5) < 2 or len(v20) < 2:
                        continue
                    r = max(v20) - min(v20)
                    if r < 1e-10:
                        continue
                    sep = abs(float(np.mean(v5)) - float(np.mean(v20))) / r
                    sep_items.append((sep, param, float(np.mean(v5)), float(np.mean(v20))))
                sep_items.sort(reverse=True)
                if sep_items[:3]:
                    lines.append(
                        f"    top{n5pct}/top{n20pct} HP separation "
                        f"(high=critical for last-mile improvement):"
                    )
                    for sep, param, m5, m20 in sep_items[:3]:
                        lines.append(
                            f"      {param:<30} sep={sep:.3f}"
                            f"  top{n5pct}%_mean={m5:.4g}"
                            f"  top{n20pct}%_mean={m20:.4g}"
                        )

    # ------------------------------------------------------------------
    # Section 4b: TPE HEALTH — detect collapsed parameter search
    # ------------------------------------------------------------------
    def _compute_tpe_health(
        completed_trials: list,
        window: int = 20,
        threshold: float = 0.05,
    ) -> dict:
        """Compute TPE health metrics from completed trials.

        Returns dict with keys: status, param_std_ratio, collapse_onset,
        wasted_trials, wasted_seconds.
        """
        sorted_trials = sorted(completed_trials, key=lambda t: t.number)
        n = len(sorted_trials)
        if n < window:
            return {"status": "insufficient", "param_std_ratio": None,
                    "collapse_onset": None, "wasted_trials": 0, "wasted_seconds": 0.0}

        # Identify numeric params present across trials.
        all_param_names: set[str] = set()
        for t in sorted_trials:
            for k, v in t.params.items():
                if isinstance(v, (int, float)):
                    all_param_names.add(k)

        if not all_param_names:
            return {"status": "insufficient", "param_std_ratio": None,
                    "collapse_onset": None, "wasted_trials": 0, "wasted_seconds": 0.0}

        # Pre-compute full-history range per param.
        full_ranges: dict[str, tuple[float, float]] = {}
        for p in all_param_names:
            vals = [t.params[p] for t in sorted_trials if p in t.params
                    and isinstance(t.params[p], (int, float))]
            if len(vals) >= 2:
                r = max(vals) - min(vals)
                if r > 1e-10:
                    full_ranges[p] = (min(vals), r)

        if not full_ranges:
            return {"status": "insufficient", "param_std_ratio": None,
                    "collapse_onset": None, "wasted_trials": 0, "wasted_seconds": 0.0}

        def _ratio_at(end_idx: int) -> float:
            """Mean normalised std for window ending at end_idx (exclusive)."""
            start = end_idx - window
            if start < 0:
                return 1.0
            win = sorted_trials[start:end_idx]
            ratios_w: list[float] = []
            for p, (_, rng) in full_ranges.items():
                wv = [t.params[p] for t in win if p in t.params
                      and isinstance(t.params[p], (int, float))]
                if len(wv) < 2:
                    continue
                ratios_w.append(float(np.std(wv)) / rng)
            return float(np.mean(ratios_w)) if ratios_w else 1.0

        # Current ratio (last window).
        current_ratio = _ratio_at(n)

        if current_ratio >= threshold:
            return {"status": "healthy", "param_std_ratio": current_ratio,
                    "collapse_onset": None, "wasted_trials": 0, "wasted_seconds": 0.0}

        # Collapse detected — find onset by scanning backwards.
        onset_idx = n  # trial index where collapse started
        for end in range(n - 1, window - 1, -1):
            r = _ratio_at(end)
            if r >= threshold:
                onset_idx = end
                break
        else:
            onset_idx = window  # collapsed from the very start

        onset_trial_num = sorted_trials[onset_idx].number if onset_idx < n else sorted_trials[-1].number
        wasted = n - onset_idx

        # Estimate wasted time.
        durations = [
            (t.datetime_complete - t.datetime_start).total_seconds()
            for t in sorted_trials
            if t.datetime_start and t.datetime_complete
        ]
        avg_dur = sum(durations) / len(durations) if durations else 0.0
        wasted_secs = wasted * avg_dur

        return {
            "status": "COLLAPSED",
            "param_std_ratio": current_ratio,
            "collapse_onset": onset_trial_num,
            "wasted_trials": wasted,
            "wasted_seconds": wasted_secs,
        }

    tpe_health_data: list[tuple[str, dict]] = []
    for mn in models_order:
        study = _studies.get(mn)
        if study is None:
            continue
        completed = [
            t for t in study.trials
            if t.state == _optuna.trial.TrialState.COMPLETE
        ]
        health = _compute_tpe_health(completed)
        tpe_health_data.append((mn, health))

    if tpe_health_data:
        lines.append(_h("4b. TPE HEALTH"))
        for mn, h in tpe_health_data:
            status = h["status"]
            ratio = h["param_std_ratio"]
            if status == "insufficient":
                lines.append(f"  {mn:<14}  insufficient (<20 completed trials)")
            elif status == "healthy":
                lines.append(f"  {mn:<14}  healthy (param_std_ratio={ratio:.3f})")
            elif status == "COLLAPSED":
                onset = h["collapse_onset"]
                wasted = h["wasted_trials"]
                wasted_s = h["wasted_seconds"]
                lines.append(
                    f"  {mn:<14}  COLLAPSED at trial ~{onset} "
                    f"(param_std_ratio={ratio:.3f}, ~{wasted} trials wasted, "
                    f"~{_fmt_time(wasted_s)} lost)"
                )

    # ------------------------------------------------------------------
    # Section 5: Fold-Level Score Matrix
    # ------------------------------------------------------------------
    fold_score_data: dict[str, list[float]] = {}
    for mn in models_order:
        fs = model_results.get(mn, {}).get("best_fold_scores", [])
        if fs:
            fold_score_data[mn] = fs

    if fold_score_data:
        lines.append(_h("5. FOLD-LEVEL SCORE MATRIX (best composite per model)"))
        n_folds = max(len(v) for v in fold_score_data.values())
        col_w = 8
        hdr_parts = [f"{'model':<16}"] + [f"f{i:>5}" for i in range(n_folds)] + [f"{'mean':>7}"]
        lines.append("  " + "".join(hdr_parts))
        lines.append("  " + "-" * (16 + (n_folds + 1) * col_w))
        all_fold_scores: list[list[float]] = []
        for mn, scores in fold_score_data.items():
            mean_s = sum(scores) / len(scores) if scores else 0.0
            row = f"  {mn:<16}" + "".join(f"{s:>7.4f} " for s in scores) + f"{mean_s:>7.4f}"
            lines.append(row)
            all_fold_scores.append(scores)
        # Per-fold mean (hard fold detection)
        if all_fold_scores and len(set(len(s) for s in all_fold_scores)) == 1:
            col_means = [
                sum(row[i] for row in all_fold_scores) / len(all_fold_scores)
                for i in range(n_folds)
            ]
            overall_mean = sum(col_means) / len(col_means)
            row = f"  {'(fold_mean)':<16}" + "".join(f"{v:>7.4f} " for v in col_means) + f"{overall_mean:>7.4f}"
            lines.append(row)
    else:
        lines.append(_h("5. FOLD-LEVEL SCORE MATRIX"))
        lines.append("  (not available — global selection mode or no data)")

    # ------------------------------------------------------------------
    # Section 6: Assembly Diagnostics
    # ------------------------------------------------------------------
    lines.append(_h("6. ASSEMBLY DIAGNOSTICS"))
    for mn in models_order:
        res = model_results.get(mn)
        if res is None:
            lines.append(f"  {mn:<14}: (no data — standalone mode)")
            continue
        mode = res.get("selection_mode", "global")
        n_final = len(res.get("oof_preds", []))
        if mode == "fold_coverage":
            n_committed = res.get("n_committed")
            committed_str = f"committed={n_committed}" if n_committed is not None else "committed=?"
            lines.append(f"  {mn:<14}: fold_coverage  {committed_str}  selected={n_final}")
        elif mode == "per_fold":
            n_raw = res.get("n_composites_raw")
            raw_str = f"raw={n_raw}→dedup={n_final}" if n_raw is not None else f"final={n_final}"
            lines.append(f"  {mn:<14}: per_fold  composites {raw_str}")
        else:
            n_configs = len(res.get("top_configs", []))
            lines.append(f"  {mn:<14}: global  top_configs={n_configs}  prediction_arrays={n_final}")

    # ------------------------------------------------------------------
    # Section 7: Substudy → Main Transfer
    # ------------------------------------------------------------------
    lines.append(_h("7. SUBSTUDY → MAIN TRANSFER EFFICIENCY"))
    if not _sub_studies:
        lines.append("  (no substudy DBs found)")
    for mn, sub_study in _sub_studies.items():
            main_study = _studies.get(mn)
            sub_completed = [
                t for t in sub_study.trials if t.state == _optuna.trial.TrialState.COMPLETE
            ]
            if not sub_completed:
                lines.append(f"  {mn}: substudy has no completed trials")
                continue
            sub_best = sorted(sub_completed, key=lambda t: t.value, reverse=maximize)[0]
            sub_val = sub_best.value
            sub_params = {k: v for k, v in sub_best.params.items() if k != "scaler"}

            match_val = None
            if main_study:
                for t in main_study.trials:
                    if t.state != _optuna.trial.TrialState.COMPLETE:
                        continue
                    main_params = {k: v for k, v in t.params.items() if k != "scaler"}
                    if main_params == sub_params:
                        match_val = t.value
                        break

            if match_val is not None:
                delta = match_val - sub_val
                quality = "✓ good" if abs(delta) < 0.005 else ("↑ improved" if delta > 0 else "↓ degraded")
                lines.append(
                    f"  {mn}: sub_best={sub_val:.6f} (trial#{sub_best.number})"
                    f"  →  main={match_val:.6f}  delta={delta:+.4f}  {quality}"
                )
            else:
                lines.append(
                    f"  {mn}: sub_best={sub_val:.6f} (trial#{sub_best.number})"
                    f"  →  config not yet evaluated in main study"
                )

    # ------------------------------------------------------------------
    # Section 8: Cross-Model Diversity
    # ------------------------------------------------------------------
    _synth_div: dict[str, float] = {}  # model → avg Spearman corr with others (for Section 12)
    if all_oof and model_labels:
        lines.append(_h("8. CROSS-MODEL DIVERSITY (tuned OOF Spearman correlations)"))
        # Build per-model average OOF
        model_avg_oof: dict[str, list[np.ndarray]] = {}
        for arr, label in zip(all_oof, model_labels):
            mn_key = label.rsplit("_", 1)[0] if "_" in label else label
            model_avg_oof.setdefault(mn_key, []).append(arr)
        avg_labels = list(model_avg_oof.keys())
        avg_arrays = [np.mean(arrs, axis=0) for arrs in model_avg_oof.values()]
        if len(avg_arrays) > 1:
            corr_mat = compute_correlation_matrix(avg_arrays)
            lw = max(len(lb) for lb in avg_labels) + 2
            hdr_line = " " * lw + "".join(f"{lb:>10s}" for lb in avg_labels)
            lines.append("  " + hdr_line)
            lines.append("  " + "-" * len(hdr_line))
            for i, rl in enumerate(avg_labels):
                row_str = f"  {rl:<{lw}}" + "".join(f"{corr_mat[i, j]:>10.3f}" for j in range(len(avg_labels)))
                lines.append(row_str)
            from src.ensemble.diversity import effective_ensemble_size
            n_eff = effective_ensemble_size(corr_mat)
            lines.append(f"\n  N_eff = {n_eff:.3f} / {len(avg_labels)}")
            # Most/least correlated pairs
            max_corr, min_corr = -float("inf"), float("inf")
            max_pair, min_pair = ("", ""), ("", "")
            for i in range(len(avg_labels)):
                for j in range(i + 1, len(avg_labels)):
                    c = corr_mat[i, j]
                    if c > max_corr:
                        max_corr, max_pair = c, (avg_labels[i], avg_labels[j])
                    if c < min_corr:
                        min_corr, min_pair = c, (avg_labels[i], avg_labels[j])
            lines.append(f"  Most correlated : {max_pair[0]} — {max_pair[1]} ({max_corr:.3f})")
            lines.append(f"  Least correlated: {min_pair[0]} — {min_pair[1]} ({min_corr:.3f})")
            # Capture per-model avg correlation with others (for Section 12)
            _synth_div: dict[str, float] = {}
            for i, lbl in enumerate(avg_labels):
                others = [corr_mat[i, j] for j in range(len(avg_labels)) if j != i]
                _synth_div[lbl] = float(np.mean(others)) if others else 1.0
    else:
        lines.append(_h("8. CROSS-MODEL DIVERSITY"))
        lines.append("  (not available — standalone mode or single model)")

    # ------------------------------------------------------------------
    # Section 9: Ensemble Result + NSGA-II Weights
    # ------------------------------------------------------------------
    _synth_ensemble: dict = {}  # populated below, used in Section 12
    lines.append(_h("9. ENSEMBLE RESULT"))
    if ensemble_score is not None and chosen_strategy is not None:
        metric_display = metric.replace("neg_", "")
        lines.append(f"  Strategy : {chosen_strategy}")
        lines.append(f"  OOF {metric_display:<8}: {ensemble_score:.6f}")
    else:
        lines.append("  (not available — standalone mode)")

    if nsga2_info is not None and model_labels:
        sel = nsga2_info.get("selected_models", [])
        wts = nsga2_info.get("weights", [])
        blend_score = nsga2_info.get("blend_score")
        meta_scores = nsga2_info.get("meta_scores", {})
        best_meta_name = nsga2_info.get("best_meta_name", "blend")
        n_sel = nsga2_info.get("n_selected", len(sel))
        n_tot = nsga2_info.get("n_total", len(model_labels))
        ens_neff = nsga2_info.get("effective_size")

        # Meta-model comparison
        if blend_score is not None or meta_scores:
            lines.append(f"  Meta-model comparison ({n_sel}/{n_tot} arrays → meta stage):")
            if blend_score is not None:
                lines.append(f"    {'blend':<10}: {blend_score:.6f}  (baseline)")
            for mn_m, ms in sorted(meta_scores.items(), key=lambda kv: -kv[1]):
                delta = ms - (blend_score or ms)
                winner = "  ← winner" if mn_m == best_meta_name and mn_m != "blend" else ""
                lines.append(f"    {mn_m:<10}: {ms:.6f}  ({delta:+.6f}){winner}")

        # Per-model array counts
        model_counts: dict[str, list[int]] = {}  # [selected, total]
        for label in model_labels:
            mn_k = label.rsplit("_", 1)[0] if "_" in label else label
            if mn_k not in model_counts:
                model_counts[mn_k] = [0, 0]
            model_counts[mn_k][1] += 1
        model_weight_sum: dict[str, float] = {}
        for idx, w in zip(sel, wts):
            if idx < len(model_labels):
                mn_k = model_labels[idx].rsplit("_", 1)[0] if "_" in model_labels[idx] else model_labels[idx]
                model_weight_sum[mn_k] = model_weight_sum.get(mn_k, 0.0) + w
                model_counts[mn_k][0] += 1

        neff_str = f"  N_eff={ens_neff:.3f}" if ens_neff else ""
        lines.append(f"  NSGA-II weights:{neff_str}")
        all_model_names = sorted(model_counts.keys(),
                                 key=lambda m: -model_weight_sum.get(m, 0.0))
        for mn_k in all_model_names:
            w = model_weight_sum.get(mn_k, 0.0)
            s_n, t_n = model_counts[mn_k]
            bar = "█" * max(1, round(w * 20)) if w > 1e-4 else ""
            excl = "  ← EXCLUDED" if s_n == 0 else ""
            lines.append(f"    {mn_k:<16}: {w:.4f}  {bar:<12}  ({s_n}/{t_n} arrays){excl}")

        _synth_ensemble = {
            "n_eff": ens_neff,
            "model_weights": model_weight_sum,
            "model_counts": model_counts,
            "blend_score": blend_score,
            "meta_scores": meta_scores,
            "best_meta": best_meta_name,
        }

    # ------------------------------------------------------------------
    # Section 10: Config Similarity Across GBMs
    # ------------------------------------------------------------------
    lines.append(_h("10. CONFIG SIMILARITY ACROSS GBMS"))
    gbm_names = [mn for mn in ["catboost", "xgboost", "lightgbm"] if mn in _studies]
    if not gbm_names:
        lines.append("  (no GBM Optuna DBs found)")
    if gbm_names:
        gbm_params = ["depth", "max_depth", "num_leaves", "learning_rate", "n_estimators", "iterations"]
        header_row = f"  {'model':<14}" + "".join(f"{p:>16}" for p in gbm_params)
        lines.append(header_row)
        lines.append("  " + "-" * (14 + 16 * len(gbm_params)))
        for mn in gbm_names:
            study = _studies[mn]
            completed = [t for t in study.trials if t.state == _optuna.trial.TrialState.COMPLETE]
            if not completed:
                continue
            best = sorted(completed, key=lambda t: t.value, reverse=maximize)[0]
            row = f"  {mn:<14}"
            for p in gbm_params:
                val = best.params.get(p, "-")
                if isinstance(val, float):
                    row += f"{val:>16.4g}"
                else:
                    row += f"{str(val):>16}"
            lines.append(row)

        # Qualitative note
        depths: list[int] = []
        for mn in gbm_names:
            study = _studies[mn]
            completed = [t for t in study.trials if t.state == _optuna.trial.TrialState.COMPLETE]
            if not completed:
                continue
            best = sorted(completed, key=lambda t: t.value, reverse=maximize)[0]
            for param in ("depth", "max_depth"):
                if param in best.params:
                    depths.append(int(best.params[param]))
                    break
            if "num_leaves" in best.params:
                leaves = int(best.params["num_leaves"])
                # num_leaves ≈ 2^depth for balanced trees
                import math
                depths.append(int(math.log2(max(leaves, 1))))
        if depths:
            d_min, d_max = min(depths), max(depths)
            if d_max - d_min <= 2:
                lines.append(f"\n  → All GBMs converged to similar depth ({d_min}-{d_max})")
            else:
                lines.append(f"\n  → GBMs show divergent depth preferences ({d_min}-{d_max})")

    # ------------------------------------------------------------------
    # Section 11: OOF-LB Gap
    # ------------------------------------------------------------------
    lines.append(_h("11. OOF-LB GAP"))
    lb = pipeline_config.output.lb_score
    if lb is not None and ensemble_score is not None:
        gap = ensemble_score - lb
        lines.append(f"  OOF  : {ensemble_score:.6f}")
        lines.append(f"  LB   : {lb:.6f}")
        lines.append(f"  Gap  : {gap:+.4f}  ({'normal' if gap < 0.005 else 'large — possible overfit'})")
    elif lb is not None:
        lines.append(f"  LB score set: {lb:.6f}  (no OOF score in standalone mode)")
    else:
        lines.append("  Set output.lb_score in pipeline.yaml after submission to enable this section.")

    # ------------------------------------------------------------------
    # Section 12: Strategy Synthesis
    # ------------------------------------------------------------------
    lines.append(_h("12. STRATEGY SYNTHESIS"))

    # Per-model status table
    mw = _synth_ensemble.get("model_weights", {})
    mc = _synth_ensemble.get("model_counts", {})
    if _synth_models:
        lines.append(
            "  MODEL STATUS  (gain_rate: SATURATED<0.15 / slowing<0.40 / improving≥0.40):"
        )
        lines.append(
            f"  {'model':<12}  {'status':<12}  {'best':>9}  {'trials':>7}  "
            f"{'gain_rate':>10}  {'wt%':>6}  {'arrays':>10}"
        )
        lines.append("  " + "-" * 75)
        for mn in models_order:
            sd = _synth_models.get(mn)
            if sd is None:
                continue
            st = sd.get("status", "unknown")
            best = sd.get("best")
            n_t = sd.get("n_trials", 0)
            gr = sd.get("gain_rate")
            w = mw.get(mn, 0.0)
            s_n, t_n = mc.get(mn, [0, 0]) if mc else (0, 0)
            gr_str = f"{gr:.2f}" if gr is not None else "n/a"
            best_str = f"{best:.6f}" if best is not None else "n/a"
            arr_str = f"{s_n}/{t_n}" if mc else "n/a"
            excl = " EXCLUDED" if s_n == 0 and t_n > 0 else ""
            lines.append(
                f"  {mn:<12}  [{st:<10}]  {best_str}  {n_t:>7}  "
                f"{gr_str:>10}  {w*100:>5.1f}%  {arr_str:>10}{excl}"
            )

    # Ensemble health
    lines.append("")
    lines.append("  ENSEMBLE HEALTH:")
    if _synth_ensemble:
        n_eff = _synth_ensemble.get("n_eff")
        n_models = len(mw) if mw else len(models_order)
        if n_eff is not None:
            if n_eff < 1.1:
                neff_label = "severe dominance"
            elif n_eff < 1.5:
                neff_label = "low diversity"
            elif n_eff < 2.5:
                neff_label = "moderate diversity"
            else:
                neff_label = "good diversity"
            dom_model = max(mw.items(), key=lambda kv: kv[1])[0] if mw else "?"
            dom_w = mw.get(dom_model, 0.0)
            lines.append(
                f"    N_eff={n_eff:.3f}/{n_models} — {neff_label} "
                f"({dom_model} {dom_w*100:.0f}% weight)"
            )
        if _synth_div:
            best_mn = max(_synth_models.keys(), key=lambda m: _synth_models[m].get("best") or 0.0)
            for mn in sorted(_synth_div.keys(), key=lambda m: _synth_div[m]):
                if mn == best_mn:
                    continue
                corr = _synth_div.get(mn, 1.0)
                lines.append(
                    f"    {mn}: avg_corr_with_others={corr:.3f}"
                    + (" (lowest corr = highest diversity value)" if corr == min(_synth_div.values()) else "")
                )
        excluded = [mn for mn, (sn, tn) in mc.items() if sn == 0 and tn > 0] if mc else []
        if excluded:
            lines.append(f"    EXCLUDED: {', '.join(excluded)} — contributed 0 arrays to ensemble")
        meta_scores = _synth_ensemble.get("meta_scores", {})
        blend_s = _synth_ensemble.get("blend_score")
        best_meta = _synth_ensemble.get("best_meta", "blend")
        if meta_scores and blend_s is not None:
            best_meta_s = max(meta_scores.values()) if meta_scores else blend_s
            delta = best_meta_s - blend_s
            if best_meta == "blend":
                lines.append(f"    Meta-stacking: no improvement over blend")
            else:
                lines.append(
                    f"    Meta-stacking ({best_meta}): {delta:+.6f} over blend "
                    f"({'marginal' if delta < 0.0002 else 'meaningful'})"
                )
    else:
        lines.append("    (not available — standalone mode)")

    # Recommended budget (cross-reference with TPE health)
    _tpe_health_map: dict[str, dict] = {mn: h for mn, h in tpe_health_data}
    lines.append("")
    lines.append("  RECOMMENDED BUDGET FOR NEXT ROUND:")
    for mn in models_order:
        sd = _synth_models.get(mn)
        if sd is None:
            continue
        st = sd.get("status", "unknown")
        s_n, t_n = mc.get(mn, [0, 0]) if mc else (0, 0)
        excluded_flag = (s_n == 0 and t_n > 0)
        if excluded_flag and st == "SATURATED":
            rec = "DROP (SATURATED + excluded from ensemble)"
        elif excluded_flag:
            rec = "20min token budget (excluded from ensemble)"
        elif st == "SATURATED":
            rec = "20min token budget (saturated)"
        elif st == "slowing":
            rec = "reduce budget (slowing convergence)"
        elif st == "improving":
            rec = "keep/increase budget (still improving)"
        elif st == "few_trials":
            rec = "keep full budget (needs more exploration)"
        else:
            rec = "reassess after more trials"
        # Append collapse_restart hint if TPE collapsed.
        health = _tpe_health_map.get(mn, {})
        if health.get("status") == "COLLAPSED":
            rec += "  *** COLLAPSED — consider enabling collapse_restart + raising tpe.gamma"
        lines.append(f"    {mn:<12} → {rec}")

    lines.append("\n" + _sep())
    report = "\n".join(lines)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        _logging.getLogger("maestro").info(f"Round report saved: {out}")

    return report


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
           b. In manual mode, strategy is read from strategy_input_path (not saved again).
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
    log_dir = Path(pipeline_config.output.results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = str(log_dir / f"{pipeline_config.run_name}_{_ts}.log")
    logger = setup_logging(pipeline_config.runtime.verbose, log_file=log_file)

    logger.info(f"Pipeline config loaded: {pipeline_yaml_path}")
    logger.info(f"Run name: {pipeline_config.run_name} | Log: {log_file}")
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
    # Step 3 & 4: EDA + Strategy (skip EDA if eda_report already exists)
    # -------------------------------------------------------------------------
    # Skip EDA when eda_report.json already exists — preserve the original report.
    # Never overwrite an existing EDA with an empty placeholder.
    eda_start = time.time()
    strategy_mode = pipeline_config.strategy.mode
    manual_cfg = pipeline_config.strategy.manual or {}
    strategy_input_path = manual_cfg.get("strategy_input_path")
    eda_report_path = results_dir / "eda_report.json"
    skip_eda = eda_report_path.exists()

    if skip_eda:
        logger.info(
            f"EDA report exists: {eda_report_path} — skipping EDA"
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

    # Auto-map binary string targets (e.g., "Yes"/"No" → 0/1) when no
    # target_mapping is configured.  EDA already does this internally, but
    # the skip-EDA path and any future code paths also need it.
    target_col = pipeline_config.target_column
    if (
        not pipeline_config.target_mapping
        and target_col in train.columns
        and not pd.api.types.is_numeric_dtype(train[target_col])
        and pipeline_config.task_type == "binary_classification"
    ):
        unique_vals = sorted(train[target_col].dropna().unique().tolist())
        if len(unique_vals) == 2:
            auto_map = {unique_vals[0]: 0, unique_vals[1]: 1}
            train[target_col] = train[target_col].map(auto_map).astype(int)
            logger.info(f"Auto-mapped string target '{target_col}': {auto_map}")

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

    # Log where the strategy lives (manual mode: already at strategy_input_path)
    if strategy_input_path:
        logger.info(f"Strategy loaded from manual mode.")

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
    _nsga2_info: dict | None = None

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
        _nsga2_info = first_info
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
        # Populate nsga2_info with meta-comparison data for round report
        _nsga2_info["blend_score"] = blend_score
        _nsga2_info["meta_scores"] = {}
        _nsga2_info["n_selected"] = len(sel)
        _nsga2_info["n_total"] = len(all_oof)

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
                _nsga2_info["meta_scores"][meta_name] = m_score
            except Exception as exc:
                logger.warning(f"  Meta-model '{meta_name}' failed: {exc}")

        _nsga2_info["best_meta_name"] = best_meta_name
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

    # Round report (replaces diversity_report — includes diversity + run history + HP convergence)
    round_report_path = results_dir / f"round_report_{pipeline_config.run_name}_{_ts}.txt"
    _generate_round_report(
        model_results=model_results,
        pipeline_config=pipeline_config,
        all_oof=all_oof,
        model_labels=model_labels,
        y_true=y_true,
        metric=metric,
        ensemble_score=ensemble_score,
        chosen_strategy=chosen_strategy,
        nsga2_info=_nsga2_info,
        output_path=round_report_path,
    )

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
