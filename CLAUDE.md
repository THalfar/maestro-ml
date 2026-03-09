# CLAUDE.md — Maestro ML

## Project
maestro-ml — LLM-orchestrated AutoML framework for tabular data competitions.

## Architecture

Three layers:
- **Layer 1: EDA** (`src/eda/`) — Pure data profiling (pandas/numpy). No ML, no randomness.
- **Layer 2: Strategy** (`src/strategy/`) — LLM reads EDA report, generates feature engineering plan and model selection as YAML. Supports API mode (automatic) and manual mode (human-in-the-loop).
- **Layer 3: Engine** (`src/models/`, `src/features/`, `src/ensemble/`) — Executes the strategy: feature engineering, Optuna hyperparameter optimization, diversity-aware ensemble selection.

## Key Principles

- **YAML is the source of truth.** Never hardcode hyperparameter ranges, feature lists, or model configs in Python code. Everything comes from `configs/models/*.yaml` and the strategy YAML.
- **Architect/coder pattern.** Python files contain function signatures + detailed docstrings. The docstrings ARE the specification — implement exactly what they describe.
- **Type hints everywhere.** Use `from __future__ import annotations` at the top of every file.
- **Functions over classes** except for `ModelRegistry`, `PerFoldTracker` (in `trainer.py`), and dataclasses in `src/utils/io.py`.
- **No in-place DataFrame mutation.** Every function that takes a DataFrame should return a new one unless explicitly documented otherwise.
- **Deterministic where possible.** Always pass random seeds. Same config = same results.

## Implementation Order

Implement in this order (each module depends on the ones above it):
1. `src/utils/io.py` — YAML loading, dataclasses, logging
2. `src/eda/profiler.py` — Dataset analysis, EDA report generation
3. `src/features/engineer.py` — Dynamic feature creation from YAML
4. `src/models/registry.py` — Model factory, search space, GPU checks
5. `src/models/trainer.py` — Optuna studies, CV training loop
6. `src/ensemble/blender.py` — Weight optimization, rank average, meta-model
7. `src/ensemble/diversity.py` — NSGA-II, effective ensemble size
8. `src/strategy/llm_strategist.py` — LLM integration (API + manual modes)
9. `run.py` — Full pipeline orchestration

## GPU Handling

- GPU detection at startup via micro-trial per model (fit tiny dataset with GPU params).
- Auto-fallback to CPU if GPU fails — no crashes, just a log warning.
- GPU params defined in model YAML (`gpu.params` / `gpu.fallback`), not in Python.
- **CatBoost:** `eval_metric` must be `Logloss` on GPU (AUC is not supported on GPU).
- **CatBoost:** Always pass `train_dir` to a temp/results directory to keep project root clean:
  ```python
  CatBoostClassifier(..., train_dir=str(results_dir / "catboost_info"))
  ```
- **XGBoost:** v2.0+ uses `device="cuda"`, not the deprecated `tree_method="gpu_hist"`.
- **LightGBM:** Default pip/conda installs are CPU-only. GPU requires special build.
- **Neural net QMC warmup**: TabM/RealMLP trials take ~8-10 min each on large datasets (595k rows). Keep `qmc_warmup_trials` low (≤10) to leave budget for TPE exploration. 30 QMC trials can consume the entire timeout.
- **Neural net fold_timeout**: Per-fold training on large datasets needs generous timeouts. 120s is too aggressive for 595k rows — use 300s+ to avoid excessive trial pruning.

## Model-Specific Quirks

- **Seed parameter names vary**: CatBoost uses `random_seed`, all others use `random_state`.
- **CatBoost**: `eval_metric` goes in the constructor, not in `fit()`.
- **XGBoost/LightGBM**: `eval_metric` goes in `fit()` via `eval_set` / callbacks.
- **LightGBM**: Early stopping uses `callbacks=[lgb.early_stopping(...)]`, NOT the deprecated `early_stopping_rounds` parameter.
- **RealMLP**: Uses rectangular architecture (`hidden_sizes: "rectangular"` in fixed_params) with `n_hidden_layers` and `hidden_width` as Optuna search params. Uses **per-fold selection** (`selection_mode: per_fold`) — see "Per-Fold Selection" section below.
- **RealMLP fixed_params**: Task-type-keyed (`binary_classification` / `regression`), like Ridge. `n_ens` (pytabkit internal ensemble count) is in the Optuna search space, not fixed.
- **Ridge/Elastic Net YAML**: `fixed_params` is task-type-keyed (`binary_classification` / `regression`) because Ridge wraps different sklearn classes per task.
- **AdaBoost/GaussianNB/SVM/KNN**: No early stopping, no GPU — simple sklearn models added for ensemble diversity.

## Monotone Constraints

- **Strategy YAML** defines `monotone_constraints` as a dict of `{feature: direction}` (1=increasing, -1=decreasing).
- **EDA monotonicity detection** flags features with |Spearman rho| > 0.7 on binned target rates — strategy YAML should only constrain features actually listed in the EDA monotonicity section.
- **Positional list resolution**: `run_optuna_study()` and `run_all_studies()` convert the dict to a positional `list[int]` aligned to `feature_cols`, then to `tuple()` for the model constructor.
- **Format**: XGBoost sklearn API requires `tuple(monotone_constraints)`, NOT `list` — calling `.keys()` on a list causes `AttributeError`. Always convert to tuple.
- **CatBoost GPU limitation**: `monotone_constraints` is NOT supported on CatBoost GPU. The trainer auto-skips constraints with a warning when `model_name == "catboost" and gpu == True`. Constraints work fine on CatBoost CPU.
- **Supported models**: CatBoost (CPU only), XGBoost, LightGBM. Other models ignore constraints.

## Per-Fold Selection (RealMLP)

Neural networks benefit from stochasticity — a trial may excel on one fold but not others. Per-fold selection exploits this:

- **During Optuna**: Each fold's trained model immediately predicts on test data. `PerFoldTracker` maintains a bounded per-fold leaderboard (top `n_top_trials` per fold), including predictions from pruned trials.
- **After Optuna — Assembly**: Two modes build composite prediction arrays from per-fold bests. **No retraining needed.**
  - `mode: rank` (default) — Composite k uses the k-th best trial per fold. Deterministic, fast.
  - `mode: nsga2` — Two-phase approach: (1) NSGA-II explores fold combinations using a fast trial-source diversity proxy, (2) greedy selection from Pareto front using the **actual** `diversity_metric` (pearson_neff, spearman_neff, or ambiguity). Each step adds the composite maximising `(1-dw)*score + dw*diversity`.
- **Assembly NSGA-II config** (in model YAML `optuna.assembly`):
  ```yaml
  assembly:
    mode: nsga2               # "rank" | "nsga2"
    n_composites: 20          # how many composites from Pareto front
    n_generations: 50         # NSGA-II generations
    pop_size: 100             # NSGA-II population size
    diversity_metric: spearman_neff  # actual metric used in greedy selection
    diversity_weight: 0.3     # 0=pure score, 1=pure diversity
  ```
- **Assembly speed**: NSGA-II phase: 5000 evals × O(n_folds) each → <1s. Greedy phase: pre-ranks OOFs once (spearman_neff) then pairwise Pearson on ranks → <1s. Total <2s.
- **fold_timeout**: Per-fold time limit (seconds). If a single fold exceeds this, the trial is pruned (but completed folds' predictions are still saved to the tracker).
- **Config**: Set `selection_mode: per_fold` and `fold_timeout: 180` in the model YAML's `optuna` section.
- **NSGA-II integration**: Per-fold composites have the same format as global-mode arrays — `(n_train,)` OOF + `(n_test,)` test. The outer model-level NSGA-II (in `run.py`) doesn't know or care about the difference.
- **Two-layer NSGA-II**: Inner (fold-level, in `trainer.py`) optimizes fold combinations into composites. Outer (model-level, in `run.py/diversity.py`) optimizes model weights across all prediction arrays.
- **Global mode** (`selection_mode: global`, default): Existing behaviour — retrain top N configs with multiple seeds. All other models use this.

## Tracker Diversity Mode (Tiered)

Low-signal data (e.g., Porto Seguro, max corr 0.054) causes neural nets to converge to nearly identical predictions regardless of hyperparameters. A pure top-N-by-score tracker fills with redundant entries, making NSGA-II assembly trivial.

- **Two modes**: `"vanilla"` (default, pure top-N by score) and `"tiered"` (diversity-aware).
- **Tiered mode**: Tier-1 (`tier1_size` slots) is protected — always the best by pure score. Tier-2 (remaining slots) uses diversity-aware insertion: redundant entries (max |corr| ≥ `tier2_corr_threshold` with existing tier-2 entry) only replace their closest match when the new score is better. Diverse entries are inserted normally.
- **Config** (in model YAML `optuna.tracker` or strategy YAML overrides):
  ```yaml
  optuna:
    tracker:
      diversity_mode: tiered    # "vanilla" | "tiered"
      tier1_size: 5             # score-protected slots
      tier2_corr_threshold: 0.99  # correlation threshold for cluster replacement
  ```
- **Backward compatible**: Default is `"vanilla"` — existing behaviour unchanged.

## Diversity Pruning

During Optuna trials, redundant trials (highly correlated predictions with existing tracker entries) can be pruned early to save compute. Especially useful for low-signal data where score-based pruning doesn't help (all trials score ~AUC ± 0.003).

- **Score-gate**: Never diversity-prunes if the trial is a new best score. Only prunes "ok but not exceptional" trials that are also redundant.
- **Warmup**: Diversity pruning activates only after `warmup_entries` entries exist per fold (prevents premature pruning).
- **Consecutive folds**: Requires `n_consecutive` consecutive folds to be flagged as redundant before pruning. Two folds is sufficient — if fold 1 and 2 both correlate >0.995, fold 3-5 are almost certainly the same.
- **Config** (in model YAML `optuna.diversity_pruning` or strategy YAML overrides):
  ```yaml
  optuna:
    diversity_pruning:
      corr_threshold: 0.995     # max |correlation| to flag as redundant
      warmup_entries: 5         # min entries per fold before activation
      n_consecutive: 2          # consecutive redundant folds to trigger prune
      score_tolerance: 0.001    # fraction of best score for score-gate
  ```
- **Disabled by default**: No `diversity_pruning` key → no diversity pruning. Opt-in only.
- **Only works with per-fold mode**: Requires `selection_mode: per_fold` and a `PerFoldTracker`.

## Ensemble & Diversity

- **SOTA models only**: Only include strong models in the ensemble. Weak models (gaussian_nb, svm, knn, adaboost) dilute ensemble quality — their noisy predictions can hurt the meta-learner and final blend even with NSGA-II selection. Prefer: catboost, xgboost, lightgbm, realmlp, random_forest, extra_trees. NSGA-II handles diversity within strong models.
- **NSGA-II → meta-model chain** (in `run.py`): NSGA-II selects diverse models, then trains meta-models on selected OOFs. Compares linear blend vs each meta-model on OOF score — picks the winner automatically.
- **NSGA-II knee-point selection** (default): `run_nsga2_ensemble()` uses `use_knee=True` by default. pymoo's `HighTradeoffPoints` finds the optimal tradeoff point on the Pareto front between metric score and diversity — no need to manually tune `diversity_weight`. The `diversity_weight` parameter serves as fallback when knee-point detection fails (< 3 Pareto solutions) or for explicit linear weighting.
- **`diversity_weight`** can be a single float or a list of floats. A single float uses knee-point selection (default, recommended). A list of floats creates multiple submissions by re-selecting from the same Pareto front with linear weighting — useful for exploring different metric/diversity tradeoffs without re-running NSGA-II.
- **`select_from_pareto()`** in `diversity.py` re-selects from cached Pareto data with different `diversity_weight` or `use_knee`.
- **Meta-model C optimization**: `optimize_meta_C()` in `blender.py` uses Optuna to search LogisticRegression/Ridge C on log scale (0.001–100). Never hardcode C=1.0.
- **XGBoost meta-learner**: `optimize_meta_xgb()` in `blender.py` searches 8 hyperparameters via Optuna for non-linear meta-stacking. Can capture model interaction effects that linear meta-learners miss.
- **Configurable meta-models**: `EnsembleConfig.meta_models` (list, default `["logreg"]`) and `meta_trials` (int or dict per meta-model). Pipeline tries all configured meta-models and picks the best.
- **Meta-model CV**: Configurable via `meta_cv_folds` in ensemble config. Defaults to `2 × pipeline CV folds` if not set. Controls how many folds the meta-model uses for its own OOF predictions.
- **Three diversity metrics** (`diversity_metric` in pipeline YAML):
  - `pearson_neff` — Effective ensemble size from Pearson error correlation eigenvalues (default).
  - `spearman_neff` — Same but Spearman rank correlation. Better for AUC (ranking metric).
  - `ambiguity` — Weighted prediction variance. Directly measures ensemble benefit.

## Extra Data (Original Datasets)

- **`extra_data`** in pipeline YAML: List of original datasets to concatenate with train data.
- Each entry: `path`, optional `drop_columns`, `column_mapping`, `sample_weight`, `target_column`.
- `_concat_extra_data()` in `run.py` handles column matching, target mapping, dtype coercion, missing id columns.
- Adds `_is_original` (bool) and `_sample_weight` (float) metadata columns to train.
- **Metadata columns are stripped before `build_features()`** and re-attached after — prevents dtype leakage into ordinal encoder (test doesn't have these columns).
- Sample weights are passed to models that support them (CatBoost, XGBoost, LightGBM).
- Kaggle Playground Series uses synthetic data generated from real datasets — weighting the original data higher (e.g., `sample_weight: 10.0`) improves signal quality.

## NaN Imputation

- Models with `handles_missing: false` (RealMLP, TabM, Ridge, KNN, etc.) get automatic median imputation in `run_optuna_study()` and `train_with_config()`.
- Fitted on train, applied to both train and test via `sklearn.impute.SimpleImputer`.
- Models with `handles_missing: true` (CatBoost, XGBoost, LightGBM) handle NaN natively — no imputation applied.

## Preprocessing & Scaling

- **Scaler as Optuna parameter**: Models with `needs_scaling: true` (Ridge, Elastic Net, KNN, SVM, Gaussian NB) get an Optuna `suggest_categorical("scaler", choices)` parameter. Optuna decides the best scaler per trial.
- **Scaler types**: `"none"`, `"standard"` (StandardScaler), `"robust"` (RobustScaler), `"quantile"` (QuantileTransformer normal output).
- **Per-fold fitting**: Scaler is fit on each fold's training data and transforms train/val/test independently. Same pattern as NaN imputation.
- **Retrain consistency**: `train_with_config()` reads the `"scaler"` key from trial params and applies the same scaler type. The key is popped from hparams so it's not passed to the model constructor.
- **Only continuous columns scaled**: `_identify_scale_cols()` excludes binary (≤2 unique) and ordinal (≤20 unique integers) columns. Target-encoded features are included.
- **LLM controls search space**: Strategy YAML `preprocessing.scaler_choices` narrows the scaler options (e.g., `["robust"]` if EDA shows outliers). `"none"` is always added so Optuna can opt out.
- **Per-model override**: Strategy YAML `preprocessing.per_model.<name>.needs_scaling` can force scaling on/off, overriding model YAML default.
- **Tree models don't scale**: CatBoost, XGBoost, LightGBM, Random Forest, Extra Trees have `needs_scaling: false` — scaling is never applied.
- **RealMLP/TabM**: `needs_scaling: false` by default (internal preprocessing). LLM can override via strategy if EDA shows extreme outliers.

### Strategy YAML Preprocessing Section
```yaml
preprocessing:
  scaler_choices: ["robust", "quantile"]    # LLM narrows based on EDA
  per_model:
    ridge:
      scaler_choices: ["robust"]            # Force robust for ridge
    realmlp:
      needs_scaling: true                   # Override: enable scaling for RealMLP
      scaler_choices: ["none", "robust"]    # Let Optuna decide
```

## EDA Preprocessing Signals

- **`preprocessing_summary`** in EDA report: Aggregated signals for LLM scaling decisions.
  - `scale_range_ratio`: Max/min feature range ratio — high values indicate features on different scales.
  - `high_skew_features`: Features with |skewness| > 1.0 — candidates for log transform.
  - `high_outlier_features`: Features with outlier% > 3% — candidates for clipping/RobustScaler.
  - `sentinel_features`: Features with detected sentinel values (-1, -999, etc.) — likely masked NaN.
  - `suggested_scalers`: EDA's automatic suggestion based on data characteristics.
- **Per-column additions**: `skewness_label` ("symmetric"/"moderate"/"high"), `sentinels` (list of detected sentinel values with counts), `range` in stats.

## EDA Advanced Analyses

- **Duplicate detection** (`duplicates`): Hash-based detection of exact duplicate rows and conflicting duplicates (same features, different target). Signal-only mode excludes noise columns (IV < 0.001 AND |AUC-0.5| < 0.01) before hashing — reveals duplicates hidden by random noise features (e.g., ps_calc_* in Porto Seguro). Conflicting duplicates set a hard ceiling on achievable performance.
- **Unseen categories** (`unseen_categories`): Per-categorical count of test values not in train, with affected row percentage. Guides target encoding smoothing alpha and encoding strategy.
- **Monotonicity detection** (`monotonicity`): Bins numeric features and computes Spearman rho on binned target rates. Features with |rho| > 0.7 flagged as monotonic — candidates for `monotone_constraints` in gradient boosting.
- **Cardinality profiles** (`cardinality_profiles`): Top-K share, Shannon entropy, normalized entropy per categorical. Classifies distribution shape as "uniform", "moderate", or "long_tail". Long-tail categoricals need frequency encoding or rare-category binning.
- **Target encoding preview** (`te_preview`): OOF target encoding simulation (5-fold, alpha=10). Reports Pearson correlation and AUC of encoded column with target — concrete numbers for whether TE is worthwhile.
- **Quick model baseline** (`quick_model`): 3-fold CV RandomForest (n_estimators=100, max_depth=8) with label-encoded categoricals. Returns baseline AUC/RMSE (performance floor without feature engineering) and feature importances (sees non-linear effects and interactions unlike univariate metrics). Subsamples to 50k rows for speed. Fully deterministic (seeded).
- **Prediction diversity probe** (`prediction_diversity`): Trains 3 RF models with different seeds and measures signal-noise ratio (SNR = within_seed_std / across_seed_std). Within-seed std = how much predictions vary across samples (signal). Across-seed std = how much seeds change predictions (noise). Classification by SNR: `very_low` (<3), `low` (3-8), `moderate` (8-15), `high` (>15). **Note**: Pearson correlation alone is misleading for RF — noise features inflate seed-to-seed variation. SNR is robust because noise inflates both stds proportionally. Also reports Pearson correlations and Fisher z CI for reference. Subsamples to 50k. Guides LLM in setting tiered tracker/diversity pruning parameters.
- **Fold context**: Dataset header shows per-fold train/val sizes for 5-fold and 10-fold CV to help calibrate min_leaf parameters.

## Pipeline YAML Key Features

- **`extra_data`**: List of extra datasets to concat with train (see "Extra Data" section).
- **`target_mapping`**: Converts string labels to numeric (`{Yes: 1, No: 0}`).
- **`log_transform_target`**: Applies `log1p()` to regression targets (for RMSLE optimization).
- **Timeout strings**: `model_timeouts` accepts human-readable strings like `"1h30m"`, `"45m"`, `"90s"` — parsed by `parse_timeout()` in `io.py`.
- **`diversity_metric`**: Selects NSGA-II diversity objective per competition (`pearson_neff`, `spearman_neff`, `ambiguity`).
- **`selection_mode`**: Per-model in model YAML (`global` or `per_fold`). Controls whether top configs are retrained (global) or per-fold composites are assembled (per_fold).
- **`fold_timeout`**: Per-model in model YAML. Max seconds per CV fold — exceeding prunes the trial.
- **`assembly`**: Per-model in model YAML. Dict with `mode` (`rank`/`nsga2`), `n_composites`, `n_generations`, `pop_size`, `diversity_metric`, `diversity_weight`. Controls how per-fold composites are built.
- **`meta_cv_folds`**: Number of CV folds for meta-model training. Defaults to `2 × pipeline n_folds` if not set. Set explicitly for large datasets (e.g., 15 folds for 595k samples).
- **Model execution order**: Strategy YAML `models` dict overrides pipeline YAML `models` list. Python dict preserves insertion order — the first key in strategy YAML runs first. To control execution order, reorder the strategy YAML `models` dict.

## CV and OOF Predictions

- Same `StratifiedKFold(n_folds, seed)` (or `KFold` for regression) used everywhere.
- Target encoding uses the **same CV folds** as model training — prevents leakage.
- OOF predictions indexed correctly: `oof[val_idx] = preds`.
- Test predictions averaged across folds: `test_preds += preds / n_folds`.
- Meta-model uses its own CV on OOF predictions (no leakage from base models).

## Configuration Files

- `configs/schemas/` — YAML schema documentation (pipeline, model, EDA report)
- `configs/models/` — One YAML per model type (12 models: catboost, xgboost, lightgbm, realmlp, ridge, elastic_net, knn, random_forest, extra_trees, adaboost, gaussian_nb, svm)
- `configs/templates/` — Ready-to-use pipeline configurations (binary_classification, regression)
- `competitions/` — Per-competition pipeline configs + strategy YAML (house_prices, ps-s6e2, ps-s6e3)

## Dependencies

Core: catboost, xgboost, lightgbm, scikit-learn, optuna, pandas, numpy
Neural nets: pytabkit (RealMLP), torch
Multi-objective: pymoo (NSGA-II)
LLM: anthropic, openai, python-dotenv
Config: pyyaml
Viz: matplotlib, seaborn
Scientific: scipy
Testing: pytest

## Shell & Python Environment

- **Shell: Git Bash on Windows** — NOT PowerShell, NOT cmd.exe
- **Always use Unix-style paths**: `/c/Users/...` never `C:\Users\...`
- **Conda environment**: `maestro`
- **Project root**: `/c/Projektit/maestro-ml`

Run Python scripts:
```bash
conda run -n maestro python script.py
conda run -n maestro python check_imports.py
```

Run tests:
```bash
conda run -n maestro pytest tests/ -v
```

## Testing

- Run tests: `conda run -n maestro pytest tests/ -v`
- Expected: **723 passed, 22 skipped, ~30s**
- `test_gpu.py` tests GPU availability — do not modify
- Each module has a corresponding test file (see `tests/CLAUDE.md` for patterns)
- `tests/conftest.py` handles Windows-specific torch/OpenMP DLL workarounds
- All test data is tiny synthetic data (n_samples ≤ 50) — no real CSVs in tests
- **Known**: torch DLL loading may print a traceback on Windows — this is non-fatal, tests still pass

## Common Patterns

### Loading a model config and creating an instance
```python
from src.models.registry import ModelRegistry
registry = ModelRegistry("configs/models")
model = registry.get_model("catboost", hparams={"depth": 6}, gpu=True)
```

### Running the pipeline
```bash
conda run -n maestro python run.py --config configs/templates/binary_classification.yaml
conda run -n maestro python run.py --config pipeline.yaml --strategy manual
```

## File Structure Quick Reference
```
src/utils/io.py          — load_yaml, load_pipeline_config, load_model_config, save_*, parse_timeout
src/eda/profiler.py      — run_eda, format_eda_for_llm
src/features/engineer.py — build_features, target encoding, interactions, ratios
src/models/registry.py   — ModelRegistry class (register, get_model, get_search_space)
src/models/trainer.py    — PerFoldTracker, run_optuna_study, train_with_config, run_all_studies, reassemble_int_lists
src/ensemble/blender.py  — optimize_blend_weights, rank_average, train_meta_model, optimize_meta_C, optimize_meta_xgb
src/ensemble/diversity.py — run_nsga2_ensemble, select_from_pareto, effective_ensemble_size, _compute_diversity
src/strategy/llm_strategist.py — generate_strategy (API + manual modes)
run.py                   — main pipeline orchestrator (includes NSGA-II→meta chain logic)
```
