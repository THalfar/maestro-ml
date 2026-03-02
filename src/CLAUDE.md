# src/ — Implementation Modules

## Module Dependency Order

Changes to earlier modules cascade downward:

```
utils/io.py
    └── eda/profiler.py
            └── features/engineer.py
                    └── models/registry.py
                            └── models/trainer.py
                                    └── ensemble/blender.py
                                            └── ensemble/diversity.py
                                                    └── strategy/llm_strategist.py
```

## Module Quick Reference

| File | Key Public Functions | Dataclasses / Classes |
|------|---------------------|-----------------------|
| `utils/io.py` | `load_yaml`, `load_pipeline_config`, `load_model_config`, `save_*` | `PipelineConfig`, `ModelConfig`, `EDAReport` |
| `eda/profiler.py` | `run_eda`, `format_eda_for_llm` | — |
| `features/engineer.py` | `build_features` | — |
| `models/registry.py` | `ModelRegistry.get_model`, `.get_search_space` | `ModelRegistry` |
| `models/trainer.py` | `run_optuna_study`, `train_with_config`, `run_all_studies` | — |
| `ensemble/blender.py` | `optimize_blend_weights`, `rank_average`, `train_meta_model` | — |
| `ensemble/diversity.py` | `effective_ensemble_size`, `run_nsga2_ensemble` | — |
| `strategy/llm_strategist.py` | `generate_strategy` | — |

## Required File Header

Every file must start with:
```python
from __future__ import annotations
```
Full type hints on all function signatures.

## Critical Patterns

### No DataFrame Mutation
Every function receiving a DataFrame must return a new one. Never `df['col'] = ...`.
```python
# CORRECT
result = df.copy()
result['new_col'] = values
return result

# WRONG
df['new_col'] = values  # mutates caller's DataFrame
```

### OOF Indexing
```python
oof = np.zeros(len(X))
for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    model.fit(X[train_idx], y[train_idx])
    oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]  # index by val_idx
```

### Seed Parameter Names (vary by model)
- CatBoost: `random_seed`
- All others: `random_state`

## Per-Module Gotchas

### `models/registry.py`
- GPU detection uses a micro-trial: fit a 100-row synthetic dataset with GPU params. If it raises, fall back to CPU params. No crashes.
- Registry auto-loads all YAMLs from `configs/models/` — no registration code needed.

### `models/trainer.py`
- Optuna flow: **Phase 1 QMC warmup** (space-filling, `qmc_warmup_trials` trials) → **Phase 2 TPE** (Bayesian).
- After study: top `n_top_trials` configs retrained with `n_seeds` seeds for stability.
- Test predictions: `test_preds += preds / n_folds` (average across folds).

### `features/engineer.py`
- Target encoding **must use the same CV folds as training** to prevent leakage. OOF encoding: compute mean on train fold, apply to val fold.
- Ratio features: always divide with epsilon (`/ (denominator + 1e-8)`).

### `ensemble/diversity.py`
- `effective_ensemble_size` uses correlation matrix eigenvalues: `n_eff = (sum(λ))² / sum(λ²)`.
- NSGA-II objectives: maximize OOF metric AND maximize effective ensemble size (minimize -n_eff).

### `strategy/llm_strategist.py`
- **API mode**: calls Anthropic/OpenAI API, parses YAML from response.
- **Manual mode**: prints prompt to console, waits for user to paste strategy YAML, reads from file path in config.
- Strategy YAML must have keys: `features`, `models`, `overrides`, `reasoning`.

## CatBoost-Specific Rules

```python
# eval_metric in CONSTRUCTOR (not fit())
CatBoostClassifier(
    eval_metric="Logloss",   # GPU: always Logloss (AUC unsupported)
    # eval_metric="AUC",     # CPU only
    train_dir=str(results_dir / "catboost_info"),  # keep root clean
)
model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

## LightGBM-Specific Rules

```python
# Early stopping via callbacks (NOT early_stopping_rounds param)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)
```

## XGBoost-Specific Rules

```python
# GPU: device="cuda" (NOT deprecated tree_method="gpu_hist")
XGBClassifier(device="cuda", ...)
# eval_metric in fit() (not constructor)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```
