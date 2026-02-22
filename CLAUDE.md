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
- **Functions over classes** except for `ModelRegistry` and dataclasses in `src/utils/io.py`.
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

## CV and OOF Predictions

- Same `StratifiedKFold(n_folds, seed)` (or `KFold` for regression) used everywhere.
- Target encoding uses the **same CV folds** as model training — prevents leakage.
- OOF predictions indexed correctly: `oof[val_idx] = preds`.
- Test predictions averaged across folds: `test_preds += preds / n_folds`.
- Meta-model uses its own CV on OOF predictions (no leakage from base models).

## Configuration Files

- `configs/schemas/` — YAML schema documentation (pipeline, model, EDA report)
- `configs/models/` — One YAML per model type (catboost, xgboost, lightgbm, ridge, knn, random_forest, extra_trees)
- `configs/templates/` — Ready-to-use pipeline configurations (binary_classification, regression)

## Dependencies

Core: catboost, xgboost, lightgbm, scikit-learn, optuna, pandas, numpy
LLM: anthropic, openai, python-dotenv
Config: pyyaml
Viz: matplotlib, seaborn
Scientific: scipy
Testing: pytest

## Testing

- Run tests: `pytest`
- `test_gpu.py` tests GPU availability — do not modify
- Each module should get its own test file in a `tests/` directory

## Common Patterns

### Loading a model config and creating an instance
```python
from src.models.registry import ModelRegistry
registry = ModelRegistry("configs/models")
model = registry.get_model("catboost", hparams={"depth": 6}, gpu=True)
```

### Running the pipeline
```bash
python run.py --config configs/templates/binary_classification.yaml
python run.py --config pipeline.yaml --strategy manual
```

## File Structure Quick Reference
```
src/utils/io.py          — load_yaml, load_pipeline_config, load_model_config, save_*
src/eda/profiler.py      — run_eda, format_eda_for_llm
src/features/engineer.py — build_features, target encoding, interactions, ratios
src/models/registry.py   — ModelRegistry class (register, get_model, get_search_space)
src/models/trainer.py    — run_optuna_study, train_with_config, run_all_studies
src/ensemble/blender.py  — optimize_blend_weights, rank_average, train_meta_model
src/ensemble/diversity.py — effective_ensemble_size, run_nsga2_ensemble
src/strategy/llm_strategist.py — generate_strategy (API + manual modes)
run.py                   — main pipeline orchestrator
```
