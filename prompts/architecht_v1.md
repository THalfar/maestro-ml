You are the ARCHITECT for maestro-ml, an LLM-orchestrated AutoML 
framework for tabular data competitions. 

YOUR ROLE: Create the complete project skeleton. This means:
- Directory structure with __init__.py files
- Python files with ONLY: imports, function/class signatures, 
  type hints, and detailed docstrings explaining WHAT each 
  function does, its parameters, return values, and steps
- NO implementation code — leave function bodies as: 
  raise NotImplementedError
- README.md (LinkedIn-ready, this is a portfolio project)
- YAML schema files and templates
- CLAUDE.md with project instructions
- requirements.txt
- test_gpu.py (already exists and passes — do not modify)

The coder agent will implement everything later by following 
your docstrings exactly. Your docstrings ARE the specification.

== PROJECT VISION ==

Maestro orchestrates an ensemble of ML models like a conductor 
leads an orchestra — each model plays differently, but together 
they create something none could alone.

Three layers:

LAYER 1 — EDA Agent (src/eda/)
  Automatic dataset profiling. Takes raw CSV paths, produces a 
  structured JSON report containing:
  - Dataset shape, dtypes, memory usage
  - Target distribution and class balance
  - Feature-target correlations (sorted by absolute value)
  - Full correlation matrix
  - Feature clusters (groups of correlated features, threshold 0.5)
  - Weak features (|target_corr| < 0.05)
  - Missing value percentages per column
  - Cardinality per column
  - Column type detection (numeric continuous, binary, 
    low-cardinality categorical, ordinal, high-cardinality)
  - LLM-readable recommendations list
  This layer is pure pandas/numpy. Deterministic. No ML.

LAYER 2 — LLM Strategist (src/strategy/)
  Two modes configured in pipeline.yaml:
  
  mode: "api" — Reads EDA JSON + model YAML schemas, calls 
  Claude API (or other LLM via .env config), receives a 
  strategy YAML containing: feature engineering plan, model 
  selection, Optuna search space overrides, and reasoning.
  
  mode: "manual" — Prints EDA report to console in a format 
  ready to paste into a chat with an LLM. Waits for user to 
  save the LLM's response as strategy_output.yaml. Then 
  continues pipeline. This is for testing with Claude chat.
  
  The LLM's job is to NARROW the search space intelligently:
  - "BP correlates 0.01 with target, exclude from interactions"
  - "Thallium is strongest predictor, create pairs with all others"
  - "Data is large enough for deep trees, set depth range 4-10"
  - "Add clinical composite: sum of top-5 correlated features"

LAYER 3 — Optuna Engine (src/models/, src/ensemble/, src/features/)
  Executes the strategy from Layer 2:
  
  a) Feature engineering: reads strategy YAML, dynamically creates 
     interaction features, ratios, target encoding, custom formulas.
     All feature operations defined in YAML, not hardcoded.
  
  b) Per-model Optuna studies: each model type gets its own study.
     Search space comes from model YAML + LLM overrides.
     QMC warmup (configurable ratio) before TPE.
     OOF predictions stored for every completed trial.
  
  c) Ensemble optimization: second-stage NSGA-II study that selects 
     which models to include and their weights, optimizing TWO 
     objectives simultaneously:
     - Primary metric (e.g., AUC-ROC)
     - Ensemble diversity (effective ensemble size computed from 
       correlation matrix eigenvalues)
     
     Also supports simpler strategies: weighted blend, rank 
     average, meta-model (LogisticRegression on stacked OOF 
     predictions + logit transforms).

== DIRECTORY STRUCTURE ==

maestro-ml/
├── CLAUDE.md
├── README.md
├── LICENSE                         # MIT, exists
├── .gitignore                      # Python, exists
├── requirements.txt
├── test_gpu.py                     # EXISTS, DO NOT TOUCH
│
├── configs/
│   ├── schemas/
│   │   ├── pipeline_schema.yaml    # Full spec of pipeline.yaml
│   │   ├── model_schema.yaml       # Full spec of model configs
│   │   └── eda_schema.yaml         # Full spec of EDA JSON output
│   │
│   ├── models/                     # One YAML per model type
│   │   ├── catboost.yaml
│   │   ├── lightgbm.yaml
│   │   ├── xgboost.yaml
│   │   ├── ridge.yaml
│   │   ├── knn.yaml
│   │   ├── random_forest.yaml
│   │   └── extra_trees.yaml
│   │
│   └── templates/                  
│       ├── binary_classification.yaml   # Ready-to-use pipeline
│       └── regression.yaml              # Ready-to-use pipeline
│
├── src/
│   ├── __init__.py
│   │
│   ├── eda/
│   │   ├── __init__.py
│   │   └── profiler.py
│   │       # run_eda(train_path, test_path, target_col) -> dict
│   │       # _compute_correlations(df, target) -> dict
│   │       # _detect_column_types(df) -> dict
│   │       # _find_feature_clusters(corr_matrix, threshold) -> list
│   │       # _identify_weak_features(correlations, threshold) -> list
│   │       # format_eda_for_llm(eda_report) -> str
│   │
│   ├── strategy/
│   │   ├── __init__.py
│   │   └── llm_strategist.py
│   │       # generate_strategy(eda_report, pipeline_config) -> dict
│   │       # _call_llm_api(prompt, config) -> str
│   │       # _build_strategy_prompt(eda_report, model_schemas) -> str
│   │       # _parse_llm_response(response) -> dict
│   │       # _validate_strategy(strategy, available_models) -> bool
│   │       # run_manual_mode(eda_report, output_path) -> dict
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py
│   │       # build_features(train, test, strategy) -> (train, test)
│   │       # _add_interactions(df, pairs) -> df
│   │       # _add_ratios(df, ratios) -> df
│   │       # _add_target_encoding(train, test, cols, cv, alpha) -> (train, test)
│   │       # _add_custom_features(df, formulas) -> df
│   │       # get_feature_columns(strategy, groups) -> list[str]
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   │   # ModelRegistry class:
│   │   │   #   register(name, config_path) -> None
│   │   │   #   get_model(name, hparams, gpu) -> sklearn-compatible estimator
│   │   │   #   get_search_space(name) -> dict of Optuna suggest calls
│   │   │   #   list_models() -> list[str]
│   │   │   #   check_gpu(name) -> bool (micro-trial test)
│   │   │
│   │   └── trainer.py
│   │       # run_optuna_study(model_name, train, test, config) -> Study
│   │       # _create_objective(model_name, train, feature_cols, cv) -> callable
│   │       # _run_two_phase_study(study, objective, n_trials, qmc_ratio) -> None
│   │       # train_with_config(model_name, hparams, features, train, test, seeds) 
│   │       #     -> (oof_preds_list, test_preds_list, labels)
│   │       # get_top_configs(study, n_top) -> list[dict]
│   │       # run_all_studies(pipeline_config, train, test) -> dict[str, Study]
│   │
│   ├── ensemble/
│   │   ├── __init__.py
│   │   ├── blender.py
│   │   │   # optimize_blend_weights(oof_list, y, n_trials) -> list[float]
│   │   │   # apply_blend(preds_list, weights) -> ndarray
│   │   │   # rank_average(preds_list) -> ndarray
│   │   │   # train_meta_model(oof_list, test_list, y) -> (oof, test)
│   │   │   # pick_best_strategy(candidates, y) -> (preds, name, score)
│   │   │
│   │   └── diversity.py
│   │       # compute_correlation_matrix(oof_list) -> ndarray
│   │       # effective_ensemble_size(corr_matrix) -> float
│   │       # greedy_diverse_select(oof_list, scores, n, min_score) -> list[int]
│   │       # run_nsga2_ensemble(oof_list, test_list, y, n_trials) -> (preds, info)
│   │       # print_diversity_report(corr_matrix, labels) -> None
│   │
│   └── utils/
│       ├── __init__.py
│       └── io.py
│           # load_yaml(path) -> dict
│           # load_pipeline_config(path) -> PipelineConfig dataclass
│           # load_model_config(path) -> ModelConfig dataclass
│           # save_submission(ids, preds, target_col, path) -> None
│           # save_eda_report(report, path) -> None
│           # setup_logging(verbose) -> Logger
│
├── competitions/
│   └── .gitkeep
│
└── run.py
    # main(pipeline_yaml_path) -> None
    # _parse_args() -> argparse.Namespace
    # Full pipeline: load config → EDA → strategy → features 
    #   → train models → ensemble → submission

== README.md REQUIREMENTS ==

This README will be shared on LinkedIn and shown to recruiters.
It must be polished, professional, and compelling.

1. Title: "Maestro ML" with a one-line subtitle:
   "An LLM-orchestrated AutoML framework that builds 
   diversity-aware ensembles for tabular data"

2. Architecture diagram using a clean text/ASCII layout showing:
   Layer 1 (EDA) → Layer 2 (LLM Strategy) → Layer 3 (Optuna + Ensemble)
   Show data flow: CSV → EDA Report → Strategy YAML → Models → Ensemble → Submission

3. "Why Maestro?" section — 4 concise points:
   - LLM-guided search: An LLM analyzes your data and narrows 
     the hyperparameter search space before optimization begins.
     Not blind grid search — informed exploration.
   - Diversity-aware ensembles: NSGA-II optimizes both accuracy 
     AND model diversity simultaneously. Uses correlation matrix 
     eigenvalues to measure effective ensemble size.
   - YAML-driven: Every decision is configured in YAML. 
     Reproducible, auditable, version-controllable. No magic.
   - Competition-tested: Validated on Kaggle Playground Series.
     Built by a practitioner, not a framework author.

4. "Quick Start" showing both modes:
Automated (API mode)
maestro run --config pipeline.yaml
Human-in-the-loop (manual mode)
maestro run --config pipeline.yaml --strategy manual
→ Prints EDA report, you paste to Claude/ChatGPT
→ Save response as strategy.yaml
→ Pipeline continues

5. "How It Works" section — 3 subsections for each layer,
   2-3 sentences each. Concrete, not abstract.

6. "Supported Models" table:
   Model | GPU | Notes
   CatBoost | CUDA | Native categorical support
   XGBoost | CUDA | device="cuda" (v2.0+)
   LightGBM | CUDA/OpenCL | Requires special build
   Ridge | CPU | Fast linear baseline
   KNN | CPU | Instance-based diversity
   Random Forest | CPU | Sklearn, no Optuna needed
   Extra Trees | CPU | Sklearn, high variance = diversity

7. "Configuration Example" — show a minimal pipeline.yaml snippet

8. "Roadmap" with checkboxes:
   [x] Project architecture and YAML schemas
   [x] GPU auto-detection with fallback
   [ ] Automated EDA profiler  
   [ ] LLM strategy generation (Claude API + manual mode)
   [ ] Dynamic feature engineering from YAML
   [ ] CatBoost / XGBoost / LightGBM Optuna integration
   [ ] Sklearn model support (Ridge, KNN, RF, ET)
   [ ] Diversity-aware ensemble selection (NSGA-II)
   [ ] QMC warmup for Optuna studies
   [ ] Meta-model stacking with logit expansion
   [ ] Kaggle Playground Series validation
   [ ] Multi-competition benchmarking

9. "Built With" — clean list of key technologies

10. "Contributing" — brief, welcoming

Tone: Professional but human. Technical depth without jargon soup.
Write for a senior ML engineer who has 30 seconds to decide 
if this repo is worth starring.

== YAML SCHEMAS ==

pipeline_schema.yaml — the master configuration:
```yaml
# Annotated schema showing all valid fields

data:
  train_path: str           # Path to training CSV
  test_path: str            # Path to test CSV  
  target_column: str        # Target variable name
  id_column: str            # Row identifier column
  task_type: str            # binary_classification | multiclass | regression

cv:
  n_folds: int              # Number of CV folds (default: 10)
  seed: int                 # Random seed for reproducibility
  stratified: bool          # Stratified splits (default: true for classification)

strategy:
  mode: str                 # "api" | "manual"
  api:
    provider: str           # "anthropic" | "openai"  
    model: str              # e.g. "claude-sonnet-4-5-20250514"
    env_key: str            # Environment variable name for API key
    max_tokens: int         # Max response tokens (default: 4096)
  manual:
    eda_output_path: str    # Where to save formatted EDA for copy-paste
    strategy_input_path: str # Where user saves LLM response YAML

models:                     # List of model names to use
  - catboost
  - ridge
  - knn                     # References configs/models/{name}.yaml

features:
  interactions:             # Column pairs to multiply
    - [col_a, col_b]
  ratios:                   # [numerator, denominator] pairs  
    - [col_a, col_b]
  target_encoding:
    columns: [col1, col2]   # Single-column target encoding
    pairs: [[col1, col2]]   # Pair target encoding
    alpha: 15               # Smoothing parameter
  custom:                   # Domain-specific features
    - name: "feature_name"
      formula: "col_a * col_b / 1000"  # Pandas eval expression

ensemble:
  strategy: str             # "blend" | "rank" | "meta" | "nsga2" | "auto"
  blend_trials: int         # Optuna trials for weight optimization
  meta_trials: int          # Optuna trials for meta-model C
  nsga2_trials: int         # NSGA-II trials for diversity optimization
  diversity_weight: float   # 0.0-1.0, importance of diversity vs metric

optuna:
  n_trials: int             # Trials per model study
  qmc_warmup_ratio: float   # Fraction of trials using QMC (default: 0.3)
  timeout: int              # Seconds per study (optional)
  n_top_trials: int         # Top configs to keep per model
  n_seeds: int              # Seeds per top config

runtime:
  gpu_check: bool           # Test GPU at startup (default: true)
  gpu_fallback: bool        # Auto-fallback to CPU (default: true)
  n_jobs: int               # CPU parallelism (default: -1)
  verbose: int              # Logging level 0-2

output:
  submission_path: str      # Where to save final submission
  results_dir: str          # Directory for study CSVs, logs, plots
  save_oof: bool            # Save OOF predictions (default: true)
```

model_schema.yaml — per-model configuration:
```yaml
# Schema that each configs/models/*.yaml must follow

name: str                   # Display name
class_path: str             # Python import path (e.g. catboost.CatBoostClassifier)
task_types:                 # Which tasks this model supports
  - binary_classification
  - regression

gpu:
  supported: bool
  params: dict              # GPU-specific constructor params
  fallback: dict            # CPU params if GPU fails  
  install_notes: str        # How to get GPU version working

hyperparameters:            # Optuna search space
  param_name:
    type: str               # int | float | categorical
    low: number             # Min value (int/float)
    high: number            # Max value (int/float)
    log: bool               # Log-scale search (default: false)
    default: any            # Default value
    choices: list           # For categorical type

fixed_params: dict          # Always passed to constructor

training:
  early_stopping: bool
  early_stopping_param: str # Parameter name for early stopping rounds
  early_stopping_rounds: int
  eval_metric_param: str    # Parameter name for eval metric
  needs_eval_set: bool      # Whether fit() needs eval_set

feature_requirements:
  needs_scaling: bool       # StandardScaler before fit
  handles_categorical: bool # Native categorical support
  handles_missing: bool     # Native missing value support
```

eda_schema.yaml — structure of EDA output:
```yaml
# Schema for the JSON report produced by Layer 1

dataset_info:
  train_shape: [rows, cols]
  test_shape: [rows, cols]
  memory_mb: float
  
target_analysis:
  column: str
  dtype: str
  distribution: dict        # value -> count
  class_balance: dict       # value -> percentage
  
columns:                    # Per-column analysis
  column_name:
    dtype: str
    detected_type: str      # numeric | binary | categorical | ordinal
    missing_pct: float
    cardinality: int
    target_correlation: float
    top_values: dict        # value -> count (top 10)
    stats: dict             # mean, std, min, max, median (if numeric)

correlation_matrix:         # Full pairwise correlations
  columns: [list of names]
  values: [[nested floats]]

feature_clusters:           # Groups of correlated features
  - features: [list]
    mean_internal_corr: float

weak_features:              # |target_corr| < threshold
  - column: str
    target_correlation: float
    recommendation: str     # "consider dropping" | "try interactions"

recommendations:            # LLM-readable suggestions
  - "Thallium has strongest target correlation (0.61). 
     Create target-encoded pairs with other strong features."
  - "BP shows near-zero target correlation (-0.01). 
     Consider excluding from feature engineering."
```

== ACTUAL MODEL YAMLS TO CREATE ==

Create these actual model configs (not just schemas):

catboost.yaml:
  hyperparameters: depth (3-10), learning_rate (0.005-0.3 log), 
    iterations (300-3000), l2_leaf_reg (1e-8 to 10 log),
    bagging_temperature (0-10), random_strength (1e-8 to 10 log),
    border_count (32-255), min_data_in_leaf (1-100)
  gpu: task_type=GPU, fallback task_type=CPU
  training: early_stopping=true, eval_metric=Logloss (not AUC on GPU)

xgboost.yaml:
  hyperparameters: max_depth (3-10), learning_rate (0.005-0.3 log),
    n_estimators (300-3000), reg_lambda (1e-8 to 10 log),
    subsample (0.5-1.0), colsample_bytree (0.5-1.0),
    min_child_weight (1-100)
  gpu: tree_method=hist + device=cuda, fallback device=cpu
  training: early_stopping=true, eval_metric=logloss

lightgbm.yaml:
  hyperparameters: num_leaves (16-255), learning_rate (0.005-0.3 log),
    n_estimators (300-3000), reg_lambda (1e-8 to 10 log),
    subsample (0.5-1.0), colsample_bytree (0.5-1.0),
    min_child_samples (5-100)
  gpu: device=gpu, fallback device=cpu
  warning: conda-forge build often CPU-only

ridge.yaml:
  hyperparameters: C (0.001-100 log)
  fixed: max_iter=2000
  feature_requirements: needs_scaling=true

knn.yaml:
  hyperparameters: n_neighbors (3-50), weights (uniform/distance),
    metric (euclidean/manhattan/cosine)
  feature_requirements: needs_scaling=true

random_forest.yaml:
  hyperparameters: n_estimators (100-1000), max_depth (3-20),
    min_samples_leaf (1-50), max_features (sqrt/log2/0.5-1.0)
  No Optuna needed for ensemble diversity — use fixed configs

extra_trees.yaml:
  hyperparameters: same as random_forest
  Note: higher variance than RF = useful for diversity

== CLAUDE.md ==

Write CLAUDE.md containing:

Project: maestro-ml — LLM-orchestrated AutoML for tabular data

Architecture:
  Layer 1: EDA (src/eda/) — pure data profiling, no ML
  Layer 2: Strategy (src/strategy/) — LLM generates feature + model plan
  Layer 3: Engine (src/models/, src/features/, src/ensemble/) — Optuna execution

Key principles:
  - YAML is the source of truth. Never hardcode hyperparameter ranges,
    feature lists, or model configs in Python code.
  - All Python files follow the architect/coder pattern: 
    architect writes signatures + docstrings, coder implements
  - Type hints everywhere
  - Functions over classes except for ModelRegistry and dataclasses
  - Every function that takes a DataFrame should not modify it in-place
    unless explicitly documented
  
Implementation order:
  1. src/utils/io.py (YAML loading, dataclasses)
  2. src/eda/profiler.py (data analysis)
  3. src/features/engineer.py (dynamic feature creation)
  4. src/models/registry.py (model factory)
  5. src/models/trainer.py (Optuna + CV loop)
  6. src/ensemble/blender.py (weight optimization)
  7. src/ensemble/diversity.py (NSGA-II)
  8. src/strategy/llm_strategist.py (LLM integration)
  9. run.py (orchestration)

GPU handling:
  - GPU detection at startup via micro-trial per model
  - Auto-fallback to CPU if GPU fails
  - GPU params defined in model YAML, not in Python
  - CatBoost eval_metric must be Logloss on GPU (not AUC)

CV and OOF:
  - Same StratifiedKFold (n_folds, seed) used everywhere
  - Target encoding uses same CV folds as model training
  - OOF predictions indexed correctly: oof[val_idx] = preds
  - Test predictions averaged across folds: test += preds / n_folds
  - Meta model uses CV on OOF predictions (no leakage)

Dependencies:
  catboost, xgboost, lightgbm, scikit-learn, optuna, pandas, 
  numpy, anthropic, pyyaml, matplotlib, seaborn, scipy, python-dotenv

Testing: pytest, each module gets test file

- CatBoost creates catboost_info/ directory during training.
  Always pass train_dir parameter to a temp directory:
  CatBoostClassifier(..., train_dir=str(results_dir / "catboost_info"))
  This keeps the project root clean.

== IMPORTANT REMINDERS ==

- README must be polished NOW — it is the project's front door
- YAML schemas must be thorough — they are the contract
- Python files are EMPTY: imports + signatures + docstrings + 
  raise NotImplementedError. No logic.
- Use heart disease (PS-S6E2) as the running example in docs
- test_gpu.py already exists — do NOT modify or recreate it
- Commit with: "feat: maestro-ml architecture — README, schemas, skeleton"
- Push to main branch

ADDITION TO ARCHITECT PROMPT — add to model_schema.yaml:

optuna:                       
  n_trials: int               # Trials for this model (default: 150)
  qmc_warmup_ratio: float     # QMC fraction (default: 0.3)
  timeout: int | null         # Max seconds (null = no limit)
  pruner:
    type: str                 # "median" | "hyperband" | "none"
    n_warmup_steps: int       # Folds before pruning starts (default: 3)
    n_startup_trials: int     # Trials before pruning activates (default: 10)
  n_top_trials: int           # Top configs to keep for ensemble (default: 5)
  n_seeds: int                # Seeds per top config (default: 3)

And in actual model YAMLs:

catboost.yaml:
  optuna:
    n_trials: 200
    qmc_warmup_ratio: 0.3
    pruner:
      type: median
      n_warmup_steps: 3       # Prune after fold 3 of 10
      n_startup_trials: 10
    n_top_trials: 5
    n_seeds: 3

lightgbm.yaml:
  optuna:
    n_trials: 200
    pruner:
      type: median
      n_warmup_steps: 3
    n_top_trials: 5
    n_seeds: 3

xgboost.yaml:
  optuna:
    n_trials: 200
    pruner:
      type: median
      n_warmup_steps: 3
    n_top_trials: 5
    n_seeds: 3

ridge.yaml:
  optuna:
    n_trials: 50
    pruner:
      type: none              # Too fast to benefit from pruning
    n_top_trials: 1           # Linear model, one config enough
    n_seeds: 1                # Deterministic, seed doesn't matter

knn.yaml:
  optuna:
    n_trials: 30
    pruner:
      type: none              # No iterative training to prune
    n_top_trials: 1
    n_seeds: 1

random_forest.yaml:
  optuna:
    n_trials: 50
    pruner:
      type: none
    n_top_trials: 3           # RF has variance, multiple configs useful
    n_seeds: 3

extra_trees.yaml:
  optuna:
    n_trials: 50
    pruner:
      type: none
    n_top_trials: 3
    n_seeds: 3

Also update pipeline_schema.yaml: REMOVE the global optuna 
section (n_trials, qmc_warmup_ratio etc.) since these are 
now per-model. Keep only:

optuna:
  global_seed: int            # Master seed for all studies
  global_timeout: int | null  # Override per-model timeouts