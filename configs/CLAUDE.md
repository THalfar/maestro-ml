# configs/ — YAML Configuration

## Directory Structure

```
configs/
├── models/       # One file per model — hyperparameters, fixed params, GPU, training
├── schemas/      # Schema docs: pipeline_schema.yaml, model_schema.yaml, eda_schema.yaml
└── templates/    # Ready-to-use: binary_classification.yaml, regression.yaml
```

## Model YAML Structure

Full annotated example:

```yaml
name: "Random Forest"
task_types: [binary_classification, regression]

# Python import path, keyed by task type (required when classifier ≠ regressor)
class_path:
  binary_classification: "sklearn.ensemble.RandomForestClassifier"
  regression: "sklearn.ensemble.RandomForestRegressor"
# Or a single string if same class handles both:
# class_path: "some.module.SomeEstimator"

# Optuna search space — each param defines type + range/choices
hyperparameters:
  n_estimators:
    type: int
    low: 100
    high: 1000
  learning_rate:
    type: float
    low: 0.01
    high: 0.3
    log: true        # log-scale: good for learning rates, regularization
  max_features:
    type: categorical
    choices: [sqrt, log2, 0.5]

# Always passed to constructor — can be flat dict or task-type keyed
fixed_params:
  n_jobs: -1
  # OR task-type keyed:
  # binary_classification:
  #   class_weight: balanced
  # regression:
  #   criterion: squared_error

gpu:
  supported: false   # true for catboost, xgboost, realmlp
  params: {}         # GPU-specific constructor params (only used if gpu=true)
  fallback: {}       # CPU fallback params (used if GPU micro-trial fails)

training:
  early_stopping: false   # true for tree boosters (uses eval_set)
  seed_param: random_state  # catboost uses random_seed
  # eval_metric: auc       # for early stopping models

feature_requirements:
  needs_scaling: false      # true for ridge, knn, svm, elastic_net, realmlp
  handles_categoricals: false
  handles_missing: false

optuna:
  n_trials: 50
  qmc_warmup_ratio: 0.3    # 30% QMC space-filling, then 70% TPE
  n_top_trials: 5           # top configs to retrain with multiple seeds
  n_seeds: 3
  timeout: 1800             # seconds (None = unlimited)
  pruner: null              # or: {type: MedianPruner, n_startup_trials: 5}
```

## Adding a New Model

1. Create `configs/models/{name}.yaml` following the structure above.
2. **No Python changes needed** — `ModelRegistry` auto-loads all YAMLs from `configs/models/`.
3. Add a test in `tests/test_new_models.py` (see existing examples).
4. Add to `configs/templates/binary_classification.yaml` and `regression.yaml` if appropriate.

## Key Conventions

### Log-scale hyperparameters
Use `log: true` for: `learning_rate`, `C`, `alpha`, `var_smoothing`, `l1_ratio` (sometimes), regularization strengths. Log-scale distributes Optuna samples evenly across orders of magnitude.

### Task-type keyed fixed_params
Ridge/LogReg is the canonical example — completely different class, different fixed params:
```yaml
class_path:
  binary_classification: "sklearn.linear_model.LogisticRegression"
  regression: "sklearn.linear_model.Ridge"
fixed_params:
  binary_classification:
    solver: lbfgs
    max_iter: 1000
  regression:
    fit_intercept: true
```

### GPU config pattern (CatBoost example)
```yaml
gpu:
  supported: true
  params:
    task_type: GPU
    eval_metric: Logloss   # GPU forces Logloss (AUC unsupported on GPU)
  fallback:
    task_type: CPU
    eval_metric: AUC
```

### Models requiring scaling
Set `feature_requirements.needs_scaling: true` for: ridge, elastic_net, knn, svm, realmlp.
The trainer applies `StandardScaler` before fitting these models.

## Templates

`templates/binary_classification.yaml` and `templates/regression.yaml` are starting points for new competitions. Copy to `competitions/{name}/pipeline.yaml` and edit:
- `data.*` — paths, target column, task type, target mapping
- `models` — which models to include
- `output.*` — submission and results paths
- `optuna.global_timeout` — total budget
