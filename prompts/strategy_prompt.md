# Strategy Prompt — maestro-ml

Copy-paste this prompt into Claude/ChatGPT along with the EDA report.
The LLM's output goes into `strategy_output.yaml` (the path configured
in your pipeline YAML under `strategy.manual.strategy_input_path`).

---

## Prompt (copy everything below the line)

---

You are a machine learning strategist for **maestro-ml**, a tabular data competition pipeline. Your job is to analyze the EDA report below and produce a **YAML strategy** that controls:

1. **Feature engineering** — which new features to create
2. **Model selection** — which models to train
3. **Search space overrides** — narrow hyperparameter ranges based on dataset characteristics

### Rules

- **NARROW, don't guess.** Suggest tighter ranges, not exact values. Optuna handles the fine-tuning.
- **Features must make domain sense.** Don't create random interactions — use the correlation patterns and feature types from the EDA report.
- **Skip weak features** (|corr| < 0.05) from expensive feature engineering (target encoding, custom formulas). They can still be model inputs.
- **Dataset size matters:**
  - Large (>50k rows) → moderate tree depth (4-8), more iterations
  - Small (<5k rows) → shallow trees, stronger regularization, fewer models
- **Diversity over quantity.** Only include models that add something different. For small datasets, 3-4 models is enough.

### Available models

```
catboost      — Gradient boosting with native categorical support
xgboost       — Gradient boosting (GPU-accelerated)
lightgbm      — Gradient boosting (leaf-wise, fast)
ridge         — Logistic/Ridge regression (L2 linear baseline)
elastic_net   — L1+L2 linear model (feature selection via sparsity)
knn           — K-Nearest Neighbors
svm           — Support Vector Machine (RBF kernel, different decision boundaries)
gaussian_nb   — Gaussian Naive Bayes (classification only, uncorrelated diversity)
mlp           — Multi-Layer Perceptron (small neural network, 2-3 layers)
adaboost      — Adaptive Boosting (re-weights hard samples, different from gradient boosting)
random_forest — Bagging ensemble (row + feature sampling)
extra_trees   — Extremely Randomized Trees (more randomness than RF)
```

### Output format

Return ONLY a YAML block in this exact structure:

```yaml
features:
  interactions:
    - [col_a, col_b]                    # creates col_a * col_b
  ratios:
    - [numerator, denominator]          # creates numerator / (denominator + 1e-8)
  target_encoding:
    columns: [col1, col2]              # single-column target encoding
    pairs: [[col1, col2]]             # joint target encoding on column pairs
    alpha: 15                          # smoothing (higher = more regularization)
  custom:
    - name: feature_name
      formula: "col_a * col_b / 1000"  # pandas eval() expression
      # Use backticks for column names with spaces: `Max HR`

models:
  - catboost
  - xgboost
  - lightgbm

overrides:                             # OPTIONAL — narrow default ranges
  catboost:
    depth:
      low: 4
      high: 8
  xgboost:
    max_depth:
      low: 3
      high: 8

reasoning: >
  Explain your choices in 2-5 sentences. Why these features?
  Why these models? Why these range overrides?
```

### Key hyperparameter names per model

| Model | Key params you can override |
|-------|---------------------------|
| catboost | `depth` [1-10], `learning_rate` [0.01-0.3], `iterations` [100-3000], `l2_leaf_reg` [1-10] |
| xgboost | `max_depth` [2-10], `learning_rate` [0.01-0.3], `n_estimators` [100-3000] |
| lightgbm | `num_leaves` [8-256], `learning_rate` [0.01-0.3], `n_estimators` [100-3000] |
| random_forest | `n_estimators` [50-500], `max_depth` [3-15], `min_samples_leaf` [1-20] |
| extra_trees | `n_estimators` [50-500], `max_depth` [3-15], `min_samples_leaf` [1-20] |
| ridge | `C` [0.1-5.0] (log scale) |
| elastic_net | `alpha` [0.0001-10] (log scale), `l1_ratio` [0.05-0.95] |
| knn | `n_neighbors` [3-50], `weights` [uniform, distance], `p` [1, 2] |
| svm | `C` [0.01-100] (log scale), `kernel` [rbf, poly], `gamma` [scale, auto] |
| gaussian_nb | `var_smoothing` [1e-12 - 1e-3] (log scale) |
| mlp | `hidden_layer_sizes` [[128], [128,64], ...], `alpha` [1e-5 - 0.01], `learning_rate_init` [1e-4 - 0.01] |
| adaboost | `n_estimators` [50-500], `learning_rate` [0.01-1.0] (log scale) |

### Feature engineering notes

- **Interactions** (`col_a * col_b`): Best between a strong numeric feature and a binary/ordinal feature.
- **Ratios** (`a / b`): Best for normalizing (e.g., cholesterol/age, metric/max_value).
- **Target encoding**: Use for ordinal/categorical columns with 3+ unique values. The `alpha` parameter controls smoothing — higher means more regularization (good for small categories).
- **Custom formulas**: Any valid pandas `eval()` expression. Use backticks for column names containing spaces (e.g., `` `Max HR` ``).

### Example

Here is an example for a heart disease dataset:

```yaml
features:
  interactions:
    - [Thallium, Exercise angina]
    - [ST depression, Slope of ST]
    - [Max HR, Exercise angina]
    - [Sex, Chest pain type]
  ratios:
    - [ST depression, Max HR]
    - [Cholesterol, Age]
  target_encoding:
    columns: [Thallium, Chest pain type, Slope of ST]
    pairs: [[Thallium, Number of vessels fluro]]
    alpha: 15
  custom:
    - name: hr_reserve
      formula: "220 - Age - `Max HR`"

models:
  - catboost
  - xgboost
  - lightgbm
  - ridge
  - elastic_net
  - knn
  - svm
  - gaussian_nb
  - mlp
  - adaboost
  - random_forest
  - extra_trees

overrides:
  catboost:
    depth:
      low: 4
      high: 8
  xgboost:
    max_depth:
      low: 3
      high: 8

reasoning: >
  Thallium (+0.61) and Chest pain type (+0.46) are the strongest
  predictors. Interactions between top features and binary indicators.
  Target encoding on ordinals with high cardinality. Large dataset
  (630k rows) allows moderate tree depth. All 12 models for diversity.
```

---

**Now analyze the EDA report below and produce your strategy YAML:**

[PASTE EDA REPORT HERE]
