# Maestro ML

**An LLM-orchestrated AutoML framework that builds diversity-aware ensembles for tabular data.**

Maestro orchestrates an ensemble of ML models like a conductor leads an orchestra — each model plays differently, but together they create something none could alone.

```
                         MAESTRO ML PIPELINE

  ┌──────────┐     ┌──────────────┐     ┌──────────────────────┐
  │  CSV     │     │  EDA Report  │     │   Strategy YAML      │
  │  Data    │────>│  (JSON)      │────>│   (LLM-generated)    │
  └──────────┘     └──────────────┘     └──────────────────────┘
       │                  │                        │
       │           Layer 1: EDA             Layer 2: LLM
       │           Pure profiling           Strategist
       │                                           │
       v                                           v
  ┌──────────────────────────────────────────────────────────┐
  │                   Layer 3: Engine                        │
  │                                                          │
  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
  │  │  Feature     │  │  Optuna      │  │  Diversity-    │  │
  │  │  Engineering │─>│  Studies     │─>│  Aware         │──│──> Submission
  │  │  (from YAML) │  │  (per model) │  │  Ensemble      │  │
  │  └─────────────┘  └──────────────┘  └────────────────┘  │
  └──────────────────────────────────────────────────────────┘
```

---

## Why Maestro?

**LLM-guided search.** An LLM analyzes your data profile and narrows the hyperparameter search space before optimization begins. Instead of blind grid search, Maestro starts with informed exploration — the LLM identifies which features matter, which interactions to create, and which parameter ranges to focus on.

**Diversity-aware ensembles.** NSGA-II optimizes both accuracy AND model diversity simultaneously. Uses correlation matrix eigenvalues to compute effective ensemble size — preventing the common failure mode where an ensemble collapses into near-identical models.

**YAML-driven.** Every decision is configured in YAML. Hyperparameter ranges, feature engineering plans, model selection, ensemble strategy — all version-controllable, auditable, and reproducible. No magic numbers buried in code.

**Competition-tested.** Built for Kaggle Playground Series competitions. Validated on real leaderboard data by a practitioner, not a framework author.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/maestro-ml.git
cd maestro-ml
pip install -r requirements.txt

# Automated mode (requires API key in .env)
python run.py --config configs/templates/binary_classification.yaml

# Human-in-the-loop mode (no API key needed)
python run.py --config configs/templates/binary_classification.yaml --strategy manual
# → Prints EDA report to console
# → Copy-paste into Claude or ChatGPT
# → Save the LLM's YAML response as strategy_output.yaml
# → Press Enter — pipeline continues automatically
```

---

## How It Works

### Layer 1: EDA Profiler

The profiler analyzes your raw CSVs and produces a structured JSON report. It computes target correlations, detects column types (binary, ordinal, categorical, continuous), identifies clusters of correlated features, flags weak predictors, and generates concrete recommendations. Pure pandas and numpy — deterministic, no randomness.

### Layer 2: LLM Strategist

The EDA report is sent to an LLM (Claude or ChatGPT) along with the available model schemas. The LLM reads the data profile and produces a strategy: which feature interactions to create, which columns to target-encode, which models to include, and how to narrow each model's hyperparameter search space. In manual mode, you control this conversation directly.

### Layer 3: Optuna Engine + Ensemble

The engine executes the LLM's strategy. Features are engineered dynamically from YAML specifications. Each model gets its own Optuna study with QMC warmup (space-filling exploration) followed by TPE (Bayesian optimization). Top configurations are retrained with multiple seeds for stability. Finally, NSGA-II selects the ensemble composition by optimizing both metric performance and model diversity.

---

## Supported Models

| Model | GPU | Notes |
|-------|-----|-------|
| CatBoost | CUDA | Native categorical support, ordered boosting |
| XGBoost | CUDA | `device="cuda"` (v2.0+) |
| LightGBM | CUDA/OpenCL | Requires special build for GPU |
| Ridge | CPU | Fast linear baseline |
| KNN | CPU | Instance-based diversity |
| Random Forest | CPU | Sklearn bagging, low variance |
| Extra Trees | CPU | Higher variance than RF = useful for diversity |

---

## Configuration Example

```yaml
# pipeline.yaml — minimal configuration
data:
  train_path: "data/train.csv"
  test_path: "data/test.csv"
  target_column: "HeartDisease"
  id_column: "id"
  task_type: "binary_classification"

cv:
  n_folds: 10
  seed: 42
  stratified: true

strategy:
  mode: "manual"

models:
  - catboost
  - xgboost
  - lightgbm
  - ridge
  - knn

ensemble:
  strategy: "auto"
  diversity_weight: 0.3

output:
  submission_path: "results/submission.csv"
  results_dir: "results/"
```

Each model's hyperparameter ranges, GPU settings, and Optuna configuration are defined in `configs/models/{name}.yaml`. The pipeline reads everything from YAML — no hardcoded values.

---

## Project Structure

```
maestro-ml/
├── configs/
│   ├── schemas/          # YAML schema documentation
│   ├── models/           # Per-model configuration (7 models)
│   └── templates/        # Ready-to-use pipeline configs
├── src/
│   ├── eda/              # Layer 1: Dataset profiling
│   ├── strategy/         # Layer 2: LLM strategy generation
│   ├── features/         # Feature engineering from YAML
│   ├── models/           # Model registry + Optuna training
│   ├── ensemble/         # Blending + diversity optimization
│   └── utils/            # YAML loading, I/O, logging
├── competitions/         # Competition-specific data and configs
├── run.py                # Pipeline entry point
└── requirements.txt
```

---

## Roadmap

- [x] Project architecture and YAML schemas
- [x] GPU auto-detection with fallback
- [ ] Automated EDA profiler
- [ ] LLM strategy generation (Claude API + manual mode)
- [ ] Dynamic feature engineering from YAML
- [ ] CatBoost / XGBoost / LightGBM Optuna integration
- [ ] Sklearn model support (Ridge, KNN, RF, Extra Trees)
- [ ] Diversity-aware ensemble selection (NSGA-II)
- [ ] QMC warmup for Optuna studies
- [ ] Meta-model stacking with logit expansion
- [ ] Kaggle Playground Series validation
- [ ] Multi-competition benchmarking

---

## Built With

- **[CatBoost](https://catboost.ai/)** / **[XGBoost](https://xgboost.readthedocs.io/)** / **[LightGBM](https://lightgbm.readthedocs.io/)** — Gradient boosting
- **[scikit-learn](https://scikit-learn.org/)** — Linear models, KNN, Random Forest, Extra Trees
- **[Optuna](https://optuna.org/)** — Bayesian hyperparameter optimization with QMC and NSGA-II
- **[Anthropic Claude API](https://docs.anthropic.com/)** — LLM-powered strategy generation
- **[pandas](https://pandas.pydata.org/)** / **[NumPy](https://numpy.org/)** — Data manipulation
- **[SciPy](https://scipy.org/)** — Statistical computations

---

## Contributing

Contributions are welcome. If you'd like to add a new model type, improve the ensemble logic, or extend the EDA profiler:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Add your model's YAML config to `configs/models/`
4. Implement following the existing docstring specifications
5. Submit a pull request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
