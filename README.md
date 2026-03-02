# Maestro ML

**LLM-orchestrated AutoML framework that builds diversity-aware ensembles for tabular data competitions.**

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

## Features

- **LLM-guided search** — An LLM analyzes your data profile and narrows the hyperparameter search space before Optuna begins. Informed exploration instead of blind grid search.
- **Diversity-aware ensembles** — NSGA-II optimizes both accuracy AND model diversity simultaneously. Correlation matrix eigenvalues compute effective ensemble size, preventing the common failure mode where ensembles collapse into near-identical models. NSGA-II selected models are then meta-stacked (LogisticRegression) and compared against the linear blend — the best wins automatically.
- **YAML-driven** — Every decision is configured in YAML. Hyperparameter ranges, feature engineering plans, model selection, ensemble strategy — all version-controllable and reproducible.
- **Two modes** — Fully automated (LLM API) or human-in-the-loop (manual mode where you control the LLM conversation).
- **GPU auto-detection** — Per-model micro-trial detects CUDA availability at startup, with automatic CPU fallback.

---

## Installation

```bash
git clone https://github.com/yourusername/maestro-ml.git
cd maestro-ml
pip install -r requirements.txt
```

**Requirements:** Python 3.10+

| Category | Packages |
|----------|----------|
| ML | catboost, xgboost, lightgbm, scikit-learn |
| Neural Nets | pytabkit (RealMLP, PyTorch) |
| Optimization | optuna |
| Data | pandas, numpy, scipy |
| LLM | anthropic, openai, python-dotenv |
| Config | pyyaml |
| Testing | pytest |

---

## Quick Start

### Option A: Human-in-the-loop (recommended first run)

```bash
python run.py --config configs/templates/binary_classification.yaml --strategy manual
```

This will:
1. Run EDA on your data and print a formatted report to the console
2. Pause and point you to `prompts/strategy_prompt.md` — copy the full prompt into Claude/ChatGPT, replacing the placeholder with the EDA report
3. Save the LLM's YAML response to `strategy_output.yaml` (see `competitions/ps-s6e2/strategy_output.yaml` for a complete example)
4. Press Enter — the pipeline runs feature engineering, Optuna, and ensemble automatically

### Option B: Fully automated (requires API key)

```bash
# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run with API mode
python run.py --config configs/templates/binary_classification.yaml
```

The LLM generates the strategy automatically and the pipeline runs end-to-end.

---

## Running a Kaggle Competition

Here's a walkthrough using [Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2) (Heart Disease Prediction) as an example.

### 1. Set up the competition directory

```bash
# Download competition data from Kaggle
kaggle competitions download -c playground-series-s6e2
unzip playground-series-s6e2.zip -d data/ps-s6e2/

# Copy the template and customize it
cp configs/templates/binary_classification.yaml competitions/ps-s6e2/pipeline.yaml
```

### 2. Edit the pipeline config

Edit `competitions/ps-s6e2/pipeline.yaml` to match your competition:

```yaml
data:
  train_path: "data/ps-s6e2/train.csv"
  test_path: "data/ps-s6e2/test.csv"
  target_column: "Heart Disease"
  id_column: "id"
  task_type: "binary_classification"
  target_mapping:        # map string labels → numeric
    Presence: 1
    Absence: 0

strategy:
  mode: "manual"
  manual:
    strategy_input_path: "competitions/ps-s6e2/strategy_output.yaml"

models:
  - catboost
  - xgboost
  - lightgbm
  - ridge
  - knn
  - random_forest
  - extra_trees

output:
  submission_path: "competitions/ps-s6e2/results/submission.csv"
  results_dir: "competitions/ps-s6e2/results/"
```

### 3. Run the pipeline

```bash
python run.py --config competitions/ps-s6e2/pipeline.yaml
```

The pipeline will:
1. **EDA** — Profile the dataset, print a formatted report
2. **Strategy** — Pause for you to generate a strategy YAML via LLM (manual mode). Use `prompts/strategy_prompt.md` as the LLM prompt template
3. **Features** — Create interaction, ratio, and target-encoded features from the strategy
4. **Optuna** — Run per-model hyperparameter optimization (QMC warmup + TPE)
5. **Ensemble** — NSGA-II selects diverse models, then compares linear blend vs meta-model stacking; picks the best on OOF score
6. **Output** — Save `submission.csv` and OOF predictions with full timing breakdown

### 4. Submit

```bash
kaggle competitions submit -c playground-series-s6e2 \
  -f competitions/ps-s6e2/results/submission.csv \
  -m "Maestro ML ensemble"
```

---

## How It Works

### Layer 1: EDA Profiler (`src/eda/profiler.py`)

Analyzes raw CSVs and produces a structured JSON report:
- Per-column statistics (type detection, missing %, cardinality, distribution)
- Target correlations sorted by importance
- Feature clusters (groups of highly correlated features)
- Weak feature identification
- Concrete LLM-readable recommendations

Pure pandas/numpy. Deterministic, no randomness.

### Layer 2: LLM Strategist (`src/strategy/llm_strategist.py`)

Takes the EDA report and produces a strategy YAML that controls Layer 3:
- **Feature engineering plan** — which interactions, ratios, target encodings to create
- **Model selection** — which models to include
- **Search space overrides** — narrowed hyperparameter ranges per model
- **Reasoning** — explanation of the LLM's choices

The LLM's job is to *narrow* the search space, not to do ML.

### Layer 3: Engine

**Feature Engineering** (`src/features/engineer.py`)
- Interaction features (column products)
- Ratio features (column divisions with epsilon)
- OOF target encoding with smoothing (no leakage — uses same CV folds as training)
- Custom features via pandas eval expressions

**Optuna Training** (`src/models/trainer.py`)
- Per-model independent studies
- Phase 1: QMC warmup (space-filling exploration)
- Phase 2: TPE (Bayesian optimization)
- Top configs retrained with multiple seeds for stability
- OOF predictions stored for ensemble

**Ensemble** (`src/ensemble/blender.py`, `src/ensemble/diversity.py`)
- Optimized weighted blend (Optuna)
- Rank averaging
- Meta-model stacking (LogisticRegression/Ridge with logit features)
- NSGA-II diversity-aware selection → meta-model stacking chain (best of blend vs stacking)
- Auto mode: tries all strategies, picks the best on OOF score

---

## Supported Models

| Model | GPU | Early Stopping | Notes |
|-------|-----|----------------|-------|
| CatBoost | CUDA | eval_set | Native categoricals, ordered boosting |
| XGBoost | CUDA | eval_set | `device="cuda"` (v2.0+) |
| LightGBM | Special build | callbacks | CPU-only by default |
| RealMLP | CUDA | patience | PyTorch neural net via pytabkit, dynamic layer depth (1-4), label smoothing |
| Ridge / LogReg | CPU | — | Fast linear baseline |
| Elastic Net | CPU | — | L1+L2 linear regression |
| KNN | CPU | — | Instance-based, adds diversity |
| Random Forest | CPU | — | Bagging, low variance |
| Extra Trees | CPU | — | Higher variance than RF, useful for diversity |
| AdaBoost | CPU | — | Boosted shallow trees |
| Gaussian NB | CPU | — | Probabilistic, fast, diverse predictions |
| SVM | CPU | — | RBF/polynomial kernel |

Each model is configured in `configs/models/{name}.yaml` with its hyperparameter ranges, GPU settings, fixed parameters, and Optuna study config.

---

## Project Structure

```
maestro-ml/
├── run.py                    # Pipeline entry point
├── configs/
│   ├── models/               # Per-model YAML configs (12 models)
│   │   ├── catboost.yaml
│   │   ├── xgboost.yaml
│   │   ├── lightgbm.yaml
│   │   ├── realmlp.yaml
│   │   ├── ridge.yaml
│   │   ├── elastic_net.yaml
│   │   ├── knn.yaml
│   │   ├── random_forest.yaml
│   │   ├── extra_trees.yaml
│   │   ├── adaboost.yaml
│   │   ├── gaussian_nb.yaml
│   │   └── svm.yaml
│   ├── schemas/              # YAML schema documentation
│   └── templates/            # Ready-to-use pipeline templates
│       ├── binary_classification.yaml
│       └── regression.yaml
├── src/
│   ├── utils/io.py           # YAML loading, dataclasses, logging
│   ├── eda/profiler.py       # Layer 1: dataset profiling
│   ├── strategy/llm_strategist.py  # Layer 2: LLM integration
│   ├── features/engineer.py  # Feature engineering from YAML
│   ├── models/
│   │   ├── registry.py       # Model factory + search space
│   │   └── trainer.py        # Optuna studies + CV training
│   └── ensemble/
│       ├── blender.py        # Weight optimization + stacking
│       └── diversity.py      # NSGA-II + effective ensemble size
├── prompts/
│   └── strategy_prompt.md    # LLM prompt template for manual mode
├── competitions/             # Competition-specific configs
│   ├── ps-s6e2/              # Kaggle PS S6E2 Heart Disease
│   │   ├── pipeline.yaml
│   │   └── strategy_output.yaml
│   └── ps-s6e3/              # Kaggle PS S6E3 Customer Churn
│       ├── pipeline.yaml
│       └── strategy_output.yaml
├── tests/                    # 240 tests
├── requirements.txt
└── CLAUDE.md                 # AI-assisted development instructions
```

---

## Testing

```bash
pytest tests/ -v
```

```
240 passed in ~4s
```

Tests cover all modules: YAML loading, EDA profiling, feature engineering (including OOF leakage checks), model registry, Optuna training (including dynamic_int_list for neural net architecture search), ensemble blending, diversity metrics, LLM strategy parsing, and end-to-end pipeline integration.

---

## Configuration Reference

### Pipeline YAML

| Section | Key Fields | Description |
|---------|-----------|-------------|
| `data` | `train_path`, `test_path`, `target_column`, `id_column`, `task_type`, `target_mapping` | Dataset paths and metadata. `target_mapping` converts string targets to numeric (e.g., `{Presence: 1, Absence: 0}`) |
| `cv` | `n_folds`, `seed`, `stratified` | Cross-validation settings |
| `strategy` | `mode` (`api`/`manual`), `api.provider`, `api.model` | LLM strategy mode |
| `models` | List of model names | Which models to train |
| `features` | `interactions`, `ratios`, `target_encoding`, `custom` | Feature engineering (populated by LLM) |
| `ensemble` | `strategy` (`auto`/`blend`/`rank`/`meta`/`nsga2`), `diversity_weight` | Ensemble selection |
| `optuna` | `global_seed`, `global_timeout` | Global Optuna settings |
| `runtime` | `gpu_check`, `gpu_fallback`, `n_jobs`, `verbose` | Runtime environment. `verbose`: 0=WARNING, 1=INFO (progress + timing), 2=DEBUG (per-fold details) |
| `output` | `submission_path`, `results_dir`, `save_oof` | Output paths |

### Model YAML

Each model config in `configs/models/` defines:
- `class_path` — Python import path per task type
- `hyperparameters` — Optuna search space (type, low, high, log, choices)
- `fixed_params` — Always-on parameters (can be task-type-keyed)
- `gpu` — GPU params and CPU fallback
- `training` — Early stopping, eval metric, seed parameter name
- `optuna` — Per-model trial budget, QMC warmup trials, pruner settings

---

## CLI Usage

```
python run.py --config <path-to-pipeline.yaml> [--strategy manual|api]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to pipeline YAML configuration |
| `--strategy` | No | Override strategy mode from config (`manual` or `api`) |

**Examples:**

```bash
# Manual mode — you control the LLM conversation
python run.py --config competitions/ps-s6e2/pipeline.yaml --strategy manual

# API mode — LLM generates strategy automatically
python run.py --config competitions/ps-s6e2/pipeline.yaml --strategy api

# Use mode from config file
python run.py --config competitions/ps-s6e2/pipeline.yaml
```

---

## Roadmap

- [x] Three-layer pipeline architecture
- [x] YAML-driven model and pipeline configuration
- [x] EDA profiler with LLM-readable output
- [x] LLM strategy generation (Claude/OpenAI API + manual mode)
- [x] Dynamic feature engineering (interactions, ratios, target encoding, custom)
- [x] OOF target encoding with CV fold isolation (no leakage)
- [x] Per-model Optuna studies with QMC warmup + TPE
- [x] GPU auto-detection with per-model micro-trials
- [x] Multi-seed retraining for stability
- [x] Four ensemble strategies (blend, rank, meta-model, NSGA-II)
- [x] NSGA-II → meta-model stacking chain (diversity selection + non-linear stacking)
- [x] Diversity-aware selection via correlation matrix eigenvalues
- [x] Dynamic neural net architecture search (Optuna chooses layer count + widths)
- [x] End-to-end pipeline orchestrator
- [x] 240 tests with full coverage
- [ ] Kaggle Playground Series validation runs
- [ ] Multi-competition benchmarking
- [ ] Feature importance analysis and selection
- [ ] Automated post-hoc analysis report

---

## Built With

- **[CatBoost](https://catboost.ai/)** / **[XGBoost](https://xgboost.readthedocs.io/)** / **[LightGBM](https://lightgbm.readthedocs.io/)** — Gradient boosting
- **[scikit-learn](https://scikit-learn.org/)** — Linear models, KNN, Random Forest, Extra Trees
- **[Optuna](https://optuna.org/)** — Bayesian hyperparameter optimization with QMC and NSGA-II
- **[Anthropic Claude API](https://docs.anthropic.com/)** — LLM-powered strategy generation
- **[pandas](https://pandas.pydata.org/)** / **[NumPy](https://numpy.org/)** / **[SciPy](https://scipy.org/)** — Data and scientific computing

---

## Contributing

Contributions are welcome. To add a new model type, improve the ensemble logic, or extend the EDA profiler:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Add your model's YAML config to `configs/models/`
4. Implement following the existing docstring specifications
5. Run `pytest tests/ -v` to verify all tests pass
6. Submit a pull request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
