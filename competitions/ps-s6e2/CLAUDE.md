# PS-S6E2 — Playground Series Season 6 Episode 2

## Competition

[Kaggle PS S6E2](https://www.kaggle.com/competitions/playground-series-s6e2) — Heart Disease Prediction
**Task**: Binary classification
**Metric**: ROC-AUC

## Data

| File | Path | Description |
|------|------|-------------|
| Train | `data/ps-s6e2/train.csv` | ~70k rows, synthetic |
| Test | `data/ps-s6e2/test.csv` | Test set for submission |
| Target | `Heart Disease` | `Presence` → 1, `Absence` → 0 |

Key features: Age, Sex, Chest pain type, Resting BP, Cholesterol, Fasting blood sugar, Resting ECG, Max HR, Exercise angina, ST depression, Slope of ST, Vessels, Thallium.

## Pipeline Configuration (`pipeline.yaml`)

- **CV**: 5 folds, stratified, seed=42
- **Strategy**: manual mode → reads from `strategy_output.yaml`
- **Models** (8): realmlp, catboost, xgboost, ridge, elastic_net, knn, gaussian_nb, adaboost
- **Ensemble**: nsga2, diversity_weight=[0.3, 0.4, 0.5], 15k trials
- **GPU**: enabled, fallback=true

### Optuna Timeouts (tuned for RTX 5090, ~10h total)
| Model | Timeout |
|-------|---------|
| realmlp | 4h |
| catboost | 2h |
| xgboost | 1h15m |
| ridge | 5min |
| elastic_net | 5min |
| knn | 5min |
| gaussian_nb | 5min |
| adaboost | 30min |

## Strategy (`strategy_output.yaml`)

LLM-generated strategy, Iteration 4. Key choices:

**Features**:
- Interactions: Thallium×Exercise_angina, Chest_pain_type×Sex, ST_depression×Exercise_angina, Max_HR×Age, Vessels×Thallium
- Ratios: ST_depression/Max_HR, Max_HR/Age, Cholesterol/Age
- Target encoding: Thallium, Chest_pain_type, Slope_of_ST, Vessels, Thallium×Vessels, Thallium×Exercise_angina (alpha=5)
- Custom: `hr_reserve_pct`, `ischemia_score`, `cardiac_risk_score`

**Model overrides** (narrowed search spaces):
- catboost: depth 4–7, lr 0.02–0.12, iterations 800–2500
- xgboost: max_depth 2–5, lr 0.005–0.05, n_estimators 1500–3000
- realmlp: hidden [64,64]–[128,128,64], lr 0.001–0.3, batch {2048,4096,8192,16384}

## Results

| File | Description |
|------|-------------|
| `results/submission.csv` | Final predictions for Kaggle submission |
| `results/oof_predictions.npy` | Out-of-fold predictions (shape: [n_train, n_models]) |
| `results/eda_report.json` | Structured EDA data |
| `results/strategy.yaml` | Executed strategy (copy of strategy_output.yaml at run time) |
| `results_realmlp_test/` | Results from alternative RealMLP-only test run |

## Run Command

```bash
conda run -n maestro python run.py --config competitions/ps-s6e2/pipeline.yaml
```

For a fresh run with manual strategy override:
```bash
conda run -n maestro python run.py --config competitions/ps-s6e2/pipeline.yaml --strategy manual
```
