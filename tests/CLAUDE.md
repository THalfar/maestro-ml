# tests/ — Test Suite

## Running Tests

```bash
# All tests
conda run -n maestro pytest tests/ -v

# Single file
conda run -n maestro pytest tests/test_trainer.py -v

# By name pattern
conda run -n maestro pytest -k "test_blend" -v

# Stop at first failure
conda run -n maestro pytest tests/ -x -v
```

Expected: **~173 tests, ~2s** (no GPU, no real Optuna studies — all use tiny synthetic data).

## Test File → Source Module Mapping

| Test File | Source Module |
|-----------|--------------|
| `test_io.py` | `src/utils/io.py` |
| `test_profiler.py` | `src/eda/profiler.py` |
| `test_engineer.py` | `src/features/engineer.py` |
| `test_registry.py` | `src/models/registry.py` |
| `test_trainer.py` | `src/models/trainer.py` |
| `test_blender.py` | `src/ensemble/blender.py` |
| `test_diversity.py` | `src/ensemble/diversity.py` |
| `test_llm_strategist.py` | `src/strategy/llm_strategist.py` |
| `test_new_models.py` | `configs/models/{adaboost,elastic_net,gaussian_nb,realmlp,svm}.yaml` |
| `test_run.py` | `run.py` (end-to-end integration) |

## conftest.py

- Sets `KMP_DUPLICATE_LIB_OK=TRUE` before any import (Windows OpenMP/libomp vs libiomp5md conflict).
- Pre-imports `torch` so `shm.dll` loads before pytabkit imports it via `importlib` (Windows only).
- Suppresses all pytorch_lightning output.

**If adding new fixtures** that require real model fits, keep datasets tiny (n_samples ≤ 50).

## Test Patterns

### Mode 1 — Pure logic (no ML fits)
For YAML parsing, dataclass validation, pure computation:
```python
def test_load_yaml():
    config = load_yaml("configs/models/ridge.yaml")
    assert config["name"] == "Ridge"
```

### Mode 2 — Tiny sklearn fits
For functions that need actual model predictions:
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.random.randn(30, 3)
y = np.array([0, 1] * 15)
model = LogisticRegression()
# ... call the function under test
```

### Testing OOF leakage (engineer.py)
The target encoding tests check that val-fold values were computed only from train-fold data. See `test_engineer.py` for the pattern.

### Testing new model YAMLs (test_new_models.py)
Each new model config needs at minimum:
1. YAML loads without error
2. `ModelRegistry.get_model(name, hparams={}, gpu=False)` returns a fitted-able estimator
3. Search space is non-empty

## Adding Tests for a New Module

1. Create `tests/test_{module}.py`
2. Import `from __future__ import annotations`
3. Use synthetic data — no real CSVs
4. Target <2s total runtime for the file
5. Cover: happy path, edge cases (empty input, wrong task_type, missing keys)
