"""Live test for FT-Transformer — runs actual fit/predict with pytabkit.

This script tests FTT outside the test suite to verify the model works
end-to-end. Not part of the CI test suite (slow, requires torch+GPU).

Usage:
    conda run -n maestro python scripts/test_ftt_live.py
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

# Windows OpenMP workaround (same as tests/conftest.py)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Pre-load shm.dll (Windows torch workaround)
torch_dir = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
shm_path = os.path.join(torch_dir, "shm.dll")
if os.path.exists(shm_path):
    try:
        ctypes.CDLL(shm_path)
        print("[OK] shm.dll loaded")
    except OSError as e:
        print(f"[FAIL] shm.dll: {e}")
        sys.exit(1)

import numpy as np
import pandas as pd
import torch

print(f"[INFO] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

from pytabkit import FTT_D_Classifier, FTT_D_Regressor

print("[OK] FTT_D_Classifier and FTT_D_Regressor imported")

# ---------------------------------------------------------------------------
# Test 1: Binary classification — fit + predict_proba
# ---------------------------------------------------------------------------
print("\n=== Test 1: Binary classification ===")
rng = np.random.default_rng(42)
n_train = 200
X_train = rng.normal(0, 1, (n_train, 5))
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

X_test = rng.normal(0, 1, (50, 5))

hparams = {
    "module_d_token": 64,
    "module_d_ffn_factor": 1.5,
    "module_n_layers": 1,
    "module_n_heads": 4,
    "module_attention_dropout": 0.0,
    "module_ffn_dropout": 0.0,
    "module_residual_dropout": 0.0,
    "lr": 0.001,
    "optimizer_weight_decay": 0.0,
    "batch_size": 256,
    "max_epochs": 10,
    "es_patience": 3,
    "n_cv": 1,
    "n_refit": 0,
    "use_checkpoints": False,
    "val_metric_name": "cross_entropy",
    "verbose": 0,
    "random_state": 42,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
hparams["device"] = device

print(f"  Device: {device}")
print(f"  Training on {n_train} samples, 5 features...")

t0 = time.time()
clf = FTT_D_Classifier(**hparams)
clf.fit(X_train, y_train)
t1 = time.time()
print(f"  Fit time: {t1 - t0:.2f}s")

proba = clf.predict_proba(X_train)
print(f"  predict_proba shape: {proba.shape}")
print(f"  proba range: [{proba.min():.4f}, {proba.max():.4f}]")
print(f"  proba sum check: {np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)}")

test_proba = clf.predict_proba(X_test)
print(f"  Test predict_proba shape: {test_proba.shape}")

assert proba.shape == (n_train, 2), f"Expected ({n_train}, 2), got {proba.shape}"
assert proba.min() >= 0.0, f"Negative probability: {proba.min()}"
assert proba.max() <= 1.0, f"Probability > 1: {proba.max()}"
assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), "Probabilities don't sum to 1"
print("[PASS] Binary classification")

# ---------------------------------------------------------------------------
# Test 2: Regression — fit + predict
# ---------------------------------------------------------------------------
print("\n=== Test 2: Regression ===")
y_reg = X_train[:, 0] * 2 + X_train[:, 1] + rng.normal(0, 0.1, n_train)

reg_hparams = {
    "module_d_token": 64,
    "module_d_ffn_factor": 1.5,
    "module_n_layers": 1,
    "module_n_heads": 4,
    "module_attention_dropout": 0.0,
    "module_ffn_dropout": 0.0,
    "module_residual_dropout": 0.0,
    "lr": 0.001,
    "optimizer_weight_decay": 0.0,
    "batch_size": 256,
    "max_epochs": 10,
    "es_patience": 3,
    "n_cv": 1,
    "n_refit": 0,
    # FTT regressor does NOT accept val_metric_name — uses MSE internally
    "use_checkpoints": False,
    "verbose": 0,
    "random_state": 42,
    "device": device,
}

t0 = time.time()
reg = FTT_D_Regressor(**reg_hparams)
reg.fit(X_train, y_reg)
t1 = time.time()
print(f"  Fit time: {t1 - t0:.2f}s")

preds = reg.predict(X_train)
print(f"  predict shape: {preds.shape}")
print(f"  preds range: [{preds.min():.4f}, {preds.max():.4f}]")
print(f"  All finite: {np.isfinite(preds).all()}")

assert preds.shape == (n_train,), f"Expected ({n_train},), got {preds.shape}"
assert np.isfinite(preds).all(), "Non-finite predictions"
print("[PASS] Regression")

# ---------------------------------------------------------------------------
# Test 3: Via ModelRegistry (end-to-end YAML → model)
# ---------------------------------------------------------------------------
print("\n=== Test 3: Via ModelRegistry ===")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.registry import ModelRegistry

registry = ModelRegistry("configs/models")
assert "ftt" in registry.list_models(), "ftt not found in registry"
print(f"  FTT registered: YES")

space = registry.get_search_space("ftt")
print(f"  Search space params: {sorted(space.keys())}")

optuna_cfg = registry.get_optuna_config("ftt")
print(f"  selection_mode: {optuna_cfg['selection_mode']}")
print(f"  fold_timeout: {optuna_cfg['fold_timeout']}")
print(f"  n_trials: {optuna_cfg['n_trials']}")

# Instantiate via registry
model = registry.get_model(
    "ftt",
    hparams={
        "module_d_token": 64,
        "module_d_ffn_factor": 1.5,
        "module_n_layers": 1,
        "module_n_heads": 4,
        "module_attention_dropout": 0.0,
        "module_ffn_dropout": 0.0,
        "module_residual_dropout": 0.0,
        "lr": 0.001,
        "optimizer_weight_decay": 0.0,
        "batch_size": 256,
    },
    task_type="binary_classification",
    gpu=torch.cuda.is_available(),
)
print(f"  Model class: {type(model).__name__}")
assert isinstance(model, FTT_D_Classifier)

# Fit via registry-created model
model.fit(X_train, y_train)
proba = model.predict_proba(X_train)
assert proba.shape == (n_train, 2)
assert proba.min() >= 0.0
print("[PASS] ModelRegistry integration")

# ---------------------------------------------------------------------------
# Test 4: DataFrame input (as pipeline uses it)
# ---------------------------------------------------------------------------
print("\n=== Test 4: DataFrame input ===")
df_train = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(5)])
df_train["target"] = y_train
feature_cols = [f"f{i}" for i in range(5)]

model2 = registry.get_model(
    "ftt",
    hparams={
        "module_d_token": 64,
        "module_d_ffn_factor": 1.5,
        "module_n_layers": 1,
        "module_n_heads": 4,
        "module_attention_dropout": 0.0,
        "module_ffn_dropout": 0.0,
        "module_residual_dropout": 0.0,
        "lr": 0.001,
        "optimizer_weight_decay": 0.0,
        "batch_size": 256,
    },
    task_type="binary_classification",
    gpu=torch.cuda.is_available(),
)
model2.fit(df_train[feature_cols], df_train["target"])
proba2 = model2.predict_proba(df_train[feature_cols])
assert proba2.shape == (n_train, 2)
print(f"  DataFrame predict_proba shape: {proba2.shape}")
print("[PASS] DataFrame input")

# ---------------------------------------------------------------------------
# Test 5: GPU micro-trial via registry
# ---------------------------------------------------------------------------
print("\n=== Test 5: GPU check via registry ===")
gpu_ok = registry.check_gpu("ftt")
print(f"  GPU available for FTT: {gpu_ok}")
print("[PASS] GPU check")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ALL FT-TRANSFORMER TESTS PASSED")
print("=" * 60)
