"""Check if the IBM Telco original dataset is compatible for use as extra_data."""
from __future__ import annotations

import sys
import pandas as pd

ORIG_PATH = "competitions/ps-s6e3/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TRAIN_PATH = "competitions/ps-s6e3/train.csv"

orig = pd.read_csv(ORIG_PATH)
train = pd.read_csv(TRAIN_PATH)

print(f"=== ORIGINAL  shape={orig.shape} ===")
print(f"Churn unique: {sorted(orig['Churn'].unique())}")
print(f"Cols: {list(orig.columns)}\n")

print(f"=== TRAIN  shape={train.shape} ===")
print(f"Churn unique: {sorted(train['Churn'].unique())}")
print(f"Cols: {list(train.columns)}\n")

# Feature columns expected in train (drop id + target)
train_features = set(train.columns) - {"id", "Churn"}
orig_cols = set(orig.columns)

# Original has 'customerID' instead of 'id', and Churn is Yes/No string
print("=== COLUMN MAPPING CHECK ===")
print(f"Orig has 'customerID': {'customerID' in orig_cols}")
print(f"Train features: {sorted(train_features)}")

missing_from_orig = train_features - orig_cols
extra_in_orig = orig_cols - train_features - {"customerID", "Churn"}
print(f"\nTrain features MISSING from orig: {sorted(missing_from_orig)}")
print(f"Extra cols in orig (not in train): {sorted(extra_in_orig)}")

print("\n=== DTYPE COMPARISON ===")
for col in sorted(train_features & orig_cols):
    t = train[col].dtype
    o = orig[col].dtype
    flag = " *** MISMATCH ***" if t != o else ""
    print(f"  {col:30s}  train={t}  orig={o}{flag}")

print("\n=== ORIG TotalCharges SAMPLE (may be string?) ===")
print(orig["TotalCharges"].dtype, orig["TotalCharges"].head(5).tolist())
print("NaN count:", orig["TotalCharges"].isna().sum())
print("Empty string count:", (orig["TotalCharges"].astype(str).str.strip() == "").sum())

print("\n=== CONCLUSION ===")
issues = []
if missing_from_orig:
    issues.append(f"Missing columns in orig: {missing_from_orig}")
if orig["Churn"].dtype == object:
    issues.append("Churn is string (Yes/No) — needs target_column + target_mapping in extra_data config")

if not issues:
    print("OK — compatible. Add extra_data to pipeline.yaml with column_mapping and sample_weight.")
else:
    for i in issues:
        print(f"  ISSUE: {i}")
