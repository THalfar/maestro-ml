"""Check RealMLP params and GPU."""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch  # noqa: F401 — pre-import for shm.dll
import inspect
from pytabkit import RealMLP_TD_Classifier

sig = inspect.signature(RealMLP_TD_Classifier.__init__)
for name, p in sig.parameters.items():
    if name == "self":
        continue
    print(f"  {name}: {p.default}")
