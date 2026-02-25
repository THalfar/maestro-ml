"""Shared test configuration.

Fixes OpenMP DLL conflict on Windows (libomp vs libiomp5md) by setting
KMP_DUPLICATE_LIB_OK before torch is imported. Must happen at conftest
level so it's set before any test module imports pytabkit/torch.
"""
from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Pre-import torch so shm.dll loads correctly before pytabkit
try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pass

# Suppress ALL pytorch_lightning / lightning output
import logging as _logging
import warnings
for _name in ("pytorch_lightning", "lightning.pytorch", "lightning",
              "lightning.fabric", "pytorch_lightning.utilities.rank_zero"):
    _pl = _logging.getLogger(_name)
    _pl.setLevel(_logging.CRITICAL)
    _pl.propagate = False
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.jit.script.*is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
