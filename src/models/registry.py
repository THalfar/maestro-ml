"""
Model Registry — Central factory for creating ML model instances.

Loads model configurations from YAML files in configs/models/ and provides
a unified interface for creating sklearn-compatible estimator instances
with the right hyperparameters, GPU settings, and task-type awareness.

This is the bridge between YAML configs and Python model objects.
"""

from __future__ import annotations

import atexit
import copy
import importlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import ModelConfig, load_model_config

logger = logging.getLogger("maestro")

_TASK_TYPES = {"binary_classification", "multiclass", "regression"}


class ModelRegistry:
    """Registry that maps model names to their configs and creates instances.

    The registry loads model YAML configs at startup and provides methods
    to create configured model instances, get Optuna search spaces, and
    check GPU availability per model.

    Usage:
        registry = ModelRegistry(configs_dir='configs/models')
        registry.register('catboost', 'configs/models/catboost.yaml')
        model = registry.get_model('catboost', hparams={'depth': 6}, gpu=True)
        space = registry.get_search_space('catboost')
    """

    def __init__(self, configs_dir: str | Path = "configs/models") -> None:
        """Initialize the registry, optionally loading all configs from a directory.

        Args:
            configs_dir: Path to directory containing model YAML files.
                         All *.yaml files in this directory will be auto-registered.

        Steps:
            1. Initialize internal dict: self._configs = {}.
            2. Initialize GPU status cache: self._gpu_status = {}.
            3. If configs_dir exists, iterate over all .yaml files and
               call self.register for each one.
        """
        self._configs: dict[str, ModelConfig] = {}
        self._gpu_status: dict[str, bool] = {}
        self._catboost_tmp: str | None = None

        configs_dir = Path(configs_dir)
        if configs_dir.exists():
            for yaml_path in sorted(configs_dir.glob("*.yaml")):
                name = yaml_path.stem  # filename without extension
                self.register(name, yaml_path)

    def register(self, name: str, config_path: str | Path) -> None:
        """Register a model by loading its YAML configuration.

        Args:
            name: Model name (e.g., 'catboost'). Used as the lookup key.
            config_path: Path to the model's YAML configuration file.

        Steps:
            1. Call load_model_config(config_path) to parse the YAML.
            2. Store the ModelConfig in self._configs[name].
            3. Log the registration.
        """
        config = load_model_config(config_path)
        self._configs[name] = config
        logger.debug(f"Registered model: {name} ({config.name})")

    def get_model(
        self,
        name: str,
        hparams: dict[str, Any],
        task_type: str = "binary_classification",
        gpu: bool = False,
        results_dir: Path | None = None,
    ) -> Any:
        """Create and return a configured model instance.

        Args:
            name: Registered model name.
            hparams: Hyperparameters from Optuna trial or manual config.
            task_type: Task type to select the correct class_path and
                       fixed_params.
            gpu: Whether to apply GPU params (True) or CPU fallback (False).
            results_dir: Directory for model artifacts (e.g., CatBoost's
                         train_dir). If None, uses a temp directory.

        Returns:
            Instantiated sklearn-compatible estimator with all params set.

        Raises:
            KeyError: If the model name is not registered.
            ImportError: If the model's class cannot be imported.

        Steps:
            1. Look up the ModelConfig from self._configs[name].
            2. Select the class_path for the given task_type.
            3. Dynamically import the class using importlib.
            4. Build the params dict:
               a. Start with fixed_params (selecting task-specific ones
                  if fixed_params is a dict keyed by task_type).
               b. Merge in hparams (Optuna-suggested values).
               c. If gpu=True, merge gpu.params. Else merge gpu.fallback.
               d. For CatBoost: set train_dir to results_dir/catboost_info.
            5. Instantiate and return the model class with the params.
        """
        if name not in self._configs:
            raise KeyError(f"Model '{name}' not registered. Available: {self.list_models()}")

        config = self._configs[name]

        # Select class_path for task_type
        class_path = config.class_path.get(task_type)
        if class_path is None:
            if not config.class_path:
                raise KeyError(f"Model '{name}' has empty class_path")
            # Fallback to first available
            class_path = next(iter(config.class_path.values()))
            logger.warning(
                f"Model '{name}' has no class_path for '{task_type}', "
                f"using '{class_path}'"
            )

        # Dynamically import the class
        module_path, class_name = class_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(f"Could not import '{class_path}': {exc}") from exc

        # Build params dict
        fixed = config.fixed_params

        # Handle task-type-specific fixed_params (e.g., ridge.yaml).
        # Detection: all top-level keys are known task_type strings.
        if fixed and isinstance(fixed, dict) and set(fixed.keys()) <= _TASK_TYPES:
            fixed = fixed.get(task_type, {}) or {}

        params: dict[str, Any] = {}
        params.update(fixed or {})
        params.update(hparams)

        # GPU / CPU params
        gpu_cfg = config.gpu or {}
        if gpu:
            params.update(gpu_cfg.get("params", {}) or {})
        else:
            params.update(gpu_cfg.get("fallback", {}) or {})

        # CatBoost: set train_dir
        if "catboost" in class_path.lower():
            if results_dir is not None:
                catboost_dir = Path(results_dir) / "catboost_info"
            else:
                if self._catboost_tmp is None:
                    self._catboost_tmp = tempfile.mkdtemp()
                    atexit.register(shutil.rmtree, self._catboost_tmp, ignore_errors=True)
                catboost_dir = Path(self._catboost_tmp) / "catboost_info"
            catboost_dir.mkdir(parents=True, exist_ok=True)
            params["train_dir"] = str(catboost_dir)

        return model_class(**params)

    def get_search_space(
        self,
        name: str,
        overrides: dict[str, dict[str, Any]] | None = None,
        task_type: str = "binary_classification",
    ) -> dict[str, dict[str, Any]]:
        """Get the Optuna search space definition for a model.

        The search space can be optionally narrowed by LLM strategy
        overrides (e.g., the LLM may restrict depth to 4-8 instead
        of the default 3-10).

        Args:
            name: Registered model name.
            overrides: Optional dict of parameter overrides from the LLM
                       strategy. Each key is a param name, value is a dict
                       with 'low', 'high', and/or 'choices' to override.
            task_type: Task type to select correct sub-dict for models with
                       task-type-keyed hyperparameters (e.g., ridge).

        Returns:
            Dictionary of {param_name: param_spec} where param_spec has
            keys: type, low, high, log, choices (matching model_schema.yaml
            hyperparameters format). Overrides are merged on top of defaults.

        Steps:
            1. Look up the ModelConfig from self._configs[name].
            2. Deep-copy the hyperparameters dict.
            3. If overrides is provided, merge each override into the
               corresponding parameter spec (override low/high/choices).
            4. Return the merged search space.
        """
        if name not in self._configs:
            raise KeyError(f"Model '{name}' not registered.")

        config = self._configs[name]
        space = copy.deepcopy(config.hyperparameters)

        # Handle task-type-specific hyperparameters (e.g., ridge.yaml).
        # Same detection logic as fixed_params: if all top-level keys are
        # known task_type strings, select the sub-dict for this task.
        if space and isinstance(space, dict) and set(space.keys()) <= _TASK_TYPES:
            space = space.get(task_type, {}) or {}

        if overrides:
            for param_name, override_spec in overrides.items():
                if isinstance(override_spec, dict):
                    if param_name in space:
                        space[param_name].update(override_spec)
                    else:
                        space[param_name] = override_spec
                else:
                    # Scalar override → fix the parameter to that value
                    space[param_name] = {"type": "fixed", "value": override_spec}

        return space

    def get_optuna_config(self, name: str) -> dict[str, Any]:
        """Get the per-model Optuna study configuration.

        Args:
            name: Registered model name.

        Returns:
            Dictionary with n_trials, qmc_warmup_trials, timeout, pruner,
            n_top_trials, n_seeds, selection_mode, fold_timeout
            (from the model's optuna YAML section).

        Steps:
            1. Look up the ModelConfig from self._configs[name].
            2. Return the optuna config as a dict.
        """
        if name not in self._configs:
            raise KeyError(f"Model '{name}' not registered.")

        optuna_cfg = self._configs[name].optuna
        return {
            "n_trials": optuna_cfg.n_trials,
            "qmc_warmup_trials": optuna_cfg.qmc_warmup_trials,
            "timeout": optuna_cfg.timeout,
            "pruner": optuna_cfg.pruner,
            "n_top_trials": optuna_cfg.n_top_trials,
            "n_seeds": optuna_cfg.n_seeds,
            "selection_mode": optuna_cfg.selection_mode,
            "fold_timeout": optuna_cfg.fold_timeout,
        }

    def get_training_config(self, name: str) -> dict[str, Any]:
        """Get the training configuration dict for a model.

        Args:
            name: Registered model name.

        Returns:
            Training config dict (needs_eval_set, early_stopping_rounds, etc.)
        """
        if name not in self._configs:
            raise KeyError(f"Model '{name}' not registered.")
        return copy.deepcopy(self._configs[name].training)

    def list_models(self) -> list[str]:
        """Return a sorted list of all registered model names.

        Returns:
            Sorted list of model name strings.
        """
        return sorted(self._configs.keys())

    def check_gpu(self, name: str, task_type: str = "binary_classification") -> bool:
        """Test whether GPU acceleration works for a model via micro-trial.

        Creates a tiny dataset and attempts to fit the model with GPU
        params. If it succeeds, returns True. If it fails (CUDA error,
        missing driver, etc.), returns False and caches the result.

        Args:
            name: Registered model name.
            task_type: Task type for selecting the correct model class.

        Returns:
            True if GPU training succeeded, False otherwise.

        Steps:
            1. Check self._gpu_status cache. If already tested, return
               cached result.
            2. Look up the ModelConfig. If gpu.supported is False,
               cache False and return.
            3. Create a tiny synthetic dataset (20 rows, 3 features).
            4. Try to instantiate the model with GPU params and fit
               on the tiny dataset.
            5. If successful, cache True and return True.
            6. If any exception occurs, log the error, cache False,
               and return False.
        """
        cache_key = f"{name}_{task_type}"
        if cache_key in self._gpu_status:
            return self._gpu_status[cache_key]

        if name not in self._configs:
            self._gpu_status[cache_key] = False
            return False

        config = self._configs[name]
        if not (config.gpu or {}).get("supported", False):
            self._gpu_status[cache_key] = False
            return False

        # Tiny synthetic dataset
        rng = np.random.default_rng(0)
        X = rng.random((20, 3)).astype(np.float32)
        if task_type == "regression":
            y = rng.random(20).astype(np.float32)
        else:
            y = (rng.random(20) > 0.5).astype(np.int32)

        try:
            model = self.get_model(name, hparams={}, task_type=task_type, gpu=True)
            model.fit(X, y)
            logger.info(f"GPU check passed for '{name}'")
            self._gpu_status[cache_key] = True
            return True
        except Exception as exc:
            logger.warning(f"GPU check failed for '{name}': {exc}. Using CPU fallback.")
            self._gpu_status[cache_key] = False
            return False

    def get_feature_requirements(self, name: str) -> dict[str, bool]:
        """Get the feature requirements for a model.

        Args:
            name: Registered model name.

        Returns:
            Dictionary with keys: needs_scaling, handles_categorical,
            handles_missing.

        Steps:
            1. Look up the ModelConfig from self._configs[name].
            2. Return the feature_requirements dict.
        """
        if name not in self._configs:
            raise KeyError(f"Model '{name}' not registered.")

        return dict(self._configs[name].feature_requirements)
