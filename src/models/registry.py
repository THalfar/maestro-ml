"""
Model Registry — Central factory for creating ML model instances.

Loads model configurations from YAML files in configs/models/ and provides
a unified interface for creating sklearn-compatible estimator instances
with the right hyperparameters, GPU settings, and task-type awareness.

This is the bridge between YAML configs and Python model objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.utils.io import ModelConfig


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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def get_search_space(
        self,
        name: str,
        overrides: dict[str, dict[str, Any]] | None = None,
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
        raise NotImplementedError

    def get_optuna_config(self, name: str) -> dict[str, Any]:
        """Get the per-model Optuna study configuration.

        Args:
            name: Registered model name.

        Returns:
            Dictionary with n_trials, qmc_warmup_ratio, timeout, pruner,
            n_top_trials, n_seeds (from the model's optuna YAML section).

        Steps:
            1. Look up the ModelConfig from self._configs[name].
            2. Return the optuna config as a dict.
        """
        raise NotImplementedError

    def list_models(self) -> list[str]:
        """Return a sorted list of all registered model names.

        Returns:
            Sorted list of model name strings.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
