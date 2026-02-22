"""
Maestro ML — Main pipeline orchestrator.

Entry point for the entire Maestro pipeline. Executes the three layers
in sequence:
  1. EDA: Profile the dataset → JSON report
  2. Strategy: LLM generates feature/model plan → strategy YAML
  3. Engine: Feature engineering → Optuna studies → Ensemble → Submission

Usage:
    python run.py --config pipeline.yaml
    python run.py --config pipeline.yaml --strategy manual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(pipeline_yaml_path: str | Path) -> None:
    """Run the complete Maestro pipeline from config to submission.

    This is the top-level orchestrator that coordinates all three layers.

    Args:
        pipeline_yaml_path: Path to the pipeline.yaml configuration file.

    Steps:
        1. Load pipeline configuration:
           a. Call load_pipeline_config(pipeline_yaml_path).
           b. Set up logging with configured verbosity.
           c. Log the pipeline configuration summary.

        2. GPU detection:
           a. If runtime.gpu_check is True:
              - Create ModelRegistry and load model configs.
              - For each model in pipeline_config.models, call
                registry.check_gpu(model_name).
              - Store GPU status dict: {model_name: bool}.
              - Log GPU availability per model.
           b. If gpu_check is False, assume CPU for all models.

        3. Layer 1 — EDA:
           a. Call run_eda(train_path, test_path, target_column).
           b. Save the EDA report to results_dir/eda_report.json.
           c. Log EDA summary (dataset shape, top correlations).

        4. Layer 2 — Strategy:
           a. Call generate_strategy(eda_report, pipeline_config).
           b. Save the strategy to results_dir/strategy.yaml.
           c. Log the strategy summary (selected features, models).
           d. Merge strategy features into pipeline_config.features
              (strategy overrides template defaults).

        5. Data loading and feature engineering:
           a. Read train.csv and test.csv into DataFrames.
           b. Create the CV splitter (StratifiedKFold or KFold).
           c. Call build_features(train, test, strategy, cv_folds).
           d. Determine the final feature column list.
           e. Log feature count (original + engineered).

        6. Layer 3 — Model training:
           a. Call run_all_studies(pipeline_config, train, test,
              feature_cols, strategy, registry, gpu_status).
           b. Collect all OOF and test prediction arrays.
           c. Log per-model best scores.

        7. Ensemble:
           a. Prepare the list of OOF and test predictions.
           b. Based on ensemble.strategy:
              - 'blend': optimize_blend_weights → apply_blend
              - 'rank': rank_average
              - 'meta': train_meta_model
              - 'nsga2': run_nsga2_ensemble
              - 'auto': try all strategies, pick_best_strategy
           c. Log the ensemble score and strategy used.
           d. Print the diversity report.

        8. Output:
           a. Save submission CSV to output.submission_path.
           b. If save_oof, save OOF predictions to results_dir.
           c. Log final summary: best score, ensemble strategy,
              number of models, submission path.
    """
    raise NotImplementedError


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace with:
        - config: path to pipeline.yaml (required)
        - strategy: optional override for strategy mode ('api' or 'manual')

    Steps:
        1. Create ArgumentParser with description.
        2. Add --config argument (required, type=str).
        3. Add --strategy argument (optional, choices=['api', 'manual']).
        4. Parse and return args.
    """
    raise NotImplementedError


if __name__ == "__main__":
    args = _parse_args()
    main(args.config)
