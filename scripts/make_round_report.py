"""
Generate a round report from existing Optuna DBs without running the pipeline.

Usage:
    conda run -n maestro python scripts/make_round_report.py \\
        --config competitions/ps-s6e3/pipeline.yaml

    conda run -n maestro python scripts/make_round_report.py \\
        --config competitions/ps-s6e3/pipeline.yaml \\
        --output competitions/ps-s6e3/results/my_report.txt

The report is written to {results_dir}/round_report_{run_name}_{timestamp}.txt
by default. It covers all 11 sections that Claude uses for strategy decisions:
run history, score trajectory, pruning analysis, HP convergence, fold-level
score matrix (if OOF pickles available), assembly diagnostics, substudy transfer
efficiency, cross-model diversity, ensemble result, GBM config similarity, and
OOF-LB gap.

Sections 5 (fold scores), 6 (assembly), 8 (diversity), and 9 (ensemble weights)
require in-memory data from a pipeline run — they are shown as "standalone mode"
when generated outside a run. All other sections read directly from Optuna DBs.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Make sure project root is on the path regardless of where script is called from
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Fix OpenMP DLL conflict on Windows before any torch/pytabkit import
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pass

from src.utils.io import load_pipeline_config
from run import _generate_round_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a round report from existing Optuna DBs."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the pipeline.yaml configuration file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the report (default: results_dir/round_report_{run_name}_{ts}.txt).",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(cfg.output.results_dir) / f"round_report_{cfg.run_name}_{ts}.txt"

    metric = "roc_auc" if cfg.task_type != "regression" else "neg_rmse"

    _generate_round_report(
        model_results={},
        pipeline_config=cfg,
        all_oof=[],
        model_labels=[],
        y_true=None,
        metric=metric,
        ensemble_score=None,
        chosen_strategy=None,
        nsga2_info=None,
        output_path=out_path,
    )

    print(f"Round report saved: {out_path}")


if __name__ == "__main__":
    main()
