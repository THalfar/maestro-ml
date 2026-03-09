"""Quick import check for all maestro-ml modules."""
import sys

modules = [
    ("src.utils.io", ["load_yaml", "load_pipeline_config", "load_model_config",
                       "save_submission", "save_eda_report", "setup_logging"]),
    ("src.eda.profiler", ["run_eda", "format_eda_for_llm"]),
    ("src.features.engineer", ["build_features", "get_feature_columns"]),
    ("src.models.registry", ["ModelRegistry"]),
    ("src.models.trainer", ["run_optuna_study", "train_with_config",
                             "get_top_configs", "run_all_studies"]),
    ("src.ensemble.blender", ["optimize_blend_weights", "apply_blend",
                               "rank_average", "train_meta_model", "pick_best_strategy"]),
    ("src.ensemble.diversity", ["compute_correlation_matrix", "effective_ensemble_size",
                                 "greedy_diverse_select", "run_nsga2_ensemble",
                                 "print_diversity_report"]),
    ("src.strategy.llm_strategist", ["generate_strategy"]),
]

all_ok = True
for module_name, symbols in modules:
    try:
        mod = __import__(module_name, fromlist=symbols)
        for sym in symbols:
            if not hasattr(mod, sym):
                print(f"MISSING: {module_name}.{sym}")
                all_ok = False
        print(f"OK: {module_name}")
    except Exception as e:
        print(f"FAIL: {module_name} — {e}")
        all_ok = False

# run.py
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("run", "run.py")
    mod = importlib.util.module_from_spec(spec)
    # Don't execute __main__ block, just check imports
    spec.loader.exec_module(mod)
    print("OK: run.py")
except Exception as e:
    print(f"FAIL: run.py — {e}")
    all_ok = False

if all_ok:
    print("\nAll imports OK!")
    sys.exit(0)
else:
    print("\nSome imports FAILED.")
    sys.exit(1)
