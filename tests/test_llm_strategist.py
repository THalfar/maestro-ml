"""Tests for src/strategy/llm_strategist.py — LLM integration."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.strategy.llm_strategist import (
    _build_strategy_prompt,
    _parse_llm_response,
    _validate_strategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_strategy() -> dict:
    return {
        "features": {
            "interactions": [["a", "b"]],
            "ratios": [["c", "d"]],
            "target_encoding": {"columns": ["cat1"], "pairs": [], "alpha": 15},
            "custom": [{"name": "f1", "formula": "a + b"}],
        },
        "models": ["catboost", "ridge"],
        "overrides": {
            "catboost": {"depth": {"low": 4, "high": 8}},
        },
        "reasoning": "Test reasoning.",
    }


@pytest.fixture
def sample_eda_report() -> dict:
    return {
        "dataset_info": {
            "train_shape": [1000, 10],
            "test_shape": [500, 9],
            "train_memory_mb": 1.5,
            "test_memory_mb": 0.8,
            "n_features": 9,
        },
        "target_analysis": {
            "dtype": "int64",
            "n_unique": 2,
            "distribution": {"0": 500, "1": 500},
            "class_balance_pct": {"0": 50.0, "1": 50.0},
            "missing_pct": 0.0,
        },
        "columns": {
            "feat_a": {
                "dtype": "float64",
                "detected_type": "numeric_continuous",
                "missing_pct": 0.0,
                "cardinality": 800,
                "top_values": {},
                "stats": {"mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0, "median": 0.0},
                "target_correlation": 0.45,
            },
        },
        "correlation_matrix": {"columns": ["feat_a"], "values": [[1.0]]},
        "feature_clusters": [],
        "weak_features": [],
        "recommendations": ["Test recommendation"],
    }


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------

class TestParseLlmResponse:
    def test_yaml_in_code_fence(self):
        response = """Here is my analysis:

```yaml
features:
  interactions:
    - [a, b]
models:
  - catboost
reasoning: "test"
```
"""
        result = _parse_llm_response(response)
        assert isinstance(result, dict)
        assert "features" in result
        assert result["models"] == ["catboost"]

    def test_bare_yaml(self):
        response = """features:
  interactions: []
models:
  - ridge
reasoning: "bare yaml"
"""
        result = _parse_llm_response(response)
        assert result["models"] == ["ridge"]

    def test_invalid_yaml_raises(self):
        with pytest.raises(Exception):
            _parse_llm_response("{{{{not yaml at all!!!!")

    def test_non_dict_raises(self):
        with pytest.raises(ValueError, match="dictionary"):
            _parse_llm_response("- just a list\n- of items\n")


# ---------------------------------------------------------------------------
# _validate_strategy
# ---------------------------------------------------------------------------

class TestValidateStrategy:
    def test_valid_passes(self, valid_strategy):
        result = _validate_strategy(valid_strategy, ["catboost", "ridge", "xgboost"])
        assert result is True

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_strategy({"features": {}}, ["catboost"])

    def test_unknown_model_raises(self, valid_strategy):
        with pytest.raises(ValueError, match="unknown models"):
            _validate_strategy(valid_strategy, ["ridge"])  # catboost not available

    def test_unknown_override_model_raises(self, valid_strategy):
        valid_strategy["overrides"]["nonexistent"] = {}
        with pytest.raises(ValueError, match="unknown model"):
            _validate_strategy(valid_strategy, ["catboost", "ridge"])

    def test_invalid_interaction_pair(self, valid_strategy):
        valid_strategy["features"]["interactions"] = [["a"]]  # only 1 element
        with pytest.raises(ValueError, match="exactly 2"):
            _validate_strategy(valid_strategy, ["catboost", "ridge"])

    def test_invalid_ratio_pair(self, valid_strategy):
        valid_strategy["features"]["ratios"] = ["not_a_list"]
        with pytest.raises(ValueError, match="exactly 2"):
            _validate_strategy(valid_strategy, ["catboost", "ridge"])

    def test_invalid_custom_feature(self, valid_strategy):
        valid_strategy["features"]["custom"] = [{"name": "x"}]  # missing formula
        with pytest.raises(ValueError, match="name.*formula"):
            _validate_strategy(valid_strategy, ["catboost", "ridge"])


# ---------------------------------------------------------------------------
# _build_strategy_prompt
# ---------------------------------------------------------------------------

class TestBuildStrategyPrompt:
    def test_contains_eda_report(self, sample_eda_report):
        prompt = _build_strategy_prompt(sample_eda_report, {})
        assert "EDA REPORT" in prompt
        assert "MAESTRO-ML" in prompt

    def test_contains_model_schemas(self, sample_eda_report):
        schemas = {
            "catboost": {
                "name": "CatBoost",
                "hyperparameters": {
                    "depth": {"type": "int", "low": 3, "high": 10},
                },
            },
        }
        prompt = _build_strategy_prompt(sample_eda_report, schemas)
        assert "catboost" in prompt
        assert "[3, 10]" in prompt

    def test_contains_output_format(self, sample_eda_report):
        prompt = _build_strategy_prompt(sample_eda_report, {})
        assert "features:" in prompt
        assert "models:" in prompt
        assert "reasoning:" in prompt
