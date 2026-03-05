"""Tests for src/strategy/llm_strategist.py — LLM integration."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.strategy.llm_strategist import (
    _build_strategy_prompt,
    _call_llm_api,
    _load_model_schemas,
    _parse_llm_response,
    _validate_strategy,
    generate_strategy,
    run_manual_mode,
)
from src.utils.io import PipelineConfig, StrategyConfig


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

    def test_categorical_hyperparameter(self, sample_eda_report):
        schemas = {
            "lightgbm": {
                "name": "LightGBM",
                "hyperparameters": {
                    "boosting_type": {
                        "type": "categorical",
                        "choices": ["gbdt", "dart"],
                    },
                },
            },
        }
        prompt = _build_strategy_prompt(sample_eda_report, schemas)
        assert "choices=" in prompt
        assert "gbdt" in prompt
        assert "dart" in prompt

    def test_log_scale_hyperparameter(self, sample_eda_report):
        schemas = {
            "xgboost": {
                "name": "XGBoost",
                "hyperparameters": {
                    "learning_rate": {
                        "type": "float",
                        "low": 0.001,
                        "high": 0.3,
                        "log": True,
                    },
                },
            },
        }
        prompt = _build_strategy_prompt(sample_eda_report, schemas)
        assert "(log scale)" in prompt
        assert "[0.001, 0.3]" in prompt

    def test_dynamic_int_list_hyperparameter(self, sample_eda_report):
        """RealMLP-style dynamic_int_list hyperparameter is formatted correctly."""
        schemas = {
            "realmlp": {
                "name": "RealMLP",
                "hyperparameters": {
                    "hidden_layers": {
                        "type": "dynamic_int_list",
                        "n_min": 1,
                        "n_max": 4,
                        "low": 64,
                        "high": 512,
                    },
                },
            },
        }
        prompt = _build_strategy_prompt(sample_eda_report, schemas)
        assert "realmlp" in prompt
        assert "1-4 layers" in prompt
        assert "[64, 512]" in prompt


# ---------------------------------------------------------------------------
# _parse_llm_response — additional edge cases
# ---------------------------------------------------------------------------

class TestParseLlmResponseEdgeCases:
    def test_generic_code_fence_without_yaml_marker(self):
        """Code fence with ``` but no yaml language marker."""
        response = "Here:\n\n```\nfeatures:\n  interactions: []\nmodels:\n  - ridge\nreasoning: test\n```\n"
        result = _parse_llm_response(response)
        assert result["models"] == ["ridge"]

    def test_multiple_code_fences_takes_first(self):
        response = (
            "```yaml\nfeatures: {}\nmodels: [catboost]\nreasoning: first\n```\n"
            "```yaml\nfeatures: {}\nmodels: [ridge]\nreasoning: second\n```\n"
        )
        result = _parse_llm_response(response)
        assert result["models"] == ["catboost"]
        assert result["reasoning"] == "first"


# ---------------------------------------------------------------------------
# _validate_strategy — additional edge cases
# ---------------------------------------------------------------------------

class TestValidateStrategyEdgeCases:
    def test_none_features_treated_as_empty(self):
        """Strategy with features=None should not crash (or {} path)."""
        strategy = {
            "features": None,
            "models": ["catboost"],
            "reasoning": "test",
        }
        assert _validate_strategy(strategy, ["catboost"]) is True

    def test_none_interactions_treated_as_empty(self):
        strategy = {
            "features": {"interactions": None, "ratios": None, "custom": None},
            "models": ["catboost"],
            "reasoning": "test",
        }
        assert _validate_strategy(strategy, ["catboost"]) is True

    def test_none_overrides_treated_as_empty(self):
        strategy = {
            "features": {},
            "models": ["catboost"],
            "overrides": None,
            "reasoning": "test",
        }
        assert _validate_strategy(strategy, ["catboost"]) is True

    def test_empty_models_list_is_valid(self):
        strategy = {
            "features": {},
            "models": [],
            "reasoning": "no models needed",
        }
        assert _validate_strategy(strategy, ["catboost"]) is True

    def test_invalid_te_columns_non_list(self):
        strategy = {
            "features": {
                "target_encoding": {"columns": "not_a_list"},
            },
            "models": [],
            "reasoning": "test",
        }
        with pytest.raises(ValueError, match="list of strings"):
            _validate_strategy(strategy, [])

    def test_invalid_te_columns_non_strings(self):
        strategy = {
            "features": {
                "target_encoding": {"columns": [1, 2, 3]},
            },
            "models": [],
            "reasoning": "test",
        }
        with pytest.raises(ValueError, match="list of strings"):
            _validate_strategy(strategy, [])

    def test_invalid_te_pair_wrong_length(self):
        """target_encoding pairs must be exactly length 2."""
        strategy = {
            "features": {
                "target_encoding": {"columns": [], "pairs": [["a", "b", "c"]]},
            },
            "models": [],
            "reasoning": "test",
        }
        with pytest.raises(ValueError, match="exactly 2"):
            _validate_strategy(strategy, [])


# ---------------------------------------------------------------------------
# generate_strategy — main entry point
# ---------------------------------------------------------------------------

class TestGenerateStrategy:
    def test_unknown_mode_raises(self, sample_eda_report):
        config = PipelineConfig(
            strategy=StrategyConfig(mode="unknown"),
        )
        with pytest.raises(ValueError, match="Unknown strategy mode"):
            generate_strategy(sample_eda_report, config)

    @patch("src.strategy.llm_strategist._load_model_schemas")
    @patch("src.strategy.llm_strategist._call_llm_api")
    def test_api_mode_dispatches(self, mock_api, mock_schemas, sample_eda_report):
        mock_schemas.return_value = {"catboost": {"name": "CatBoost", "hyperparameters": {}}}
        yaml_response = (
            "```yaml\n"
            "features: {}\n"
            "models: [catboost]\n"
            "reasoning: test\n"
            "```\n"
        )
        mock_api.return_value = yaml_response

        config = PipelineConfig(
            strategy=StrategyConfig(mode="api", api={"provider": "anthropic"}),
            models=["catboost"],
        )
        result = generate_strategy(sample_eda_report, config)

        assert result["models"] == ["catboost"]
        mock_schemas.assert_called_once_with(config)
        mock_api.assert_called_once()

    @patch("src.strategy.llm_strategist.run_manual_mode")
    def test_manual_mode_dispatches(self, mock_manual, sample_eda_report):
        mock_manual.return_value = {
            "features": {},
            "models": ["ridge"],
            "reasoning": "manual",
        }
        config = PipelineConfig(
            strategy=StrategyConfig(
                mode="manual",
                manual={"strategy_input_path": "results/strategy.yaml"},
            ),
        )
        result = generate_strategy(sample_eda_report, config)

        assert result["models"] == ["ridge"]
        mock_manual.assert_called_once()

    @patch("src.strategy.llm_strategist.run_manual_mode")
    def test_manual_mode_default_path(self, mock_manual, sample_eda_report):
        """When manual config is None/empty, default path is used."""
        mock_manual.return_value = {
            "features": {},
            "models": [],
            "reasoning": "ok",
        }
        config = PipelineConfig(
            strategy=StrategyConfig(mode="manual", manual={}),
        )
        generate_strategy(sample_eda_report, config)
        # Check the default path was used
        call_args = mock_manual.call_args
        assert call_args[0][1] == "results/strategy.yaml"


# ---------------------------------------------------------------------------
# _call_llm_api
# ---------------------------------------------------------------------------

class TestCallLlmApi:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = {"provider": "anthropic", "env_key": "ANTHROPIC_API_KEY"}
        with pytest.raises(RuntimeError, match="API key not set"):
            _call_llm_api("test prompt", config)

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "fake-key")
        config = {"provider": "gemini", "env_key": "SOME_KEY"}
        with pytest.raises(ValueError, match="Unknown provider"):
            _call_llm_api("test prompt", config)

    @patch("src.strategy.llm_strategist.time.sleep")  # skip actual sleep
    def test_anthropic_success(self, mock_sleep, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="response text")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        with patch.dict("sys.modules", {"anthropic": MagicMock()}) as _:
            import sys
            mock_anthropic = sys.modules["anthropic"]
            mock_anthropic.Anthropic.return_value = mock_client

            result = _call_llm_api("test", {"provider": "anthropic", "env_key": "ANTHROPIC_API_KEY"})
            assert result == "response text"

    @patch("src.strategy.llm_strategist.time.sleep")
    def test_openai_success(self, mock_sleep, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

        mock_choice = MagicMock()
        mock_choice.message.content = "openai response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": MagicMock()}) as _:
            import sys
            mock_openai = sys.modules["openai"]
            mock_openai.OpenAI.return_value = mock_client

            result = _call_llm_api("test", {
                "provider": "openai",
                "env_key": "OPENAI_API_KEY",
                "model": "gpt-4",
            })
            assert result == "openai response"

    @patch("src.strategy.llm_strategist.time.sleep")
    def test_retry_then_fail(self, mock_sleep, monkeypatch):
        """All 3 retries fail → RuntimeError with last error."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

        with patch.dict("sys.modules", {"anthropic": MagicMock()}) as _:
            import sys
            mock_anthropic = sys.modules["anthropic"]
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = ConnectionError("network error")
            mock_anthropic.Anthropic.return_value = mock_client

            with pytest.raises(RuntimeError, match="failed after 3 attempts"):
                _call_llm_api("test", {"provider": "anthropic", "env_key": "ANTHROPIC_API_KEY"})

            # Verify exponential backoff sleeps — no sleep after the final attempt
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1)
            mock_sleep.assert_any_call(2)


# ---------------------------------------------------------------------------
# _load_model_schemas
# ---------------------------------------------------------------------------

class TestLoadModelSchemas:
    def test_loads_existing_model_yamls(self, tmp_path):
        """Loads YAML files for models listed in pipeline config."""
        models_dir = tmp_path / "configs" / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "catboost.yaml").write_text(
            "name: CatBoost\nhyperparameters:\n  depth:\n    low: 3\n    high: 10\n",
            encoding="utf-8",
        )

        config = PipelineConfig(models=["catboost"])

        with patch("src.strategy.llm_strategist.Path") as MockPath:
            # Make Path("configs/models") return our tmp_path
            def path_side_effect(arg):
                if arg == "configs/models":
                    return models_dir
                return Path(arg)
            MockPath.side_effect = path_side_effect

            result = _load_model_schemas(config)

        assert "catboost" in result
        assert result["catboost"]["name"] == "CatBoost"

    def test_missing_yaml_logs_warning(self, tmp_path, caplog):
        """Model listed in config but no YAML → warning logged."""
        models_dir = tmp_path / "configs" / "models"
        models_dir.mkdir(parents=True)

        config = PipelineConfig(models=["nonexistent_model"])

        with patch("src.strategy.llm_strategist.Path") as MockPath:
            def path_side_effect(arg):
                if arg == "configs/models":
                    return models_dir
                return Path(arg)
            MockPath.side_effect = path_side_effect

            import logging
            with caplog.at_level(logging.WARNING, logger="maestro"):
                result = _load_model_schemas(config)

        assert result == {}
        assert "not found" in caplog.text


# ---------------------------------------------------------------------------
# run_manual_mode
# ---------------------------------------------------------------------------

class TestRunManualMode:
    def test_loads_strategy_from_yaml(self, tmp_path, monkeypatch):
        """Writes a strategy YAML and verifies run_manual_mode loads it."""
        strategy_yaml = tmp_path / "strategy.yaml"
        strategy_yaml.write_text(
            "features: {}\nmodels: [catboost]\nreasoning: manual test\n",
            encoding="utf-8",
        )

        # Mock input() and configs/models directory
        monkeypatch.setattr("builtins.input", lambda _: "")
        with patch("src.strategy.llm_strategist.Path") as MockPath:
            real_path = Path

            def path_side_effect(arg):
                if arg == "configs/models":
                    # Return a non-existent dir so available_models is empty
                    return tmp_path / "nonexistent"
                return real_path(arg)
            MockPath.side_effect = path_side_effect

            result = run_manual_mode(
                {"dataset_info": {}, "target_analysis": {"dtype": "int64", "n_unique": 2,
                 "distribution": {}, "class_balance_pct": {}, "missing_pct": 0},
                 "columns": {}, "correlation_matrix": {"columns": [], "values": []},
                 "feature_clusters": [], "weak_features": [], "recommendations": []},
                str(strategy_yaml),
            )

        assert result["models"] == ["catboost"]

    def test_file_not_found_raises(self, tmp_path, monkeypatch):
        """If strategy YAML doesn't exist after user presses Enter."""
        monkeypatch.setattr("builtins.input", lambda _: "")
        with pytest.raises(FileNotFoundError, match="Strategy YAML not found"):
            run_manual_mode(
                {"dataset_info": {}, "target_analysis": {"dtype": "int64", "n_unique": 2,
                 "distribution": {}, "class_balance_pct": {}, "missing_pct": 0},
                 "columns": {}, "correlation_matrix": {"columns": [], "values": []},
                 "feature_clusters": [], "weak_features": [], "recommendations": []},
                str(tmp_path / "nonexistent.yaml"),
            )

    def test_eof_in_noninteractive(self, tmp_path):
        """EOFError from input() is handled (conda run scenario)."""
        strategy_yaml = tmp_path / "strategy.yaml"
        strategy_yaml.write_text(
            "features: {}\nmodels: []\nreasoning: eof test\n",
            encoding="utf-8",
        )

        with patch("builtins.input", side_effect=EOFError):
            with patch("src.strategy.llm_strategist.Path") as MockPath:
                real_path = Path
                def path_side_effect(arg):
                    if arg == "configs/models":
                        return tmp_path / "nonexistent"
                    return real_path(arg)
                MockPath.side_effect = path_side_effect

                result = run_manual_mode(
                    {"dataset_info": {}, "target_analysis": {"dtype": "int64", "n_unique": 2,
                     "distribution": {}, "class_balance_pct": {}, "missing_pct": 0},
                     "columns": {}, "correlation_matrix": {"columns": [], "values": []},
                     "feature_clusters": [], "weak_features": [], "recommendations": []},
                    str(strategy_yaml),
                )

        assert result["reasoning"] == "eof test"

    def test_custom_eda_output_path(self, tmp_path, monkeypatch):
        """When eda_output_path is given, EDA report is saved there."""
        strategy_yaml = tmp_path / "strategy.yaml"
        strategy_yaml.write_text(
            "features: {}\nmodels: []\nreasoning: custom path\n",
            encoding="utf-8",
        )
        eda_out = tmp_path / "custom" / "my_eda.txt"

        monkeypatch.setattr("builtins.input", lambda _: "")
        with patch("src.strategy.llm_strategist.Path") as MockPath:
            real_path = Path
            def path_side_effect(arg):
                if arg == "configs/models":
                    return tmp_path / "nonexistent"
                return real_path(arg)
            MockPath.side_effect = path_side_effect

            run_manual_mode(
                {"dataset_info": {}, "target_analysis": {"dtype": "int64", "n_unique": 2,
                 "distribution": {}, "class_balance_pct": {}, "missing_pct": 0},
                 "columns": {}, "correlation_matrix": {"columns": [], "values": []},
                 "feature_clusters": [], "weak_features": [], "recommendations": []},
                str(strategy_yaml),
                eda_output_path=str(eda_out),
            )

        assert eda_out.exists()
        assert "MAESTRO-ML" in eda_out.read_text(encoding="utf-8")

    def test_default_eda_output_path(self, tmp_path, monkeypatch):
        """When eda_output_path is None, defaults to strategy_path.parent / eda_report.txt."""
        subdir = tmp_path / "results"
        subdir.mkdir()
        strategy_yaml = subdir / "strategy.yaml"
        strategy_yaml.write_text(
            "features: {}\nmodels: []\nreasoning: default eda path\n",
            encoding="utf-8",
        )

        monkeypatch.setattr("builtins.input", lambda _: "")
        with patch("src.strategy.llm_strategist.Path") as MockPath:
            real_path = Path
            def path_side_effect(arg):
                if arg == "configs/models":
                    return tmp_path / "nonexistent"
                return real_path(arg)
            MockPath.side_effect = path_side_effect

            run_manual_mode(
                {"dataset_info": {}, "target_analysis": {"dtype": "int64", "n_unique": 2,
                 "distribution": {}, "class_balance_pct": {}, "missing_pct": 0},
                 "columns": {}, "correlation_matrix": {"columns": [], "values": []},
                 "feature_clusters": [], "weak_features": [], "recommendations": []},
                str(strategy_yaml),
            )

        default_eda = subdir / "eda_report.txt"
        assert default_eda.exists()
        assert "MAESTRO-ML" in default_eda.read_text(encoding="utf-8")

    def test_validation_failure_warns_not_raises(self, tmp_path, monkeypatch, caplog):
        """Manual mode: invalid strategy → warning logged, strategy still returned."""
        strategy_yaml = tmp_path / "strategy.yaml"
        # Missing 'reasoning' key → validation would raise ValueError
        strategy_yaml.write_text(
            "features: {}\nmodels: [nonexistent_model]\n",
            encoding="utf-8",
        )

        monkeypatch.setattr("builtins.input", lambda _: "")
        models_dir = tmp_path / "configs" / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "catboost.yaml").write_text("name: CatBoost\n", encoding="utf-8")

        with patch("src.strategy.llm_strategist.Path") as MockPath:
            real_path = Path
            def path_side_effect(arg):
                if arg == "configs/models":
                    return models_dir
                return real_path(arg)
            MockPath.side_effect = path_side_effect

            import logging
            with caplog.at_level(logging.WARNING, logger="maestro"):
                result = run_manual_mode(
                    {"dataset_info": {}, "target_analysis": {"dtype": "int64", "n_unique": 2,
                     "distribution": {}, "class_balance_pct": {}, "missing_pct": 0},
                     "columns": {}, "correlation_matrix": {"columns": [], "values": []},
                     "feature_clusters": [], "weak_features": [], "recommendations": []},
                    str(strategy_yaml),
                )

        # Strategy is returned despite validation failure
        assert result["models"] == ["nonexistent_model"]
        # Warning was logged
        assert "validation warning" in caplog.text.lower() or "missing required keys" in caplog.text


# ---------------------------------------------------------------------------
# _parse_llm_response — empty / whitespace edge cases
# ---------------------------------------------------------------------------

class TestParseLlmResponseEmpty:
    def test_empty_string_raises(self):
        """Empty string parses to None via safe_load → ValueError."""
        with pytest.raises(ValueError, match="dictionary"):
            _parse_llm_response("")

    def test_whitespace_only_raises(self):
        """Whitespace-only string parses to None → ValueError."""
        with pytest.raises(ValueError, match="dictionary"):
            _parse_llm_response("   \n\n  ")


# ---------------------------------------------------------------------------
# _call_llm_api — retry then succeed
# ---------------------------------------------------------------------------

class TestCallLlmApiRetrySuccess:
    @patch("src.strategy.llm_strategist.time.sleep")
    def test_succeed_on_second_attempt(self, mock_sleep, monkeypatch):
        """First call fails, second succeeds — should return result."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="retry success")]
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            ConnectionError("transient"),
            mock_msg,
        ]

        with patch.dict("sys.modules", {"anthropic": MagicMock()}) as _:
            import sys
            mock_anthropic = sys.modules["anthropic"]
            mock_anthropic.Anthropic.return_value = mock_client

            result = _call_llm_api(
                "test", {"provider": "anthropic", "env_key": "ANTHROPIC_API_KEY"}
            )

        assert result == "retry success"
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(1)


# ---------------------------------------------------------------------------
# _validate_strategy — models=None edge case
# ---------------------------------------------------------------------------

class TestValidateStrategyModelsNone:
    def test_none_models_treated_as_empty(self):
        """strategy['models'] = None should not crash."""
        strategy = {
            "features": {},
            "models": None,
            "reasoning": "test",
        }
        assert _validate_strategy(strategy, ["catboost"]) is True


# ---------------------------------------------------------------------------
# generate_strategy — eda_output_path passthrough
# ---------------------------------------------------------------------------

class TestGenerateStrategyEdaOutputPath:
    @patch("src.strategy.llm_strategist.run_manual_mode")
    def test_eda_output_path_forwarded(self, mock_manual, sample_eda_report):
        """eda_output_path from manual config is forwarded to run_manual_mode."""
        mock_manual.return_value = {
            "features": {},
            "models": [],
            "reasoning": "ok",
        }
        config = PipelineConfig(
            strategy=StrategyConfig(
                mode="manual",
                manual={
                    "strategy_input_path": "results/strategy.yaml",
                    "eda_output_path": "results/eda.txt",
                },
            ),
        )
        generate_strategy(sample_eda_report, config)
        call_kwargs = mock_manual.call_args
        assert call_kwargs.kwargs["eda_output_path"] == "results/eda.txt"
