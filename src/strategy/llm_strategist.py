"""
LLM Strategist — Layer 2 of the Maestro pipeline.

Takes the EDA report from Layer 1 and produces a strategy YAML that
controls Layer 3 (feature engineering, model selection, search space
overrides). Supports two modes:

- API mode: Calls Claude/OpenAI API automatically.
- Manual mode: Prints the EDA report for the user to paste into an
  LLM chat, then waits for the user to save the response as YAML.

The LLM's job is to NARROW the search space — not to do ML, but to
make informed decisions about which features to engineer and which
hyperparameter ranges to focus on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import PipelineConfig


def generate_strategy(
    eda_report: dict,
    pipeline_config: PipelineConfig,
) -> dict:
    """Generate a strategy YAML by dispatching to API or manual mode.

    This is the main entry point for Layer 2. Based on the strategy.mode
    setting in pipeline_config, it either calls the LLM API or enters
    manual (human-in-the-loop) mode.

    Args:
        eda_report: Complete EDA report dict from Layer 1.
        pipeline_config: Pipeline configuration with strategy settings.

    Returns:
        Strategy dictionary containing:
        - features: dict with interactions, ratios, target_encoding, custom
        - models: list of model names to use
        - overrides: dict of per-model hyperparameter range overrides
        - reasoning: string explaining the LLM's choices

    Raises:
        ValueError: If strategy mode is not 'api' or 'manual'.
        RuntimeError: If API call fails after retries.

    Steps:
        1. Check pipeline_config.strategy.mode.
        2. If 'api':
           a. Load model schemas from configs/models/*.yaml.
           b. Call _build_strategy_prompt with EDA report and schemas.
           c. Call _call_llm_api with the prompt.
           d. Call _parse_llm_response to extract strategy dict.
           e. Call _validate_strategy to ensure the strategy is valid.
           f. Return the strategy dict.
        3. If 'manual':
           a. Call run_manual_mode.
           b. Return the strategy dict.
        4. Otherwise raise ValueError.
    """
    raise NotImplementedError


def _call_llm_api(prompt: str, config: dict) -> str:
    """Call the LLM API (Anthropic or OpenAI) and return the response text.

    Args:
        prompt: The complete prompt string to send to the LLM.
        config: API configuration dict with keys:
                - provider: 'anthropic' or 'openai'
                - model: model identifier string
                - env_key: environment variable name holding the API key
                - max_tokens: maximum response tokens

    Returns:
        Raw response text from the LLM.

    Raises:
        RuntimeError: If the API key environment variable is not set.
        RuntimeError: If the API call fails after 3 retries.

    Steps:
        1. Read the API key from os.environ using config['env_key'].
        2. If provider == 'anthropic':
           a. Import anthropic, create client.
           b. Call client.messages.create with the model, max_tokens,
              and a single user message containing the prompt.
           c. Extract and return the text content.
        3. If provider == 'openai':
           a. Import openai, create client.
           b. Call client.chat.completions.create with the model
              and a user message.
           c. Extract and return the response text.
        4. Wrap in retry logic: up to 3 attempts with exponential backoff.
    """
    raise NotImplementedError


def _build_strategy_prompt(
    eda_report: dict,
    model_schemas: dict[str, dict],
) -> str:
    """Build the complete prompt for the LLM strategist.

    The prompt instructs the LLM to analyze the EDA report and produce
    a YAML strategy. It includes the EDA summary, available model schemas
    with their hyperparameter ranges, and explicit instructions for the
    output format.

    Args:
        eda_report: Complete EDA report dict.
        model_schemas: Dict of {model_name: model_config_dict} for all
                       available models.

    Returns:
        Complete prompt string ready to send to the LLM.

    Steps:
        1. Start with a system-level instruction explaining the LLM's role
           as a machine learning strategist.
        2. Include the formatted EDA report (use format_eda_for_llm).
        3. List available models with their hyperparameter ranges from
           the model schemas.
        4. Specify the exact YAML output format expected:
           - features: interactions, ratios, target_encoding, custom
           - models: which models to include
           - overrides: per-model hyperparameter range adjustments
           - reasoning: explanation of choices
        5. Add guidelines:
           - Focus on narrowing search spaces, not guessing exact values.
           - Create features based on correlations and domain patterns.
           - Exclude weak features from expensive operations.
           - Consider dataset size when setting tree depth ranges.
        6. Return the assembled prompt string.
    """
    raise NotImplementedError


def _parse_llm_response(response: str) -> dict:
    """Parse the LLM's text response into a strategy dictionary.

    The LLM is expected to return a YAML block (possibly wrapped in
    markdown code fences). This function extracts and parses it.

    Args:
        response: Raw text response from the LLM.

    Returns:
        Parsed strategy dictionary.

    Raises:
        ValueError: If no valid YAML block is found in the response.
        yaml.YAMLError: If the extracted YAML is malformed.

    Steps:
        1. Look for YAML content between ```yaml and ``` markers.
        2. If found, extract the content between the markers.
        3. If not found, try to parse the entire response as YAML.
        4. Parse the YAML string with yaml.safe_load.
        5. Validate that the result is a dictionary with expected keys.
        6. Return the parsed dictionary.
    """
    raise NotImplementedError


def _validate_strategy(
    strategy: dict,
    available_models: list[str],
) -> bool:
    """Validate that a strategy dictionary is well-formed and references
    only available models.

    Args:
        strategy: Parsed strategy dictionary from the LLM.
        available_models: List of model names that have YAML configs.

    Returns:
        True if the strategy is valid.

    Raises:
        ValueError: If the strategy references unknown models.
        ValueError: If required keys are missing.
        ValueError: If feature specifications are malformed (e.g.,
                    interaction pairs are not length-2 lists).

    Steps:
        1. Check that 'features', 'models', 'reasoning' keys exist.
        2. Verify all model names in strategy['models'] are in
           available_models.
        3. If 'overrides' exists, verify each key is a valid model name.
        4. Validate feature specifications:
           - interactions: each must be a list of exactly 2 column names.
           - ratios: each must be a list of exactly 2 column names.
           - target_encoding: columns must be a list of strings.
           - custom: each must have 'name' and 'formula' keys.
        5. Return True if all checks pass.
    """
    raise NotImplementedError


def run_manual_mode(
    eda_report: dict,
    output_path: str | Path,
) -> dict:
    """Run the human-in-the-loop manual strategy mode.

    Prints the formatted EDA report to console and waits for the user
    to save their LLM's response as a YAML file.

    Args:
        eda_report: Complete EDA report dict from Layer 1.
        output_path: Path where the user should save the strategy YAML.

    Returns:
        Parsed strategy dictionary from the user-provided YAML file.

    Steps:
        1. Format the EDA report using format_eda_for_llm.
        2. Print the formatted report to console with clear headers.
        3. Also save the formatted report to the eda_output_path
           configured in the pipeline (so user can copy from file).
        4. Print instructions telling the user to:
           a. Copy the EDA report above.
           b. Paste it into Claude/ChatGPT.
           c. Save the LLM's YAML response to output_path.
           d. Press Enter when done.
        5. Wait for user input (input()).
        6. Load and parse the YAML file from output_path.
        7. Validate the strategy with _validate_strategy.
        8. Return the parsed strategy dict.
    """
    raise NotImplementedError
