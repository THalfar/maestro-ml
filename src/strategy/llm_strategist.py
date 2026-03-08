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

import logging
import os
import re
import time
from pathlib import Path

import yaml

from src.utils.io import PipelineConfig

logger = logging.getLogger("maestro")


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
    mode = pipeline_config.strategy.mode

    if mode == "api":
        # Load model schemas
        model_schemas = _load_model_schemas(pipeline_config)

        prompt = _build_strategy_prompt(eda_report, model_schemas)
        response_text = _call_llm_api(prompt, pipeline_config.strategy.api)
        strategy = _parse_llm_response(response_text)
        available_models = list(model_schemas.keys())
        _validate_strategy(strategy, available_models)
        logger.info("Strategy generated via API mode.")
        return strategy

    elif mode == "manual":
        manual_cfg = pipeline_config.strategy.manual or {}
        strategy_input_path = manual_cfg.get(
            "strategy_input_path", "results/strategy.yaml"
        )
        eda_output_path = manual_cfg.get("eda_output_path")
        strategy = run_manual_mode(eda_report, strategy_input_path, eda_output_path=eda_output_path)
        logger.info("Strategy loaded from manual mode.")
        return strategy

    else:
        raise ValueError(f"Unknown strategy mode: '{mode}'. Must be 'api' or 'manual'.")


def _load_model_schemas(pipeline_config: PipelineConfig) -> dict[str, dict]:
    """Load all model YAML schemas for the models in pipeline_config."""
    from src.utils.io import load_yaml

    schemas: dict[str, dict] = {}
    configs_dir = Path("configs/models")
    for model_name in pipeline_config.models:
        yaml_path = configs_dir / f"{model_name}.yaml"
        if yaml_path.exists():
            schemas[model_name] = load_yaml(yaml_path)
        else:
            logger.warning(f"Model config not found: {yaml_path}")
    return schemas


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
    provider = config.get("provider", "anthropic")
    model_id = config.get("model", "claude-sonnet-4-6")
    env_key = config.get("env_key", "ANTHROPIC_API_KEY")
    max_tokens = int(config.get("max_tokens", 4096))

    api_key = os.environ.get(env_key)
    if not api_key:
        raise RuntimeError(
            f"API key not set. Expected environment variable: '{env_key}'"
        )

    max_retries = 3
    last_exc = None

    if provider not in ("anthropic", "openai"):
        raise ValueError(f"Unknown provider: '{provider}'")

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    else:  # openai
        import openai
        client = openai.OpenAI(api_key=api_key)

    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                message = client.messages.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text

            else:  # openai
                response = client.chat.completions.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content

        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # exponential backoff
                logger.warning(
                    f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {exc}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

    raise RuntimeError(
        f"LLM API call failed after {max_retries} attempts. Last error: {last_exc}"
    )


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
           - preprocessing (optional): scaler_choices list and per_model
             overrides for needs_scaling / scaler_choices
           - monotone_constraints (optional): dict mapping feature name to
             direction (-1, 0, 1) for catboost/xgboost/lightgbm
           - drop_columns (optional): list of column names to drop before
             feature engineering (e.g. noisy calculated features)
        5. Add guidelines:
           - Focus on narrowing search spaces, not guessing exact values.
           - Create features based on correlations and domain patterns.
           - Exclude weak features from expensive operations.
           - Consider dataset size when setting tree depth ranges.
        6. Return the assembled prompt string.
    """
    from src.eda.profiler import format_eda_for_llm

    eda_text = format_eda_for_llm(eda_report)

    # Format model schemas
    model_info_lines = []
    for model_name, schema in model_schemas.items():
        hparams = schema.get("hyperparameters", {})
        hparam_lines = []
        for param, spec in hparams.items():
            if spec.get("type") == "categorical":
                desc = f"choices={spec.get('choices')}"
            elif spec.get("type") == "dynamic_int_list":
                n_min = spec.get("n_min", 1)
                n_max = spec.get("n_max", 4)
                lo = spec.get("low")
                hi = spec.get("high")
                desc = f"{n_min}-{n_max} layers of [{lo}, {hi}]"
            else:
                lo = spec.get("low")
                hi = spec.get("high")
                log_str = " (log scale)" if spec.get("log") else ""
                desc = f"[{lo}, {hi}]{log_str}"
            hparam_lines.append(f"    {param}: {desc}")
        model_info_lines.append(
            f"  {model_name} ({schema.get('name', model_name)}):\n" + "\n".join(hparam_lines)
        )
    models_text = "\n".join(model_info_lines)

    prompt = f"""You are a machine learning strategist for a tabular data competition pipeline.
Your job is to analyze an EDA report and produce a YAML strategy that guides
feature engineering and model hyperparameter search.

IMPORTANT GUIDELINES:
- Your role is to NARROW search spaces, not to guess exact values.
- Create interaction/ratio features based on the correlations and domain patterns you observe.
- Skip weak features (|corr| < 0.05) from expensive operations.
- For large datasets (>50k rows), prefer shallower trees (depth 4-8).
- For small datasets (<5k rows), deeper trees are risky — suggest stronger regularization.
- Only include models that will add diversity; skip near-duplicate models for small datasets.

=== EDA REPORT ===
{eda_text}

=== AVAILABLE MODELS AND HYPERPARAMETER RANGES ===
{models_text}

=== YOUR TASK ===
Based on the EDA report, produce a YAML strategy with the following structure:

```yaml
features:
  interactions:
    - [col_a, col_b]    # col_a * col_b
  ratios:
    - [numerator, denominator]  # numerator / (denominator + 1e-8)
  target_encoding:
    columns: [col1, col2]  # categorical columns to target-encode
    pairs: [[col1, col2]]  # column pairs to jointly target-encode
    alpha: 15              # smoothing strength
  custom:
    - name: new_feature
      formula: "col_a * col_b / 1000"  # pandas eval expression

models:
  - catboost
  - xgboost
  - lightgbm

overrides:
  catboost:
    depth:
      low: 4
      high: 8
  xgboost:
    max_depth:
      low: 4
      high: 8

# Optional: control scaler search for linear/distance models.
# scaler_choices narrows Optuna's scaler options globally or per model.
# Use "robust" if EDA shows outliers, "quantile" if high skew.
# per_model overrides can also force scaling on/off for a specific model.
preprocessing:
  scaler_choices: ["robust", "quantile"]  # omit section if no strong signal
  per_model:
    ridge:
      scaler_choices: ["robust"]
    realmlp:
      needs_scaling: true
      scaler_choices: ["none", "robust"]

# Optional: monotone constraints for gradient boosting (catboost/xgboost/lightgbm).
# Only set when EDA shows a clear monotonic relationship (|rho| > 0.7).
# Values: 1 = increasing, -1 = decreasing, 0 = unconstrained (omit instead).
monotone_constraints:
  feature_name: 1    # example: higher value → higher target probability

# Optional: drop noisy or leaky columns before feature engineering.
drop_columns:
  - col_to_drop

reasoning: >
  Brief explanation of your choices here.
```

Only include features that make domain sense based on the correlation patterns.
Only include models that will add meaningful diversity.
Omit optional sections (preprocessing, monotone_constraints, drop_columns) when
the EDA shows no strong signal justifying them.
Return ONLY the YAML block above, with no other text before or after it.
"""
    return prompt


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
    # Try to extract from markdown code fence
    pattern = r"```(?:yaml)?\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        yaml_str = match.group(1).strip()
    else:
        # Try to parse the entire response
        yaml_str = response.strip()

    parsed = yaml.safe_load(yaml_str)

    if not isinstance(parsed, dict):
        raise ValueError(
            f"LLM response did not parse to a dictionary. Got: {type(parsed)}"
        )

    return parsed


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
    required_keys = {"features", "models", "reasoning"}
    missing = required_keys - set(strategy.keys())
    if missing:
        raise ValueError(f"Strategy missing required keys: {missing}")

    # Validate model names
    strategy_models = strategy.get("models", []) or []
    unknown = [m for m in strategy_models if m not in available_models]
    if unknown:
        raise ValueError(
            f"Strategy references unknown models: {unknown}. "
            f"Available: {available_models}"
        )

    # Validate overrides
    overrides = strategy.get("overrides", {}) or {}
    for override_model in overrides.keys():
        if override_model not in available_models:
            raise ValueError(
                f"Override references unknown model: '{override_model}'"
            )

    # Validate feature specs
    features = strategy.get("features", {}) or {}

    for pair in (features.get("interactions", []) or []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"Interaction pair must be a list of exactly 2 column names, got: {pair}"
            )

    for pair in (features.get("ratios", []) or []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"Ratio pair must be a list of exactly 2 column names, got: {pair}"
            )

    te = features.get("target_encoding", {}) or {}
    te_columns = te.get("columns", []) or []
    if not isinstance(te_columns, list) or not all(isinstance(c, str) for c in te_columns):
        raise ValueError(f"target_encoding.columns must be a list of strings, got: {te_columns}")

    for pair in (te.get("pairs", []) or []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"target_encoding pair must be a list of exactly 2 column names, got: {pair}"
            )

    for item in (features.get("custom", []) or []):
        if not isinstance(item, dict) or "name" not in item or "formula" not in item:
            raise ValueError(
                f"Custom feature must have 'name' and 'formula' keys, got: {item}"
            )

    return True


def run_manual_mode(
    eda_report: dict,
    strategy_path: str | Path,
    eda_output_path: str | Path | None = None,
) -> dict:
    """Run the human-in-the-loop manual strategy mode.

    Prints the formatted EDA report to console and waits for the user
    to save their LLM's response as a YAML file.

    Args:
        eda_report: Complete EDA report dict from Layer 1.
        strategy_path: Path where the user should save the strategy YAML
                       (the file this function reads after the user presses Enter).
        eda_output_path: Optional path to save the formatted EDA report as a
                         text file. Defaults to <strategy_path parent>/eda_report.txt.

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
    from src.eda.profiler import format_eda_for_llm
    from src.utils.io import load_yaml

    strategy_path = Path(strategy_path)
    formatted_eda = format_eda_for_llm(eda_report)

    print(formatted_eda)

    # Save EDA to file for easy copy
    eda_text_path = Path(eda_output_path) if eda_output_path else strategy_path.parent / "eda_report.txt"
    eda_text_path.parent.mkdir(parents=True, exist_ok=True)
    eda_text_path.write_text(formatted_eda, encoding="utf-8")
    print(f"\n[EDA report also saved to: {eda_text_path}]")

    prompt_file = Path("prompts/strategy_prompt.md")
    print("\n" + "=" * 60)
    print("MANUAL STRATEGY MODE")
    print("=" * 60)
    print("Instructions:")
    print(f"  1. Open the strategy prompt:  {prompt_file}")
    print(f"     Copy the full prompt into Claude / ChatGPT.")
    print(f"  2. Replace [PASTE EDA REPORT HERE] with the EDA report above")
    print(f"     (or from: {eda_text_path})")
    print(f"  3. Save the LLM's YAML response to: {strategy_path}")
    print("  4. Press Enter here when done.")
    print("=" * 60)

    try:
        input("\n>>> Press Enter when you have saved the strategy YAML... ")
    except EOFError:
        # Non-interactive context (e.g. conda run). Proceed if file exists.
        pass

    if not strategy_path.exists():
        raise FileNotFoundError(
            f"Strategy YAML not found at: {strategy_path}. "
            "Please save the LLM's YAML response to that path and run again."
        )

    strategy = load_yaml(strategy_path)

    configs_dir = Path("configs/models")
    available_models = (
        [p.stem for p in sorted(configs_dir.glob("*.yaml"))]
        if configs_dir.exists()
        else []
    )
    try:
        _validate_strategy(strategy, available_models=available_models)
    except ValueError as exc:
        logger.warning(f"Strategy validation warning: {exc}")

    return strategy
