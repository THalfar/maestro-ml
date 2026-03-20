# Post-Run Strategy Update — maestro-ml

Use this prompt after an overnight Optuna run. Paste the EDA report, the
latest round report, and the current strategy YAML. The LLM produces an
**updated** strategy YAML for the next run.

---

## Prompt (copy everything below the line)

---

You are an AI strategist reviewing a completed Optuna hyperparameter search. You receive three inputs and must produce an **updated strategy YAML** for the next round.

### Inputs

1. **EDA report** — dataset characteristics (features, distributions, baseline scores)
2. **Round report** — Optuna results: trial counts, HP convergence, TPE health, pruning, ensemble diagnostics
3. **Current strategy YAML** — the strategy that produced these results

### Decision framework

Read the round report carefully. For **each model**, determine its status and action:

| Status | Trigger (from round report) | Actions |
|--------|---------------------------|---------|
| **COLLAPSED** | Section 4b: `COLLAPSED`, param_std_ratio < 0.05 | Enable `collapse_restart`; raise `tpe.gamma` (30-50); fix all converged HPs to reduce dimensionality; consider DROP if score ceiling also reached |
| **SATURATED** | Section 2: gain_rate < 0.15, landscape smooth | Reduce budget (token run); fix ALL converged HPs; DROP if not in ensemble |
| **IMPROVING** | Section 2: gain_rate > 0.40, or best at last trial | Keep/increase budget; fix only HIGH-convergence categoricals; enable substudy if not already |
| **HEALTHY** | No collapse, gain_rate 0.15-0.40 | No changes needed; minor narrowing if Section 4 shows clear convergence |

### TPE gamma tuning

`gamma` is the fixed number of trials TPE classifies as "good" in its surrogate model.

- **Default**: `gamma: 25` — reasonable for 100-300 trial studies.
- **Increase to 30-50**: When TPE collapsed (too few "good" trials → over-exploitation). More "good" trials = broader exploration.
- **Decrease to 10-15**: When landscape is smooth, model near ceiling, and you want faster exploitation of the known good region.
- **Very large (>50)**: Essentially turns TPE into quasi-random search. Use only as a last resort for highly collapsed models.

```yaml
overrides:
  xgboost:
    optuna:
      tpe:
        gamma: 30                    # more exploration (collapsed last run)
        multivariate: true           # consider HP correlations
        n_ei_candidates: 48          # more EI candidates (default 24)
```

### collapse_restart tuning

Enable for models with >200 trials where TPE HEALTH shows COLLAPSED. The callback detects collapse during the run and injects restart trials:

- **With substudy**: Samples from substudy history (exponential distribution) — best "map" to escape local minima.
- **Without substudy**: Samples randomly from search space bounds.

```yaml
overrides:
  xgboost:
    optuna:
      collapse_restart:
        window: 20          # last N trials to check
        threshold: 0.05     # param_std_ratio threshold
        n_restart: 5        # trials to inject per event
        cooldown: 10        # min trials between restarts
        top_n: 2            # guaranteed best from substudy
        temperature: 0.3    # exp sampling sharpness (0.1=top-heavy, 0.5=diverse)
```

- `temperature: 0.3` (default) — balanced; some good, some exploratory.
- `temperature: 0.5` — more diverse restarts; better for severely collapsed models.
- `top_n: 2` — ensures quality reference points among restart trials.
- `cooldown: 10` — prevents over-restarting; increase to 15-20 for very long runs.

### Multivariate TPE and search space changes

**CRITICAL**: With `multivariate: true` (project default), you must NEVER narrow HP ranges on models with persistent DB history. Narrowing creates a "dynamic search space" → Optuna silently falls back to RandomSampler for affected params, defeating multivariate TPE entirely.

**Decision table for HP control:**

| Situation | Action | Example |
|-----------|--------|---------|
| HP fully converged (100% top-20%) | **Fix** as scalar override | `d_embedding: 12` |
| HP needs guided exploration | **Targeted enqueue** — explore specific values | `targeted_enqueue: [{param: lr, range: [0.0001, 0.001], log: true}]` |
| HP needs full range change | **Substudy reset** — re-run substudy with new ranges | `substudy: {reset: true, ...}` |
| Drastic change unavoidable | **Disable multivariate** for that model | `tpe: {multivariate: false}` |

**Targeted enqueue** generates trials that explore a specific HP range while keeping other params from the study's best trials. The search space stays unchanged → multivariate works.

```yaml
optuna:
  targeted_enqueue:
    - param: lr
      range: [0.00008, 0.0003]    # auto-generate log-spaced points
      n_points: 5
      log: true                    # log-spaced (for log-scale HPs)
      n_base: 3                    # pair with 3 top trial configs
      temperature: 0.2             # low = prefer best trials
    - param: batch_size
      values: [2000, 3000, 4000, 5000]  # explicit values
      n_base: 2
```

**Guidelines**:
- Target 1-2 HPs per round maximum (combinatorial explosion: 5 values × 3 bases = 15 trials per HP)
- Use `log: true` for log-scale HPs (learning_rate, weight_decay, etc.)
- Use low `temperature` (0.1-0.2) when you want to explore around the best configs
- Use higher `temperature` (0.3-0.5) when you want more diverse base configs
- Prefer **fixing** over **targeted enqueue** when HP is clearly converged — it's simpler and reduces search dimensionality

**Substudy reset** (`substudy.reset: true`): Deletes the old substudy DB and re-runs with current ranges. Use when you need a completely different exploration range. Substudies are cheap (~15min) so resetting is low-cost.

### HP fixing rules (from Section 4: HP CONVERGENCE)

- **Fix categorical**: 100% of top-20% AND >75% of all trials use same value → safe to fix as scalar override.
- **Fix numeric**: top-20% range < 10% of full range AND >100 trials completed → fix to best trial's value.
- **Narrow numeric range**: top-20% range < 30% of full range → narrow search bounds in override.
- **Never fix**: HPs with LOW convergence or high separation score (these are still being explored productively).

### Budget allocation

- Total budget from pipeline YAML `model_timeouts`.
- **Prioritize**: improving models > diverse/ensemble-contributing models > saturated models.
- **DROP**: gain_rate=0.00 AND excluded from ensemble (0 arrays selected by NSGA-II).
- **Token budget (15-20min)**: For ceiling models — lets TPE do a few exploratory trials without serious investment.

### Substudy considerations

- Enable substudy for slow models (>3 min/trial) that are still improving.
- `sample_fraction: 1.0` with `n_folds: 2` gives ~5x speedup per trial.
- Lock scaler for models that benefit from it; disable for models with internal preprocessing (RealMLP, TabM).

### Ensemble health (Section 8-9)

- N_eff < 1.5: severe dominance — prioritize diversity (different model families, collapse_restart for diverse exploration).
- N_eff 1.5-3.0: moderate — keep current mix.
- N_eff > 3.0: good diversity — focus on score improvement.
- If meta-learner (Section 9) beat blend: keep meta comparison active.

### Available Optuna/TPE settings (per model via `overrides.<model>.optuna`)

```yaml
optuna:
  # TPE sampler configuration
  tpe:
    gamma: 25                     # fixed "good" trials count
    n_startup_trials: 0           # random trials before TPE
    n_ei_candidates: 24           # expected improvement candidates
    multivariate: true            # project default: always consider HP correlations
    consider_prior: true          # use prior distribution
    consider_magic_clip: true     # Optuna internal clipping
    consider_endpoints: false     # include endpoints in EI
    constant_liar: false          # parallel-friendly TPE
    group: false                  # group-decomposed TPE

  # Collapse detection and restart injection (opt-in)
  collapse_restart:
    window: 20
    threshold: 0.05
    n_restart: 5
    cooldown: 10
    top_n: 2
    temperature: 0.3

  # Substudy warm-start (opt-in)
  substudy:
    enabled: true
    sample_fraction: 0.10
    n_folds: 3
    timeout: "15m"
    n_enqueue: 20
    top_n: 3
    temperature: 0.3
    lock_scaler: true

  # Targeted HP exploration (multivariate-safe)
  targeted_enqueue:
    - param: lr
      range: [0.0001, 0.001]
      n_points: 5
      log: true
      n_base: 3
      temperature: 0.2

  # Manual trial injection
  enqueue_trials:
    - max_depth: 6
      learning_rate: 0.03

  # Tracker diversity (per_fold mode only)
  tracker:
    diversity_mode: tiered
    tier1_size: 5
    tier2_corr_threshold: 0.99

  # Diversity pruning (per_fold mode only)
  diversity_pruning:
    corr_threshold: 0.995
    warmup_entries: 5
    n_consecutive: 2
    score_tolerance: 0.001
```

### Output format

Return the **complete updated strategy YAML** (same schema as input). Add comments explaining each change with reference to round report data. For example:

```yaml
overrides:
  xgboost:
    # R5: TPE COLLAPSED at trial ~252 (param_std_ratio=0.02) — enable restart
    # R5: raise gamma 25→40 for broader exploration
    optuna:
      tpe:
        gamma: 40
      collapse_restart:
        window: 20
        threshold: 0.05
        n_restart: 5
        temperature: 0.3
```

---

**Now analyze the round report and strategy YAML below and produce your updated strategy:**

[PASTE EDA REPORT HERE]

[PASTE ROUND REPORT HERE]

[PASTE CURRENT STRATEGY YAML HERE]
