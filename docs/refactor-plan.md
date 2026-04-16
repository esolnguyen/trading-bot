# Refactor Plan — 500-line cap, structure cleanup, dedup

## Goal

Bring every Python file under 500 lines, split oversized modules into focused folders,
eliminate duplicated helpers, and ensure services / utils sit in the correct layer.
No behavioral changes — this refactor is purely structural.

## Scope

11 files currently exceed 500 lines. Total to restructure: ~8,000 lines across
the worst offenders.

| Lines | File |
|---|---|
| 1303 | `src/services/trading/trading_loop.py` |
| 1109 | `src/infrastructure/storage/vector_memory.py` |
|  892 | `src/services/analysis/indicators/base/technical_indicators.py` |
|  879 | `dashboard.py` |
|  822 | `src/core/config/settings.py` |
|  753 | `src/services/trading/brain_service.py` |
|  674 | `src/services/trading/executor.py` |
|  615 | `src/infrastructure/ai/llm_manager.py` |
|  595 | `src/services/analysis/indicators/momentum/momentum_indicators.py` |
|  561 | `src/services/trading/trading_strategy.py` |
|  550 | `src/app.py` |

## Guiding Rules

1. **Keep public imports stable.** Each split folder gets an `__init__.py` that
   re-exports the original symbols so callers like
   `from src.services.trading.trading_loop import TradingLoop` keep working.
   Tests do not move.
2. **One class or one cohesive module per file**, ≤400 lines target (hard cap 500).
3. **Folder-per-split** when breaking up a file: `foo.py` → `foo/` package with
   `core.py`, `helpers.py`, etc.
4. **Run `pytest` after each phase**; one split = one commit.
5. **No behavior changes in this refactor** — pure structural moves + dedup.

---

## Per-File Splits

### 1. `trading_loop.py` (1303 → folder `src/services/trading/loop/`)

`TradingLoop` has 26 methods grouped into 5 concerns. Extract helper classes that
are injected into `TradingLoop` so the orchestrator stays thin.

- `loop/orchestrator.py` — `TradingLoop.__init__`, `run`, `stop`,
  `run_cycle_once`, `trigger_immediate_cycle`
- `loop/snapshot_collector.py` — `_collect_snapshots`, `_collect_patterns`,
  `_collect_htf_analyses`, `_build_ml_context`, `_build_rag_context_per_symbol`
- `loop/decision_router.py` — `_decide_per_symbol`, `_decide_all_symbols_single_call`,
  `_decide`, `_build_balance_context`, `_get_snapshot_price`
- `loop/position_manager.py` — `_update_open_positions`,
  `_reconcile_open_positions`, `_format_position_context`,
  `_sync_position_from_strategy`
- `loop/position_monitor.py` — `run_position_monitor`,
  `_monitor_check_positions`, `_monitor_execute_close`,
  `_update_loss_tracking`, `_attach_pnl`, `_check_slippage`
- `loop/regime.py` — `_refresh_regime`

### 2. `vector_memory.py` (1109 → folder `src/infrastructure/storage/vector_memory/`)

- `vector_memory/service.py` — init, `store_experience`,
  `retrieve_similar_experiences`, `get_context_for_prompt`
- `vector_memory/rules.py` — `store_semantic_rule`, `get_active_rules`,
  `get_relevant_rules`, `get_anti_patterns_for_prompt`, `semantic_rule_count`
- `vector_memory/statistics.py` — `compute_confidence_stats`,
  `compute_adx_performance`, `compute_factor_performance`,
  `get_confidence_recommendation`
- `vector_memory/thresholds.py` — `compute_optimal_thresholds`,
  `_learn_position_size_threshold`, `_learn_confluence_thresholds`,
  `_learn_alignment_thresholds`
- `vector_memory/internal.py` — `_sanitize_metadata`,
  `_calculate_recency_score`, `_generate_synthetic_insight`,
  `_get_trade_metadatas`

### 3. `technical_indicators.py` (892 → folder `src/services/analysis/indicators/base/`)

Keep the `TechnicalIndicators` facade as the public entry point; move math
helpers to category modules. Free functions that take `(ohlcv, ...)` are
preferred so the facade stays thin.

- `indicators/base/facade.py` — class shell + data-loading
  (`get_data`, `open/high/low/...` properties)
- `indicators/base/oscillators.py` — rsi, macd, stochastic, roc, momentum,
  williams_r, tsi, rmi, ppo, uo, kst, coppock
- `indicators/base/moving_averages.py` — ema, sma, ewma
- `indicators/base/statistics.py` — kurtosis, skew, stdev, variance, zscore,
  mad, quantile, entropy, hurst, linreg
- `indicators/base/levels.py` — support/resistance variants, fib, pivots
- `indicators/base/trend.py` — adx, supertrend, ichimoku, parabolic_sar,
  vortex, trix
- `indicators/base/volatility.py` — atr, bollinger, chandelier, keltner,
  donchian, choppiness
- `indicators/base/volume.py` — cci, mfi, obv (+slope), pvt, cmf, ad,
  force_index, eom, volume_profile, vwap, twap

**Dedup note:** `momentum_indicators.py` (numba) already hosts the hot-path
implementations — remove any redundant pure-Python copies from
`technical_indicators.py`.

### 4. `dashboard.py` (879 → folder `src/dashboard/`, root shim kept)

- `dashboard/__main__.py` — `st.set_page_config`, nav/layout, main render
- `dashboard/data_loaders.py` — `load_cycle_logs`, `load_trades`,
  `load_position`, `load_statistics`, `load_trade_history`,
  `load_closed_trades`, `load_ohlcv`, `load_api_costs`, `available_symbols`
- `dashboard/panels/statistics.py` — `STAT_META`, `_fmt_stat`, `_stat_color`,
  statistics panel render
- `dashboard/panels/position.py` — `_position_gauge`, position panel
- `dashboard/panels/trades.py` — trades table, `_color_decision`, `_color_pnl`
- `dashboard/panels/logs.py` — cycle logs panel
- `dashboard/panels/charts.py` — OHLCV / chart rendering

### 5. `settings.py` (822 → files inside `src/core/config/`)

- `config/settings.py` — the `Settings` dataclass only (fields +
  `bot_enabled`, `bot_dry_run`, `use_signal_scorer`, `effective_*` properties)
- `config/parsers.py` — `_parse_bool`, `_parse_int`, `_parse_float`,
  `_parse_list`, `_parse_trading_engine`, `_parse_bot_mode`
- `config/loader.py` — `Settings.from_env` (kept as a classmethod shim that
  calls a free function here)
- `config/validation.py` — `_validate_required_fields`, `_validate_ranges`,
  `__post_init__` helper
- `config/models.py` — `get_model_config` (provider-specific model config
  assembly)

### 6. `brain_service.py` (753 → folder `src/services/trading/brain/`)

- `brain/service.py` — `TradingBrainService.__init__`,
  `update_from_closed_trade`, `get_context`, `get_parameter_suggestions`,
  `get_dynamic_thresholds`
- `brain/context_builder.py` — `_build_rich_context_string`, `get_vector_context`
- `brain/reflection.py` — `_trigger_reflection`, `_trigger_loss_reflection`
- `brain/tracking.py` — `track_position_update`, `_extract_factor_scores`,
  `_count_strong_confluences`, `_count_patterns`
- `brain/cache.py` — `_get_cached_stats`

### 7. `executor.py` (674 → folder `src/services/trading/execution/`)

- `execution/executor.py` — class + `execute`, lifecycle (`initialize`,
  `close`, `__aenter__`, `__aexit__`)
- `execution/bracket_orders.py` — `place_bracket_orders`,
  `cancel_bracket_orders`, `_place_futures_bracket`, `_await_limit_fill`
- `execution/filters.py` — `_cache_filters`, `_get_step_size`,
  `_get_market_step_size`, `_get_tick_size`, `_prewarm_filters`,
  `_format_quantity`, `_format_price`
- `execution/leverage.py` — `_apply_leverage`, `get_live_position_size`,
  `_is_demo_fapi`
- `execution/client_factory.py` — `_ensure_api_client`,
  `_default_api_client_factory`, `_extract_price`

### 8. `llm_manager.py` (615 → folder `src/infrastructure/ai/llm/`)

- `llm/manager.py` — `LLMManager.__init__`, `decide`, `send_prompt`,
  `send_prompt_with_chart_analysis`, `close`
- `llm/azure_backend.py` — `_decide_azure`, `_get_azure_client`,
  `_complete_azure`, `_normalize_azure_openai_endpoint`,
  `_should_use_azure_foundry_anthropic_client`, `_AnthropicResponseAdapter`
- `llm/multi_provider_backend.py` — `_decide_multi_provider`,
  `_get_orchestrator`, `_get_unified_parser`, `_process_orchestrator_result`
- `llm/decision_parser.py` — `_parse_decision`, `_parse_decision_from_dict`,
  `_fallback_decision`, `_strip_markdown_fence`, `ParseError`, `ResponseShapeError`
- `llm/messages.py` — `_build_messages`, `_build_log_prompt`, `_to_b64`,
  `_truncate_for_log`, `_extract_content`, `_extract_usage`, `_is_retryable_error`

### 9. `momentum_indicators.py` (595 → files inside `src/services/analysis/indicators/momentum/`)

Split numba helpers by family:

- `momentum/rsi.py` — `rsi_numba`, `detect_rsi_divergence`
- `momentum/macd.py` — `macd_numba`, `ppo_numba`
- `momentum/stochastic.py` — `stochastic_numba`, `williams_r_numba`
- `momentum/trend.py` — `tsi_numba`, `coppock_curve_numba`
- `momentum/composite.py` — `uo_numba`, `kst_numba`, `_uo_numba`,
  `UltimateOscillatorConfig`
- `momentum/simple.py` — `roc_numba`, `momentum_numba`, `rmi_numba`,
  `calculate_relative_strength_numba`
- `momentum/_njit.py` — numba fallback shim (dedup target — used across all
  indicator modules)

### 10. `trading_strategy.py` (561 → folder `src/services/trading/strategy/`)

- `strategy/strategy.py` — init + `check_position`, `close_position`,
  `reconcile`, `get_position_context`
- `strategy/trailing.py` — `_update_trailing_stop`, `_trigger_partial_tp`,
  `_update_position_parameters`
- `strategy/conditions.py` — `_build_conditions_from_position`

### 11. `app.py` (550 → split into `src/bootstrap/` + thin entry)

- `bootstrap/logging.py` — `_attach_timeframe_file_handler`
- `bootstrap/services_factory.py` — all `_try_build_*` helpers
- `bootstrap/runtime.py` — `build_runtime`, `_close_runtime`, `_run_guarded`
- `app.py` — thin entry: `main`, `run`, top-level wiring only (target <150 lines)

---

## Cross-Cutting Dedup

Checked during execution, confirmed hot spots:

- **Numba fallback `njit` shim** — redefined in each `*_indicators.py`. Move to
  `src/services/analysis/indicators/_numba_compat.py` and import from there.
- **Symbol slug helper** — `symbol.replace("/", "").replace(":", "").lower()`
  appears in `persistence.py` at least 5x. Extract to
  `src/shared/symbol_utils.py::slugify_symbol`.
- **JSON read/write helpers** — `_read_json` / `_write_json` duplicated in
  `persistence.py` and `dashboard.py`. Move to `src/shared/json_io.py`.
- **`_parse_bool`** — currently in `settings.py`; also re-implemented in
  `app.py` for `LOGGER_DEBUG`. Reuse from `config.parsers`.
- **Executor simulated-outcome** — same block appears twice in `executor.py`
  (lines 62 and 88). Extract `_simulated_outcome(decision, dry_run=True)`.
- **Per-symbol JSON path builders** — `position_{slug}.json`,
  `trade_history_{slug}.json`, `statistics_{slug}.json`, `last_response_{slug}.json`,
  `last_analysis_{slug}.json` in `persistence.py` — collapse into
  `_slug_path(kind: str, symbol: str) -> Path`.

## Folder Hygiene

Current layout (domain / services / infrastructure / interfaces / shared / core)
is sound. Issues to fix:

- **`dashboard.py` at repo root** — move internals to `src/dashboard/`. Leave
  a thin `dashboard.py` at root as the Streamlit entry point so
  `streamlit run dashboard.py` still works.
- **`scripts/backtest.py` and `scripts/retrain_all.py`** — keep in `scripts/`
  (they are CLI entry points) but verify they only import from `src/services/`
  and don't duplicate service code.
- **`src/interfaces/notifiers/filehandler_components/`** — this pattern
  (subpackage for helper components) is a good template; reuse it for the
  splits above.

---

## Phasing

One PR per phase. `pytest` must be green before moving to the next.

| Phase | Scope | Risk |
|---|---|---|
| 1 | Dedup pass: `symbol_utils`, `json_io`, `_numba_compat`, `_simulated_outcome`, `_slug_path` | Low |
| 2 | Split `app.py` → `bootstrap/` | Low (startup-only code) |
| 3 | Split `settings.py` → files inside `core/config/` | Medium (wide import surface) |
| 4 | Split `dashboard.py` → `src/dashboard/` | Low (no tests depend on it) |
| 5 | Split `llm_manager.py`, `executor.py`, `trading_strategy.py` | Medium |
| 6 | Split `brain_service.py`, `vector_memory.py` | Medium |
| 7 | Split `technical_indicators.py`, `momentum_indicators.py` | Low (mostly pure functions) |
| 8 | Split `trading_loop.py` — biggest, do last with full test suite | High |

Each phase keeps public symbols exported from the old module path via
`__init__.py` re-exports. Tests continue referencing
`from src.services.trading.trading_loop import TradingLoop` unchanged.

---

## Preconditions

1. **Green baseline.** `tests/test_dry_trading_cycle.py::test_dry_trading_cycle_logs_without_real_trade`
   currently fails with `FakeAggregator.snapshot() got an unexpected keyword
   argument 'timeframe'`. Fix this unrelated test first so the refactor has a
   reliable pass/fail signal.
2. **Confirm `dashboard.py` placement** — root shim preserved, internals moved
   (recommended), vs. full move to `src/dashboard/__main__.py`.
3. **Commit cadence** — one logical split per commit, phase-level PRs.

## Acceptance Criteria

- No Python file under `src/`, `scripts/`, or at repo root exceeds 500 lines.
- `pytest` passes at each phase boundary.
- No public import paths break (old `from src.x.y import Z` still resolves).
- No duplicated helpers remain from the dedup list above.
- Each split folder has a clear `__init__.py` defining its public surface.
