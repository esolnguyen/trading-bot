# Config split — migration plan

Rolling plan for finishing the config refactor started during the
MCP-first migration. Each phase leaves the tree in a shippable state;
do not start a phase until the previous one is merged.

## Target architecture

Three top-level folders under `src/` own their own config:

```
src/
  config/                  # cross-cutting sub-settings (shared by ≥2 processes)
    __init__.py
    binance.py             # BinanceSettings           [done]
    storage.py             # StorageSettings           [done]
  mcp_servers/             # MCP runtime (reads: feed, indicators, models)
  enrich_knowledge/        # write path (Chroma ingestion, ML training)
    config/                # IngestionSettings, MLTrainingSettings
  trading_bot/             # trading loop runtime
    config/                # TraderSettings, LLMProviderSettings, DiscordSettings
```

Rule: if a sub-setting is used by exactly one folder, it lives under
that folder's `config/`. If it is used by two or more, it is promoted
to `src/config/`. Today that's `BinanceSettings` (MCP + trading_bot)
and `StorageSettings` (MCP + enrich_knowledge + trading_bot).

Sibling CLAUDE.md invariant: the "two folders under src/" rule must be
updated to reflect the three-folder reality + `src/config/` before
this refactor lands. Do it in the same commit that deletes the
monolithic Settings (Phase 5) so the docs never lie about current
state.

## Status

- [x] Phase 0 — scaffold `src/config/` with `BinanceSettings` +
      `StorageSettings`, wire MCP-side consumers.
- [ ] Phase 1 — extract `enrich_knowledge` sub-settings.
- [ ] Phase 2 — extract `trading_bot` sub-settings.
- [ ] Phase 3 — migrate remaining `Settings` callers off the monolith.
- [ ] Phase 4 — delete `src/mcp_servers/config/` (the legacy monolith).
- [ ] Phase 5 — housekeeping: update CLAUDE.md, fix skills path bug.

## Phase 0 — scaffold [done]

Done in this session:

- `src/config/__init__.py` re-exports `BinanceSettings`,
  `StorageSettings`.
- `src/config/binance.py` — pydantic-settings with per-field `alias=`
  matching legacy env var names (`BINANCE_API_KEY`, …). Field names
  keep the `binance_` prefix so the `_HasBinanceCreds` Protocol in
  `src/mcp_servers/shared/infrastructure/binance/client.py` accepts it
  unchanged.
- `src/config/storage.py` — `chroma_path`, `data_dir`, `log_dir`, plus
  `ohlcv_csv_path(symbol, interval)` moved off the monolith.
- `src/mcp_servers/base.py` — added `get_binance_settings()`,
  `get_storage_settings()`; kept `get_settings()` until Phase 4.
- All 5 MCP services migrated: `binance`, `analysis`, `rag`, `ml`
  take narrow sub-settings; `skills` takes none.

Smoke test (run after any config change):

```bash
python -c "
from src.mcp_servers.binance.server import mcp as b
from src.mcp_servers.analysis.server import mcp as a
from src.mcp_servers.rag.server import mcp as r
from src.mcp_servers.ml.server import mcp as m
from src.mcp_servers.skills.server import mcp as s
print(b.name, a.name, r.name, m.name, s.name)
"
```

## Phase 1 — enrich_knowledge sub-settings

Create `src/enrich_knowledge/config/` with two dataclasses. Keep
`src/config/storage.py` as the owner of `chroma_path` and `data_dir` —
enrich_knowledge composes them, does not re-declare.

### `IngestionSettings` (Chroma writer pipeline)

```
cryptocompare_api_key           CRYPTOCOMPARE_API_KEY      required
coingecko_api_key               COINGECKO_API_KEY
news_interval                   NEWS_INTERVAL              900
macro_interval                  MACRO_INTERVAL             1800
ohlcv_interval                  OHLCV_INTERVAL             3600
rag_update_interval_hours       RAG_UPDATE_INTERVAL_HOURS  4
rag_categories_update_interval_hours   ...                 24
rag_coingecko_update_interval_hours    ...                 24
rag_defillama_update_interval_hours    ...                 0.25
rag_news_limit                  RAG_NEWS_LIMIT             5
rag_article_max_tokens          RAG_ARTICLE_MAX_TOKENS     256
rag_density_penalty_threshold   ...                        300
rag_density_boost_threshold     ...                        1000
rag_density_penalty_multiplier  ...                        0.5
rag_density_boost_multiplier    ...                        1.2
rag_cooccurrence_multiplier     ...                        1.5
```

### `MLTrainingSettings`

```
ml_timeframe                    ML_TIMEFRAME               4h
trading_symbols                 TRADING_SYMBOLS            BTCUSDT,ETHUSDT
candle_limit                    CANDLE_LIMIT               200
```

(`trading_symbols` is shared with `trading_bot` — promote to
`src/config/` only if a second writer needs it. For now duplicate:
ingestion wants the full fleet, trading wants the subset it trades.)

### Composition

```python
# src/enrich_knowledge/config/__init__.py
from dataclasses import dataclass
from src.config import StorageSettings
from .ingestion import IngestionSettings
from .ml_training import MLTrainingSettings

@dataclass(frozen=True)
class EnrichKnowledgeSettings:
    storage: StorageSettings
    ingestion: IngestionSettings
    ml_training: MLTrainingSettings

    @classmethod
    def load(cls) -> "EnrichKnowledgeSettings":
        return cls(StorageSettings(), IngestionSettings(),
                   MLTrainingSettings())
```

### Done when

- `python -c "from src.enrich_knowledge.config import EnrichKnowledgeSettings; EnrichKnowledgeSettings.load()"` succeeds against `.env`.
- No enrich_knowledge code imports from
  `src.mcp_servers.config.settings`.

### `src/enrich_knowledge/` layout

Config is only the scaffolding — the folder's real job is to own the
**write path** (Chroma ingestion + ML training) that today lives
under `src/legacy/services/rag/` and `scripts/train_*.py`. Target:

```
src/enrich_knowledge/
  config/                   # sub-settings (above)
  rag_ingestion/            # Chroma writers
    __init__.py
    loop.py                 # was: legacy/services/rag/ingestion_loop.py
    filter.py               # was: legacy/services/rag/filter.py
    memory_manager.py       # was: legacy/services/rag/memory_manager.py
    ohlcv_writer.py         # was: legacy/services/rag/ohlcv_writer.py
    sources/                # was: legacy/services/rag/sources/
      cryptocompare.py
      coingecko.py
      defillama.py
      alternative_me.py
      ohlcv_history.py
  ml_training/              # model fit drivers (predict stays in mcp_servers/ml/)
    __init__.py
    anomaly.py              # was: scripts/train_anomaly.py
    direction.py            # was: scripts/train_direction.py
    outcome.py              # was: scripts/train_outcome.py
    regime.py               # was: scripts/train_regime.py
    key_levels.py           # was: scripts/fit_key_levels.py
    retrain_all.py          # was: scripts/retrain_all.py
  runners/                  # CLI entrypoints (thin wrappers)
    __init__.py
    run_ingestion.py        # long-running rag writer
    run_training.py         # retrain_all driver
```

### Read/write asymmetry — Chroma + OHLCV

`ChromaStore` and the OHLCV CSV helpers are used on both sides of the
system (MCP reads, enrich_knowledge writes). Two options:

1. **Import across the boundary** — enrich_knowledge imports
   `ChromaStore` from `src.mcp_servers.rag.storage`. The CLAUDE.md
   rule ("never import trading-loop code from MCP") is one-directional;
   enrich → MCP is not forbidden. Cheap, lowest churn, but leaves the
   Chroma wrapper misfiled under `mcp_servers/` as soon as a 3rd
   consumer shows up.
2. **Promote to `src/storage/`** — a sibling of `src/config/` with
   `ChromaStore` + OHLCV helpers. MCP and enrich_knowledge both
   import from there. Cleaner long term, but adds a 4th top-level
   folder, which conflicts with the current architectural rule.

Go with option 1 for Phase 1. Revisit in Phase 5 if trading_bot
starts reading Chroma directly (it shouldn't — it should hit
bot-mcp-rag). Drop a comment at the top of
`mcp_servers/rag/storage/chroma_store.py` noting that
enrich_knowledge is a legitimate consumer so the next reader doesn't
try to enforce the boundary.

### ML training vs inference boundary

- **Model class** (`mcp_servers/ml/services/anomaly.py` etc.) owns
  both `.fit()` and `.predict()` — one class, two entrypoints.
- **Training driver** (`enrich_knowledge/ml_training/anomaly.py`)
  imports the model class, orchestrates data load → fit → persist to
  disk via the shared `model_store`.
- **Inference** stays in the MCP handler — loads the persisted model,
  calls `.predict()`.

This split keeps the heavy numba/sklearn imports out of the MCP fast
path at startup: the MCP server imports the class but only touches
sklearn when `.predict()` runs.

### Done when (full folder)

- `python -m src.enrich_knowledge.runners.run_training --help` works.
- `python -m src.enrich_knowledge.runners.run_ingestion --help` works.
- No module in `src.legacy.services.rag` or `src.legacy.services.ml`
  is imported by anything outside `legacy/` — at which point that
  subtree can be deleted in Phase 4.
- `scripts/train_*.py` and `scripts/backfill_ohlcv.py` are thin
  wrappers that call into `src.enrich_knowledge.*` (or get deleted
  entirely in favour of the `runners/` entrypoints).

## Phase 1.5 — Triggering + pipeline invariants

How enrich_knowledge actually runs. Locked in before Phase 1 code
lands so every source writes to the same contract.

### Three-layer pipeline shape

```
src/enrich_knowledge/rag_ingestion/
  sources/       # fetch + parse raw (one file per upstream API)
  transforms/    # pure fns: raw → chunked/scored records (no I/O)
  writers/       # load into Chroma / CSV (idempotent upsert only)
```

`sources/` owns rate limits + retries. `transforms/` is pure (unit
test without network). `writers/` owns the idempotency key. Each
layer testable in isolation.

### Invariants every source must honour

- **Idempotent by natural key.** News → hash of URL. OHLCV →
  `(symbol, timeframe, timestamp)`. Upsert, never blind append. Safe
  to re-run any job.
- **Resumable.** Each source persists `last_fetched_at` (Chroma
  metadata or a small state JSON under `data/ingestion_state/`).
  Restarts continue, don't duplicate.
- **Fragment-level failure.** One broken source logs + skips; the
  scheduler keeps running others. A CoinGecko 429 must not halt the
  news writer.
- **Dry-run flag on every runner.** Prints what it *would* write,
  exits clean. Mandatory for onboarding a new source safely.
- **No in-place model overwrites.** Training writes
  `data/models/<name>_<ts>.pkl` and flips a `current` symlink after
  validation. Rollback = `ln -sf`.

### Two runners, one library

- `runners/run_ingestion.py` — long-running loop, APScheduler with
  per-source intervals from `config/schedule.py` (news=15m,
  macro=30m, ohlcv=1h). One process, many jobs.
- `runners/run_training.py` — batch CLI, exits when done. Flags:
  `--model {all|anomaly|direction|outcome|regime|key_levels}`,
  `--dry-run`.

Both runners are thin shells over the library code; everything
testable without the scheduler.

### Scheduler choice: APScheduler

- Misfire handling + jitter + coalescing out of the box — replaces
  hand-rolled `time.sleep` loops.
- Schedule definitions are *code* (`config/schedule.py`), versioned
  with the pipeline. Intervals are logic, not secrets; do not move
  them into `.env`.

### Production trigger: systemd (preferred) or Docker+cron

Recommended for this project: **systemd**. Linux host, no extra
runtime, logs go to journald for free.

```ini
# /etc/systemd/system/bot-ingest.service
[Unit]
Description=Bot knowledge ingestion loop
After=network-online.target

[Service]
WorkingDirectory=/home/tnguyen/source/personal/bot
ExecStart=/usr/bin/python -m src.enrich_knowledge.runners.run_ingestion
Restart=always
RestartSec=30
EnvironmentFile=/home/tnguyen/source/personal/bot/.env

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/bot-train.service
[Unit]
Description=Bot model retraining (oneshot)

[Service]
Type=oneshot
WorkingDirectory=/home/tnguyen/source/personal/bot
ExecStart=/usr/bin/python -m src.enrich_knowledge.runners.run_training --model all
EnvironmentFile=/home/tnguyen/source/personal/bot/.env

# /etc/systemd/system/bot-train.timer
[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

Alternative (portable, if you end up containerising): `docker
compose up ingest` + host cron running `docker compose run train`.
Same contract, different wrapper.

### Boundaries — what NOT to do

- **No firing ingestion from the trading loop or an MCP tool.**
  Read and write paths have independent lifecycles. A stalled writer
  must never stall trade decisions.
- **No shared Python process for ingestion + training.** Training
  loads heavy sklearn/numba; keep it out of the ingestion hot path.
  Separate systemd units = separate memory footprints.
- **No hand-rolled `time.sleep` loops.** If you find one, replace
  with an APScheduler job.
- **No state in-process.** All checkpointing (last_fetched_at, model
  version pointers) lives on disk so any worker restart is a no-op.

### Done when

- `systemctl --user status bot-ingest` shows a healthy long-running
  loop with jobs firing on schedule.
- `systemctl --user list-timers` shows `bot-train.timer` scheduled.
- Killing and restarting `bot-ingest` produces zero duplicate rows
  across three test runs.
- A new source can be added by creating one file in `sources/` +
  one entry in `config/schedule.py` — nothing else.

## Phase 2 — trading_bot sub-settings

Create `src/trading_bot/config/` with three dataclasses.

### `TraderSettings`

Everything the trading loop needs that isn't LLM/Discord.

```
bot_mode                        BOT_MODE                    off
bot_interval_seconds            BOT_INTERVAL_SECONDS        0
max_order_usdt                  MAX_ORDER_USDT              50
timeframe                       TIMEFRAME                   1h
candle_limit                    CANDLE_LIMIT                200
trading_symbols                 TRADING_SYMBOLS             BTCUSDT,ETHUSDT
single_symbol_decision          SINGLE_SYMBOL_DECISION      true
trading_engine                  TRADING_ENGINE              llm_enriched
trader_skills                   TRADER_SKILLS               [5 defaults]

# Risk guards
max_daily_loss_pct              MAX_DAILY_LOSS_PCT          0.05
max_consecutive_losses          MAX_CONSECUTIVE_LOSSES      3
min_confidence_threshold        MIN_CONFIDENCE_THRESHOLD    0.0

# Trailing / partial TP
trailing_stop_enabled           TRAILING_STOP_ENABLED       false
trailing_stop_activation_pct    TRAILING_STOP_ACTIVATION_PCT 0.01
trailing_stop_distance_pct      TRAILING_STOP_DISTANCE_PCT  0.005
partial_tp_enabled              PARTIAL_TP_ENABLED          false
partial_tp1_atr_multiplier      PARTIAL_TP1_ATR_MULTIPLIER  2.0
partial_tp1_size_pct            PARTIAL_TP1_SIZE_PCT        0.5

# Scoring / RSI / regime
scoring_entry_threshold         ...                         0.30
scoring_exit_threshold          ...                         0.20
scoring_w_signal                ...                         0.25
scoring_w_direction             ...                         0.25
scoring_w_trend                 ...                         0.15
scoring_w_momentum              ...                         0.15
scoring_w_volume                ...                         0.10
scoring_w_key_levels            ...                         0.10
scoring_choppiness_penalty      ...                         0.3
choppiness_threshold            ...                         61.8
signal_rsi_strong_buy/_buy/_sell/_strong_sell
reentry_cooldown_cycles         ...                         3

# Execution
limit_order_timeout_seconds     ...                         300
max_slippage_pct                MAX_SLIPPAGE_PCT            0.005
position_monitor_enabled        ...                         true
position_monitor_interval       ...                         15
spot_sl_limit_offset_pct        ...                         0.01
spot_tp_limit_offset_pct        ...                         0.002
futures_leverage                FUTURES_LEVERAGE            1
default_stop_loss_pct           DEFAULT_STOP_LOSS_PCT       0.02
default_take_profit_pct         DEFAULT_TAKE_PROFIT_PCT     0.04
default_position_size           DEFAULT_POSITION_SIZE       0.02

# Trade memory
trade_memory_max_entries        ...                         500

# Multi-timeframe
htf_timeframe                   HTF_TIMEFRAME               4h
htf_confirmation_enabled        ...                         false
```

Keep `bot_enabled` / `bot_dry_run` / `effective_bot_interval` /
`effective_rsi_thresholds` as methods on `TraderSettings` — they are
pure computations over the settings themselves and belong with the
fields they read.

### `LLMProviderSettings`

Everything the LLM abstraction reads. Split into nested provider
blocks so `get_model_config(model_name)` can route cleanly.

```
provider                        PROVIDER                    azure
model_supports_vision           MODEL_SUPPORTS_VISION       false

azure_endpoint / azure_api_key / azure_deployment / azure_api_version

google_studio_api_key / google_studio_paid_api_key / google_studio_model
google_max_tokens / google_temperature / google_top_p / google_top_k
google_thinking_level / google_code_execution

openrouter_api_key / openrouter_base_url / openrouter_base_model /
openrouter_fallback_model

lm_studio_base_url / lm_studio_model / lm_studio_streaming

blockrun_wallet_key / blockrun_base_url / blockrun_model

model_temperature / model_max_tokens / model_top_p / model_top_k
model_freq_penalty / model_pres_penalty
```

`get_model_config(model_name, overrides)` stays as a method on
`LLMProviderSettings`.

### `DiscordSettings`

```
discord_bot_enabled             DISCORD_BOT_ENABLED         false
bot_token_discord               BOT_TOKEN_DISCORD
guild_id_discord                GUILD_ID_DISCORD
main_channel_id                 MAIN_CHANNEL_ID
temporary_channel_id_discord    TEMPORARY_CHANNEL_ID_DISCORD
admin_user_ids                  ADMIN_USER_IDS              []
file_message_expiry             FILE_MESSAGE_EXPIRY         604800
```

### Composition

```python
# src/trading_bot/config/__init__.py
@dataclass(frozen=True)
class TradingBotSettings:
    binance: BinanceSettings
    storage: StorageSettings
    trader: TraderSettings
    llm: LLMProviderSettings
    discord: DiscordSettings

    @classmethod
    def load(cls) -> "TradingBotSettings":
        return cls(BinanceSettings(), StorageSettings(),
                   TraderSettings(), LLMProviderSettings(),
                   DiscordSettings())
```

### llm_trader module overrides

The standalone `llm_trader/` module reads several `LLM_TRADER_*` env
vars that shadow the shared ones (see `.env.example:39-53`). These
belong on `TraderSettings` under an `llm_trader_*` prefix — keep the
existing naming so env compatibility is preserved.

### Done when

- `TradingBotSettings.load()` works against `.env`.
- The trading loop entrypoint (wherever it wires settings today)
  consumes `TradingBotSettings` instead of `Settings`.

## Phase 3 — migrate remaining Settings callers

Grep for `from src.mcp_servers.config` outside of
`src/mcp_servers/config/` itself. Every hit must be re-pointed to the
matching sub-setting. Likely hotspots:

- `src/services/llm_trader/` (moves under `trading_bot/` in this
  phase — now reads `TraderSettings` + `LLMProviderSettings`).
- `src/services/trading/` (moves under `trading_bot/`).
- `src/services/backtest/` (decide: enrich_knowledge or trading_bot;
  backtests read history but run without live trading → probably
  `enrich_knowledge/`).
- `scripts/backfill_ohlcv.py` — reads `StorageSettings` only.
- `src/app.py` — the composition root. Should construct
  `TradingBotSettings.load()` and pass sub-settings into the
  services that need them.

Do this in topological order (leaves first). Each commit should flip
one subsystem and re-run tests before the next starts.

## Phase 4 — delete monolithic Settings

Preconditions:

- No module outside `src/mcp_servers/config/` imports `Settings` or
  `load_settings_from_env`.
- `grep -r "from src.mcp_servers.config" src/ tests/ scripts/` returns
  nothing outside the folder itself.

Actions:

1. Delete `src/mcp_servers/config/` (settings.py, loader.py, parsers.py,
   validation.py, __init__.py).
2. Delete `get_settings()` from `src/mcp_servers/base.py`; keep
   `get_binance_settings()` / `get_storage_settings()`.
3. Update `CLAUDE.md`:
   - Replace "two folders directly under `src/`" with three folders
     + `src/config/`.
   - Add a note: "Sub-settings live with their primary owner; shared
     sub-settings go to `src/config/`. Never add a monolithic
     Settings back."
4. Delete any `.env.example` lines that no longer correspond to a
   field.

## Phase 5 — housekeeping

- **Skills path bug** — `src/mcp_servers/skills/service.py:21` builds
  `_SKILLS_DIR` as `parents[2] / "services" / "llm_trader" / "skills"`
  which resolves to `src/services/llm_trader/skills/`. After Phase 2
  the skills files move under `src/trading_bot/skills/` (or stay put
  as a standalone package). Fix the path once the target location is
  settled. Until then, the MCP server logs `Skills dir missing` on
  startup — that warning is the tripwire that reminds us.
- **CLAUDE.md** — see Phase 4 step 3. Do not land until the monolith
  is actually gone; premature docs will mislead future sessions.
- **`.env.example`** — re-group by sub-setting so the file reads like
  the code it configures.

## Anti-goals

- Do not introduce a "global" settings singleton outside
  `src/config/`. The whole point of this refactor is that each
  process constructs exactly the sub-settings it reads.
- Do not add backwards-compat shims (re-exporting `Settings` from the
  new modules, for instance). The monolith is deleted outright in
  Phase 4; any remaining caller is a bug to fix, not to paper over.
- Do not add per-field migration flags. Env var names are preserved
  through aliases, so `.env` keeps working during every phase.
