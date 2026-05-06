# Project overview

A personal crypto research workspace. The repo houses three things
that share the same on-disk state:

1. Five **MCP servers** that wrap live market data, technical
   analysis, ML inference, RAG retrieval, and trading playbooks.
2. A **LangGraph trading agent** (`src/trading_bot/`) that consumes
   those servers each tick and emits one `TradingDecision` per
   symbol.
3. A **write-path package** (`src/enrich_knowledge/`) that runs
   continuously to keep the RAG corpus and the ML model files fresh.

The same MCP servers also power the **`/trade`** Claude skill, which
turns the toolchain into a hands-on analyst you can drive from
natural language.

## Goals and non-goals

**Goals**

- Keep the trading hot path **one-shot, deterministic, and cheap** —
  one tick = one Claude call → one decision → sleep.
- Expose every reusable capability as an MCP tool so the trading
  loop, the `/trade` skill, MCP Inspector, or any other MCP client
  consume them through the same interface.
- Make the write path (ingestion + training) **independent and
  restartable**. Failure of any single source or model never blocks
  the trading loop.

**Non-goals**

- The trading loop, risk manager, and safety tracker are **not**
  exposed as MCP servers — they're stateful orchestration, not
  tool-shaped.
- Order placement is **not** wired in yet (Phase 7). `BOT_MODE=live`
  is refused at startup. `cancel_order` is the only write tool that
  ships today, on the principle that cancellation can only shrink
  exposure.

## Runtime topology

Three independent processes touch two shared on-disk resources:

```
┌────────────────────────┐                       ┌──────────────────────┐
│ enrich_knowledge       │  writes               │ data/ohlcv/*.csv     │
│   run_ingestion (cron) │ ───────────────────▶  │ chroma_db/           │
│   run_training (timer) │                       │ models/<tf>/*.joblib │
└────────────────────────┘                       └──────────┬───────────┘
                                                            │ reads
                                                            ▼
   ┌────────────────────┐                ┌──────────────────────────────┐
   │ trading_bot.runner │  spawns stdio  │ 5 MCP servers (read-only)    │
   │  (LangGraph ReAct) │ ─────────────▶ │  binance · analysis · ml ·   │
   └────────────────────┘                │  rag · skills                │
                                         └──────────────────────────────┘
                                                            ▲
                                                            │ also called by
                                         ┌──────────────────────────────┐
                                         │ /trade skill in Claude Code  │
                                         │ or Claude Desktop            │
                                         └──────────────────────────────┘
```

If training never runs, the ML tools return `model_unavailable` and
the agent falls back to TA-only reasoning. If ingestion never runs,
the RAG tools return empty. Neither degradation crashes the loop.

## Top-level layout

```
src/
├── mcp_servers/        # 5 read-path MCP servers
│   ├── binance_mcp/    # live Binance feed + safe cancel_order
│   ├── analysis_mcp/   # indicators, signal, patterns, chart, backtest
│   ├── ml_mcp/         # 7 trained models exposed as tools
│   ├── rag_mcp/        # ChromaDB readers (news / macro / trade memory)
│   ├── skills_mcp/     # 8 SKILL.md playbooks served as MCP prompts
│   └── shared/         # BinanceFeed, IndicatorCalculator, etc.
├── trading_bot/        # LangGraph agent + daemon runner
├── enrich_knowledge/   # RAG ingestion + ML training
└── config/             # Cross-cutting BinanceSettings / StorageSettings

scripts/
├── mcp/run_mcp_<name>.py   # one stdio launcher per MCP server
├── enrich/                 # ad-hoc training entrypoints
├── install_mcp_servers.sh  # one-shot Claude Code registration
└── run_trading_bot.py      # legacy convenience wrapper

data/ohlcv/             # OHLCV CSVs the trainers read
models/<timeframe>/     # joblib artifacts the ml MCP loads
chroma_db/              # ChromaDB collections (news, macro, trade_memory)
history/                # Saved analyses written by /trade
logs/                   # Ingestion + training log files
```

## Configuration

`.env` is the single source of truth. The high-leverage knobs:

| Var                                 | Default             | Purpose                                       |
|-------------------------------------|---------------------|-----------------------------------------------|
| `BOT_MODE`                          | `dry_run`           | `off` / `dry_run` / `live` (live is refused)  |
| `TRADING_SYMBOLS`                   | `BTCUSDT`           | CSV, uppercased on load                       |
| `TRADING_TIMEFRAME`                 | `15m`               | Primary TF the agent focuses on each tick     |
| `TRADING_DECISION_INTERVAL_SECONDS` | `900`               | Sleep between ticks                           |
| `TRADING_MIN_CONVICTION`            | `6`                 | Below this, LONG/SHORT gets gated to HOLD     |
| `LLM_MODEL`                         | `claude-sonnet-4-6` | Any model `langchain-anthropic` accepts       |
| `LLM_MAX_ITERATIONS`                | `12`                | Recursion cap inside the ReAct graph          |
| `ML_TIMEFRAMES`                     | `15m,1h,4h,1d`      | Timeframes trained by `run_training`          |
| `TRAINING_SYMBOLS`                  | `BTCUSDT,ETHUSDT`   | Symbol universe for training                  |
| `BINANCE_API_KEY` / `_SECRET`       |                     | Required for the binance MCP                  |
| `ANTHROPIC_API_KEY`                 |                     | Required for the trading agent                |
| `CRYPTOCOMPARE_API_KEY`             |                     | News ingestion only                           |

See `.env.example` for the full list. Each subpackage owns its own
README with the specific knobs it consumes.

## Daily cadence

Crypto is 24/7 but liquidity and signal quality are not:

| Layer                              | Cadence                | Why                                                               |
|------------------------------------|------------------------|-------------------------------------------------------------------|
| Price / liquidation guard          | every 5–15 s           | At 50× leverage a 1% move is ~50% PnL.                            |
| 1H signal refresh                  | every 1H close (+ +55) | No new structural info arrives mid-candle.                        |
| ML direction + anomaly inference   | every 1H               | Models were trained on 1H features.                               |
| MTF alignment                      | every 4H close         | That's when the 4H flips.                                         |
| Cycle / key levels                 | every 1D               | Daily-trained, doesn't move intraday.                             |
| RAG news / macro                   | every 15–30 min        | ETF-flow / headline catalysts are the main intraday risk.         |
| Funding check                      | every 8H               | Decide whether to hold through funding.                           |
| ML training (`run_training`)       | once a day, 07:15 VN   | Daily candle just closed; 15-min buffer past funding settlement.  |

Avoid running the agent through funding settlements (07:00 / 15:00 /
23:00 Vietnam, ±5 min) and macro releases (FOMC / CPI / NFP, ±30
min). Weekend volume is thin and structure reads degrade.

## Where to go next

- [docs/mcp-servers.md](mcp-servers.md) — the 5 servers, their tool
  inventories, and how to register them with a Claude client.
- [docs/ml-models.md](ml-models.md) — what each trained model does,
  the features it consumes, the labelling rule, and the validated
  edge from walk-forward backtests.
- [docs/training.md](training.md) — how to backfill OHLCV and run
  the trainers, including the staleness-triggered orchestrator.
- [docs/data.md](data.md) — the on-disk shape of OHLCV CSVs, model
  artifacts, ChromaDB collections, and the history archive.
- [docs/trade-skill.md](trade-skill.md) — using the `/trade` skill
  in Claude Code or Claude Desktop.
