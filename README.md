# trading-bot

A LangChain/LangGraph trading agent fed by five in-process MCP servers.
Every tick the agent pulls market data, indicators, ML predictions, and
retrieved context from the servers, cross-checks signals, and emits one
`TradingDecision` per symbol.

Live order execution is **gated** — the bot refuses to start with
`BOT_MODE=live` until the Phase 7 execution MCP ships. Today it runs as
either `off` (always HOLD) or `dry_run` (logs a `[would-trade]` line).

## Layout

```
src/
├── trading_bot/        # LangGraph agent + daemon runner  (the bot itself)
├── mcp_servers/        # 5 read-path MCP servers         (ml · binance · analysis · rag · skills)
├── enrich_knowledge/   # Write path: RAG ingestion + ML training
└── config/             # Cross-cutting BinanceSettings / StorageSettings
```

Three independent processes at runtime:

```
┌──────────────────────────┐    ┌────────────────────────────┐
│ enrich_knowledge         │───▶│ chroma_db/ + models/       │
│   run_ingestion (long)   │    │ (shared on disk)           │
│   run_training (one-off) │    └──────────────┬─────────────┘
└──────────────────────────┘                   │
                                               ▼
                       ┌────────────────────────────────────┐
                       │ trading_bot/runner                 │
                       │   spawns 5 MCP servers via stdio   │
                       │   → ReAct agent → TradingDecision  │
                       └────────────────────────────────────┘
```

Deeper docs live in [`docs/`](./docs) — start with `docs/running.md`
for operational setup and `docs/mcp-migration-plan.md` for the MCP
architecture.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in Anthropic + Binance keys
```

Minimum `.env` for a dry-run tick:

```dotenv
BOT_MODE=dry_run
ANTHROPIC_API_KEY=sk-ant-...
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
TRADING_SYMBOLS=BTCUSDT,ETHUSDT
```

Run the bot:

```bash
python -m src.app                          # or: python scripts/run_trading_bot.py
```

Ingest news/macro into ChromaDB (long-running):

```bash
python -m src.enrich_knowledge.runners.run_ingestion
```

Train all ML models (one-shot; backfills OHLCV then writes
`models/<tf>/<family>_<symbol>.joblib`):

```bash
python -m src.enrich_knowledge.runners.run_training --model all
```

## Smoke tests

```bash
python -m pytest tests/                    # 24 unit tests, no network

python -c "                                # imports + config gate
from dotenv import load_dotenv; load_dotenv()
from src.trading_bot.config import TradingBotSettings
TradingBotSettings().assert_runnable()
print('OK')"
```

End-to-end tick (costs Anthropic tokens — shorten the interval first):

```bash
TRADING_DECISION_INTERVAL_SECONDS=60 python -m src.app
```

## Configuration

`.env` is the single source of truth. The important knobs:

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

See `.env.example` for the full list including Binance + ingestion
keys. Each package also has its own README with the knobs it owns
([`src/trading_bot/README.md`](./src/trading_bot/README.md),
[`src/mcp_servers/README.md`](./src/mcp_servers/README.md),
[`src/enrich_knowledge/README.md`](./src/enrich_knowledge/README.md)).

## Repository conventions

- Python 3.12, pydantic v2, pydantic-settings v2.
- `dict` / `list` / `X | None` — no `typing.Dict` / `Optional[X]`.
- `Callable` / `Iterable` / `Mapping` from `collections.abc`.
- MCP server files (`server.py` / `handlers.py` / `schemas.py`)
  **never** get `from __future__ import annotations` — FastMCP's
  `TypeAdapter` can't resolve stringified hints.
- Comments explain *why*, not *what*. Default to no comment.

More rules live in `CLAUDE.md`; it's the canonical reference.
