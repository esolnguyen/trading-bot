# Crypto Trading Bot

A personal crypto research and trading workspace built around five
in-process **MCP servers**. The same servers feed:

- a LangGraph **trading agent** (`src/trading_bot/`) that emits one
  `TradingDecision` per symbol per tick, and
- the **`/trade`** Claude skill, a hands-on analyst you call from
  Claude Code or Claude Desktop to research a coin, validate a setup,
  or backtest a rule.

Live order execution is gated. `BOT_MODE=live` is refused at startup
until the Phase 7 execution server ships; today the bot only runs in
`off` (always HOLD) or `dry_run` (logs `[would-trade]`).

## Layout

```
src/
├── mcp_servers/        # 5 read-path MCP servers (binance · analysis · ml · rag · skills)
├── trading_bot/        # LangGraph agent + daemon runner
├── enrich_knowledge/   # Write path: RAG ingestion + ML training
└── config/             # Cross-cutting BinanceSettings / StorageSettings
```

See [docs/overview.md](docs/overview.md) for the full architecture and
[docs/mcp-servers.md](docs/mcp-servers.md) for per-server tool surfaces.

Three independent processes at runtime:

```
┌──────────────────────────┐    ┌────────────────────────────┐
│ enrich_knowledge         │───▶│ chroma_db/  +  models/     │
│   run_ingestion (daemon) │    │ data/ohlcv/  (shared disk) │
│   run_training (one-off) │    └──────────────┬─────────────┘
└──────────────────────────┘                   │
                                               ▼
                       ┌────────────────────────────────────┐
                       │ trading_bot.runner   OR   /trade   │
                       │   spawns 5 MCP servers via stdio   │
                       │   → ReAct / Claude → decision      │
                       └────────────────────────────────────┘
```

## Documentation map

Start with the doc that matches your goal:

| If you want to…                                        | Read              |
|--------------------------------------------------------|-------------------|
| Understand the project at a glance                     | [docs/overview.md](docs/overview.md)         |
| Wire up the 5 MCP servers and call them                | [docs/mcp-servers.md](docs/mcp-servers.md)   |
| Know what each ML model does and how it was validated  | [docs/ml-models.md](docs/ml-models.md)       |
| Train (or retrain) those models                        | [docs/training.md](docs/training.md)         |
| Understand the data on disk                            | [docs/data.md](docs/data.md)                 |
| Use the `/trade` skill in Claude                       | [docs/trade-skill.md](docs/trade-skill.md)   |

Project conventions for Claude sessions live in
[`CLAUDE.md`](CLAUDE.md).

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env       # fill Anthropic + Binance keys
```

Minimum `.env` for a dry-run tick:

```dotenv
BOT_MODE=dry_run
ANTHROPIC_API_KEY=sk-ant-...
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
TRADING_SYMBOLS=BTCUSDT,ETHUSDT
```

Three commands, three runtimes:

```bash
# Trading agent (one decision per tick) — see docs/overview.md
python -m src.app

# News/macro ingestion into ChromaDB (long-running) — see docs/data.md
python -m src.enrich_knowledge.runners.run_ingestion

# Train every ML model (one-shot; backfills OHLCV first) — see docs/training.md, docs/ml-models.md
python -m src.enrich_knowledge.runners.run_training --model all
```

Register the 5 MCP servers with Claude Code in one shot:

```bash
bash scripts/install_mcp_servers.sh    # adds bot-binance, bot-analysis, bot-ml, bot-rag, bot-skills
```

After that, run `/trade BTC 4h, what's the read?` in Claude — see
[docs/trade-skill.md](docs/trade-skill.md).

## Smoke tests

```bash
python -m pytest tests/                       # unit tests, no network

python -c "                                   # imports + config gate
from dotenv import load_dotenv; load_dotenv()
from src.trading_bot.config import TradingBotSettings
TradingBotSettings().assert_runnable()
print('OK')"
```

End-to-end tick (costs Anthropic tokens — shorten the interval first):

```bash
TRADING_DECISION_INTERVAL_SECONDS=60 python -m src.app
```

## Repository conventions

- Python 3.12, pydantic v2, pydantic-settings v2.
- `dict` / `list` / `X | None` — no `typing.Dict` / `Optional[X]`.
- `Callable` / `Iterable` / `Mapping` from `collections.abc`.
- MCP server modules (`server.py` / `handlers.py` / `schemas.py`)
  must **never** add `from __future__ import annotations` — FastMCP's
  `TypeAdapter` cannot resolve stringified hints.
- Comments explain *why*, not *what*. Default to no comment.
- No emojis in code, comments, or commit messages unless explicitly
  requested.

