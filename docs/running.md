# Running the bot

Operational guide for the three runtimes in this repo and how they
share state. For a one-screen overview of the architecture, read the
top-level `README.md` first; for the MCP-server-specific rules, read
`src/mcp_servers/README.md`.

## The three processes

Nothing in this repo is a single-process application. Three
independent runtimes touch two shared on-disk resources:

| Process                               | Command                                                          | Purpose                              | Owns                           |
|---------------------------------------|------------------------------------------------------------------|--------------------------------------|--------------------------------|
| **trading_bot**                       | `python -m src.app`                                              | Decision loop (one tick per wake-up) | nothing on disk                |
| **enrich_knowledge — ingestion**      | `python -m src.enrich_knowledge.runners.run_ingestion`           | Keep RAG fresh                       | `chroma_db/`                   |
| **enrich_knowledge — training**       | `python -m src.enrich_knowledge.runners.run_training --model all`| Fit ML models                        | `data/ohlcv/*.csv`, `models/<tf>/*.joblib` |

The trading_bot only **reads** from `chroma_db/` and `models/` (via
the `rag` and `ml` MCP servers). If training never runs, the ML tools
return "model unavailable" and the agent works without them; if
ingestion never runs, the RAG tools return empty. Either case
degrades the agent gracefully — it won't crash.

## Prerequisites

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in Anthropic, Binance, CryptoCompare
```

Minimum `.env` to get a dry-run tick:

```dotenv
BOT_MODE=dry_run
ANTHROPIC_API_KEY=sk-ant-...
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
BINANCE_TESTNET=true
TRADING_SYMBOLS=BTCUSDT,ETHUSDT
CRYPTOCOMPARE_API_KEY=...   # only needed for news ingestion
```

First-run ordering matters only if you want the MCP tools to return
useful data on the first tick:

1. Backfill + train once: `python -m src.enrich_knowledge.runners.run_training --model all`.
2. Start ingestion so `chroma_db/` isn't empty.
3. Only then start `trading_bot` — otherwise expect "no docs found"
   and "model unavailable" responses during the first ticks.

## Day-to-day operation

### trading_bot

Foreground (development):

```bash
python -m src.app
```

Defaults to 15-minute ticks. For a fast sanity loop:

```bash
TRADING_DECISION_INTERVAL_SECONDS=60 python -m src.app
```

Shutdown is SIGINT/SIGTERM-clean — the in-flight tick drains, the
MCP clients close, and the process exits 0. Don't kill the process
with `-9` unless you want zombie stdio pipes to the MCP subprocesses.

Log lines to watch for per tick (per symbol):

```
INFO tick: BTCUSDT
INFO [would-trade] BTCUSDT LONG conviction=8 sl=1.5 tp=3.0 — EMA stack aligned, FVG reclaim
INFO [would-trade] ETHUSDT HOLD conviction=4 sl=None tp=None  (gated: below min_conviction) — …
```

`[would-trade]` = `BOT_MODE=dry_run`; `[decision]` would appear in
`live` mode (gated today, see `assert_runnable()`).

### enrich_knowledge — ingestion

```bash
python -m src.enrich_knowledge.runners.run_ingestion
```

APScheduler runs each job on its own cadence
(`macro.coingecko` every 30 min, `news.cryptocompare` every 15 min,
etc. — see `src/enrich_knowledge/README.md` for the table). A failing
upstream fails that cycle only; the scheduler keeps running.

Dry-run prints the schedule and exits:

```bash
python -m src.enrich_knowledge.runners.run_ingestion --dry-run
```

### enrich_knowledge — training

One-shot, not a daemon:

```bash
python -m src.enrich_knowledge.runners.run_training --model all
python -m src.enrich_knowledge.runners.run_training --model anomaly
python -m src.enrich_knowledge.runners.run_training --model retrain_all
```

On every invocation the runner first backfills OHLCV (incremental
append) for every `(symbol × timeframe)` pair, then dispatches to the
requested driver. Skip the backfill with `--skip-backfill` when the
CSVs are known fresh.

Model artifacts are written to `models/<tf>/<family>_<symbol>.joblib`;
the `retrain_all` orchestrator globs this path for staleness checks,
so don't "organise" model files into other subfolders by hand.

## systemd

Put `trading_bot` and `ingestion` under a supervisor; run `training`
as a timer. Never put training inside the ingestion loop.

```ini
# /etc/systemd/system/trading-bot.service
[Unit]
Description=Trading bot decision loop
After=network-online.target

[Service]
WorkingDirectory=/home/tnguyen/source/personal/bot
EnvironmentFile=/home/tnguyen/source/personal/bot/.env
ExecStart=/usr/bin/python -m src.app
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/bot-ingestion.service
[Service]
WorkingDirectory=/home/tnguyen/source/personal/bot
EnvironmentFile=/home/tnguyen/source/personal/bot/.env
ExecStart=/usr/bin/python -m src.enrich_knowledge.runners.run_ingestion
Restart=on-failure
```

```ini
# /etc/systemd/system/bot-retrain.service
[Service]
Type=oneshot
WorkingDirectory=/home/tnguyen/source/personal/bot
EnvironmentFile=/home/tnguyen/source/personal/bot/.env
ExecStart=/usr/bin/python -m src.enrich_knowledge.runners.run_training --model retrain_all
```

```ini
# /etc/systemd/system/bot-retrain.timer
[Timer]
OnCalendar=Sun 02:00
Persistent=true

[Install]
WantedBy=timers.target
```

## Troubleshooting

**`RuntimeError: BOT_MODE=live is not supported`** — expected.
Phase 7 (execution MCP) hasn't shipped. Use `BOT_MODE=dry_run`.

**`ANTHROPIC_API_KEY is empty`** — the key isn't in `.env` or the
process didn't read it. `python -m src.app` calls `load_dotenv` on
start; systemd doesn't — use `EnvironmentFile=`.

**`agent returned no structured_response for BTCUSDT`** — the ReAct
graph hit the recursion limit before emitting a `TradingDecision`.
Raise `LLM_MAX_ITERATIONS` (cap is `×2` inside the loop) or narrow
the prompt.

**Skills MCP logs `Skills dir missing`** — the `skills_mcp` server
expects its SKILL.md files at
`src/mcp_servers/skills_mcp/skills/`. If you relocate them, update
`_SKILLS_DIR` in `src/mcp_servers/skills_mcp/service.py`.

**MCP server fails to start on stdio** — run its launcher standalone:

```bash
python scripts/run_mcp_binance.py
```

The FastMCP handshake should print on stdout; any import error
surfaces immediately instead of being swallowed by the adapter.

**`ModuleNotFoundError: langgraph.prebuilt`** — the `langgraph-prebuilt`
distribution's files are missing from site-packages (a known broken
install). Reinstall:

```bash
pip install --force-reinstall --no-deps langgraph-prebuilt
```

**Tests can't find `.env`** — they deliberately shouldn't. The
`clean_env` fixture in `tests/conftest.py` chdirs to a tmp path and
strips every trading-bot env var so unit tests don't leak your local
credentials.
