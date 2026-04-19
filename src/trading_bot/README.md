# trading_bot

The LangGraph trading agent. Spawns the five MCP servers over stdio,
hands their tools to a ReAct graph, and emits one
`TradingDecision` per symbol each tick. Never owns long-lived state
across ticks — cooldowns/open positions are still TODO (lands with
Phase 7).

## Folder layout

```
src/trading_bot/
├── runner.py      # Daemon: sleep/wake loop, signal handling, lifecycle
├── loop.py        # One-shot decision cycle (spawns MCP, runs agent, tears down)
├── config.py      # TradingBotSettings + startup gate (assert_runnable)
└── agent/
    ├── graph.py     # TradingDecision schema + build_agent (create_react_agent)
    ├── tools.py     # MCP_SERVER_LAUNCHERS + build_mcp_tools (async)
    └── prompts.py   # SYSTEM_PROMPT — the agent's decision rules
```

## Lifecycle

1. **`runner.main()`** — loads `.env`, builds `TradingBotSettings`,
   calls `assert_runnable()` (refuses `BOT_MODE=live` and empty
   `ANTHROPIC_API_KEY`). Installs SIGINT/SIGTERM handlers, then loops:
   `run_cycle(settings)` → sleep `TRADING_DECISION_INTERVAL_SECONDS`.
2. **`loop.run_cycle()`** — per tick: `build_mcp_tools()` spawns the 5
   MCP servers, `build_agent()` wires them into a ReAct graph, then for
   each symbol the agent is invoked with a `_tick_prompt` and the
   resulting `structured_response` (a `TradingDecision`) is logged.
   Every tick tears the MCP clients down in a `finally` block.
3. **Log line** — in `dry_run` mode each decision prints as
   `[would-trade] BTCUSDT LONG conviction=8 sl=1.5 tp=3.0 — <rationale>`;
   a decision whose conviction falls below `TRADING_MIN_CONVICTION` is
   annotated `(gated: below min_conviction)`.

## TradingDecision schema

The ReAct graph is constrained to emit a
`TradingDecision` via `response_format`:

| Field              | Type                              | Notes                                                        |
|--------------------|-----------------------------------|--------------------------------------------------------------|
| `symbol`           | `str`                             | Echoed from the tick prompt.                                 |
| `side`             | `"LONG" \| "SHORT" \| "HOLD"`     | HOLD is the safe default; agent should prefer it when unsure.|
| `conviction`       | `int` in `[1, 10]`                | Below `TRADING_MIN_CONVICTION` → gated to HOLD.              |
| `rationale`        | `str`                             | Which signals agreed / conflicted and how the thesis invalidates. |
| `stop_loss_pct`    | `float \| None`                   | Positive percent from entry; required for LONG/SHORT.        |
| `take_profit_pct`  | `float \| None`                   | Positive percent from entry; required for LONG/SHORT.        |

## The five MCP servers

`MCP_SERVER_LAUNCHERS` maps each logical name to the launcher script
under `scripts/`:

| Name       | Launcher                        | What it exposes                          |
|------------|---------------------------------|------------------------------------------|
| `ml`       | `scripts/run_mcp_ml.py`         | direction / anomaly / regime / key-levels / outcome / sentiment |
| `binance`  | `scripts/run_mcp_binance.py`    | OHLCV, ticker, funding, open interest, balance |
| `analysis` | `scripts/run_mcp_analysis.py`   | indicators, multi-TF, patterns, chart    |
| `rag`      | `scripts/run_mcp_rag.py`        | `retrieve_news` / `retrieve_macro` / `retrieve_memory` |
| `skills`   | `scripts/run_mcp_skills.py`     | `list_skills` + one MCP prompt per SKILL.md |

The trading loop never imports MCP server code directly — it only
talks to them through `langchain-mcp-adapters`
`MultiServerMCPClient`, identical to how any other MCP client would
consume them.

## Runtime gate

`assert_runnable()` refuses to start the loop when any of the
following is true:

- `BOT_MODE=live` — Phase 7 (order execution MCP) hasn't shipped, so
  a live mode would silently become dry-run-without-logs.
- `ANTHROPIC_API_KEY` is empty — the first tick would crash inside
  `ChatAnthropic` with an unhelpful error.
- `TRADING_SYMBOLS` is empty — nothing to decide on.

Fail-fast at startup beats discovering any of these three after a
tick lands in logs.

## Adding tools the agent can call

You don't. The agent picks up whichever MCP tools the five servers
advertise at startup; there is no allowlist in this package. To add
a tool:

1. Add the handler to the relevant MCP server
   (see `src/mcp_servers/README.md`).
2. Restart the bot. The tool appears in `build_mcp_tools()`'s return
   list automatically.

The agent's appetite for tools is capped by `LLM_MAX_ITERATIONS` —
the recursion limit passed to `agent.ainvoke`.

## Configuration

`TradingBotSettings` reads the `.env` keys summarised in the top-level
README. One pydantic-settings detail worth remembering: the
`trading_symbols` field is annotated with `NoDecode` so the CSV
string from `TRADING_SYMBOLS` flows through the `_split_csv`
`field_validator` instead of pydantic's default JSON-parse path.

## Testing

```bash
python -m pytest tests/test_trading_bot_config.py tests/test_trading_bot_agent.py -v
```

No LLM calls, no network. The tests cover the settings gate, the
CSV parser, the `TradingDecision` bounds, and the shape of
`MCP_SERVER_LAUNCHERS`. End-to-end behaviour (real Anthropic + Binance
calls) stays out of the test suite on purpose; run it interactively
per the "End-to-end tick" section of the top-level README.
