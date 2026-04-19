# mcp_servers

Read-path package. Every subfolder under `src/mcp_servers/` is a
FastMCP server exposing one concern to MCP clients (Claude Code,
Claude Desktop, MCP Inspector, agent frameworks). Servers never own
transport — launchers live in `scripts/run_mcp_<name>.py`.

Write side (ingestion + training) is in `src/enrich_knowledge/`;
this package only reads.

## Servers at a glance

| Server     | Tools                                                                                                       | Needs                         |
|------------|-------------------------------------------------------------------------------------------------------------|-------------------------------|
| `binance`  | `get_ohlcv`, `get_ticker`, `get_order_book`, `get_funding_rate`, `get_open_interest`, `get_balance`, …      | `BINANCE_API_KEY/SECRET`      |
| `analysis` | `compute_indicators`, `analyze_multi_tf`, `get_snapshot`, `analyze_signal`, `detect_patterns`, `render_chart`, per-category computes | Binance creds                 |
| `ml`       | `predict_direction`, `detect_anomaly`, `classify_cycle`, `get_key_levels`, `percentile_rank`, `predict_outcome`, `score_sentiment` | Trained model files in `models/` |
| `rag`      | `retrieve_news`, `retrieve_macro`, `retrieve_memory`, `ingestion_status`                                    | `CHROMA_PATH` populated by ingestion |
| `skills`   | `list_skills` + one MCP **prompt** per SKILL.md                                                             | `skills/` directory           |

## Canonical 4-file layout

Every server lives at `src/mcp_servers/<name>/` with exactly these
files:

```
<name>/
├── server.py    # Wiring only. Builds FastMCP, constructs service, registers handlers.
├── handlers.py  # Thin @mcp.tool() fns — validate inputs, delegate to service, return model.model_dump()
├── service.py   # All long-lived state (feed handles, model caches, locks) + business logic
└── schemas.py   # Pydantic response models + Literal aliases (Timeframe, Side, …)
```

Handlers never reach past the service. The service never touches the
transport. This lets the same `mcp` object be reused across
stdio / HTTP / SSE by swapping the launcher.

## Running a server

Every server has a launcher in `scripts/run_mcp_<name>.py`:

```bash
# Local smoke test with MCP Inspector
npx @modelcontextprotocol/inspector python scripts/run_mcp_binance.py

# Direct stdio (what MCP clients register)
python scripts/run_mcp_rag.py
```

Registering with a Claude Code client (`~/.claude/mcp.json`):

```json
{
  "mcpServers": {
    "bot-binance": {
      "command": "python",
      "args": ["scripts/run_mcp_binance.py"],
      "cwd": "/home/tnguyen/source/personal/bot"
    },
    "bot-rag": {
      "command": "python",
      "args": ["scripts/run_mcp_rag.py"],
      "cwd": "/home/tnguyen/source/personal/bot"
    }
  }
}
```

Five servers × one launcher each — enable whichever you need.

## Shared vs. server-specific code

- **`shared/`** — code used by ≥2 servers. Re-exports the public
  surface from `shared/__init__.py`:
  ```python
  from src.mcp_servers.shared import (
      BinanceFeed, IndicatorCalculator, MarketSnapshot, OHLCVCandle,
  )
  ```
  Internal layout:
  ```
  shared/
  ├── domain/          # Dataclasses: MarketSnapshot, OHLCVCandle
  ├── infrastructure/  # Binance SDK wrapper (BinanceFeed)
  ├── services/        # IndicatorCalculator, MarketAggregator
  └── utils/
  ```
- **Server-specific** (used by exactly one server) lives under that
  server's folder — e.g. `ml/services/` (model loaders, anomaly
  detector), `rag/storage/` (ChromaStore wrapper),
  `analysis/indicators/` (category computations), `skills/skills/`
  (SKILL.md files).

**Never** import from `src.trading_bot.*` inside `src/mcp_servers/`.
MCP servers are consumed by the trading loop, never the other way
around — anything both sides need belongs in `shared/` or
`src/config/`.

## Adding a new tool

1. Add the Pydantic response model to `schemas.py`.
2. Add a thin handler to `handlers.py`:
   ```python
   @mcp.tool()
   async def my_tool(
       symbol: Annotated[str, Field(min_length=3, max_length=20)],
       limit: Annotated[int, Field(ge=1, le=500)] = 100,
   ) -> dict:
       """One-sentence docstring — this becomes the tool description."""
       try:
           data = await service.do_thing(symbol, limit)
           return MyResponse(**data).model_dump()
       except Exception as exc:
           logger.exception("my_tool failed")
           return tool_error(f"{type(exc).__name__}: {exc}")
   ```
3. Add the business logic method to `service.py`.
4. Keep it to one concern — if the tool needs a new long-lived
   resource, that belongs on the service, not captured via a closure.

## Adding a new server

1. Create `src/mcp_servers/<name>/{server,handlers,service,schemas}.py`.
2. Create the launcher `scripts/run_mcp_<name>.py` (copy an existing
   one; change the server import).
3. Shared types go in `shared/`; anything one-server stays local.

## FastMCP tripwires

- **Never add `from __future__ import annotations`** to `server.py`,
  `handlers.py`, or `schemas.py`. FastMCP builds tool schemas via
  pydantic's `TypeAdapter`, which cannot resolve stringified hints.
  `service.py` may use it (no pydantic schema introspection there).
- **Lazy-init async resources.** `BinanceFeed` and anything with a
  live SDK client must be `start()`-ed inside the running event loop,
  behind an `asyncio.Lock`. Never construct at import time.
- **Tool responses are JSON-serializable dicts.** Always return
  `model.model_dump()` from handlers. On failure wrap with
  `tool_error(message, **extra)` from `base.py`.
- **Surface freshness.** Any cached / ingested data should include
  `freshness_seconds` in the response so the client can discount
  stale rows.

## Conventions (Python 3.12)

- Use `dict`, `list`, `tuple`, `type` — not `Dict`, `List`, `Tuple`,
  `Type`.
- Write `X | None` / `A | B` — not `Optional[X]` / `Union[A, B]`.
- `Callable`, `Iterable`, `Mapping`, … come from `collections.abc`.
- `typing` is still correct for `Any`, `Annotated`, `Literal`,
  `TypeVar`, `ClassVar`, `TYPE_CHECKING`, `Protocol`, and runtime
  `Union` used in identity checks.
- Package-level re-exports are the public surface. Consumers import
  from the package, not the inner module:
  `from src.mcp_servers.shared import BinanceFeed` — not
  `…shared.infrastructure.binance.feed import BinanceFeed`.

## Data flow with enrich_knowledge

The `rag` server reads ChromaDB; `enrich_knowledge/runners/run_ingestion.py`
writes it. They agree on:

- the path (`StorageSettings.chroma_path`),
- the collection names (`news`, `macro`, `trade_memory`),
- the embedding function (both use `build_default_embedding_function()`,
  which resolves to the same physical collection suffix),
- the metadata keys (`published_at`/`timestamp`, `source`, `symbol`,
  `action`, `reasoning`, `outcome_pnl`, `sentiment_score`).

`ml` reads model files from `models/` that `enrich_knowledge/ml_training/`
fits. If a tool returns "model unavailable", run the matching driver
first:

```bash
python -m src.enrich_knowledge.runners.run_training --model anomaly
```
