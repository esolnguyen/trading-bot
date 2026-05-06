# MCP servers

Five FastMCP servers under `src/mcp_servers/`. Each exposes one
concern. The trading agent and the `/trade` skill consume them
through the standard MCP protocol, so any other MCP client (MCP
Inspector, Claude Desktop, a custom agent) works the same way.

## Server inventory

| Server     | Folder                       | Launcher                     | Needs                                |
|------------|------------------------------|------------------------------|--------------------------------------|
| binance    | `src/mcp_servers/binance_mcp/`  | `scripts/mcp/run_mcp_binance.py`  | `BINANCE_API_KEY` / `_SECRET`        |
| analysis   | `src/mcp_servers/analysis_mcp/` | `scripts/mcp/run_mcp_analysis.py` | Binance creds (pulls candles)        |
| ml         | `src/mcp_servers/ml_mcp/`       | `scripts/mcp/run_mcp_ml.py`       | Trained model files in `models/`     |
| rag        | `src/mcp_servers/rag_mcp/`      | `scripts/mcp/run_mcp_rag.py`      | `chroma_db/` populated by ingestion  |
| skills     | `src/mcp_servers/skills_mcp/`   | `scripts/mcp/run_mcp_skills.py`   | `skills_mcp/skills/` directory       |

Every server follows the canonical 4-file layout:

```
<server_name>/
├── server.py    # Wiring only. Builds FastMCP, constructs service, registers handlers.
├── handlers.py  # Thin @mcp.tool() functions; validate inputs, delegate to service.
├── service.py   # All long-lived state (feed handles, model caches, locks).
└── schemas.py   # Pydantic response models + Literal aliases.
```

Transport selection lives in the launcher (`scripts/mcp/run_mcp_<name>.py`)
so the same `mcp` object can be reused across stdio / HTTP / SSE.

---

## bot-binance — live market & account data

Wraps `BinanceFeed` (UM-futures by default). Read-only **except**
for `cancel_order`, intentionally included on the principle that
cancellation can only shrink exposure.

| Tool                | Returns                                             |
|---------------------|-----------------------------------------------------|
| `get_ohlcv`         | Up to 1500 candles for arbitrary symbol/timeframe   |
| `get_ticker`        | 24h last/change/bid/ask/volume                      |
| `get_order_book`    | L2 depth (bid/ask book with aggregated sizes)       |
| `get_funding_rate`  | Perpetual funding + mark/index                      |
| `get_open_interest` | Open interest in base units                         |
| `get_balance`       | Account balances                                    |
| `get_open_positions`| Currently open perp positions                       |
| `get_order_status`  | Status of a specific order id                       |
| `cancel_order`      | Cancel an open order (safe write)                   |

Order placement and bracket setup are deliberately **not** here —
they belong to the gated execution server (Phase 7).

---

## bot-analysis — indicators, signal, patterns, charts, backtest

Twelve tools, all single-symbol. Indicators run on numba-accelerated
kernels in `src/mcp_servers/shared/services/indicators/`.

| Tool                  | What it does                                                   |
|-----------------------|----------------------------------------------------------------|
| `compute_indicators`  | Canonical indicator block (RSI, MACD, BB, EMAs, ADX, ATR, OBV slope, choppiness, vol ratio, CCI). |
| `get_snapshot`        | Aggregated market snapshot — ticker + order book top + OHLCV + funding + OI. |
| `analyze_signal`      | Weighted momentum + trend + volatility + volume + structure → `STRONG_BUY…STRONG_SELL` + reasoning. |
| `analyze_multi_tf`    | Higher-timeframe alignment summary (15m / 1h / 4h / 1d).       |
| `detect_patterns`     | Double bottom/top, engulfing, nearest support & resistance.    |
| `render_chart`        | Base64 PNG (mplfinance preferred, matplotlib fallback).        |
| `backtest_signal`     | Replay `analyze_signal` bar-by-bar with fees + slippage; returns total return, CAGR, Sharpe, max DD, win rate, trade list, equity curve. |
| `compute_momentum`    | RSI, MACD, stochastic, ROC, momentum, Williams %R, TSI, RMI, PPO, Coppock, KST, UO. |
| `compute_trend`       | ADX, Supertrend, Ichimoku, Parabolic SAR, Vortex, TRIX, PFE, TD sequential. |
| `compute_volatility`  | ATR, Bollinger, Keltner, Donchian, Chandelier, Choppiness, CCI, VHF, EBSW. |
| `compute_volume`      | OBV, OBV slope, MFI, VWAP, TWAP, CMF, Force Index, PVT, A/D line, avg quote volume. |
| `compute_structure`   | Support/resistance, Fibonacci retracement, Fib Bollinger, pivot points (standard + Fibonacci). |
| `compute_statistical` | Hurst, z-score, entropy, kurtosis, skew, stdev, variance, quantile, MAD, linreg, fear & greed. |

`backtest_signal` is the historical-validation tool — it returns
honest fee/slippage-adjusted equity curves so you can sanity-check a
proposed rule before wiring it into the agent.

---

## bot-ml — trained models

Seven tools. Each loads a model file from `models/` lazily on first
call. If the artifact is missing, the tool returns
`{"success": false, "error": "model_unavailable", ...}` rather than
crashing — clients can degrade gracefully.

See [docs/ml-models.md](ml-models.md) for the full description of
each model, its features, label, and walk-forward validation.

| Tool                | Underlying model                                     |
|---------------------|------------------------------------------------------|
| `predict_direction` | Pooled XGBoost binary classifier (B2)                |
| `detect_anomaly`    | IsolationForest on volume / velocity / range (B4)    |
| `classify_cycle`    | Random Forest macro regime (A4)                      |
| `get_key_levels`    | DBSCAN-derived support/resistance cache (A3)         |
| `percentile_rank`   | Statistical percentile vs. recent history (A1)       |
| `predict_outcome`   | Logistic regression hit-rate prior (B3)              |
| `score_sentiment`   | FinBERT sentiment label + confidence (B1)            |

All tools surface `freshness_seconds` so the client can discount
stale predictions.

---

## bot-rag — curated knowledge retrieval

Reads from ChromaDB at `chroma_path` (default `./chroma_db`). Zero
external API keys — the writer (`enrich_knowledge.runners.run_ingestion`)
fills the collections.

| Tool                | Reads collection      | Use for                                  |
|---------------------|-----------------------|------------------------------------------|
| `retrieve_news`     | `news`                | Curated crypto news (filter by symbol)   |
| `retrieve_macro`    | `macro`               | Fear/greed, TVL, global mcap, BTC/ETH narrative |
| `retrieve_memory`   | `trade_memory`        | Past trade outcomes with the same setup  |
| `ingestion_status`  | all three             | Freshness check before trusting results  |

Each retrieved item carries `freshness_seconds` and a normalised
`as_iso` timestamp so consumers don't branch on storage format.
`ingestion_status` returns `latest_timestamps` per collection — page
the operator if a collection stalls.

---

## bot-skills — playbook prompts

One tool plus eight MCP prompts.

| Surface                | What it returns                                        |
|------------------------|--------------------------------------------------------|
| `list_skills` (tool)   | `[{name, description, category}]` for every playbook   |
| `skill.<folder>` (prompt × 8) | The full SKILL.md body, prefixed with a header telling the model to synthesise rather than quote. |

Current catalogue: `candlestick`, `crypto-derivatives`, `elliott-wave`,
`harmonic`, `ichimoku`, `perp-funding-basis`, `smc`, `technical-basic`.

The same SKILL.md files at `src/mcp_servers/skills_mcp/skills/` are
read by the trading agent's in-process loader, so edits propagate to
both surfaces automatically.

---

## Running a server

Smoke-test any server with MCP Inspector:

```bash
npx @modelcontextprotocol/inspector python scripts/mcp/run_mcp_binance.py
```

Run on stdio directly (what MCP clients spawn):

```bash
python scripts/mcp/run_mcp_rag.py
```

In-process, without a transport, using FastMCP's `Client`:

```python
import asyncio
from fastmcp import Client
from src.mcp_servers.ml_mcp.server import mcp

async def smoke():
    async with Client(mcp) as c:
        print(await c.call_tool("predict_direction",
                                {"symbol": "BTCUSDT", "timeframe": "15m"}))

asyncio.run(smoke())
```

## Registering with Claude

One-shot script:

```bash
bash scripts/install_mcp_servers.sh
```

Re-run safe — `claude mcp add` overwrites the existing entry. The
script registers all five servers under user scope as
`bot-binance`, `bot-analysis`, `bot-ml`, `bot-rag`, `bot-skills`.

Manual registration via `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "bot-binance":  { "command": "python", "args": ["scripts/mcp/run_mcp_binance.py"],  "cwd": "/home/tnguyen/source/personal/bot" },
    "bot-analysis": { "command": "python", "args": ["scripts/mcp/run_mcp_analysis.py"], "cwd": "/home/tnguyen/source/personal/bot" },
    "bot-ml":       { "command": "python", "args": ["scripts/mcp/run_mcp_ml.py"],       "cwd": "/home/tnguyen/source/personal/bot" },
    "bot-rag":      { "command": "python", "args": ["scripts/mcp/run_mcp_rag.py"],      "cwd": "/home/tnguyen/source/personal/bot" },
    "bot-skills":   { "command": "python", "args": ["scripts/mcp/run_mcp_skills.py"],   "cwd": "/home/tnguyen/source/personal/bot" }
  }
}
```

Verify with `claude mcp list`.

## Conventions

- **Never** add `from __future__ import annotations` to `server.py`,
  `handlers.py`, or `schemas.py`. FastMCP builds tool schemas via
  pydantic's `TypeAdapter`, which cannot resolve stringified hints.
  `service.py` may use it (no schema introspection there).
- **Lazy-init** async resources inside the running event loop behind
  an `asyncio.Lock`. `BinanceFeed` is the canonical example.
- Tool responses are JSON-serializable dicts. On failure, return the
  `tool_error(message, **extra)` envelope from `src/mcp_servers/base.py`.
- Cached or ingested data should include `freshness_seconds` so the
  client can decide whether to trust it.
- Server-specific code lives in the server's folder. Code reused by
  ≥2 servers belongs in `src/mcp_servers/shared/`. Trading-loop
  imports (`src.trading_bot.*`) are forbidden inside MCP servers —
  copy or extract the piece you need.

## Adding a new tool

1. Add the response model to `schemas.py`.
2. Add the handler to `handlers.py`:
   ```python
   @mcp.tool()
   async def my_tool(
       symbol: Annotated[str, Field(min_length=3, max_length=20)],
       limit: Annotated[int, Field(ge=1, le=500)] = 100,
   ) -> dict:
       """One-sentence docstring — becomes the tool description."""
       try:
           data = await service.do_thing(symbol, limit)
           return MyResponse(**data).model_dump()
       except Exception as exc:
           logger.exception("my_tool failed")
           return tool_error(f"{type(exc).__name__}: {exc}")
   ```
3. Add the business logic method to `service.py`.

If the tool needs a new long-lived resource, attach it to the
service — never capture it in a closure.
