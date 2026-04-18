# MCP Migration Plan

Canonical tracking doc for migrating parts of the bot into MCP servers.
Update the checkboxes as phases ship. Each phase section is self-contained
so you (or a future Claude session) can resume from any point without
re-deriving context.

---

## Goal

Expose well-bounded pieces of the bot (ML inference, market data, analysis,
RAG) as **MCP servers** so:

- The main trading loop stays **one-shot** (cheap, fast, deterministic).
- A separate **research / post-mortem agent loop** can call tools
  dynamically for exploratory work.
- The news/macro ingestion pipeline runs **independently**, and any MCP
  client reads from the shared ChromaDB.

## Non-goals

- Do **not** MCP-ify the trading loop, risk manager, or safety tracker —
  they're stateful orchestration, not tool-shaped.
- Do **not** expose order placement until Phase 7 (explicit, gated).

---

## Architecture

```
┌──────────────────────┐     ┌────────────────────────────┐
│  Ingestion pipeline  │────▶│  ChromaDB (./chroma_db)    │
│  (cron / systemd)    │     │  news + macro + memory     │
└──────────────────────┘     └──────────────┬─────────────┘
                                            │
        ┌───────────────────────────────────┼──────────────────────┐
        │                                   │                      │
        ▼                                   ▼                      ▼
┌──────────────┐   ┌───────────────┐  ┌──────────────┐   ┌─────────────────┐
│ bot-mcp-ml   │   │ bot-mcp-binan │  │ bot-mcp-ana  │   │ bot-mcp-rag     │
│ (inference)  │   │ -ce (feed +   │  │ (chart/ind)  │   │ (retriever)     │
│              │   │  cancel)      │  │              │   │                 │
└──────┬───────┘   └──────┬────────┘  └──────┬───────┘   └────────┬────────┘
       └──────────────────┼──────────────────┼────────────────────┘
                          ▼                  ▼
                 ┌────────────────┐  ┌──────────────────┐
                 │ Trading loop   │  │ Research agent   │
                 │ (one-shot)     │  │ (multi-turn)     │
                 └────────────────┘  └──────────────────┘
```

---

## Phase status

| Phase | Deliverable                              | Status |
|-------|------------------------------------------|--------|
| 0     | Shared scaffolding                       | ✅ done |
| 1     | ML inference MCP server (7 tools)        | ✅ 7/7 tools wired |
| 2     | RAG MCP reader (ingestion split deferred) | ✅ reader shipped |
| 3     | Binance feed MCP (read + safe cancel)    | ✅ 9/9 tools wired |
| 4     | Analysis MCP (indicators + chart)        | ✅ 12/12 tools wired |
| 5     | Skills as MCP prompts                    | ✅ 8 skills wired |
| 6     | Research agent loop                      | ⬜ pending |
| 7     | (Gated) order execution MCP              | ⬜ pending |

---

## Phase 0 — Shared scaffolding ✅

**Shipped:**
- `src/mcp_servers/__init__.py`
- `src/mcp_servers/base.py` — `get_settings()` (cached), `tool_error()` envelope
- Transport: **stdio** via FastMCP, launcher pattern `scripts/run_mcp_<name>.py`
- Deps: `mcp==1.9.1`, `fastmcp==2.5.1` already in `requirements.txt`

**Conventions established:**
- Servers live at `src/mcp_servers/<name>/server.py`.
- Launchers live at `scripts/run_mcp_<name>.py` and add the project root
  to `sys.path` so they can be called as bare scripts.
- Every tool returns a JSON-serializable `dict`, success or error, via the
  `tool_error()` envelope.

---

## Phase 1 — ML inference MCP server ✅

**Server name:** `bot-mcp-ml`
**Layout:** `src/mcp_servers/ml/{server,handlers,service,schemas}.py` — canonical
4-file template every future MCP server should follow (server wiring /
tool handlers / service layer / pydantic schemas).
**Launcher:** `scripts/run_mcp_ml.py` — owns transport choice (`stdio`).

**Tools:**

| Tool | Wraps | Status |
|------|-------|--------|
| `predict_direction(symbol, timeframe, candle_limit)` | `DirectionClassifier` | ✅ shipped |
| `detect_anomaly(symbol, timeframe, candle_limit)`    | `AnomalyDetector`     | ✅ shipped |
| `classify_cycle(symbol, daily_candles)`              | `CycleClassifier`     | ✅ shipped |
| `get_key_levels(symbol, max_levels)`                 | `KeyLevelDetector`    | ✅ shipped |
| `percentile_rank(symbol, timeframe, candle_limit)`   | `HistoricalPercentileScorer` | ✅ shipped (needs local OHLCV CSV) |
| `predict_outcome(symbol, side, timeframe, candle_limit)` | `OutcomePredictor` | ✅ shipped (model file optional) |
| `score_sentiment(text)`                              | `SentimentScorer`     | ✅ shipped (requires transformers+torch) |

**Verified end-to-end (live BTCUSDT call):**
- `predict_direction` 15m → 28% bullish (high-conviction bearish)
- `detect_anomaly` 15m → `is_anomaly=false`
- `classify_cycle` → `ACCUMULATION` @ 46% confidence
- `get_key_levels` → 4 clusters ranked by proximity
- `percentile_rank` 4h → RSI 81st pct, 30d momentum 100th pct
- `predict_outcome` → `model_unavailable` (expected when no
  `outcome_predictor.joblib` on disk; handler returns structured error)
- `score_sentiment` → 0.86 `bullish` on a headline

**Graceful-degradation rules (all tools):**
- Missing model file → `{"success": false, "error": "model_unavailable", ...}`
- Missing OHLCV CSV → `{"success": false, "error": "percentile_unavailable", ...}`
- Missing transformers/torch → `{"success": false, "error": "model_unavailable"}`

### 🚨 Tripwires — must-read before extending any MCP server

1. **Never use `from __future__ import annotations`** in a FastMCP server
   module. FastMCP builds tool schemas via pydantic's TypeAdapter, which
   evaluates stringified hints with a minimal namespace — `Any` / domain
   types will not resolve. Either omit the future import, or use plain
   `dict` / `list` return types.
2. **ML classes consume `IndicatorSet`, not raw candles.** Every ML tool
   must bundle the feed + `IndicatorCalculator.compute()` pipeline.
   Shared module-level `_get_feed()` + `_calculator` singleton handles
   this cleanly.
3. **`BinanceFeed` must be `start()`-ed inside the running event loop.**
   Don't build it at import time; lazy-init via `_feed_lock` + `_get_feed()`.

### Next actions (Phase 1 follow-ups)

- [x] Add remaining 6 tools, each reusing `MLToolsService`.
- [x] `freshness_seconds` field on tool responses (uniform across servers).
- [ ] Add a pytest fixture that monkeypatches the feed to return fixture
      candles — avoid live Binance hits in CI.
- [ ] Smoke test with `npx @modelcontextprotocol/inspector python scripts/run_mcp_ml.py`.
- [ ] Train & drop `models/outcome_predictor.joblib` so `predict_outcome`
      returns probabilities instead of `model_unavailable`.

---

## Phase 2 — RAG MCP reader ✅ (ingestion split deferred)

### Scope decision

The original Phase 2 bundled **(a)** extracting the ingestion loop into a
standalone `scripts/run_ingestion.py` + systemd unit, and **(b)** the
MCP reader. We shipped **(b) only** this pass. The ingestion split is
deferred into the `src/trading_bot/` migration so the ingestion loop
moves just once — both into its new home and out of the trading cycle —
rather than shuffling it twice.

### MCP reader (`bot-mcp-rag`) — shipped

**Layout:** `src/mcp_servers/rag/{server,handlers,service,schemas}.py`
**Launcher:** `scripts/run_mcp_rag.py` (owns transport).

| Tool | Reads | Status |
|------|-------|--------|
| `retrieve_news(query, k)`                       | `news` collection         | ✅ |
| `retrieve_macro(query, k)`                      | `macro` collection        | ✅ |
| `retrieve_memory(query, symbol?, k)`            | `trade_memory` collection | ✅ |
| `ingestion_status()`                            | all three collections     | ✅ |

Zero external API keys — reads Chroma only.

**Timestamp normalisation.** The ingestion pipeline stores timestamps
inconsistently (`news` uses unix-epoch strings, `macro` uses ISO-8601,
`trade_memory` has `0` sentinels). `RAGToolsService.freshness_seconds()`
accepts both formats and `as_iso()` returns a uniform display value, so
clients never have to branch on format.

**Verified against live Chroma** (BTCUSDT/ETHUSDT corpus):
- `ingestion_status` → `news=1480, macro=167, trade_memory=155`
- `retrieve_news "bitcoin ETF inflow" k=2` → 2 items, freshness 256k s
- `retrieve_memory "long BTCUSDT breakout" symbol=BTCUSDT` → 2 items

### Ingestion split (deferred to trading_bot migration)

When the `src/trading_bot/` move happens, the ingestion loop moves from
`src/services/rag/ingestion_loop.py` → `src/trading_bot/rag/ingestion.py`
and gets its own launcher + systemd unit. At the same time, the
in-process ingestion call inside `futures_trader.py` / `runner.py` is
removed so the trading loop stays one-shot. See Phase 2 follow-ups
below.

### Staleness safety

- Each `retrieve_*` item carries `freshness_seconds` when parseable.
- `ingestion_status()` returns `latest_timestamps` per collection — use
  to page operators if a collection stalls.
- Research agent (Phase 6) can hard-skip docs older than a threshold.

---

## Phase 3 — Binance feed MCP ✅

**Server name:** `bot-mcp-binance`
**Layout:** `src/mcp_servers/binance/{server,handlers,service,schemas}.py`
**Launcher:** `scripts/run_mcp_binance.py` (owns transport).

Wraps methods on `BinanceFeed` (which is already async + JSON-shaped):

| Tool | Category | Wraps | Status |
|------|----------|-------|--------|
| `get_ohlcv`          | market data  | `BinanceFeed.get_ohlcv` | ✅ shipped |
| `get_ticker`         | market data  | `BinanceFeed.get_ticker` | ✅ shipped |
| `get_order_book`     | market data  | `BinanceFeed.get_order_book` | ✅ shipped |
| `get_funding_rate`   | market data  | `BinanceFeed.get_funding_rate` | ✅ shipped |
| `get_open_interest`  | market data  | `BinanceFeed.get_open_interest` | ✅ shipped |
| `get_balance`        | account view | `BinanceFeed.get_balance` | ✅ shipped |
| `get_open_positions` | account view | `BinanceFeed.get_open_positions` | ✅ shipped |
| `get_order_status`   | account view | `BinanceFeed.get_order_status` | ✅ shipped |
| `cancel_order`       | **safe write** | `BinanceFeed.cancel_order` | ✅ shipped |

**Verified end-to-end (live BTCUSDT call):**
- `get_ohlcv` 1h limit=3 → 3 candles, freshness ≈9 min
- `get_ticker` BTCUSDT → last 76 579, 24h change +1.19 %
- `get_order_book` depth=5 → best bid/ask with quantities

Feed lazy-inits inside the event loop behind the service's `_feed_lock`
(reuses the `shared/infrastructure/binance/feed.py` pattern).

### Why `cancel_order` is in this phase (not Phase 7)

- **Blast-radius asymmetry** — cancel can only shrink exposure; it never
  creates new risk. Order placement + bracket setup can open unbounded
  positions and must stay gated.
- **Operator ergonomics** — research agents need a kill-switch to close
  runaway orders without waiting for Phase 7. Pulling cancel out would
  leave a gap in incident response.
- **No dry-run scaffold needed** — the `BOT_MODE=dry_run` enforcement +
  "would-execute" logging in Phase 7 exists for order *creation*;
  cancellation doesn't benefit from it.

**Explicitly omit** order placement (`place_order`, bracket setup) —
Phase 7.

Credentials: loaded from `.env` by `get_settings()`. No secrets pass
through the MCP protocol.

---

## Phase 4 — Analysis MCP ✅

**Server name:** `bot-mcp-analysis`
**Launcher:** `scripts/run_mcp_analysis.py`
**Files:** `src/mcp_servers/analysis/{server,handlers,service,schemas}.py`

**Shipped tools (12):**

| Tool | Purpose |
|------|---------|
| `compute_indicators(symbol, timeframe, candle_limit)` | Canonical indicator set (RSI, MACD, BB, EMAs, ADX, ATR, OBV slope, choppiness, vol ratio, CCI) + `freshness_seconds`. |
| `get_snapshot(symbol, timeframe, candle_limit)` | Aggregated market snapshot — ticker + order book top + OHLCV + funding rate + open interest. |
| `analyze_signal(symbol, timeframe, …rsi thresholds)` | Directional Signal (STRONG_BUY…STRONG_SELL) + reasoning + full market-conditions dict. |
| `analyze_multi_tf(symbol, trading_timeframe, current_signal)` | Higher-timeframe alignment summary via the shared `MultiTimeframeAnalyzer` (cached per-TF). |
| `detect_patterns(symbol, timeframe, candle_limit)` | Double bottom/top, engulfing, nearest support & resistance. |
| `render_chart(symbol, timeframe, candle_limit)` | Base64 PNG chart (mplfinance when available, matplotlib fallback). |
| `compute_momentum(indicator, symbol, timeframe, params, last_n)` | RSI, MACD, stochastic, ROC, momentum, Williams %R, TSI, RMI, PPO, Coppock, KST, UO. |
| `compute_trend(indicator, ...)` | ADX, Supertrend, Ichimoku, Parabolic SAR, Vortex, TRIX, PFE, TD sequential. |
| `compute_volatility(indicator, ...)` | ATR, Bollinger, Keltner, Donchian, Chandelier, Choppiness, CCI, VHF, EBSW. |
| `compute_volume(indicator, ...)` | OBV, OBV slope, MFI, VWAP, TWAP, CMF, Force Index, PVT, A/D line, avg quote volume. |
| `compute_structure(indicator, ...)` | Support/resistance, Fibonacci retracement, Fib Bollinger, pivot points (standard + Fibonacci). |
| `compute_statistical(indicator, ...)` | Hurst, z-score, entropy, kurtosis, skew, stdev, variance, quantile, MAD, linreg, fear & greed. |

Category tools dispatch via `src/mcp_servers/analysis/indicator_catalog.py`,
which maps MCP indicator names → numba facade methods on
`TechnicalIndicators` (promoted from `src/services/analysis/indicators/`).
Overrides passed in `params` are type-coerced against the facade's
declared defaults so pydantic's float widening never breaks numba-typed
kernels.

**Live-verified** against BTCUSDT 1h (2026-04-18):
`compute_indicators` returns RSI/ADX/ATR with `freshness_seconds`;
`detect_patterns` returns support/resistance levels; `analyze_multi_tf`
returns a multi-line alignment summary; `render_chart` emits ~78 KB of
base64-encoded PNG. Tools were invoked in-process via FastMCP's
`Client(mcp)` — no real stdio transport needed for smoke checks.

**Shared vs server-specific split:**
- `IndicatorCalculator`, `PatternAnalyzer`, `ChartGenerator`,
  `MultiTimeframeAnalyzer` all live under
  `src/mcp_servers/shared/services/` so the trading loop and the
  analysis MCP share a single implementation (promoted during Phase 4
  prep — see `src/services/analysis/__init__.py` which now re-exports
  from the shared package).
- Analysis-specific code (response schemas, service wiring, tool
  handlers) is confined to `src/mcp_servers/analysis/`.

**Registration (`~/.claude/mcp.json`):**

```json
{
  "mcpServers": {
    "bot-analysis": {
      "command": "python",
      "args": ["scripts/run_mcp_analysis.py"],
      "cwd": "/home/tnguyen/source/personal/bot"
    }
  }
}
```

`npx @modelcontextprotocol/inspector python scripts/run_mcp_analysis.py`
for a local stdio smoke test.

---

## Phase 5 — Skills as MCP prompts ✅

**Server name:** `bot-mcp-skills`
**Launcher:** `scripts/run_mcp_skills.py`
**Files:** `src/mcp_servers/skills/{server,handlers,service,schemas}.py`

**Shipped surface:**

- 1 tool: `list_skills()` returns `{name, description, category}` for
  every playbook. Discovery path for clients without MCP Prompts
  support.
- 8 prompts: one per SKILL.md under
  `src/services/llm_trader/skills/`. Naming convention
  `skill.<folder-name>` (e.g. `skill.perp-funding-basis`,
  `skill.smc`). Each prompt returns the playbook body with a short
  header telling the model to synthesise rather than quote.

**Current catalogue** (auto-discovered at server startup):
`candlestick`, `crypto-derivatives`, `elliott-wave`, `harmonic`,
`ichimoku`, `perp-funding-basis`, `smc`, `technical-basic`.

**Shared vs server-specific split:**
- The server reads SKILL.md files directly from disk — it never
  imports trading-loop code. `SkillsService` owns a minimal YAML
  frontmatter parser (skill frontmatter is flat key/value).
- `src/services/llm_trader/skills_loader.py` stays in place as the
  trading-loop's in-process fallback so the bot still works without
  MCP. Both layers share the same SKILL.md files on disk, so edits
  propagate everywhere automatically.

**Live-verified** in-process via `Client(mcp)` (2026-04-18):
`prompts/list` returns all 8 skills; `get_prompt('skill.perp-funding-basis')`
returns the playbook body; `list_skills` tool returns the full catalogue
with categories.

**Registration (`~/.claude/mcp.json`):**

```json
{
  "mcpServers": {
    "bot-skills": {
      "command": "python",
      "args": ["scripts/run_mcp_skills.py"],
      "cwd": "/home/tnguyen/source/personal/bot"
    }
  }
}
```

`npx @modelcontextprotocol/inspector python scripts/run_mcp_skills.py`
for a local stdio smoke test.

---

## Phase 6 — Research agent loop ⬜

- New: `src/services/research/agent_loop.py` — multi-turn loop wiring
  all MCP servers into a single Claude agent session.
- Use case: **post-mortem / exploratory only**. Not the trading hot path.
- Entry point: `scripts/run_research.py "why did BTCUSDT stop out at 14:30?"`
- Must cap max-turns (default 10) and total token spend per session.

### Why multi-turn is unavoidable

A single LLM call can **request** a tool but cannot **execute** it.
Execution happens in the client loop:

```
turn 1: prompt + schemas → LLM → tool_use: get_ohlcv(...)
        (host runs tool, captures result)
turn 2: prompt + tool_result → LLM → tool_use: predict_direction(...)
        ...
turn N: prompt + results → LLM → final text / structured decision
```

Each arrow is a separate Claude API call. This is why we keep the
**live trading loop one-shot** and reserve MCP for research.

---

## Phase 7 — (Gated) order execution MCP ⬜

Only after Phase 6 is stable. Heavy guardrails:

- Server enforces `BOT_MODE=dry_run` unless an env flag is explicitly set.
- Every `place_order` / `set_brackets` call logs a "would-execute" line
  and requires explicit confirmation protocol.
- Tools: `place_order(symbol, side, qty, ...)`, `set_brackets(...)`
  (new entry creation only — `cancel_order` already lives in Phase 3
  as a safe write).
- Consider: per-call human approval via MCP's sampling capability.

---

## Cross-cutting checklist

Apply to every server:

- [ ] No `from __future__ import annotations` inside server modules.
- [ ] All tools return a JSON-serializable `dict` using the
      `tool_error()` envelope on failure.
- [ ] Model / prompt / DB version included in responses (debuggability).
- [ ] `freshness_seconds` on any cached / ingested data.
- [ ] pytest smoke test per server (no live Binance hits — fixture candles).
- [ ] Launcher docstring documents `mcp-inspector` + Claude Code
      registration.
- [ ] Entry in this doc's phase-status table flipped to ✅ when shipped.

---

## Dependency graph

```
Phase 0 ──► Phase 1 (ML)
         ├► Phase 2 (RAG)
         ├► Phase 3 (Binance)
         ├► Phase 4 (Analysis)
         └► Phase 5 (Prompts)
                │
                ▼
             Phase 6 (research agent)
                │
                ▼
             Phase 7 (executor, gated)
```

Phases 1–5 are independent and can be parallelized if needed.
Phases 6 and 7 depend on the earlier servers being available.

---

## Registration snippets

### Claude Code (`~/.claude/mcp.json`)

```json
{
  "mcpServers": {
    "bot-ml": {
      "command": "python",
      "args": ["scripts/run_mcp_ml.py"],
      "cwd": "/home/tnguyen/source/personal/bot"
    },
    "bot-rag": {
      "command": "python",
      "args": ["scripts/run_mcp_rag.py"],
      "cwd": "/home/tnguyen/source/personal/bot"
    },
    "bot-binance": {
      "command": "python",
      "args": ["scripts/run_mcp_binance.py"],
      "cwd": "/home/tnguyen/source/personal/bot"
    }
  }
}
```

### MCP Inspector (visual smoke test)

```bash
npx @modelcontextprotocol/inspector python scripts/run_mcp_ml.py
npx @modelcontextprotocol/inspector python scripts/run_mcp_rag.py
npx @modelcontextprotocol/inspector python scripts/run_mcp_binance.py
```

### In-process test (no MCP client)

```python
import asyncio
from fastmcp import Client
from src.mcp_servers.ml.server import mcp as ml_mcp
from src.mcp_servers.rag.server import mcp as rag_mcp

async def smoke():
    async with Client(ml_mcp) as c:
        print(await c.call_tool("predict_direction",
                                {"symbol": "BTCUSDT", "timeframe": "15m"}))
    async with Client(rag_mcp) as c:
        print(await c.call_tool("ingestion_status", {}))

asyncio.run(smoke())
```

---

## Notes for future Claude sessions resuming this work

- The working tree was dirty with the Binance SDK migration when this
  plan began. Check `git status` before layering more changes on top.
- `BINANCE_BASE_URL` must stay in `.env` — demo-fapi is **not** reachable
  via the testnet flag; see `src/infrastructure/binance/client.py`.
- `BINANCE_PRODUCT` was removed from `.env`; loader default flipped to
  `"usdt_futures"` (SDK is UM-futures only). Dead spot branches in
  `positions.py:107` and `risk_manager.py:439` are still there — remove
  when you touch those files for unrelated reasons.

---

## Post-MCP: `src/trading_bot/` restructure

Target layout — only two folders directly under `src/`:

```
src/
├── mcp_servers/     # all MCP servers (this doc's Phases 1–5, 7)
└── trading_bot/     # everything else (trading loop + RAG ingestion + infra)
    ├── bootstrap/
    ├── core/
    ├── domain/
    ├── infrastructure/
    ├── services/
    └── shared/
```

**Sequencing rule.** This restructure happens **after** all MCP servers
are shipped (Phases 1–5 ✅). Doing it earlier would require rewriting
MCP imports twice — once to the current `src.services.*` paths while
building each server, then again after the move. Ordering MCP-first
means every MCP server's `from src.services.ml.xxx import ...` becomes
one mechanical rewrite to `from src.trading_bot.services.ml.xxx import
...` at the end.

**Scope of the move** (all mechanical, no behaviour changes):

1. `git mv src/{bootstrap,core,domain,infrastructure,services,shared}
   src/trading_bot/`
2. Rewrite imports across the repo: `from src.<folder>.*` → `from
   src.trading_bot.<folder>.*`. Touch points: MCP servers, tests,
   scripts, `main.py`, any stray `import src.xxx`.
3. Update `src/trading_bot/__init__.py` (new).
4. Re-run the in-process MCP smoke tests above — zero logic should have
   changed.

**Ingestion split** (bundled with the move): when `src/services/rag/`
moves to `src/trading_bot/services/rag/`, also:

- add `scripts/run_ingestion.py` → wraps `trading_bot.services.rag.ingestion_loop`
- remove the in-process ingestion call from `futures_trader.py` /
  `runner.py` so the trading loop stays one-shot
- log to `logs/ingestion.log` so operators can monitor separately.

This is the one-and-only time we touch ingestion; bundling avoids a
second refactor.
