# Project Knowledge Base — AI Trading Bot

> Auto-generated from source code on 2026-04-03. Use this file as the authoritative
> reference for understanding the codebase structure, data flows, and design decisions.

---

## 1. Project Overview

An **AI-assisted cryptocurrency trading bot** that trades on Binance using:
- **LLM-powered decision-making** (the single decision-maker)
- **Traditional technical analysis** (indicators + patterns as LLM context)
- **RAG pipeline** (ChromaDB vector store for news, macro, and trade memory)
- **ML enrichment layer** (XGBoost, Isolation Forest, Random Forest, FinBERT — inform LLM, never override it)

### Core Design Principle
> The LLM is the *only* decision-maker. Indicators, ML models, and RAG context are translated
> into natural language and fed as prompt context. The only code that ever overrides the LLM
> is the **RiskManager** — a hard Python safety layer.

---

## 2. High-Level Trading Cycle (per tick)

```
1.  KILL SWITCH CHECK          RiskManager.check_kill_switches()
2.  DATA COLLECTION            BinanceFeed → OHLCVCandle[], ticker, funding rate, open interest
3.  ANOMALY GUARD              AnomalyDetector.is_anomaly() → skip cycle if anomalous (B4)
4.  OHLCV CSV APPEND           OHLCVWriter.append() → keep local training data fresh (E)
5.  REGIME REFRESH (daily)     CycleClassifier.predict() → injects macro regime into system prompt (A4)
6.  POSITION LIFECYCLE         TradingStrategy.check_position() → SL/TP/trailing stop/partial TP
7.  TECHNICAL ANALYSIS         IndicatorCalculator → IndicatorSet; TechnicalAnalyzer → Signal
8.  PATTERN DETECTION          PatternAnalyzer.analyze() → PatternResult
9.  MULTI-TF ALIGNMENT         MultiTimeframeAnalyzer.build_summary() (A2)
10. KEY LEVEL DETECTION        KeyLevelDetector.format_context() (A3)
11. DIRECTION PROBABILITY      DirectionClassifier.format_context() → P(bullish) (B2)
12. HISTORICAL PERCENTILE      HistoricalPercentileScorer.format_context() (A1)
13. RAG RETRIEVAL              RAGRetriever.retrieve() → news, macro, trade_memory from ChromaDB
14. CONTEXT BUILD              ContextBuilder.build() → (system_prompt, user_message)
15. LLM DECISION               LLMManager.decide() → TradeDecision
16. RISK VALIDATION            RiskManager.validate() → clamp qty, block on rules, outcome gate (B3)
17. EXECUTION                  Executor.execute() → TradeOutcome (or dry-run)
18. BRACKET ORDERS             Executor.place_bracket_orders() → exchange-native SL + TP orders
19. MEMORY RECORD              MemoryManager.record() → trade_memory in ChromaDB
20. PERSISTENCE                Persistence.append_trade() + async_save_position()
21. BRAIN LEARNING             TradingBrainService.update_from_closed_trade() (on close only)
```

A parallel **position monitor** task polls every 15 s and fires a market close if stop-loss is hit between main cycles.

---

## 3. Repository Layout

```
bot/
├── src/
│   ├── app.py                        # Entrypoint — build_runtime() + main() coroutine
│   ├── core/config/settings.py       # All runtime settings (dataclass, loaded from .env)
│   ├── domain/                       # Pure immutable models (no external deps)
│   │   ├── analysis/models.py        # Signal, IndicatorSet, TechnicalAnalysis, PatternResult
│   │   ├── market/models.py          # OHLCVCandle, MarketSnapshot
│   │   └── trading/models.py         # Action, TradeDecision, TradeOutcome, Position, TradeRecord, etc.
│   ├── services/
│   │   ├── analysis/
│   │   │   ├── indicator_calculator.py   # Computes IndicatorSet from candles (pure Python)
│   │   │   ├── technical_analyzer.py     # Maps IndicatorSet → Signal + reasoning text
│   │   │   ├── pattern_analyzer.py       # Double top/bottom, S/R levels, engulfing candles
│   │   │   ├── chart_generator.py        # mplfinance candlestick charts → base64 PNG
│   │   │   ├── context_builder.py        # Assembles system + user prompts from all signals
│   │   │   ├── market_aggregator.py      # Calls BinanceFeed for all symbols → dict[sym, MarketSnapshot]
│   │   │   ├── multi_timeframe_analyzer.py # 1H/4H/1D alignment summary (A2)
│   │   │   ├── prompt_templates.py       # build_system_prompt() — LLM contract text
│   │   │   └── indicators/               # Modular indicator library (momentum/trend/volatility/volume/etc.)
│   │   ├── ml/
│   │   │   ├── anomaly_detector.py       # B4: Isolation Forest — skip cycle on anomaly
│   │   │   ├── cycle_classifier.py       # A4: Random Forest regime — injects macro context to system prompt
│   │   │   ├── direction_classifier.py   # B2: XGBoost — P(bullish) for prompt context
│   │   │   ├── historical_percentile.py  # A1: CSV percentile scorer — where are we vs history?
│   │   │   ├── key_level_detector.py     # A3: DBSCAN S/R clusters from cache JSON
│   │   │   ├── outcome_predictor.py      # B3: Logistic Regression — hard gate in RiskManager
│   │   │   └── sentiment_scorer.py       # B1: FinBERT — news sentiment [-1,+1] at ingestion
│   │   ├── rag/
│   │   │   ├── ingestion_loop.py         # Async loop ingesting news/macro into ChromaDB
│   │   │   ├── retriever.py              # RAGRetriever — query ChromaDB for decision-time context
│   │   │   ├── memory_manager.py         # Records trade outcomes → trade_memory collection
│   │   │   ├── ohlcv_writer.py           # Appends candles to CSV for ML training
│   │   │   ├── filter.py                 # Dedup + relevance filter for news articles
│   │   │   └── sources/                  # Data source adapters (async fetch())
│   │   │       ├── cryptocompare.py      # Crypto news via CryptoCompare API
│   │   │       ├── coingecko.py          # Fear & Greed via CoinGecko
│   │   │       ├── alternative_me.py     # Fear & Greed via Alternative.me
│   │   │       ├── defillama.py          # DeFi TVL via DefiLlama
│   │   │       └── ohlcv_history.py      # Historical candle narratives
│   │   └── trading/
│   │       ├── trading_loop.py           # Main cycle orchestrator + position monitor
│   │       ├── trading_strategy.py       # Position lifecycle: open/check SL-TP/close + trailing stop
│   │       ├── executor.py               # Submits TradeDecision to Binance REST client
│   │       ├── risk_manager.py           # Hard safety gates (kill switch, corr, size clamp, outcome gate)
│   │       ├── brain_service.py          # Learn from closed trades; build adaptive brain context
│   │       ├── memory_service.py         # Rolling TradingMemory (recent decisions for prompt)
│   │       ├── statistics_service.py     # Win rate, Sharpe, drawdown stats persistence
│   │       └── statistics_calculator.py  # Pure calculation functions
│   ├── infrastructure/
│   │   ├── ai/
│   │   │   ├── llm_manager.py            # Routes prompt to active provider, parses JSON decision
│   │   │   ├── provider_orchestrator.py  # Round-robin across multiple providers ("all" mode)
│   │   │   ├── provider_types.py         # Provider enum
│   │   │   ├── unified_parser.py         # Extract TradeDecision from raw LLM text
│   │   │   └── providers/
│   │   │       ├── base.py               # Abstract BaseAIClient
│   │   │       ├── azure.py              # Azure OpenAI
│   │   │       ├── google.py             # Google Gemini (genai SDK)
│   │   │       ├── openrouter.py         # OpenRouter (base + fallback model)
│   │   │       ├── lmstudio.py           # LM Studio local inference
│   │   │       ├── blockrun.py           # BlockRun
│   │   │       └── mock.py               # Test mock provider
│   │   ├── binance/
│   │   │   ├── feed.py                   # BinanceFeed — async market data (OHLCV, ticker, funding, OI)
│   │   │   └── rest_client.py            # BinanceRestClient — HMAC-signed REST, time-sync, no ccxt
│   │   ├── ml/
│   │   │   └── model_store.py            # load(name) / load_json(name) — joblib/json from models/
│   │   └── storage/
│   │       ├── chroma_store.py           # ChromaDB wrapper (news, macro, trade_memory collections)
│   │       ├── persistence.py            # File-based persistence (positions JSON, trades CSV, stats)
│   │       └── vector_memory.py          # VectorMemoryService — semantic search over trading experiences
│   ├── interfaces/notifiers/
│   │   ├── base_notifier.py
│   │   ├── console_notifier.py
│   │   ├── logger_notifier.py
│   │   ├── discord_notifier.py           # Discord trade/signal notifications
│   │   └── filehandler.py + filehandler_components/  # Discord message lifecycle management
│   ├── mcp_server/                       # Standalone Binance MCP Server (FastMCP)
│   │   ├── server.py                     # MCP tool definitions (16 tools)
│   │   ├── security.py                   # Rate limiting + audit logging
│   │   └── tools/                        # One file per MCP tool
│   ├── dashboard/                        # FastAPI real-time dashboard (port 8000)
│   │   ├── server.py                     # DashboardServer (FastAPI + WebSocket)
│   │   ├── state.py                      # Shared dashboard state
│   │   ├── routers/                      # REST + WebSocket route handlers
│   │   └── static/                       # Frontend HTML/JS/CSS + module components
│   ├── contracts/risk_contract.py        # RiskManagerProtocol (structural typing)
│   └── shared/
│       ├── data_utils.py                 # SerializableMixin (to_dict/from_dict for dataclasses)
│       ├── decorators.py                 # Retry, timeout decorators
│       ├── format_utils.py               # FormatUtils for Discord message formatting
│       ├── timeframe_validator.py        # Validate Binance timeframe strings
│       ├── token_counter.py              # Estimate prompt token count
│       └── types.py                      # Shared type aliases
├── scripts/
│   ├── retrain_all.py                    # Retrain all ML models in sequence
│   ├── train_regime.py                   # Train Random Forest regime classifier (1D data)
│   ├── train_direction.py                # Train XGBoost direction classifier (15m data)
│   ├── train_anomaly.py                  # Train Isolation Forest anomaly detector
│   ├── train_outcome.py                  # Train Logistic Regression outcome predictor
│   ├── fit_key_levels.py                 # Run DBSCAN on historical closes → key_levels_cache.json
│   ├── backfill_ohlcv.py                 # Download historical OHLCV CSV data from Binance
│   └── apply_preset.py                   # Apply a config preset from config/presets.json to .env
├── config/presets.json                   # Named .env configuration presets
├── data/
│   ├── ohlcv/*.csv                       # Historical candle data (symbol_timeframe.csv)
│   └── position_{symbol}.json            # Saved open positions (survives restart)
├── models/
│   ├── xgboost_direction_{sym}_{tf}.joblib
│   ├── isolation_forest_{sym}_{tf}.joblib
│   ├── regime_classifier_{sym}_{tf}.joblib
│   ├── key_levels_{sym}_cache.json
│   └── (legacy: *_4h.joblib, *_5m.joblib)
├── logs/
│   ├── bot.log
│   └── trades.csv
├── tests/                                # 70 pytest tests
├── chroma_db/                            # ChromaDB persistent vector store
├── .env / .env.example
├── requirements.txt
├── pytest.ini
└── run_tests.sh
```

---

## 4. Domain Models (`src/domain/`)

### `Action` (enum)
```
BUY | SELL | HOLD | CLOSE | UPDATE | CLOSE_LONG | CLOSE_SHORT
```

### `TradeDecision` (slots dataclass)
| Field | Type | Description |
|---|---|---|
| `symbol` | str | e.g. `BTCUSDT` |
| `action` | Action | LLM output |
| `quantity` | float | Base currency quantity |
| `order_type` | str | `MARKET` or `LIMIT` |
| `price` | float\|None | Limit price |
| `reasoning` | str | LLM explanation |
| `confidence` | float | 0–100 |
| `source` | str | Provider name |

### `Position` (slots dataclass) — active position
Key fields: `entry_price`, `stop_loss`, `take_profit`, `size`, `direction` (LONG/SHORT), `symbol`, `trailing_stop_price`, `tp1_price`, `partial_tp1_hit`, `max_drawdown_pct`, `max_profit_pct` (MAE/MFE), and full confluence metadata for brain learning.

### `IndicatorSet` (slots dataclass)
`rsi_14`, `macd_line/signal/hist`, `bb_upper/mid/lower`, `ema_20`, `ema_50`, `volume_sma_20`, `atr`, `adx`, `obv_slope`, `choppiness`, `vol_ratio`, `cci_14`

### `MarketSnapshot` (slots dataclass)
`symbol`, `price`, `change_24h_pct`, `volume_24h`, `bid`, `ask`, `candles: list[OHLCVCandle]`, `funding_rate` (futures), `open_interest` (futures)

### `Signal` (enum)
`STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL`

---

## 5. Settings (`src/core/config/settings.py`)

All settings are a `@dataclass` loaded via `Settings.from_env()`.

### Key groups:

**Bot Core**
- `bot_enabled: bool = False` — master kill switch
- `bot_dry_run: bool = True` — simulate orders
- `bot_interval_seconds: int = 300` — cycle cadence
- `max_order_usdt: float = 50.0`
- `trading_symbols: list[str] = ["BTCUSDT","ETHUSDT"]`
- `crypto_pair: str = "BTC/USDT"` — primary display pair
- `timeframe: str = "1h"` — candle timeframe
- `candle_limit: int = 200`

**Binance**
- `binance_api_key`, `binance_api_secret`
- `binance_product: str = "spot"` — `spot | usdt_futures | coin_futures`
- `binance_testnet: bool = True`

**AI Provider**
- `provider: str = "azure"` — `azure | googleai | openrouter | local | blockrun | all`
- Per-provider: endpoint, API key, model name, temperature, max tokens
- `model_temperature: float = 0.7`; `model_max_tokens: int = 8192`

**Risk Management**
- `max_daily_loss_pct: float = 0.05`
- `max_consecutive_losses: int = 3`
- `min_confidence_threshold: float = 0.0` (0 = disabled)
- `default_stop_loss_pct: float = 0.02`
- `default_take_profit_pct: float = 0.04`
- `default_position_size: float = 0.02`

**Trailing Stop**
- `trailing_stop_enabled: bool = False`
- `trailing_stop_activation_pct: float = 0.01`
- `trailing_stop_distance_pct: float = 0.005`

**Partial Take-Profit**
- `partial_tp_enabled: bool = False`
- `partial_tp1_atr_multiplier: float = 2.0` (TP1 = entry ± ATR×2)
- `partial_tp1_size_pct: float = 0.5`

**Signal Thresholds**
- `choppiness_threshold: float = 61.8` — skip entries above this
- `signal_rsi_strong_buy: float = 30.0`
- `signal_rsi_buy: float = 40.0`
- `signal_rsi_sell: float = 60.0`
- `signal_rsi_strong_sell: float = 70.0`

**RAG**
- `chroma_path: str = "./chroma_db"`
- `news_interval: int = 900`, `macro_interval: int = 1800`, `ohlcv_interval: int = 3600`
- `cryptocompare_api_key`, `coingecko_api_key`

---

## 6. Trading Loop (`src/services/trading/trading_loop.py`)

### Class: `TradingLoop`

**Key attributes:**
- `_open_positions: dict[str, dict]` — in-memory position state (seeded from persistence on start)
- `_consecutive_losses: int` / `_daily_loss_pct: float` — kill-switch counters
- `_position_lock: asyncio.Lock` — prevents race between main cycle and position monitor

**Main methods:**
| Method | Purpose |
|---|---|
| `run()` | Main async loop; calls `run_cycle_once()` every `bot_interval_seconds` |
| `run_cycle_once()` | Full single-cycle pipeline (steps 1–21 above) |
| `run_position_monitor()` | Background task polling every 15s for SL hits |
| `stop()` | Sets `_stop_event` to cleanly exit |
| `_collect_snapshots()` | Parallel-fetch `MarketSnapshot` for all symbols |
| `_reconcile_open_positions()` | Detect positions closed by exchange TP/SL since last cycle |
| `_update_loss_tracking()` | Increment streak / daily loss on stop_loss or trailing_stop close |
| `_check_slippage()` | Log warning if executed price > `max_slippage_pct` from reference |

**Position state duality:**
- `_open_positions` dict — simple executor-path positions (SL/TP stored as raw floats)
- `trading_strategy.current_position` — rich `Position` dataclass with trailing stop, brain metadata

---

## 7. Risk Manager (`src/services/trading/risk_manager.py`)

Hard safety layer. Runs **after** the LLM, **before** the executor.

### Gates (in order):
1. **`check_kill_switches()`** — `consecutive_losses >= max` or `daily_loss_pct >= max` → halt entire cycle
2. **`bot_enabled=False`** → always HOLD
3. **Outcome predictor gate (B3)** — `OutcomePredictor.should_block()` → HOLD if `P(win) < 0.40`
4. **Confidence threshold** — HOLD if `decision.confidence < min_confidence_threshold` (when > 0)
5. **Correlation guard** — won't open BTCUSDT + ETHUSDT same-direction positions simultaneously
6. **Symbol allowlist** — only `trading_symbols` are accepted
7. **LIMIT order price** — required; block if missing
8. **Size clamp** — `notional = qty × price`; if `> max_order_usdt`, clamp quantity

### `CORRELATED_PAIRS`:
```python
frozenset({"BTCUSDT", "ETHUSDT"})
```

---

## 8. Executor (`src/services/trading/executor.py`)

Submits validated `TradeDecision` objects to Binance.

### Key methods:
| Method | Purpose |
|---|---|
| `execute(decision, dry_run)` | MARKET/LIMIT order placement; returns `TradeOutcome` |
| `place_bracket_orders(symbol, entry_side, sl_price, tp_price, qty)` | Places exchange-native SL + TP |
| `cancel_bracket_orders(symbol, sl_id, tp_id)` | Cancel open bracket orders |
| `_await_limit_fill(outcome, client)` | Poll until LIMIT order fills or timeout (then cancel) |
| `_format_quantity(symbol, qty, client)` | Rounds to exchange step size |
| `_apply_leverage(client)` | Set leverage on futures (from settings) |

**Futures bracket orders:** `STOP_MARKET` + `TAKE_PROFIT_MARKET` with `closePosition=true`  
**Spot bracket orders:** `STOP_LOSS_LIMIT` + `TAKE_PROFIT_LIMIT` with explicit quantity

---

## 9. LLM Manager (`src/infrastructure/ai/llm_manager.py`)

### Class: `LLMManager`

Handles: provider routing, prompt submission, JSON parsing, fallback logic.

**Providers** (set via `PROVIDER` env var):
| Value | Class |
|---|---|
| `azure` | Azure OpenAI |
| `googleai` | Google Gemini |
| `openrouter` | OpenRouter (base + fallback model) |
| `local` | LM Studio |
| `blockrun` | BlockRun |
| `all` | Round-robin via `ProviderOrchestrator` |

**`decide(system, user)` → `TradeDecision`:**
1. Calls active provider's `chat_completion()`
2. Extracts JSON from response via `UnifiedParser`
3. Maps to `TradeDecision`; falls back to HOLD on parse failure

**Vision support:** When `model_supports_vision=True`, chart base64 PNG is included in the user message.

---

## 10. ML Services (`src/services/ml/`)

All ML services:
- Load from `models/` directory using `model_store.load()` (joblib) or `load_json()`
- Load **per-symbol** bundles (e.g. `xgboost_direction_btcusdt_15m.joblib`) with a shared fallback
- Degrade gracefully if model files are missing (return `None`)
- Are **informational** — they add context to LLM prompts, never make final decisions (except B3/B4)

### ML Layer IDs:
| ID | Class | Role |
|---|---|---|
| A1 | `HistoricalPercentileScorer` | Where is current RSI/ATR/volume vs. 90-day history? |
| A2 | `MultiTimeframeAnalyzer` | 1H/4H/1D alignment summary for prompt |
| A3 | `KeyLevelDetector` | DBSCAN S/R cluster levels from 6-month history |
| A4 | `CycleClassifier` | Random Forest macro regime → system prompt suffix |
| B1 | `SentimentScorer` | FinBERT [-1,+1] news sentiment at ingestion |
| B2 | `DirectionClassifier` | XGBoost P(bullish) for prompt context |
| B3 | `OutcomePredictor` | Logistic Regression win gate in RiskManager (hard gate) |
| B4 | `AnomalyDetector` | Isolation Forest — skip entire cycle on anomaly |

### Regime labels (`CycleClassifier`):
- `BULL_TRENDING` — favour LONG on pullbacks
- `BULL_CORRECTION` — look for LONG re-entry at support
- `BEAR_TRENDING` — short only, tighten stops
- `ACCUMULATION` — range-bound, reduce frequency

---

## 11. RAG Pipeline (`src/services/rag/`)

### ChromaDB Collections
| Collection | Contents |
|---|---|
| `news` | CryptoCompare news articles (deduplicated by URL SHA256) |
| `macro` | Fear & Greed, TVL, OHLCV narratives |
| `trade_memory` | Past trade outcomes (kept to `trade_memory_max_entries`) |

### Ingestion (`IngestionLoop`)
- Runs 5 coroutines in parallel, each on its own cadence
- News: relevance-filtered, FinBERT sentiment scored, deduped
- Macro/OHLCV: timestamp-keyed, deduped

### Retrieval (`RAGRetriever`)
- Query built from `f"{symbol} price {price} signal {signal} {reasoning[:200]}"`
- Returns up to 5 news, 3 macro, 3 trade_memory docs
- Per-symbol keyword filtering on news (btc/bitcoin, eth/ethereum, etc.)
- Hard max: 4000 chars total output

### Embedding
- Default: `sentence-transformers/all-MiniLM-L6-v2` (via ChromaDB's `SentenceTransformerEmbeddingFunction`)
- Fallback: `DeterministicEmbeddingFunction` (SHA256-based, for offline/test)

---

## 12. Trading Strategy (`src/services/trading/trading_strategy.py`)

Manages the rich `Position` lifecycle for the LLM-trader path.

### Position checks (per cycle):
1. Update MAE/MFE metrics (`position.update_metrics()`)
2. Update trailing stop (ratchet as price improves)
3. Check stop-loss (includes trailing stop if active)
4. Check partial TP1 (if `partial_tp_enabled` and `tp1_price > 0`)
5. Check full take-profit
6. On any close: call `TradingBrainService.update_from_closed_trade()`

### Trailing Stop Logic:
- Activates when `pnl_pct >= trailing_stop_activation_pct`
- LONG: trail = `current_price × (1 - distance_pct)`, only moves up
- SHORT: trail = `current_price × (1 + distance_pct)`, only moves down

### Partial TP1:
- Closes `tp1_size_pct` (default 50%) of position at `tp1_price`
- Moves stop-loss to break-even after TP1 hit
- Records `PARTIAL_TP1_{direction}` trade record to persistence

---

## 13. Brain Service (`src/services/trading/brain_service.py`)

The adaptive learning layer. Stores and retrieves trading experiences via ChromaDB vector search.

### Key methods:
| Method | Purpose |
|---|---|
| `update_from_closed_trade(position, close_price, reason, ...)` | Store experience in vector memory |
| `get_context(current_conditions)` | Retrieve similar past trades for prompt injection |
| `get_parameter_suggestions()` | Suggest SL/TP/size based on winning patterns |
| `get_dynamic_thresholds()` | Adaptive RSI/confidence thresholds from win-rate analysis |
| `_trigger_reflection()` | Every 10 trades: synthesize win patterns into semantic rules |

### Context in LLM prompt:
- Similar past trades (cosine similarity search via `VectorMemoryService`)
- Temporal decay: older trades weighted less (90-day half-life)
- Win patterns, losing patterns, dynamic threshold suggestions

---

## 14. Context Builder (`src/services/analysis/context_builder.py`)

Assembles the final `(system_prompt, user_message)` pair.

### User message sections (priority order, budget = 8000 chars):
0. Market Snapshot (symbol, price, 24h%, signal, funding rate, OI)
1. Patterns Detected (double top/bottom, S/R, engulfing)
2. Current Position (entry, SL, TP, direction)
3. Adaptive Brain Insights / ML Context
4. Dynamic Thresholds
5. Trading History (recent decisions)
6. RAG Context (news, macro, trade_memory)
7. Chart Images (base64 PNG, if `model_supports_vision=True`)

Sections that don't fit within budget are replaced with `[OMITTED: section too large...]`.

### System prompt:
Built by `build_system_prompt(max_order_usdt, regime_suffix)` from `prompt_templates.py`.
- Defines allowed symbols, max order size, output format (JSON only)
- Appended with `CycleClassifier` regime instruction when model has been trained

---

## 15. Binance Client (`src/infrastructure/binance/`)

### `BinanceRestClient`
Pure Python HMAC-signed REST client (no ccxt).

**Base URLs:**
- Spot: `https://api.binance.com` / testnet: `https://testnet.binance.vision`
- Futures: `https://fapi.binance.com` / demo: `https://demo-fapi.binance.com`

**Key methods:** `get_ticker`, `get_klines`, `get_order_book`, `get_account`, `create_order`, `get_funding_rate`, `get_open_interest`, `get_position_risk`

**Time sync:** On init, fetches server time and computes `_time_offset_ms` for timestamp accuracy.

### `BinanceFeed`
Async wrapper around `BinanceRestClient`. Provides:
- `get_ohlcv(symbol, timeframe, limit)` → `list[OHLCVCandle]`
- `get_ticker(symbol)` → normalized dict with `last_price`, `bid_price`, etc.
- `get_order_book(symbol)` → bids/asks
- `get_funding_rate(symbol)` → futures only
- `get_open_interest(symbol)` → futures only

---

## 16. Persistence (`src/infrastructure/storage/persistence.py`)

| Method | Persists To |
|---|---|
| `append_trade(outcome, ts)` | `logs/trades.csv` |
| `load_position(symbol)` / `async_save_position(symbol, data)` | `data/position_{symbol}.json` |
| `async_save_trade_decision(record)` | `data/trade_decisions.json` |
| `save_statistics(stats)` / `load_statistics()` | `data/trading_statistics.json` |
| `save_memory(memory)` / `load_memory()` | `data/trading_memory.json` |
| `append_bot_log(entry)` | `logs/bot.log` |

---

## 17. Technical Indicators (`src/services/analysis/indicator_calculator.py`)

All implemented in pure Python (no TA-Lib dependency). Requires ≥50 candles.

| Indicator | Period | Purpose |
|---|---|---|
| RSI | 14 | Overbought/oversold |
| MACD | 12/26/9 | Momentum and trend direction |
| Bollinger Bands | 20, 2σ | Volatility envelope |
| EMA | 20, 50 | Short and medium trend |
| ADX | 14 | Trend strength (>25 = trending) |
| ATR | 14 | Volatility magnitude |
| OBV Slope | 20 | Volume-confirmed accumulation/distribution |
| Choppiness Index | 14 | Regime filter (>61.8 = choppy) |
| CCI | 14 | Commodity Channel Index |
| Volume Ratio | 20-SMA | `last_volume / vol_sma_20` |

---

## 18. Dashboard (`src/dashboard/`)

FastAPI server at port 8000, launched separately from the trading bot.

### Routers:
| Router | Path | Purpose |
|---|---|---|
| `ws_router` | `/ws` | WebSocket live feed |
| `monitor_router` | `/api/monitor` | Bot status, positions |
| `perf_router` | `/api/performance` | Trade statistics |
| `brain_router` | `/api/brain` | Brain insights |
| `visuals_router` | `/api/visuals` | Chart data |

Static frontend served from `src/dashboard/static/`.  
Frontend modules: `websocket.js`, `position_panel.js`, `statistics_panel.js`, `synapse_viewer.js`, `performance_chart.js`, `news_panel.js`, `log_viewer.js`, `vector_panel.js`.

---

## 19. MCP Server (`src/mcp_server/`)

Standalone **Binance MCP Server** built with FastMCP. Exposes 16 tools for external LLM clients.

### Tools:
**Market Data:** `get_ticker_price`, `get_ticker`, `get_order_book`, `get_available_assets`, `get_fee_info`  
**Account:** `get_balance`, `get_account_snapshot`  
**Trading:** `create_order`, `get_orders`  
**Portfolio:** `get_position_info`, `get_pnl`  
**Wallet:** `get_deposit_address`, `get_deposit_history`, `get_withdraw_history`, `get_universal_transfer_history`, `get_liquidation_history`

Security: rate limiting, input validation, audit logging (`security.py`).

---

## 20. Training Scripts (`scripts/`)

| Script | Model Output | Algorithm |
|---|---|---|
| `train_regime.py` | `regime_classifier_{sym}_1d.joblib` | Random Forest |
| `train_direction.py` | `xgboost_direction_{sym}_15m.joblib` | XGBoost |
| `train_anomaly.py` | `isolation_forest_{sym}_15m.joblib` | Isolation Forest |
| `train_outcome.py` | `outcome_predictor.joblib` | Logistic Regression |
| `fit_key_levels.py` | `key_levels_{sym}_cache.json` | DBSCAN clustering |
| `backfill_ohlcv.py` | `data/ohlcv/{sym}_{tf}.csv` | Binance klines download |
| `retrain_all.py` | All of the above | Orchestrates all training |

All scripts read from `data/ohlcv/*.csv`. Run `backfill_ohlcv.py` first.

---

## 21. Configuration Presets (`config/presets.json`)

Named bundles of `.env` settings. Apply with `python scripts/apply_preset.py <preset_name>`.

---

## 22. Testing

```bash
python3 -m pytest tests/ -q     # 70 tests
bash run_tests.sh                # Convenience wrapper
```

Test files are numbered by subsystem (`test_s0` through `test_s14`):
- `s1`: config/settings
- `s2`: contracts
- `s3`: BinanceFeed
- `s4`: indicators
- `s5`: patterns
- `s6`: ingestion
- `s7`: retriever
- `s8`: context builder
- `s9`: LLM manager
- `s10`: risk manager
- `s11`: executor
- `s12`: memory/brain
- `s13`: persistence
- `s14`: app wiring
- `test_dry_trading_cycle.py`: end-to-end dry run
- `test_binance_rest_client.py`: REST client unit tests

---

## 23. Key Design Patterns

### Graceful Degradation
All optional services (`VectorMemoryService`, `TradingBrainService`, `DiscordNotifier`, all ML services) are wrapped in `_try_build_*()` helpers that catch all exceptions and return `None`. The system continues at reduced capability.

### Async Architecture
- `trading_loop.run()` and `ingestion_loop.run()` run as concurrent `asyncio.Task`s
- Position monitor is a third concurrent task
- Discord notifier is a fourth
- All tasks are isolated via `_run_guarded()` (crash in one doesn't kill others)

### Dependency Injection
`build_runtime(settings)` in `app.py` wires all ~30 components and returns a dict. Testing injects mock versions.

### Prompt Budget Management
`ContextBuilder` enforces `MAX_USER_MESSAGE_CHARS = 8000`. Sections are sorted by priority; lower-priority sections are omitted with a placeholder when budget is exhausted.

### SerializableMixin
Domain models that need JSON persistence (Position, TradeRecord, etc.) inherit `SerializableMixin` which provides `to_dict()` / `from_dict()` via `dataclasses.asdict`.

---

## 24. Environment Setup

```bash
cp .env.example .env
# Fill in: BINANCE_API_KEY, BINANCE_API_SECRET, CRYPTOCOMPARE_API_KEY
# Set provider: PROVIDER=azure|googleai|openrouter|local|blockrun

# Install dependencies
pip install -r requirements.txt

# Train ML models (after backfilling data)
python scripts/backfill_ohlcv.py
python scripts/retrain_all.py

# Run (dry-run mode is safe default)
BOT_DRY_RUN=true python -m src.app
```

Optional heavy deps (uncomment in `requirements.txt` if needed):
- `numba` — JIT-compiled indicator speedup
- `torch` + `sentence-transformers` — neural embeddings for ChromaDB
- `discord.py` — Discord notifications

---

## 25. Data Flow Diagram

```
Binance REST ──► BinanceFeed ──► MarketAggregator ──► MarketSnapshot[]
                                                              │
                              ┌───────────────────────────────┤
                              ▼                               ▼
                    IndicatorCalculator              PatternAnalyzer
                              │                               │
                              ▼                               ▼
                    TechnicalAnalyzer                 PatternResult[]
                    (Signal + reasoning)
                              │
         ┌────────────────────┼─────────────────────┐
         ▼                    ▼                     ▼
    ML Services          RAGRetriever           ContextBuilder
 (A1-A4, B1-B4)        (ChromaDB query)        (prompt assembly)
         │                    │                     │
         └────────────────────┴──────────────────►──┘
                                                   │
                                                   ▼
                                            LLMManager.decide()
                                                   │
                                                   ▼
                                            TradeDecision
                                                   │
                                                   ▼
                                          RiskManager.validate()
                                                   │
                                          ┌────────┴────────┐
                                         PASS             BLOCK
                                          │               (→ HOLD)
                                          ▼
                                    Executor.execute()
                                          │
                                          ▼
                                     TradeOutcome
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                       MemoryManager           Persistence
                    (trade_memory ChromaDB)  (trades.csv, position JSON)
```
