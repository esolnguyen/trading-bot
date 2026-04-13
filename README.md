# AI Trading Bot

An AI-assisted cryptocurrency trading bot that trades on Binance using LLM-powered decision-making, traditional technical analysis, and a RAG pipeline for market context enrichment.

## How It Works

Each trading cycle runs the following pipeline:

```
1.  Market Data        BinanceFeed fetches OHLCV candles, funding rate, open interest
2.  Technical Analysis IndicatorCalculator → RSI, MACD, BB, EMA, ADX, ATR, OBV, Choppiness
3.  Pattern Detection  PatternAnalyzer → double top/bottom, S/R levels, engulfing candles
4.  ML Enrichment      Direction / regime / anomaly / outcome classifiers + signal scorer
5.  RAG Context        RAGRetriever queries ChromaDB → news, macro data, fear/greed, TVL
6.  Context Assembly   ContextBuilder merges all signals into a structured LLM prompt
7.  LLM Decision       LLMManager.decide() → TradeDecision (action, symbol, qty, confidence)
8.  Risk Validation    RiskManager → kill switches, futures-aware sizing, re-entry cooldown
9.  Execution          Executor places order on Binance (or simulates in dry-run mode)
10. Position Mgmt      TradingStrategy tracks stops, trailing stops, partial take-profit
11. Learning           TradingBrainService stores outcomes in vector memory for future cycles
```

The LLM is the single decision-maker. Indicators and ML outputs are translated into semantic language and fed as context — they never compete with or override the LLM. The risk manager enforces hard safety gates in Python.

## Architecture

```
src/
├── app.py                   # Entrypoint — wires all ~20 components and runs the main loop
├── core/config/             # Settings (pydantic dataclass, loaded from .env)
├── domain/                  # Pure domain models, no external dependencies
│   ├── analysis/            # Signal, IndicatorSet, TechnicalAnalysis, PatternResult
│   ├── market/              # OHLCVCandle, MarketSnapshot
│   └── trading/             # Action, TradeDecision, Position, TradeRecord, TradingStatistics
├── services/                # Business logic and orchestration
│   ├── analysis/            # IndicatorCalculator, TechnicalAnalyzer, PatternAnalyzer, ChartGenerator
│   │   └── indicators/      # Modular indicator library (momentum, trend, volatility, volume)
│   ├── ml/                  # Direction/regime/anomaly/outcome classifiers, key-level detector,
│   │                        #   historical percentile, sentiment scorer
│   ├── rag/                 # IngestionLoop, RAGRetriever, MemoryManager, data sources
│   └── trading/             # TradingLoop, Executor, RiskManager, TradingStrategy, BrainService,
│                            #   SignalScorer, StatisticsService
├── infrastructure/          # External system adapters
│   ├── ai/                  # LLM providers (Azure, Google, OpenRouter, LM Studio, BlockRun)
│   ├── binance/             # Direct REST client (no ccxt dependency)
│   ├── ml/                  # ModelStore — loads joblib classifiers from models/
│   └── storage/             # ChromaDB vector store, file-based persistence
├── interfaces/notifiers/    # Output adapters (console, logger, Discord)
├── contracts/               # Protocol/interface definitions
└── shared/                  # Generic utilities (formatting, token counting, decorators)

scripts/                     # Offline tooling (not part of the runtime loop)
├── backtest.py              # Replay historical candles through the strategy
├── backfill_ohlcv.py        # Seed ChromaDB / model training data
├── train_direction.py       # XGBoost direction classifier
├── train_regime.py          # Regime classifier
├── train_anomaly.py         # Isolation-forest anomaly detector
├── train_outcome.py         # Trade-outcome predictor
├── fit_key_levels.py        # Support/resistance level fitter
├── retrain_all.py           # Runs every trainer end-to-end
└── apply_preset.py          # Apply a saved config preset to .env
```

## Technical Indicators

| Indicator | Purpose |
|---|---|
| RSI (14) | Overbought/oversold detection |
| MACD (line, signal, histogram) | Momentum and trend direction |
| Bollinger Bands (20, 2σ) | Volatility envelope and mean reversion |
| EMA 20 / EMA 50 | Short and medium trend |
| ADX (14) | Trend strength (>25 = strong trend) |
| ATR (14) | Volatility magnitude for stop placement |
| OBV Slope | Volume-confirmed accumulation/distribution |
| Choppiness Index (14) | Regime filter (>61.8 = choppy, suppress entries) |

## RAG Data Sources

| Source | Data | Cadence |
|---|---|---|
| CryptoCompare | Crypto news articles | `NEWS_INTERVAL` |
| CoinGecko | Fear & Greed index | `MACRO_INTERVAL` |
| Alternative.me | Fear & Greed (alt source) | `OHLCV_INTERVAL` |
| DefiLlama | DeFi TVL data | `OHLCV_INTERVAL` |
| OHLCV History | Historical candle narratives | `OHLCV_INTERVAL` |

## AI Providers

Configured via the `PROVIDER` env var. Supported values:

| Value | Backend |
|---|---|
| `azure` | Azure OpenAI (default) |
| `googleai` | Google Gemini |
| `openrouter` | OpenRouter (cloud aggregator) |
| `local` | LM Studio (local inference) |
| `blockrun` | BlockRun |
| `all` | Round-robin across all configured providers |

## Setup

### 1. Clone and install

```bash
git clone git@github.com:esolnguyen/trading_bot.git bot
cd bot
pip install -r requirements.txt
```

For optional heavy dependencies:
```bash
pip install numba        # JIT-compiled indicators (faster)
pip install torch        # Neural embeddings for ChromaDB
pip install discord.py   # Discord notifier
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your credentials. Required fields:

```env
BINANCE_API_KEY=
BINANCE_API_SECRET=
CRYPTOCOMPARE_API_KEY=
```

### 3. Run

```bash
# Dry run (simulated orders, safe to test)
BOT_DRY_RUN=true python -m src.app

# Live trading (real orders)
BOT_DRY_RUN=false BOT_ENABLED=true python -m src.app
```

## Configuration Reference

### Core Bot

| Variable | Default | Description |
|---|---|---|
| `BOT_ENABLED` | `false` | Enable/disable trading (kill switch) |
| `BOT_DRY_RUN` | `true` | Simulate orders without placing them |
| `BOT_INTERVAL_SECONDS` | `300` | Seconds between trading cycles |
| `MAX_ORDER_USDT` | `50` | Maximum order size in USDT |
| `TRADING_SYMBOLS` | `BTCUSDT,ETHUSDT` | Comma-separated symbols to trade |
| `CRYPTO_PAIR` | `BTC/USDT` | Primary pair for context display |
| `TIMEFRAME` | `1h` | Candle timeframe |
| `CANDLE_LIMIT` | `200` | Number of candles per fetch |

### Binance

| Variable | Default | Description |
|---|---|---|
| `BINANCE_API_KEY` | — | Required |
| `BINANCE_API_SECRET` | — | Required |
| `BINANCE_PRODUCT` | `usdt_futures` | `spot`, `usdt_futures`, or `coin_futures` |
| `BINANCE_TESTNET` | `true` | Use Binance testnet |
| `BINANCE_BASE_URL` | _(auto)_ | Override endpoint URL |

### AI Provider

| Variable | Default | Description |
|---|---|---|
| `PROVIDER` | `azure` | Active provider (see table above) |
| `AZURE_ENDPOINT` | — | Azure OpenAI endpoint |
| `AZURE_API_KEY` | — | Azure API key |
| `AZURE_DEPLOYMENT` | — | Azure deployment name |
| `GOOGLE_STUDIO_API_KEY` | — | Google AI Studio key |
| `OPENROUTER_API_KEY` | — | OpenRouter key |
| `OPENROUTER_BASE_MODEL` | `google/gemini-2.5-pro` | Primary model |
| `OPENROUTER_FALLBACK_MODEL` | `deepseek/deepseek-r1:free` | Fallback model |
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | Local inference endpoint |
| `BLOCKRUN_WALLET_KEY` | — | BlockRun wallet key |
| `MODEL_TEMPERATURE` | `0.7` | Sampling temperature |
| `MODEL_MAX_TOKENS` | `8192` | Max output tokens |

### Risk Management

| Variable | Default | Description |
|---|---|---|
| `MAX_DAILY_LOSS_PCT` | `0.05` | Halt after losing this % of capital in one day |
| `MAX_CONSECUTIVE_LOSSES` | `3` | Halt after this many losses in a row |
| `MIN_CONFIDENCE_THRESHOLD` | `0.0` | Skip LLM decisions below this confidence (0–100; 0 = disabled) |
| `DEFAULT_STOP_LOSS_PCT` | `0.02` | Default stop-loss distance (2%) |
| `DEFAULT_TAKE_PROFIT_PCT` | `0.04` | Default take-profit distance (4%) |
| `DEFAULT_POSITION_SIZE` | `0.02` | Default position size as fraction of capital |
| `REENTRY_COOLDOWN_CYCLES` | `3` | Cycles to wait after a closed position before re-entering the same symbol |

### Trailing Stop

| Variable | Default | Description |
|---|---|---|
| `TRAILING_STOP_ENABLED` | `false` | Enable trailing stop |
| `TRAILING_STOP_ACTIVATION_PCT` | `0.01` | Profit % required to activate |
| `TRAILING_STOP_DISTANCE_PCT` | `0.005` | Trail distance from best price |

### Partial Take-Profit

| Variable | Default | Description |
|---|---|---|
| `PARTIAL_TP_ENABLED` | `false` | Enable partial take-profit |
| `PARTIAL_TP1_ATR_MULTIPLIER` | `2.0` | TP1 = entry ± (ATR × this); full TP uses 4× |
| `PARTIAL_TP1_SIZE_PCT` | `0.5` | Fraction of position closed at TP1 |

### RAG / ChromaDB

| Variable | Default | Description |
|---|---|---|
| `CRYPTOCOMPARE_API_KEY` | — | Required for news ingestion |
| `COINGECKO_API_KEY` | — | Optional (higher rate limits) |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB storage path |
| `NEWS_INTERVAL` | `900` | News fetch cadence (seconds) |
| `MACRO_INTERVAL` | `1800` | Macro data fetch cadence (seconds) |
| `OHLCV_INTERVAL` | `3600` | OHLCV history fetch cadence (seconds) |

### Discord (optional)

| Variable | Default | Description |
|---|---|---|
| `DISCORD_BOT_ENABLED` | `false` | Enable Discord notifier |
| `BOT_TOKEN_DISCORD` | — | Discord bot token |
| `GUILD_ID_DISCORD` | — | Discord server ID |
| `MAIN_CHANNEL_ID` | — | Channel for trade notifications |

## Persistence

| Path | Contents |
|---|---|
| `data/position_{symbol}.json` | Open positions (survives restarts) |
| `logs/trades.csv` | Full trade history log |
| `chroma_db/` | Vector embeddings for RAG and brain memory |
| `models/` | Trained joblib classifiers (direction, regime, anomaly) and key-level caches |

## Backtesting

Replay the strategy against historical candles without placing orders:

```bash
python3 scripts/backtest.py --csv data/ohlcv/btcusdt_15m.csv --symbol BTCUSDT
```

## Tests

```bash
python3 -m pytest tests/ -q
```

91 tests. ChromaDB pydantic deprecation warnings are harmless.
