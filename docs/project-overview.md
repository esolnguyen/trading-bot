# Trading Bot — Project Overview

## What It Is

An AI-assisted cryptocurrency trading bot that trades BTC/USDT and ETH/USDT on Binance. It combines traditional technical analysis with LLM-powered decision-making and a RAG (Retrieval-Augmented Generation) pipeline for market context enrichment.

## Architecture

The project follows a **src-first clean architecture**:

```
src/
├── app.py                  # Entrypoint — wires all dependencies and runs the main loop
├── core/config/            # Settings loaded from .env
├── domain/                 # Pure domain models (no external dependencies)
│   ├── analysis/           # Signal, IndicatorSet, TechnicalAnalysis, PatternResult
│   ├── market/             # OHLCVCandle, MarketSnapshot
│   └── trading/            # Action, TradeDecision, Position, TradeRecord, TradingStatistics
├── services/               # Business logic and orchestration
│   ├── analysis/           # Indicator calculation, technical analysis, pattern detection, charting
│   │   └── indicators/     # Modular indicator library (momentum, trend, volatility, volume, etc.)
│   ├── rag/                # Ingestion loop, retriever, memory manager, data sources
│   └── trading/            # Trading loop, executor, risk manager, strategy, brain, statistics
├── infrastructure/         # External system adapters
│   ├── ai/                 # LLM providers (Google, OpenRouter, LM Studio, BlockRun, Azure)
│   ├── binance/            # Binance market data feed
│   ├── market_data/        # Additional market data adapters
│   └── storage/            # ChromaDB vector store, file-based persistence, vector memory
├── interfaces/notifiers/   # Output adapters (console, logger, Discord)
├── contracts/              # Protocol/interface definitions (e.g., RiskManagerProtocol)
├── shared/                 # Generic utilities (formatting, token counting, decorators)
├── dashboard/              # FastAPI web dashboard with live state
└── mcp_server/             # MCP tool server for external integrations
```

## How a Trading Cycle Works

```
1. Market Data        BinanceFeed → MarketAggregator
                      Fetches OHLCV candles, funding rate, open interest for each symbol

2. Technical Analysis IndicatorCalculator → TechnicalAnalyzer
                      Computes RSI, MACD, Bollinger Bands, EMA 20/50, ADX, ATR,
                      OBV slope, Choppiness Index → produces a directional Signal

3. Pattern Detection  PatternAnalyzer
                      Scans for double top/bottom, support/resistance levels,
                      engulfing candles on recent price action

4. RAG Context        RAGRetriever queries ChromaDB
                      Returns relevant news, macro data, fear/greed index,
                      DeFi TVL, historical OHLCV narratives

5. Context Assembly   ContextBuilder
                      Merges market snapshot, indicators, patterns, RAG context,
                      position state, brain insights, trading history, and
                      dynamic thresholds into a structured LLM prompt

6. LLM Decision       LLMManager.decide()
                      Sends system + user prompt to the configured AI provider
                      Parses response into a TradeDecision (action, symbol,
                      quantity, confidence, reasoning)

7. Risk Validation    RiskManager.validate()
                      Checks kill switches (consecutive losses, daily loss cap),
                      validates symbol, clamps position size, blocks if bot disabled

8. Execution          Executor
                      Places order on Binance (or simulates in dry-run mode)

9. Position Mgmt      TradingStrategy
                      Tracks open positions, monitors stop-loss / take-profit /
                      trailing stop, handles partial closes

10. Learning          TradingBrainService
                      Stores trade outcomes in vector memory, extracts insights,
                      adjusts thresholds for future cycles
```

## Key Components

### Technical Indicators
| Indicator | Purpose |
|-----------|---------|
| RSI (14) | Overbought/oversold detection |
| MACD (line, signal, histogram) | Momentum and trend direction |
| Bollinger Bands (20, 2σ) | Volatility envelope and mean reversion |
| EMA 20 / EMA 50 | Short and medium trend |
| ADX (14) | Trend strength (>25 = strong trend) |
| ATR (14) | Volatility magnitude for stop placement |
| OBV Slope | Volume-confirmed accumulation/distribution |
| Choppiness Index (14) | Regime filter (>61.8 = choppy, suppress entries) |

### RAG Data Sources
| Source | Data | Cadence |
|--------|------|---------|
| CryptoCompare | Crypto news articles | `news_interval` |
| CoinGecko | Fear & Greed index | `macro_interval` |
| Alternative.me | Fear & Greed (alt source) | `ohlcv_interval` |
| DefiLlama | DeFi TVL data | `ohlcv_interval` |
| OHLCV History | Historical candle narratives | `ohlcv_interval` |

### Risk Management
- Kill switches: max consecutive losses, max daily loss percentage
- Position sizing clamped to `max_order_usdt`
- Correlated pair protection (won't hold same-direction BTC + ETH)
- Bot enable/disable flag
- Trailing stop with configurable activation
- Partial take-profit levels

### AI Providers
Supports multiple LLM backends via `ProviderOrchestrator`:
- Google (Gemini)
- OpenRouter
- LM Studio (local)
- BlockRun
- Azure OpenAI

### Adaptive Brain
The `TradingBrainService` stores trade outcomes as vector embeddings in ChromaDB. On each closed trade it records market conditions, entry reasoning, and P&L. This builds a searchable memory that:
- Provides historical context to the LLM ("last time conditions were similar, we lost 2%")
- Suggests dynamic parameter adjustments
- Triggers periodic self-reflection after every N trades

### Persistence
- Positions saved to `data/position_{symbol}.json` — survives restarts
- Trade history logged to `logs/trades.csv`
- ChromaDB stores vector embeddings in `chroma_db/`

## Running

```bash
# Configure
cp .env.example .env
# Edit .env with API keys and settings

# Install
pip install -r requirements.txt

# Run
python -m src.app
```

## Current Decision-Making Approach

The bot uses an **LLM-as-decision-maker** pattern. Traditional indicators and patterns are computed deterministically, then packaged into a structured prompt. The LLM interprets all signals together and outputs a JSON trade decision. Rule-based risk management acts as a hard safety layer between the LLM output and actual execution.

There are **no traditional ML models** (classifiers, regressors, neural nets) in the current pipeline. All "intelligence" comes from the LLM plus hand-coded indicator thresholds.
