# Running the Bot

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Create your `.env`

```bash
cp .env.example .env
```

Fill in the required keys:

```env
# Binance
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BINANCE_PRODUCT=usdt_futures
BINANCE_TESTNET=true          # false for live trading

# AI provider — pick ONE
PROVIDER=googleai
GOOGLE_STUDIO_API_KEY=your_key
# or PROVIDER=openrouter + OPENROUTER_API_KEY=...
# or PROVIDER=azure + AZURE_ENDPOINT/AZURE_API_KEY/AZURE_DEPLOYMENT

# News/RAG
CRYPTOCOMPARE_API_KEY=your_key

# Symbols
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT
SINGLE_SYMBOL_DECISION=false  # true = one LLM call evaluates all symbols
```

---

## Step 3 — Pick a trading engine

`TRADING_ENGINE` selects which decision path drives the main loop:

| Value | Decision maker | Uses indicators? | Uses ML? | Uses RAG? | Cost |
|---|---|---|---|---|---|
| `scorer` | Deterministic signal scorer | yes | no | no | free (no LLM) |
| `llm_skills` | LLM with playbook skills + chart only | no | no | no | low |
| `llm_enriched` | LLM with skills + indicators + ML + RAG | yes | yes | yes | high |

```env
TRADING_ENGINE=scorer
TRADER_SKILLS=candlestick,technical-basic,smc,crypto-derivatives,perp-funding-basis
```

`TRADER_SKILLS` is only consumed by the LLM modes. Skill names map to markdown
files under `src/services/llm_trader/skills/`.

---

## Step 4 — Apply a preset

Presets write all timeframe-related settings (timeframe, leverage, SL/TP,
scoring weights, RSI thresholds, choppiness, etc.) into `.env` at once.

```bash
# Preview without writing
python scripts/apply_preset.py intraday_15m --dry-run

# Apply
python scripts/apply_preset.py intraday_15m
```

Available presets:

| Preset | Timeframe | Leverage hint |
|---|---|---|
| `scalping_1m` | 1m | x20–x50 |
| `scalping_5m` | 5m | x10–x20 |
| `intraday_15m` | 15m | x5–x10 |
| `swing_1h` | 1h | x2–x5 |
| `position_4h` | 4h | x1–x3 |

Leverage and scoring weights ship with each preset and are overwritten when
you apply it. API keys, `BOT_MODE`, `TRADING_ENGINE`, `TRADER_SKILLS`,
`MAX_ORDER_USDT`, `TRADING_SYMBOLS`, and `SINGLE_SYMBOL_DECISION` are
protected and never touched.

---

## Step 5 — ML setup (backfill + train)

Only needed when `TRADING_ENGINE=llm_enriched`. Downloads OHLCV history and
trains all ML models for every symbol in `TRADING_SYMBOLS`:

```bash
bash scripts/ml_setup.sh intraday_15m
```

This runs 6 steps:
1. Apply preset
2. Install dependencies
3. Backfill OHLCV data (timeframe + 1d) for each symbol
4. Fit key S/R levels per symbol
5. Train anomaly detector + direction classifier per symbol
6. Train regime classifier per symbol

Model files are saved to `models/`. Re-run weekly to keep models fresh.

To retrain direction and anomaly models without a full setup:

```bash
python scripts/train_direction.py --timeframe 15m --symbol btcusdt
python scripts/train_anomaly.py  --timeframe 15m --symbol btcusdt
# repeat for ethusdt, solusdt, bnbusdt
```

---

## Step 6 — Pick a bot mode

`BOT_MODE` is the master switch:

| Value | Effect |
|---|---|
| `off` | Master kill switch — always HOLD, no orders ever placed |
| `dry_run` | Simulate orders and log what would have happened |
| `live` | Place real orders on Binance |

```env
BOT_MODE=dry_run    # start here, switch to live when confident
```

---

## Step 7 — Run

```bash
python -m src.app
```

The bot starts the trading loop and RAG ingestion loop in parallel. Cycle
output looks like:

```
Cycle 1 2026-04-02T16:10:00Z BTC:NEUTRAL ETH:SELL SOL:SELL BNB:NEUTRAL Decision:SELL
cycle=1 BTCUSDT=SELL(passed) | ETHUSDT=SELL(passed) | SOLUSDT=HOLD(passed) | BNBUSDT=HOLD(passed) mode=dry_run
```

---

## Optional — Dashboard

```bash
python -m src.dashboard.server    # FastAPI, port 8000
```

---

## Dry run → Live checklist

Before setting `BOT_MODE=live`:

- [ ] Orders appearing correctly in dry-run logs
- [ ] `FUTURES_LEVERAGE` set to intended value (applied by preset)
- [ ] `BINANCE_TESTNET=false`
- [ ] `BINANCE_BASE_URL` empty (auto-selects live endpoint)
- [ ] `MAX_ORDER_USDT` set to your actual position size
- [ ] `BOT_MODE=live`
