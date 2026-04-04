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

# Symbols and timeframe
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT
FUTURES_LEVERAGE=50           # for 15m x50
```

---

## Step 3 — Apply a preset

Presets write all timeframe-related settings into `.env` at once.

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

> `FUTURES_LEVERAGE` is protected and never overwritten by presets — set it manually.

---

## Step 4 — ML setup (backfill + train)

Downloads OHLCV history and trains all ML models for every symbol in `TRADING_SYMBOLS`:

```bash
bash scripts/ml_setup.sh intraday_15m
```

This runs 6 steps:
1. Apply preset
2. Install dependencies
3. Backfill OHLCV data (15m + 1d) for each symbol
4. Fit key S/R levels per symbol
5. Train anomaly detector + direction classifier per symbol
6. Train regime classifier per symbol

Model files are saved to `models/`. Re-run weekly to keep models fresh.

To retrain only the direction and anomaly models without a full setup:

```bash
python scripts/train_direction.py --timeframe 15m --symbol btcusdt
python scripts/train_anomaly.py  --timeframe 15m --symbol btcusdt
# repeat for ethusdt, solusdt, bnbusdt
```

---

## Step 5 — Enable the bot

In `.env`:

```env
BOT_ENABLED=true
BOT_DRY_RUN=true    # start with dry-run, switch to false when confident
```

---

## Step 6 — Run

```bash
python -m src.app
```

The bot starts the trading loop and RAG ingestion loop in parallel. Cycle output looks like:

```
Cycle 1 2026-04-02T16:10:00Z BTC:NEUTRAL ETH:SELL SOL:SELL BNB:NEUTRAL Decision:SELL
cycle=1 BTCUSDT=SELL(passed) | ETHUSDT=SELL(passed) | SOLUSDT=HOLD(passed) | BNBUSDT=HOLD(passed) dry_run=True
```

---

## Optional — Dashboard

```bash
streamlit run dashboard.py
```

---

## Dry run → Live checklist

Before setting `BOT_DRY_RUN=false`:

- [ ] Orders appearing correctly in dry-run logs
- [ ] `FUTURES_LEVERAGE` set to intended value
- [ ] `BINANCE_TESTNET=false`
- [ ] `BINANCE_BASE_URL` empty (auto-selects live endpoint)
- [ ] `MAX_ORDER_USDT` set to your actual position size
- [ ] `BOT_DRY_RUN=false`
