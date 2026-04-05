# ML Retrain Guide — Per Timeframe

How long to collect data and how often to retrain, broken down by trading timeframe.

---

## Data Requirements

Each model needs a minimum amount of OHLCV history before the first training.

| Timeframe | Candles/day | Direction + Anomaly (10K candles) | Percentile scorer (6 months) | Regime (200+ daily equiv.) |
|-----------|-------------|-----------------------------------|------------------------------|---------------------------|
| **1m**    | 1,440       | **7 days**                        | **180 days** (259,200)       | **200 days** (288,000)    |
| **5m**    | 288         | **35 days**                       | **180 days** (51,840)        | **200 days** (57,600)     |
| **15m**   | 96          | **104 days**                      | **180 days** (17,280)        | **200 days** (19,200)     |
| **1h**    | 24          | **417 days**                      | **180 days** (4,320)         | **200 days** (4,800)      |
| **4h**    | 6           | **4.5 years**                     | **180 days** (1,080)         | **200 days** (1,200)      |

> The percentile scorer doesn't require training — it scores live values against a rolling window of raw OHLCV data. But it needs **6 months** of candles to produce meaningful percentiles.

---

## Retrain Frequency

Rule of thumb from the direction classifier docstring: **retrain after every ~500 new candles.**

| Timeframe | 500 candles = | Recommended retrain |
|-----------|---------------|---------------------|
| **1m**    | ~8 hours      | **Daily** (or twice daily) |
| **5m**    | ~1.7 days     | **Every 2–3 days**  |
| **15m**   | ~5.2 days     | **Weekly**           |
| **1h**    | ~21 days      | **Every 2–3 weeks** |
| **4h**    | ~83 days      | **Monthly**          |

---

## Commands Per Timeframe

Replace `{TF}` with your timeframe (e.g. `1m`, `5m`, `15m`) and `{SYM}` with the lowercase symbol (e.g. `btcusdt`, `ethusdt`).

### Backfill data (run once)

```bash
python scripts/backfill_ohlcv.py --timeframe {TF} --symbol {SYM}
```

### Train all models

```bash
# Direction classifier (XGBoost)
python scripts/train_direction.py --timeframe {TF} --symbol {SYM}

# Anomaly detector (Isolation Forest)
python scripts/train_anomaly.py --timeframe {TF} --symbol {SYM}

# Regime classifier — only supports 15m and above
python scripts/train_regime.py --timeframe {TF} --symbol {SYM}

# Key levels (no timeframe needed — uses daily data)
python scripts/fit_key_levels.py
```

### Output files

```
models/
  xgboost_direction_{SYM}_{TF}.joblib
  isolation_forest_{SYM}_{TF}.joblib
  regime_classifier_{SYM}_{TF}.joblib
  key_levels_cache.json
```

---

## Current Limitations for 1m / 5m

| Component | Issue | Status |
|-----------|-------|--------|
| `train_regime.py` | `--timeframe` choices are `1d, 4h, 1h, 15m` — no 1m or 5m | Needs `_CPD` entries for 1m (1440) and 5m (288) added |
| `historical_percentile.py` | `_6M_CANDLES` missing 1m and 5m entries — falls back to 1,080 candles (18 hours at 1m) | Needs 1m (259,200) and 5m (51,840) entries added |
| `_CANDLES_PER_DAY` | Missing 1m and 5m | Needs 1m (1440) and 5m (288) entries added |

Until these are patched, 1m and 5m will use incorrect window sizes for percentile scoring and cannot train a regime classifier.

---

## Quick-Start Example: 1m BTC

```bash
# 1. Backfill at least 7 days of 1m data
python scripts/backfill_ohlcv.py --timeframe 1m --symbol btcusdt

# 2. Train direction + anomaly (regime not supported yet for 1m)
python scripts/train_direction.py --timeframe 1m --symbol btcusdt
python scripts/train_anomaly.py   --timeframe 1m --symbol btcusdt

# 3. Set ML_TIMEFRAME=1m in .env
#    ML_TIMEFRAME=1m

# 4. Retrain daily (or set up a cron job)
```

---

## Quick-Start Example: 5m ETH

```bash
# 1. Backfill at least 35 days of 5m data
python scripts/backfill_ohlcv.py --timeframe 5m --symbol ethusdt

# 2. Train all models (regime not supported yet for 5m)
python scripts/train_direction.py --timeframe 5m --symbol ethusdt
python scripts/train_anomaly.py   --timeframe 5m --symbol ethusdt

# 3. Set ML_TIMEFRAME=5m in .env
#    ML_TIMEFRAME=5m

# 4. Retrain every 2–3 days
```
