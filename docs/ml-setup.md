# ML Setup — Step by Step

The ML stack is already implemented in the codebase. You only need to supply data and train the models. **No code changes required.**

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

`xgboost`, `scikit-learn`, `joblib`, `transformers`, `torch`, `numpy`, `pandas`, and `scipy` are all in `requirements.txt`.

---

## Step 2 — Backfill historical OHLCV data (run once)

Downloads market data from Binance (no auth required).

```bash
python scripts/backfill_ohlcv.py
```

Creates:
- `data/btcusdt_1d.csv` — ~1,500 daily candles (4 years)
- `data/btcusdt_15m.csv` — ~70,000 fifteen-minute candles (~7 weeks)

---

## Step 3 — Fit key support/resistance levels

No training required — just DBSCAN clustering on the daily data you just downloaded.

```bash
python scripts/fit_key_levels.py
```

Creates `models/key_levels_cache.json`. Ready immediately.

---

## Step 4 — Train the models

Run in this order. Each script prints validation metrics before saving.

```bash
# Anomaly detector (needs 5,000+ 15m candles — you have 70,000 after backfill)
python scripts/train_anomaly.py

# XGBoost direction classifier (same data requirement)
python scripts/train_direction.py

# Regime/cycle classifier (needs 200+ daily candles — you have ~1,500)
python scripts/train_regime.py

# Logistic regression trade gate — SKIP for now
# Needs 50+ closed trades from live history. Run later when you have enough.
# python scripts/train_outcome.py
```

After this, `models/` will contain:

```
models/
  key_levels_cache.json       ← A3: key S/R levels
  isolation_forest.joblib     ← B4: anomaly detector
  xgboost_direction.joblib    ← B2: direction classifier
  regime_classifier.joblib    ← A4: market regime
  (outcome_predictor.joblib)  ← B3: when you have 50+ trades
```

---

## Step 5 — Start the bot normally

```bash
python -m src.app
```

`ModelStore` loads every `.joblib` file at startup automatically. If a file exists, the feature is active. If a file is missing, that feature is silently skipped — the bot runs exactly as before.

**No settings or env vars to change.**

---

## What each model does at runtime

| When | Model | Effect |
|---|---|---|
| Every cycle start | B4 Isolation Forest | Skips the entire cycle if market anomaly detected |
| Every cycle start | OHLCV writer | Appends last closed candle to `data/btcusdt_15m.csv` |
| Once per day | A4 Regime Classifier | Updates system prompt (e.g. "BEAR MARKET: no LONG positions") |
| During analysis | A1 Percentiles + A2 Multi-TF + A3 Key Levels + B2 XGBoost | Injected into LLM prompt as `## ML Context` |
| Before execution | B3 LogReg gate | Blocks trade in `RiskManager` if win probability < 40% |
| During news ingestion | B1 FinBERT | Scores each article; enables hybrid ChromaDB queries |

---

## Retraining schedule

| Model | Cadence | Command |
|---|---|---|
| B2 XGBoost + B4 Isolation Forest | Weekly | `train_direction.py` + `train_anomaly.py` |
| A3 Key Levels | Weekly | `fit_key_levels.py` |
| A4 Regime Classifier | Monthly | `train_regime.py` |
| B3 LogReg Gate | Every 50 closed trades | `train_outcome.py` |
| All at once | Auto-checks triggers | `scripts/retrain_all.py` |

---

## FinBERT (B1) — separate note

FinBERT is not trained offline — it uses pretrained Hugging Face weights (~400MB). It loads lazily the first time a news article is ingested. No script to run; it works as soon as `transformers` and `torch` are installed.
