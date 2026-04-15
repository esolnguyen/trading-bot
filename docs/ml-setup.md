# ML Setup — Step by Step

The ML stack is already implemented in the codebase. You only need to supply
data and train the models. **No code changes required.**

ML models are only needed when `TRADING_ENGINE=llm_enriched` (or when
`trading_engine=scorer` relies on B2/B3/B4 in the signal scorer). The
`llm_skills` and pure `scorer` modes run without ML files.

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

`xgboost`, `scikit-learn`, `joblib`, `transformers`, `torch`, `numpy`,
`pandas`, and `scipy` are all in `requirements.txt`.

---

## Step 2 — Apply a preset

Picks the timeframe ML will be trained for (`TIMEFRAME`, `ML_TIMEFRAME`, etc.):

```bash
python scripts/apply_preset.py intraday_15m
```

---

## Step 3 — One-shot setup script

The fastest path — backfills data, fits key levels, and trains every model
for every symbol in `TRADING_SYMBOLS`:

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

Skip to Step 7 once this completes. The manual steps below are for running
individual scripts when you want finer control.

---

## Step 4 — Backfill historical OHLCV data (manual)

Downloads market data from Binance (no auth required).

```bash
python scripts/backfill_ohlcv.py
```

Creates per-symbol CSVs under `data/ohlcv/`:
- `data/ohlcv/btcusdt_1d.csv` — ~1,500 daily candles (4 years)
- `data/ohlcv/btcusdt_15m.csv` — ~70,000 fifteen-minute candles (~7 weeks)
- …and the same pair for each symbol in `TRADING_SYMBOLS`

---

## Step 5 — Fit key support/resistance levels (manual)

No training required — just DBSCAN clustering on the daily data.

```bash
python scripts/fit_key_levels.py --symbol btcusdt
```

Creates `models/key_levels_btcusdt_cache.json`. Repeat per symbol.

---

## Step 6 — Train the models (manual)

Pass `--symbol` and `--timeframe` to each script. Repeat for every symbol.

```bash
# Anomaly detector
python scripts/train_anomaly.py --symbol btcusdt --timeframe 15m

# XGBoost direction classifier
python scripts/train_direction.py --symbol btcusdt --timeframe 15m

# Regime/cycle classifier (always uses 1d data)
python scripts/train_regime.py --symbol btcusdt

# Logistic regression trade gate — SKIP until you have 50+ closed trades
# python scripts/train_outcome.py
```

Or use the orchestrator that walks all symbols in `TRADING_SYMBOLS`:

```bash
python scripts/retrain_all.py
```

After this, `models/` will contain per-symbol bundles:

```
models/
  key_levels_btcusdt_cache.json       ← A3: key S/R levels
  isolation_forest_btcusdt_15m.joblib ← B4: anomaly detector
  xgboost_direction_btcusdt_15m.joblib ← B2: direction classifier
  regime_classifier_btcusdt_1d.joblib ← A4: market regime
  (outcome_predictor.joblib)          ← B3: when you have 50+ trades
```

---

## Step 7 — Start the bot normally

```bash
python -m src.app
```

`ModelStore` loads every `.joblib` file at startup automatically. If a file
exists, the feature is active. If a file is missing, that feature is silently
skipped — the bot runs exactly as before.

---

## What each model does at runtime

| When | Model | Effect |
|---|---|---|
| Every cycle start | B4 Isolation Forest | Skips the entire cycle if market anomaly detected |
| Every cycle start | OHLCV writer | Appends last closed candle to `data/ohlcv/{sym}_{tf}.csv` |
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
| All at once | Per-symbol | `scripts/retrain_all.py` |

---

## FinBERT (B1) — separate note

FinBERT is not trained offline — it uses pretrained Hugging Face weights
(~400MB). It loads lazily the first time a news article is ingested. No
script to run; it works as soon as `transformers` and `torch` are installed.
