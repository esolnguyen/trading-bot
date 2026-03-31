# ML & Context Augmentation — Trading Bot Roadmap

## The Core Framing

This bot is **LLM-driven with RAG**. Every ML addition must be evaluated through one question:

> *Does this make the LLM smarter, or does it try to replace it?*

### The Two-Brain Conflict

If you train a Deep Learning model (e.g., TFT) that predicts "90% chance up," and the LLM reads bad news and says "HOLD" — **who wins?** There is no principled answer. Two opaque systems disagreeing creates an architecture with no clear authority.

The correct mental model is:

```
ML Models        → pre-process and translate numbers into semantic language
RAG Pipeline     → retrieve and compress relevant memory/news
LLM              → the single decision-maker, reasoning over rich context
Risk Manager     → hard gates enforced in Python, not delegated to LLM
```

ML models should never compete with the LLM. They feed it or guard it.

### Three Categories

| Category | Role | Example |
|---|---|---|
| **Must-Haves** | Fix LLM weaknesses — math, amnesia, macro context | Percentiles, multi-TF, key levels |
| **Nice-to-Haves** | Guards/filters outside the LLM | XGBoost confidence input, LogReg kill-switch |
| **Traps** | Replace the LLM rather than improving it | GRU, TFT, TCN, N-BEATS, Autoencoder |

---

## Part A — Must-Haves: LLM Context Upgrades

These directly fix the LLM's core weaknesses. None require a trained model file. All translate numbers into semantic language the LLM reasons about naturally.

---

### A1. Historical Percentile Scorer *(MVP — do this first)*

**What it does:** For each current indicator, computes its percentile rank over the last N months. Turns raw math into semantic concepts.

**Why LLMs need this:** "RSI is 65" is meaningless without context. "RSI is at the 88th percentile over the last 6 months (historically overextended)" is a concept an LLM instantly understands and can reason about.

**Example LLM prompt injection:**
```
## Historical Context (6-month percentiles)
- RSI: 65 → 88th pct (historically overextended)
- ATR%: 2.1% → 91st pct (near extreme volatility)
- Volume: 1.3× SMA → 55th pct (normal)
- Funding Rate: +0.012% → 88th pct (longs heavily overextended)
- Price vs 6M high: -8% (near top of historical range)
- Price vs 6M low: +61% (well above historical floor)
```

**Implementation:** `numpy.percentileofscore()` on a rolling CSV window. Zero new dependencies if numpy is present. ~30 lines of code.

**Integration point:** New `HistoricalPercentileScorer` in `services/ml/historical_percentile.py`. Called in `ContextBuilder.build()`, output injected as a new section at priority 1 (above everything else).

**Data needed:** 5,000+ candles of 15m or 1H OHLCV → ~7 weeks. Start collecting immediately (see Part C).

---

### A2. Multi-Timeframe Alignment *(Spatial + Temporal Awareness)*

**What it does:** Fetches 1H, 4H, and 1D candles alongside the existing 15m, runs a lightweight indicator pass, and surfaces a compact alignment summary.

**Why LLMs need this:** A 15m BUY signal means completely different things when:
- 1H: bullish, 4H: bullish, 1D: bullish → strong alignment, high conviction long
- 1H: bullish, 4H: neutral, 1D: bearish → countertrend, macro headwind

**Example LLM prompt injection:**
```
## Timeframe Alignment
- 15m: BULLISH (current)
- 1H:  BULLISH — RSI 58, MACD cross up, above EMA20
- 4H:  NEUTRAL — ADX 22 (weak), price at BB midpoint
- 1D:  BEARISH — below 200D EMA, RSI 44, declining volume
→ Alignment: 2/4 bullish. Macro headwind. Treat as short-term signal only.
```

**Implementation:** `BinanceFeed.get_ohlcv()` already accepts `timeframe` and `limit` params. Add calls for `"1h"`, `"4h"`, `"1d"` alongside the existing `"15m"` call. Run through a stripped-down `IndicatorCalculator` pass.

**Integration point:** New `MultiTimeframeAnalyzer` in `services/analysis/multi_timeframe_analyzer.py`. Injected into `ContextBuilder` as a new section.

**Data needed:** Live fetch per cycle — no historical storage required.

---

### A3. DBSCAN Key Level Detector *(Spatial S/R Awareness)*

**What it does:** Identifies major support/resistance clusters from months of daily candle highs and lows using density-based clustering.

**Why it is better than the existing pattern analyzer:** `PatternAnalyzer` finds S/R in the last 200 candles (~2 days at 15m). This finds levels that held for months — the ones institutional traders actually watch.

**Example LLM prompt injection:**
```
## Major Historical Levels (6-month clusters)
- Resistance: $71,400–$71,800 (8 touches, last: 14d ago)
- Resistance: $69,200–$69,500 (5 touches, last: 32d ago)
- Support:    $64,800–$65,200 (11 touches, last: 3d ago)  ← near current price
- Support:    $59,000–$59,400 (6 touches, last: 61d ago)
```

**Implementation:** `sklearn.cluster.DBSCAN` on 6–12 months of daily highs + lows. Set `eps` to ~0.3% of current price. Rank clusters by touch count and recency.

**Integration point:** `services/ml/key_level_detector.py`. Run as a daily background task (not every cycle), cache result to a small JSON file. `ContextBuilder` reads the cache.

**Data needed:** ~365 rows of daily OHLCV (`data/btcusdt_1d.csv`). Tiny file, download once.

**Effort:** ~50 lines of code. `scikit-learn` is likely already a transitive dependency.

---

### A4. Regime / Cycle Classifier *(Behavioral Mode for LLM)*

**What it does:** Classifies the current macro market state. The label is injected into the **system prompt**, not the user message — dynamically changing the LLM's behavioral rules based on regime.

**Why it matters more than just context:** Knowing regime does not just inform the LLM, it changes what it is allowed to do:

```python
# Dynamic system prompt injection based on regime
if regime == "BEAR_TRENDING":
    system_prompt += "\nCURRENT REGIME: BEAR MARKET. Do not authorize LONG positions. "
                     "Only consider SHORT entries at resistance with strong confirmation."
elif regime == "ACCUMULATION":
    system_prompt += "\nCURRENT REGIME: ACCUMULATION. Reduce trade frequency. "
                     "Only trade at range extremes. Widen stops."
```

**Pick one — you do not need both initially:**

| Option | Model | Data | Effort |
|---|---|---|---|
| Short-term | Gaussian HMM (4-state) | 10,000+ 15m candles | Medium |
| Macro | Random Forest classifier | 1,500 daily candles (4 years) | Medium |

Start with the **Random Forest on daily data** — less data needed, more interpretable, maps directly to actionable regime names.

**Regime labels:** `BULL_TRENDING`, `BULL_CORRECTION`, `BEAR_TRENDING`, `ACCUMULATION`

**Features (daily):** Price vs 50D/100D/200D EMA, EMA slope, distance from 52-week high/low, ADX(14), 30D realized volatility, higher-high/lower-low count over 90 days.

**Integration point:** `services/ml/cycle_classifier.py`. Run daily. Cache regime label + confidence to disk. `TradingLoop` reads it at startup and injects into `build_system_prompt()`.

---

## Part B — Nice-to-Haves: Guards and Filters

These do not go inside the LLM prompt. They enforce hard rules in Python — acting as the safety net for LLM hallucinations.

---

### B1. News Sentiment Scorer — FinBERT *(RAG Upgrade)*

**What it does:** Scores each news article (-1 to +1) at ingestion time using FinBERT, and stores `sentiment_score` as ChromaDB metadata.

**Why this upgrades the RAG pipeline specifically:**
- Currently, raw news text is chunked into ChromaDB and retrieved wholesale — the LLM reads every word
- With `sentiment_score` as metadata, you unlock **Hybrid RAG queries**:

```python
# Before: retrieve by semantic similarity only
results = collection.query(query_texts=[query], n_results=5)

# After: filter by sentiment + semantic similarity
results = collection.query(
    query_texts=[query],
    n_results=5,
    where={"sentiment_score": {"$lt": -0.5}}  # only bearish articles
)
```

- The LLM receives pre-digested summaries instead of raw text — saves tokens, reduces hallucination from noisy content
- Aggregate recent sentiment feeds into context: "News sentiment (last 6h): -0.42 (4 bearish, 1 neutral)"

**Model:** `ProsusAI/finbert` from Hugging Face (~400MB, CPU inference, ~50ms/article).

**Watch out for:** FinBERT was trained on financial equities news, not crypto specifically. Sentiment drift is real for crypto. Consider monitoring accuracy over time.

**Integration point:** `services/ml/sentiment_scorer.py`. Called in `IngestionLoop.ingest_news_once()` before storing. Aggregate score surfaced in `RAGRetriever` output.

**Effort:** Low-medium. Requires `transformers` (+ `torch` already optional in codebase).

---

### B2. XGBoost Direction Classifier *(Quantitative Baseline for LLM)*

**What it does:** Predicts direction probability from the current indicator snapshot. The score is fed **into the LLM prompt as context**, not used as an override.

**How to use it correctly:**
```
## Quantitative Signal
- XGBoost direction model: 68% bullish (moderate conviction)
  → Use this as a baseline. If RAG news contradicts, weight news heavily.
```

The LLM can agree, disagree, or discount the XGBoost signal based on news context. This is the correct role — one input among many, not a decision-maker.

**Input features:** RSI, MACD hist, ADX, ATR%, OBV slope, choppiness, BB position, funding rate, OI, volume ratio, lagged values t-1/t-2/t-3.

**Target:** Binary — price up >0.5% within next 4–8 candles.

**Watch out for:** Data leakage — lagged features must not overlap the target look-ahead window during training.

**Integration point:** `services/ml/direction_classifier.py`. Called in `TechnicalAnalyzer.analyze()`, score added to prompt.

**Effort:** Low. `xgboost` + `scikit-learn`, serialize with `joblib`.

**Data needed:** 5,000+ 15m candles (~7 weeks).

---

### B3. Logistic Regression Trade Gate *(ML Kill-Switch in RiskManager)*

**What it does:** Predicts historical win probability for a proposed trade based on entry conditions. Used as a hard gate in `RiskManager` — **never shown to the LLM**.

**Why it belongs in RiskManager, not the prompt:**
- LLMs hallucinate; they can be convinced by their own reasoning to override statistical evidence
- A 32% historical win rate should trigger an automatic HOLD in Python code, not a suggestion the LLM can talk itself out of

```python
# In RiskManager.validate()
win_prob = outcome_predictor.predict(entry_conditions)
if win_prob < 0.40:
    return RiskDecision(approved=False, reason=f"Historical win rate {win_prob:.0%} below threshold")
```

**Input features:** All indicator values at entry, pattern flags, regime label (from A4), sentiment score (from B1), recent win/loss streak, current drawdown.

**Watch out for:** Small dataset — if the bot has been live for weeks, you may only have 50–200 trades. With 30+ features that risks overfitting badly. Use strong L2 regularization.

**Integration point:** `services/ml/outcome_predictor.py`. Called in `RiskManager.validate()` before any order is placed.

---

### B4. Isolation Forest — Anomaly Kill-Switch

**What it does:** Detects abnormal market microstructure (OI spikes, funding rate divergence, volume anomalies) and triggers an automated trading pause.

**Key principle:** Do not ask the LLM what to do during a flash crash. Just halt in Python.

```python
# In TradingLoop.run_cycle_once()
if anomaly_detector.is_anomaly(current_features):
    self.logger.warning("Anomaly detected — skipping cycle")
    await notifier.send("Trading paused: market anomaly detected")
    return  # skip LLM entirely
```

**Features:** Funding rate z-score, OI change rate, volume spike ratio, price velocity.

**Integration point:** `services/ml/anomaly_detector.py`. First check in `TradingLoop.run_cycle_once()`, before any other processing.

**Effort:** Low-medium. `sklearn.ensemble.IsolationForest`.

---

## Part C — Traps: What to Skip

These models work against the LLM-RAG architecture rather than enhancing it.

---

### ~~Part B (original) — Deep Learning Time Series: GRU, TCN, TFT, N-BEATS~~

**Why to skip:**

If you successfully train a TFT that predicts multi-horizon price movements with high accuracy, the LLM becomes a bottleneck rather than a value-add. You will have built a different product — a Deep Learning-driven bot — and the entire RAG infrastructure becomes dead weight.

These require:
- 17+ months of 15m data (TFT)
- Significant GPU training time
- A completely separate inference pipeline
- A resolution strategy when they conflict with the LLM — which has no clean answer

**The correct threshold:** Only consider sequence models if and when your XGBoost classifier (B2) consistently fails despite good features, suggesting that temporal structure matters more than snapshot features. That is the empirical signal to investigate GRU. Do not build it speculatively.

---

### ~~D5 (original) — Autoencoder Latent Vectors~~

**Why to skip:**

LLMs cannot reason about a 16-dimensional float vector like `[0.4, -0.1, 0.9, ...]`. You would need to map it to nearest historical analogues first, which is exactly what the Regime Classifier (A4) already does — in a much simpler, more interpretable way.

---

## Part D — RAG as Market Memory

The most underutilised capability in this bot is using ChromaDB as a **long-term trade memory**, not just a news store. This requires no new ML model.

### Trade Post-Mortems

Every time a trade closes, write a structured paragraph and embed it into ChromaDB:

```
"BTCUSDT SHORT closed LOSS on 2026-03-15.
Regime: BULL_CORRECTION. RSI at 82nd pct (overextended but not extreme).
Entry: RSI divergence pattern + MACD cross. 4H was neutral, 1D bullish — macro headwind ignored.
Outcome: Price continued up 2.1%. Loss: -1.8%.
Key lesson: Shorting in BULL_CORRECTION with only 2/4 timeframe alignment is low probability."
```

### RAG the Trade History

When the LLM is about to decide, query for similar past situations:

```python
query = f"regime:{regime} rsi_pct:{rsi_percentile_bucket} signal:SHORT setup:divergence"
similar_trades = vector_db.query(query, where={"outcome": {"$in": ["win", "loss"]}}, n=3)
```

### Prompt Injection

```
## Similar Historical Trades (RAG)
- 2026-02-10: SHORT in BULL_CORRECTION, RSI 85th pct → LOSS (-2.1%). LLM ignored 1D bullish.
- 2025-11-22: SHORT in BULL_CORRECTION, RSI 79th pct → LOSS (-1.4%). Pattern false signal.
- 2025-09-08: SHORT in BULL_CORRECTION, RSI 91st pct → WIN (+3.2%). Extreme overextension only.
→ 1 of 3 similar setups won. Only valid at extreme percentiles (>90th). Current RSI: 82nd pct.
```

This gives the LLM **episodic memory** — it learns from its own trading history without retraining any model.

**Integration point:** `TradingStrategy.close_position()` already has the trade outcome and market conditions. Add a post-mortem writer there that calls `VectorMemoryService.store()`.

---

## Part E — Data Collection

### Current State

**OHLCV is not persisted anywhere in the codebase.**

`BinanceFeed.get_ohlcv()` (`src/infrastructure/binance/feed.py:47`) fetches candles every cycle and they are discarded after `IndicatorCalculator` runs. The `data/` directory already exists — it is created by `Persistence.__init__()` (`src/infrastructure/storage/persistence.py:44`) — but it only holds:

```
data/
  position_{symbol}.json
  trade_history_{symbol}.json
  statistics_{symbol}.json
  last_response_{symbol}.json
  last_analysis_{symbol}.json
```

No OHLCV files exist. Both the one-time backfill and the live append need to be implemented.

---

### Step 1 — One-Time Backfill Script

Run once to seed the historical dataset. No auth required for Binance klines (market data endpoint is public). Rate limit is generous (~1,200 req/min). Downloading 4 years of daily + 7 weeks of 15m takes a few minutes.

```python
# scripts/backfill_ohlcv.py
import csv, time
from src.infrastructure.binance.rest_client import BinanceRestClient
from src.core.config import Settings

settings = Settings.from_env()
client = BinanceRestClient(
    api_key=settings.binance_api_key,
    api_secret=settings.binance_api_secret,
    product=settings.binance_product,
    testnet=settings.binance_testnet,
    base_url=settings.binance_base_url,
)

def backfill(symbol: str, interval: str, target_rows: int, output_path: str) -> None:
    rows, end_ms = [], None

    while len(rows) < target_rows:
        kwargs = dict(symbol=symbol, interval=interval, limit=1000)
        if end_ms:
            kwargs["endTime"] = end_ms
        batch = client.get_klines(**kwargs)
        if not batch:
            break
        rows = batch + rows
        end_ms = int(batch[0][0]) - 1  # move window back 1ms
        print(f"{symbol} {interval}: {len(rows)} rows, oldest={batch[0][0]}")
        time.sleep(0.1)  # stay well under rate limit

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        writer.writerows([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows])
    print(f"Done. Saved {len(rows)} rows → {output_path}")

# Daily candles — 4 years for regime classifier (A4) and key levels (A3)
backfill("BTCUSDT", "1d",  target_rows=1_500,  output_path="data/btcusdt_1d.csv")

# 15m candles — 7+ weeks for percentile scorer (A1) and XGBoost (B2)
backfill("BTCUSDT", "15m", target_rows=70_000, output_path="data/btcusdt_15m.csv")
```

---

### Step 2 — Live Append in TradingLoop

After the backfill, the bot must append each newly closed candle on every cycle so the dataset stays current. Add this to `TradingLoop.run_cycle_once()` after candles are fetched:

```python
# Append the last closed candle to the OHLCV CSV (index -1 is the current open candle)
if candles and len(candles) >= 2:
    c = candles[-2]
    with open("data/btcusdt_15m.csv", "a", newline="") as f:
        csv.writer(f).writerow([c.timestamp, c.open, c.high, c.low, c.close, c.volume])
```

No new dependency — stdlib `csv` only. The `data/` directory already exists.

### Data Requirements Summary

| Model | Timeframe | Min rows | Notes |
|---|---|---|---|
| A1 Percentiles | 15m or 1H | 5,000 | ~7 weeks |
| A2 Multi-TF | 1H / 4H / 1D | Live fetch | No storage needed |
| A3 Key Levels | Daily | ~365 | 1 year highs/lows |
| A4 Regime Classifier | Daily | ~1,500 | 4 years — tiny file |
| B1 FinBERT | News articles | N/A | Ingest at runtime |
| B2 XGBoost | 15m | 5,000 | ~7 weeks |
| B3 LogReg Gate | Trade logs | 200 trades | Depends on activity |
| B4 Isolation Forest | 15m | 5,000 | ~7 weeks |

---

## Part F — Offline Training

### The Principle

All models that require training must be trained **offline** — completely separate from the running bot. The live bot only ever does inference (loading a pre-trained file and calling `predict()`). Training is never done inside the bot process.

```
Offline (your laptop / a script):
  data/btcusdt_15m.csv  ──►  train_*.py script  ──►  models/xgboost_direction.joblib
  data/btcusdt_1d.csv   ──►  train_*.py script  ──►  models/regime_classifier.joblib

Live bot (inference only):
  startup  ──►  ModelStore.load("xgboost_direction")  ──►  model.predict(features)
```

Training is slow, memory-intensive, and produces non-deterministic results. Mixing it into the trading loop would stall cycles, corrupt state, and make failures hard to diagnose. Keep them fully separated.

---

### Model File Layout

```
models/
  xgboost_direction.joblib     ← B2: XGBoost direction classifier
  regime_classifier.joblib     ← A4: Random Forest cycle classifier
  isolation_forest.joblib      ← B4: Isolation Forest anomaly detector
  outcome_predictor.joblib     ← B3: Logistic Regression trade gate
  key_levels_cache.json        ← A3: DBSCAN output (not a model — cached result)
  regime_cache.json            ← A4: current regime label + confidence (refreshed daily)
```

`models/` sits alongside `data/` in the project root. Add both to `.gitignore` — model files can be 10–100MB and are environment-specific.

---

### Which Models Need Offline Training

| Model | Needs training? | Trigger to retrain | Script |
|---|---|---|---|
| A1 Percentiles | No — pure statistics | — | — |
| A2 Multi-TF | No — live fetch | — | — |
| A3 DBSCAN Key Levels | Refit weekly | Weekly cron or manual | `scripts/fit_key_levels.py` |
| A4 Regime Classifier | Train once, retrain monthly | After ~30 new daily candles | `scripts/train_regime.py` |
| B1 FinBERT | No — pretrained weights | — | download once via `transformers` |
| B2 XGBoost | Train once, retrain weekly | After ~500 new 15m candles | `scripts/train_direction.py` |
| B3 LogReg Gate | Retrain after every 50 closed trades | Trade count threshold | `scripts/train_outcome.py` |
| B4 Isolation Forest | Refit weekly | Same cadence as XGBoost | `scripts/train_anomaly.py` |

---

### Training Scripts

#### `scripts/train_direction.py` — XGBoost Direction Classifier (B2)

```python
"""
Offline training for the XGBoost direction classifier (B2).
Run manually or on a weekly cron after new 15m data has accumulated.

Usage: python scripts/train_direction.py
Output: models/xgboost_direction.joblib
"""
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report

# --- Load OHLCV ---
df = pd.read_csv("data/btcusdt_15m.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df = df.sort_values("timestamp").reset_index(drop=True)

# --- Build features (mirrors IndicatorCalculator output) ---
# These must exactly match what the live bot computes per-candle
close = df["close"]
df["rsi_14"]     = _compute_rsi(close, 14)           # implement or import from src
df["macd_hist"]  = _compute_macd_hist(close)
df["adx"]        = _compute_adx(df, 14)
df["atr_pct"]    = _compute_atr(df, 14) / close * 100
df["obv_slope"]  = _compute_obv_slope(df)
df["bb_pos"]     = _compute_bb_position(close, 20)
df["ema_spread"] = _compute_ema(close, 20) - _compute_ema(close, 50)
df["vol_ratio"]  = df["volume"] / df["volume"].rolling(20).mean()

# Lag features — t-1, t-2, t-3
for col in ["rsi_14", "macd_hist", "adx"]:
    for lag in [1, 2, 3]:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

# --- Target: did price rise >0.5% within the next 8 candles? ---
# IMPORTANT: use shift(-8) to look forward; never include future data in features
LOOKAHEAD = 8
df["target"] = (close.shift(-LOOKAHEAD) / close - 1 > 0.005).astype(int)

# Drop rows with NaN (from rolling indicators and lags) and the last LOOKAHEAD rows
feature_cols = [c for c in df.columns if c not in ["timestamp", "open", "high", "low",
                                                     "close", "volume", "target"]]
df = df.dropna(subset=feature_cols + ["target"])

X = df[feature_cols].values
y = df["target"].values

# --- Time-series cross-validation (never shuffle financial data) ---
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          use_label_encoder=False, eval_metric="logloss")
    model.fit(X[train_idx], y[train_idx],
              eval_set=[(X[val_idx], y[val_idx])], verbose=False)
    preds = model.predict(X[val_idx])
    print(f"Fold {fold+1}:\n{classification_report(y[val_idx], preds)}")

# --- Final model: train on all data ---
final_model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             use_label_encoder=False, eval_metric="logloss")
final_model.fit(X, y)

joblib.dump({"model": final_model, "feature_cols": feature_cols}, "models/xgboost_direction.joblib")
print(f"Saved → models/xgboost_direction.joblib  ({len(X):,} training rows)")
```

**Critical warning:** Never shuffle time-series data when splitting. Always use `TimeSeriesSplit` — a future candle must never appear in the training set of a past candle's fold.

---

#### `scripts/train_regime.py` — Market Cycle Classifier (A4)

```python
"""
Offline training for the macro regime classifier (A4).
Uses daily OHLCV. Retrain monthly or after ~30 new daily candles.

Usage: python scripts/train_regime.py
Output: models/regime_classifier.joblib
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("data/btcusdt_1d.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df = df.sort_values("timestamp").reset_index(drop=True)
close = df["close"]

# --- Features ---
for period in [50, 100, 200]:
    ema = close.ewm(span=period, adjust=False).mean()
    df[f"ema{period}_dist"]  = (close - ema) / ema         # % above/below EMA
    df[f"ema{period}_slope"] = ema.diff(5) / ema           # EMA rising/falling

df["high_52w_dist"] = (close - close.rolling(365).max()) / close
df["low_52w_dist"]  = (close - close.rolling(365).min()) / close
df["adx_14"]        = _compute_adx(df, 14)
df["realized_vol"]  = close.pct_change().rolling(30).std() * np.sqrt(365)

# Higher highs / lower lows over last 90 days
df["hh_count"] = df["high"].rolling(90).apply(
    lambda x: sum(x[i] > x[i-1] for i in range(1, len(x))), raw=True
)
df["ll_count"] = df["low"].rolling(90).apply(
    lambda x: sum(x[i] < x[i-1] for i in range(1, len(x))), raw=True
)

# --- Labels (rule-based labeler — adjust thresholds to your taste) ---
# This produces training labels from rules; the model then generalises them.
def label_regime(row) -> str:
    if row["ema200_dist"] > 0.05 and row["ema50_slope"] > 0:
        return "BULL_TRENDING"
    elif row["ema200_dist"] > -0.05 and row["high_52w_dist"] < -0.10:
        return "BULL_CORRECTION"
    elif row["ema200_dist"] < -0.05 and row["ema50_slope"] < 0:
        return "BEAR_TRENDING"
    else:
        return "ACCUMULATION"

feature_cols = [c for c in df.columns if c not in
                ["timestamp", "open", "high", "low", "close", "volume"]]
df = df.dropna(subset=feature_cols)
df["label"] = df.apply(label_regime, axis=1)

X = df[feature_cols].values
y = df["label"].values

model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                class_weight="balanced", random_state=42)
model.fit(X, y)

joblib.dump({"model": model, "feature_cols": feature_cols}, "models/regime_classifier.joblib")
print(f"Label distribution:\n{pd.Series(y).value_counts()}")
print("Saved → models/regime_classifier.joblib")
```

---

#### `scripts/fit_key_levels.py` — DBSCAN Key Levels (A3)

```python
"""
Fit key S/R levels from daily candle highs/lows. Not a trained model —
runs DBSCAN and caches the output as JSON for the bot to read.
Rerun weekly.

Usage: python scripts/fit_key_levels.py
Output: models/key_levels_cache.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

df = pd.read_csv("data/btcusdt_1d.csv")
df = df.sort_values("timestamp").tail(365)  # last 1 year
current_price = float(df["close"].iloc[-1])

# All significant price points from highs and lows
price_points = np.concatenate([df["high"].values, df["low"].values])
price_points = price_points.reshape(-1, 1)

# eps = 0.3% of current price — clusters price levels within that band
eps = current_price * 0.003
db = DBSCAN(eps=eps, min_samples=3).fit(price_points)

clusters = {}
for label, price in zip(db.labels_, price_points.flatten()):
    if label == -1:  # noise
        continue
    clusters.setdefault(label, []).append(price)

levels = []
for prices in clusters.values():
    center = float(np.mean(prices))
    levels.append({
        "center": round(center, 2),
        "low":    round(float(np.min(prices)), 2),
        "high":   round(float(np.max(prices)), 2),
        "touches": len(prices),
        "type":  "resistance" if center > current_price else "support",
    })

levels.sort(key=lambda x: x["touches"], reverse=True)

with open("models/key_levels_cache.json", "w") as f:
    json.dump({"current_price": current_price, "levels": levels[:10]}, f, indent=2)

print(f"Found {len(levels)} clusters. Top levels saved → models/key_levels_cache.json")
```

---

#### `scripts/train_outcome.py` — Logistic Regression Trade Gate (B3)

```python
"""
Offline training for the LogReg trade outcome predictor (B3).
Retrain after every 50 new closed trades.

Usage: python scripts/train_outcome.py
Output: models/outcome_predictor.joblib
"""
import json, glob
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load all closed trades from Persistence trade history files
rows = []
for path in glob.glob("data/trade_history_*.json"):
    with open(path) as f:
        trades = json.load(f)
    for t in trades:
        if not t.get("action", "").startswith("CLOSE"):
            continue
        cond = t.get("market_conditions", {})
        rows.append({
            "rsi":         cond.get("rsi", 50),
            "adx":         cond.get("adx", 20),
            "atr_pct":     cond.get("atr_percentage", 1.5),
            "trend":       1 if cond.get("trend_direction") == "BULLISH" else -1,
            "vol_state":   1 if cond.get("volume_state") == "HIGH" else 0,
            "bb_pos":      {"UPPER": 1, "MIDDLE": 0, "LOWER": -1}.get(
                               cond.get("bb_position", "MIDDLE"), 0),
            "profitable":  1 if t.get("pnl_pct", 0) > 0 else 0,
        })

if len(rows) < 50:
    print(f"Only {len(rows)} trades — need at least 50 to train. Skipping.")
    exit(0)

import pandas as pd
df = pd.DataFrame(rows)
feature_cols = [c for c in df.columns if c != "profitable"]
X = df[feature_cols].values
y = df["profitable"].values

# Pipeline: scale + L2 logistic regression (strong regularization for small datasets)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")),
])
model.fit(X, y)

joblib.dump({"model": model, "feature_cols": feature_cols}, "models/outcome_predictor.joblib")
print(f"Trained on {len(X)} trades. Saved → models/outcome_predictor.joblib")
```

---

#### `scripts/train_anomaly.py` — Isolation Forest (B4)

```python
"""
Offline fit for the Isolation Forest anomaly detector (B4).
Refit weekly on recent 15m data.

Usage: python scripts/train_anomaly.py
Output: models/isolation_forest.joblib
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

df = pd.read_csv("data/btcusdt_15m.csv")
df = df.sort_values("timestamp").tail(10_000)  # last ~7 weeks

close = df["close"]
df["vol_ratio"]   = df["volume"] / df["volume"].rolling(20).mean()
df["price_vel"]   = close.pct_change(3)  # 3-candle rate of change
df["vol_spike"]   = df["vol_ratio"] > 3
df["high_low_rng"] = (df["high"] - df["low"]) / close

feature_cols = ["vol_ratio", "price_vel", "high_low_rng"]
df = df.dropna(subset=feature_cols)
X = df[feature_cols].values

# contamination: expected fraction of anomalies (tune carefully)
# 0.01 = 1% of candles flagged — start conservative
model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
model.fit(X)

joblib.dump({"model": model, "feature_cols": feature_cols}, "models/isolation_forest.joblib")
print(f"Fitted on {len(X):,} candles. Saved → models/isolation_forest.joblib")
```

---

### Loading Models at Bot Startup

The bot loads all model files once at startup via `ModelStore`. If a file is missing the model is skipped — the bot falls back to existing behaviour.

```python
# src/infrastructure/ml/model_store.py
import joblib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
MODELS_DIR = Path("models")

def load(name: str) -> Any | None:
    """Load a joblib model by name. Returns None if file is missing."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        logger.warning("Model file not found: %s — feature disabled", path)
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        logger.error("Failed to load model %s: %s", path, exc)
        return None

def load_json(name: str) -> Any | None:
    """Load a JSON cache file by name. Returns None if file is missing."""
    path = MODELS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
```

In `app.py` `build_runtime()`, load models after `Persistence` is wired:

```python
from src.infrastructure.ml.model_store import load, load_json

direction_model  = load("xgboost_direction")   # B2 — None until trained
regime_model     = load("regime_classifier")   # A4 — None until trained
anomaly_model    = load("isolation_forest")    # B4 — None until trained
outcome_model    = load("outcome_predictor")   # B3 — None until trained
key_levels       = load_json("key_levels_cache")  # A3 — None until fit
```

Each `services/ml/` wrapper checks for `None` and returns a neutral default:

```python
# services/ml/direction_classifier.py
class DirectionClassifier:
    def __init__(self, model_data):
        self._model = model_data  # None if not yet trained

    def predict_proba(self, features: dict) -> float | None:
        if self._model is None:
            return None  # caller omits the section from the LLM prompt
        X = [[features[c] for c in self._model["feature_cols"]]]
        return float(self._model["model"].predict_proba(X)[0][1])
```

---

### Retraining Schedule

| Model | Cadence | Trigger | Script |
|---|---|---|---|
| B2 XGBoost | Weekly | ~500 new 15m candles (~5 days) | `train_direction.py` |
| A4 Regime | Monthly | ~30 new daily candles | `train_regime.py` |
| A3 Key Levels | Weekly | Price level landscape changes | `fit_key_levels.py` |
| B4 Isolation Forest | Weekly | Same as XGBoost | `train_anomaly.py` |
| B3 LogReg | Per 50 trades | Trade count check on startup | `train_outcome.py` |

Run these manually for now. Once the models are validated, they can be automated with a simple cron or a `scripts/retrain_all.py` that checks each trigger condition and reruns the appropriate script.

---

## Pragmatic Implementation Phases

### Phase 1 — Zero-ML Context Upgrades *(This weekend)*

These require no model training, no new dependencies, and immediately give the LLM macro context it currently lacks entirely.

1. **A1 — Percentile Scorer:** `numpy.percentileofscore()` on rolling CSV. ~30 lines.
2. **A2 — Multi-Timeframe Alignment:** Extra `get_ohlcv()` calls + compact summary formatter.
3. **Part E — Data Collection:** Start the CSV backfill and live append. The clock starts now.

### Phase 2 — Enhancing the RAG Pipeline *(Next week)*

4. **B1 — FinBERT Sentiment:** Score at ingestion, enable hybrid ChromaDB queries.
5. **A3 — DBSCAN Key Levels:** Script + daily cache → LLM sees major S/R zones.
6. **Part D — Trade Post-Mortems:** Write structured trade summaries into ChromaDB on close.

### Phase 3 — Lightweight ML Guards *(Next month)*

7. **A4 — Regime Classifier:** Random Forest on daily data, inject into system prompt dynamically.
8. **B3 — LogReg Trade Gate:** Add ML kill-switch to `RiskManager.validate()`.
9. **B2 — XGBoost Baseline:** Feed quantitative signal probability into prompt as one voice among many.
10. **B4 — Isolation Forest:** Automated pause on anomaly detection.

---

## Part G — RAG Pipeline: Current State & What to Improve

### Does News Actually Flow Into RAG?

**Yes — but the pipeline has meaningful limitations.** Tracing the full path:

```
CryptoCompareNewsSource.fetch()
  → returns: title, url, body, source, published_at, symbol_tags

IngestionLoop.ingest_news_once()
  → is_relevant_article(): filters by keywords ["bitcoin","btc","ethereum","eth","crypto"]
  → prepare_news_document(): truncates body to 1,000 chars  ← filter.py:16
  → store.add_document("news", text=f"{title} {body[:500]}")  ← ingestion_loop.py:80
     NOTE: only first 500 chars of body are embedded as the search text

RAGRetriever.retrieve()
  → query = f"{symbol} price {price} signal {signal} {reasoning[:200]}"
  → store.query("news", query, n_results=5)  ← pure semantic similarity, no filters
  → _format_news_section(): body truncated again to 300 chars  ← retriever.py:18
```

**What works:** Articles are fetched, deduplicated by URL hash, filtered for relevance, stored, and retrieved into the LLM prompt every cycle.

**The real limitations:**

| Issue | Where | Impact |
|---|---|---|
| Body truncated to 500 chars at embed time | `ingestion_loop.py:80` | The embedding only captures the beginning of the article; long articles are poorly represented |
| Body truncated again to 300 chars at retrieval | `retriever.py:18` | LLM only sees 300 chars of context per article |
| Query is purely semantic — no metadata filters | `retriever.py:38` | Can't filter by recency, source credibility, or sentiment; may retrieve old or off-topic articles |
| No sentiment score on documents | `ingestion_loop.py` | Hybrid queries (e.g., "only fetch bearish articles") impossible until FinBERT is added |
| Keyword filter is very broad | `filter.py:8` | Any article mentioning "bitcoin" passes — no quality signal |

**Quick wins without FinBERT:**
1. Embed the full body (remove the `[:500]` in `ingestion_loop.py:80`) — the body is already truncated to 1,000 chars in `prepare_news_document()`; embed all of it.
2. Add `published_at` as a ChromaDB `where` filter in `RAGRetriever.retrieve()` to only retrieve articles from the last 24h.
3. Surface the article count and date range in the formatted output so the LLM knows how fresh the news is.

---

### Should Charts Go Into RAG?

**Short answer: No — keep charts going directly to the LLM. But there is a valuable related idea.**

#### Why the current direct-to-LLM approach is correct

`ContextBuilder._vision_section()` (`context_builder.py:155`) sends the base64 PNG straight to the LLM message if `model_supports_vision=True`. This is the right architecture for the *current* chart:

- The chart shows **right-now** market structure — the LLM needs to see the current chart to reason about the current trade
- Vision input is processed by the LLM's own attention mechanism — it can correlate what it sees in the chart with the text context (indicators, news) simultaneously
- RAG is for **retrieval of past context**; the current chart is not past context

#### Why putting charts into RAG doesn't work well

ChromaDB embeds **text**. A base64 PNG fed as text produces a meaningless embedding — the vector would represent the literal ASCII characters of the base64 string, not the visual content of the chart. You would retrieve random documents, not visually similar charts.

To do this properly you would need a **multimodal embedding model** (e.g., CLIP) as a custom ChromaDB embedding function:

```python
# Hypothetical — complex to implement
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
embedding_fn = OpenCLIPEmbeddingFunction()  # embeds images into the same space as text
store.add_document("charts", image=png_bytes, metadata={"outcome": "+2.3%"})
# At query time: "find charts visually similar to the current one"
similar_charts = store.query_by_image("charts", current_png_bytes, n_results=3)
```

This is technically possible with ChromaDB + OpenCLIP, but it introduces:
- A ~1.5GB multimodal embedding model running at inference time every cycle
- Significant latency per query
- A completely separate chart storage + retrieval pipeline

**The effort is high and the payoff is uncertain** — you don't actually know that two visually similar charts (same candlestick pattern) predict similar outcomes without extensive backtesting.

#### The right way to get historical chart value into RAG

Instead of storing images, store **text descriptions of chart patterns + outcomes** in the trade post-mortem (Part D). This is already the plan:

```
POST-MORTEM STORED IN ChromaDB:
"BTCUSDT LONG 2026-03-10 — Entry at $67,200.
Chart: Bullish engulfing on 15m, price at lower BB, RSI divergence from 38→44.
4H: neutral, 1D: bearish — macro headwind.
Outcome: LOSS -1.8%. Price rejected at $68,400 resistance (8-touch cluster).
Key lesson: Engulfing at lower BB failed due to strong 1D headwind and nearby resistance."
```

When the LLM is about to enter a similar setup, RAG retrieves this text description. The LLM reads it and adjusts — without any image embedding infrastructure.

This gives you 80% of the value of "chart memory" at 5% of the complexity.

---

### Summary: RAG Pipeline Health Check

| Component | Status | Action needed |
|---|---|---|
| News ingestion | ✓ Working | Fix body truncation at embed time; add recency filter |
| Macro ingestion (fear/greed, DeFiLlama) | ✓ Working | No change needed |
| OHLCV history source | ⚠ Stub only | Returns placeholder text — needs real aggregator wired |
| Sentiment scoring (FinBERT) | ✗ Not implemented | Phase 2 |
| Trade post-mortems | ✗ Not implemented | Phase 2 |
| Chart → RAG | ✗ Not applicable | Keep direct-to-LLM; use text post-mortems instead |

---

## Architecture Placement

```
src/
├── services/
│   ├── analysis/
│   │   ├── indicator_calculator.py      ← feature source for all models
│   │   ├── technical_analyzer.py        ← integrate XGBoost score (B2)
│   │   ├── multi_timeframe_analyzer.py  ← NEW A2: 1H/4H/1D alignment
│   │   └── context_builder.py           ← inject A1/A2/A3/A4/B2 into LLM prompt
│   ├── ml/
│   │   ├── historical_percentile.py     ← A1: rolling percentile scorer
│   │   ├── key_level_detector.py        ← A3: DBSCAN S/R clustering
│   │   ├── cycle_classifier.py          ← A4: regime → system prompt injection
│   │   ├── sentiment_scorer.py          ← B1: FinBERT at ingestion
│   │   ├── direction_classifier.py      ← B2: XGBoost → LLM context
│   │   ├── outcome_predictor.py         ← B3: LogReg → RiskManager gate
│   │   └── anomaly_detector.py          ← B4: IsolationForest → cycle pause
│   ├── rag/
│   │   └── ingestion_loop.py            ← B1: score sentiment before storing
│   └── trading/
│       ├── risk_manager.py              ← B3: gate; B4: pause
│       ├── trading_strategy.py          ← Part D: write post-mortem on close
│       └── trading_loop.py              ← B4: anomaly check at cycle start
├── infrastructure/
│   └── ml/
│       └── model_store.py               ← NEW: load/save .joblib / .pkl files
└── data/
    ├── btcusdt_15m.csv                  ← 15m OHLCV for A1, B2, B4
    └── btcusdt_1d.csv                   ← daily OHLCV for A3, A4
```

All models are thin wrappers with a `predict(features) → result` interface. If a model fails or is unavailable, the bot falls back to existing behavior — same pattern used by `TradingBrainService` today.

---

## Part H — Implementation Reference

All models described in this document have been implemented. This section is the authoritative reference for what is in the codebase, where it lives, and how to operate it.

---

### H1. What Was Implemented

#### New source files

| File | Model | Role |
|---|---|---|
| `src/infrastructure/ml/model_store.py` | — | `load()` / `save()` / `load_json()` with graceful missing-file handling |
| `src/services/ml/historical_percentile.py` | A1 | Reads OHLCV CSV, computes rolling 6-month percentiles |
| `src/services/ml/key_level_detector.py` | A3 | Reads `models/key_levels_cache.json`, formats nearby S/R levels |
| `src/services/ml/cycle_classifier.py` | A4 | Loads regime model, injects instruction into system prompt |
| `src/services/ml/direction_classifier.py` | B2 | Loads XGBoost, returns bullish % for LLM context |
| `src/services/ml/outcome_predictor.py` | B3 | Loads LogReg, returns win probability for RiskManager gate |
| `src/services/ml/anomaly_detector.py` | B4 | Loads IsolationForest, pauses cycle on anomaly |
| `src/services/ml/sentiment_scorer.py` | B1 | Lazy-loads FinBERT, scores articles at ingestion |
| `src/services/analysis/multi_timeframe_analyzer.py` | A2 | Fetches 1H/4H/1D candles, builds alignment summary |
| `src/services/rag/ohlcv_writer.py` | E | Appends last closed candle to CSV every cycle |

#### Modified source files

| File | Change |
|---|---|
| `src/app.py` | Instantiates all ML services; wires into `RiskManager`, `IngestionLoop`, `ContextBuilder`, `TradingLoop` |
| `src/services/analysis/context_builder.py` | Added `ml_context` section in prompt; `cycle_classifier` + `regime_suffix` support |
| `src/services/analysis/prompt_templates.py` | `build_system_prompt()` accepts `regime_suffix` from `CycleClassifier` |
| `src/services/trading/trading_loop.py` | Anomaly check at cycle start; OHLCV append; daily regime refresh; `_build_ml_context()` assembles all ML sections |
| `src/services/trading/risk_manager.py` | LogReg outcome gate in `validate()` before any order is placed |
| `src/services/rag/ingestion_loop.py` | FinBERT scores news at ingestion; full body now embedded (was truncated to 500 chars) |

#### Offline training scripts

| Script | Model | Trigger |
|---|---|---|
| `scripts/backfill_ohlcv.py` | — | Run once to seed historical data |
| `scripts/fit_key_levels.py` | A3 DBSCAN | Weekly — no training, just cluster fitting |
| `scripts/train_anomaly.py` | B4 IsolationForest | Weekly — needs 5,000+ 15m candles |
| `scripts/train_direction.py` | B2 XGBoost | Weekly — needs 5,000+ 15m candles |
| `scripts/train_regime.py` | A4 RandomForest | Monthly — needs 200+ daily candles |
| `scripts/train_outcome.py` | B3 LogReg | Per 50 closed trades |
| `scripts/retrain_all.py` | all | Checks triggers automatically |

---

### H2. Full Workflow: First-Time Setup to Running Bot

#### Step 1 — Backfill historical data

Run once. Downloads years of OHLCV from Binance (no auth required for market data).

```bash
python scripts/backfill_ohlcv.py
```

This creates:
- `data/btcusdt_1d.csv` — ~1,500 daily candles (4 years)
- `data/btcusdt_15m.csv` — ~70,000 fifteen-minute candles (~7 weeks)

#### Step 2 — Fit key levels (no model training required)

```bash
python scripts/fit_key_levels.py
```

Creates `models/key_levels_cache.json` immediately. The bot can use this right away.

#### Step 3 — Train models

Run in order. Each script prints validation metrics before saving.

```bash
# Needs 5,000+ 15m candles (~7 weeks). After backfill, you have 70,000. Run now.
python scripts/train_anomaly.py
python scripts/train_direction.py

# Needs 200+ daily candles. After backfill, you have ~1,500. Run now.
python scripts/train_regime.py

# Needs 50+ closed trades from your live trading history.
# Skip until you have enough data — the bot runs fine without it.
python scripts/train_outcome.py
```

Each script writes a `.joblib` file to `models/`:

```
models/
  key_levels_cache.json       ← A3: DBSCAN clusters
  isolation_forest.joblib     ← B4: anomaly detector
  xgboost_direction.joblib    ← B2: direction classifier
  regime_classifier.joblib    ← A4: cycle/regime classifier
  outcome_predictor.joblib    ← B3: trade gate (when you have enough trades)
```

#### Step 4 — Run the bot

```bash
python -m src.app
# or however you normally start it
```

**That is all.** The bot loads every model file at startup via `ModelStore`. If a file exists, the feature is active. If a file is missing, the bot logs a debug message and runs exactly as before — no code changes, no flags, no restarts.

You do not need to change any settings or environment variables to enable ML features.

---

### H3. What the Bot Does With Each Model at Runtime

| When | What happens |
|---|---|
| **Startup** | All `.joblib` files in `models/` are loaded. Missing files → feature silently disabled. |
| **Every cycle start** | B4 (anomaly check): if anomaly detected, cycle is skipped. No LLM call made. |
| **Every cycle start** | E (OHLCV writer): last closed candle appended to `data/btcusdt_15m.csv`. |
| **Once per day (UTC)** | A4 (regime refresh): daily CSV read, regime computed, system prompt suffix updated. |
| **During analysis** | A1 percentiles + A2 multi-TF + A3 key levels + B2 direction assembled into `## ML Context` section of LLM prompt. |
| **Before order execution** | B3 (outcome gate): if win probability < 40%, `RiskManager.validate()` blocks the trade and returns HOLD. |
| **During news ingestion** | B1 (FinBERT): each article scored at ingestion; `sentiment_score` stored as ChromaDB metadata. |

---

### H4. Graceful Degradation

Every ML service follows the same pattern used by `TradingBrainService`:

```
model file exists → feature active
model file missing → feature disabled, bot runs as before
model inference fails → exception caught, logged at WARNING, feature skipped
```

This means:
- You can delete any `.joblib` file and the bot continues without that feature.
- You can retrain a model while the bot is running. The new file is loaded on the **next bot restart**.
- FinBERT (B1) requires `transformers + torch`. If not installed, sentiment scoring is silently disabled.

---

### H5. Retraining Schedule

| Model | Script | Cadence | Trigger condition |
|---|---|---|---|
| A3 Key Levels | `fit_key_levels.py` | Weekly | Cache age > 7 days |
| B4 Anomaly | `train_anomaly.py` | Weekly | Model age > 7 days |
| B2 XGBoost | `train_direction.py` | Weekly | Model age > 7 days |
| A4 Regime | `train_regime.py` | Monthly | Model age > 30 days |
| B3 LogReg | `train_outcome.py` | Per 50 trades | 50+ new closed trades |

Run `scripts/retrain_all.py` on a schedule. It checks all triggers automatically and only runs scripts that need it:

```bash
# Preview what would run without executing
python scripts/retrain_all.py --dry-run

# Run all stale models
python scripts/retrain_all.py

# Force retrain everything regardless of age
python scripts/retrain_all.py --force
```

After retraining, **restart the bot** to load the new model files. The bot does not hot-reload models during a run.

---

### H6. Installing Optional Dependencies

The ML features have optional dependencies not in the base requirements:

```bash
# Required for B2 XGBoost training and inference
pip install xgboost scikit-learn joblib

# Required for B4, A3, A4, B3 training (scikit-learn covers all of these)
pip install scikit-learn joblib

# Required for B1 FinBERT sentiment scoring
pip install transformers torch

# Required for training scripts (data manipulation)
pip install pandas numpy
```

The bot starts and runs without any of these installed. Features activate as dependencies become available.

---

### H7. File Layout After Full Setup

```
project root/
├── data/
│   ├── btcusdt_15m.csv           ← live-appended each cycle (E)
│   ├── btcusdt_1d.csv            ← used by regime + key levels
│   ├── position_*.json           ← existing Persistence files
│   ├── trade_history_*.json      ← existing Persistence files
│   └── statistics_*.json         ← existing Persistence files
├── models/
│   ├── key_levels_cache.json     ← A3 (refit weekly)
│   ├── isolation_forest.joblib   ← B4 (retrain weekly)
│   ├── xgboost_direction.joblib  ← B2 (retrain weekly)
│   ├── regime_classifier.joblib  ← A4 (retrain monthly)
│   └── outcome_predictor.joblib  ← B3 (retrain per 50 trades)
├── scripts/
│   ├── backfill_ohlcv.py
│   ├── fit_key_levels.py
│   ├── train_direction.py
│   ├── train_regime.py
│   ├── train_outcome.py
│   ├── train_anomaly.py
│   └── retrain_all.py
└── src/
    ├── infrastructure/ml/model_store.py
    ├── services/ml/               ← all ML wrappers
    ├── services/analysis/multi_timeframe_analyzer.py
    └── services/rag/ohlcv_writer.py
```

Add `models/` and `data/*.csv` to `.gitignore` — model files are environment-specific and OHLCV CSVs can be large.
