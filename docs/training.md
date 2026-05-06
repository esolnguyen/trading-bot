# Training the ML models

The training entrypoints all live in
`src/enrich_knowledge/ml_training/`. They share one batch driver,
`run_training.py`, which:

1. Loads the composed `EnrichKnowledgeSettings` from `.env`.
2. Backfills OHLCV (incremental append) for every configured
   `(symbol × timeframe)` pair.
3. Dispatches to the requested driver(s).

For what each model does, see [docs/ml-models.md](ml-models.md). For
what the data on disk looks like, see [docs/data.md](data.md).

## TL;DR

```bash
# Train everything
python -m src.enrich_knowledge.runners.run_training --model all

# Train one family
python -m src.enrich_knowledge.runners.run_training --model anomaly
python -m src.enrich_knowledge.runners.run_training --model direction

# Staleness-triggered: only retrain models whose artifact is older than its TTL
python -m src.enrich_knowledge.runners.run_training --model retrain_all

# Dry run — print the plan without writing
python -m src.enrich_knowledge.runners.run_training --model all --dry-run

# Skip the OHLCV backfill (when CSVs are known fresh)
python -m src.enrich_knowledge.runners.run_training --model anomaly --skip-backfill
```

## Driver registry

`MODEL_REGISTRY` in `src/enrich_knowledge/runners/run_training.py`:

| `--model`     | Driver module                                   | Writes                                                |
|---------------|-------------------------------------------------|-------------------------------------------------------|
| `direction`   | `ml_training/direction.py`                      | `models/<tf>/xgboost_direction.joblib` (pooled)       |
| `outcome`     | `ml_training/outcome.py`                        | `models/outcome_predictor.joblib`                     |
| `anomaly`     | `ml_training/anomaly.py`                        | `models/<tf>/isolation_forest_<symbol>.joblib`        |
| `regime`      | `ml_training/regime.py`                         | `models/1d/regime_classifier_<symbol>.joblib`         |
| `key_levels`  | `ml_training/key_levels.py`                     | `models/1d/key_levels_<symbol>_cache.json`            |
| `retrain_all` | `ml_training/retrain_all.py` (orchestrator)     | Whichever family files are stale                      |
| `all`         | All of the above                                | Everything                                            |

`retrain_all` checks artifact age vs. per-family TTL:

| Family        | Artifact glob                                    | TTL    | Min OHLCV rows |
|---------------|--------------------------------------------------|--------|----------------|
| Key Levels    | `models/1d/key_levels*cache.json`                | 7 d    | 100 rows of 1d |
| Anomaly       | `models/*/isolation_forest*.joblib`              | 7 d    | 5 000 rows of 15m |
| Direction     | `models/*/xgboost_direction*.joblib`             | 7 d    | 5 000 rows of 15m |
| Regime        | `models/1d/regime_classifier*.joblib`            | 30 d   | 200 rows of 1d |
| Outcome       | `models/outcome_predictor*.joblib`               | 7 d    | 1 500 rows of 4h |

It's the safe choice for cron — touches nothing if artifacts are
fresh, runs the cheapest set if a few are stale.

## Configuration

Training reads two pydantic-settings blocks from `.env`:

```dotenv
# src/enrich_knowledge/config/ml_training.py
ML_TIMEFRAMES=15m,1h,4h,1d         # timeframes to fit (CSV)
TRAINING_SYMBOLS=BTCUSDT,ETHUSDT   # symbol universe (CSV, uppercased)
CANDLE_LIMIT=1500                  # bars per fit (cap, not target)

# src/config/storage.py
CHROMA_PATH=./chroma_db            # not used by training, but loaded
OHLCV_DIR=./data/ohlcv             # CSV destination
MODELS_DIR=./models                # joblib destination
```

Add a symbol to the universe by appending it to `TRAINING_SYMBOLS`.
The next `run_training` invocation will:

- Backfill the missing OHLCV CSV (`data/ohlcv/<symbol>_<tf>.csv`).
- Include it in the pooled direction / outcome fits automatically
  (the `symbol_id` feature is integer-mapped at training time).
- Train its own per-symbol anomaly / regime / key-level artifacts.

## Per-driver detail

### Direction (B2)

```bash
# Production batch (every TF in ML_TIMEFRAMES, pooled across TRAINING_SYMBOLS)
python -m src.enrich_knowledge.runners.run_training --model direction

# Ad-hoc single fit (one TF, default symbols)
python -m src.enrich_knowledge.ml_training.direction --timeframe 4h

# Single-symbol inspection fit
python -m src.enrich_knowledge.ml_training.direction --timeframe 4h --symbol btcusdt

# Custom symbol pool
python -m src.enrich_knowledge.ml_training.direction --timeframe 1h --symbols btcusdt,ethusdt,solusdt
```

Notable knobs (defaults in `direction.py`):

- `MAX_HORIZON = 24`, `BARRIER_K = 1.5` — triple-barrier window.
- `DEFAULT_LOOKBACK_DAYS = 720` — pre-feature lookback per symbol.
- `RECENCY_LAMBDA = 1.0` — exponential time-decay applied to the
  final fit (CV folds are uniqueness-only).

The output is a single `models/<tf>/xgboost_direction.joblib`
bundle:

```python
{
    "model": CalibratedClassifierCV(estimator=XGBClassifier(...), method="isotonic"),
    "feature_cols": [...15 cols...],
    "symbol_map": {"BTCUSDT": 0, "ETHUSDT": 1, ...},
    "max_horizon": 24,
    "barrier_k": 1.5,
    "recency_lambda": 1.0,
}
```

### Outcome (B3)

```bash
python -m src.enrich_knowledge.runners.run_training --model outcome
python -m src.enrich_knowledge.ml_training.outcome --timeframe 4h
```

Same triple-barrier label, same uniqueness/embargo stack as B2, but
projected onto the 6-feature inference space
(`rsi, adx, atr_pct, trend, vol_state, bb_pos`). The bucketing
helpers in `ml_training/outcome.py` (`_bucket_trend`,
`_bucket_vol_state`, `_bucket_bb_pos`) **must** stay in sync with
`OutcomePredictor._conditions_to_row` — change one, change the
other.

### Anomaly (B4)

```bash
python -m src.enrich_knowledge.runners.run_training --model anomaly

# One symbol/TF
python -m src.enrich_knowledge.ml_training.anomaly --symbol btcusdt --timeframe 4h
```

`DEFAULT_LOOKBACK_DAYS = 180` (microstructure baselines drift,
training on years of candles teaches "normal" that no longer
applies). `contamination = 0.01`.

### Regime (A4)

```bash
python -m src.enrich_knowledge.runners.run_training --model regime
python -m src.enrich_knowledge.ml_training.regime --symbol btcusdt
```

Daily-only. `FORWARD_HORIZON_DAYS = 60` for the forward-realized
label rule. Per-symbol artifact at
`models/1d/regime_classifier_<symbol>.joblib`.

### Key levels (A3)

```bash
python -m src.enrich_knowledge.runners.run_training --model key_levels
python -m src.enrich_knowledge.ml_training.key_levels --symbol btcusdt
```

Not a trained model — DBSCAN clustering cache. Default
`eps_pct = 0.005` and `lookback = 365` daily candles. Output is a
JSON cache file the MCP tool reads directly:

```json
{
  "current_price": 76579.0,
  "levels": [
    {"center": 75000.0, "low": 74800.0, "high": 75200.0,
     "touches": 9, "type": "support", "dist_pct": -2.06},
    ...
  ]
}
```

## Walk-forward backtests

Every B/A model with an artifact has a matching backtest script.
These are validation tools, not deployment trainers — they emit
per-prediction trace CSVs at
`models/<tf>/backtest_<family>_<symbol>.csv` for analysis.

```bash
# Direction classifier (B2)
python -m src.enrich_knowledge.ml_training.backtest_direction \
    --symbol btcusdt --timeframe 1h

# Direction regression counterfactual (proves classification > regression)
python -m src.enrich_knowledge.ml_training.backtest_direction_regression \
    --symbol btcusdt --timeframe 1h

# Anomaly detector (B4)
python -m src.enrich_knowledge.ml_training.backtest_anomaly \
    --symbol btcusdt --timeframe 1h --horizon 4 --abs-threshold 0.02

# Key levels (A3)
python -m src.enrich_knowledge.ml_training.backtest_key_levels \
    --symbol ethusdt --near-pct 0.005 --eps-pct 0.005

# Regime classifier (A4)
python -m src.enrich_knowledge.ml_training.backtest_regime \
    --symbol btcusdt --fwd-horizon-days 30
```

`--lookback-days` and `--refit-every-days` sweep window and cadence
on all four; direction and anomaly need a `--timeframe`; key levels
runs on daily candles only.

**Engineering note.** The backtest scripts force `n_jobs=1` on
XGBoost. Default `n_jobs=-1` spends ~57 s per fit on a 1 074-row
matrix coordinating threads; `n_jobs=1` does the same fit in 0.22 s.
Walk-forward over 321 folds drops from ~8 hours to ~70 seconds.

## Recommended cadence

| Job                                        | When (Vietnam) | UTC   | Why                                                                 |
|--------------------------------------------|----------------|-------|---------------------------------------------------------------------|
| `run_training --model all` (full retrain)  | 07:15          | 00:15 | Daily candle just closed; 15-min buffer past funding settlement.    |
| `run_training --model retrain_all` (cron)  | weekly Sun 02:00 |     | Cheap; only re-fits stale families.                                 |

Second-best slot if 07:15 is inconvenient: 23:15 Vietnam (16:15
UTC) — post-funding, pre-US close. Daily candle is still 7 h from
closing, so 1d features will be slightly stale.

**Avoid:** funding settlements (07:00 / 15:00 / 23:00 Vietnam ±5
min), weekends (volume drops, fakeouts rise, backtest stats
degrade), and macro releases (FOMC / CPI / NFP ±30 min).

### Cron

```bash
15 0 * * *   cd ~/source/personal/bot && python -m src.enrich_knowledge.runners.run_training --model all >> logs/train.log 2>&1
```

### systemd

```ini
# /etc/systemd/system/bot-retrain.service
[Service]
Type=oneshot
WorkingDirectory=/home/tnguyen/source/personal/bot
EnvironmentFile=/home/tnguyen/source/personal/bot/.env
ExecStart=/usr/bin/python -m src.enrich_knowledge.runners.run_training --model retrain_all
```

```ini
# /etc/systemd/system/bot-retrain.timer
[Timer]
OnCalendar=Sun 02:00
Persistent=true

[Install]
WantedBy=timers.target
```

## Adding a new training driver

1. Create `src/enrich_knowledge/ml_training/<name>.py` with:
   - `def fit(*, symbol, timeframe, ...) -> None` — library
     entrypoint that loads the CSV, fits, writes the artifact.
   - `def train(settings, dry_run=False) -> None` — iterates the
     configured `(symbol, timeframe)` pairs and calls `fit()`
     in-process. Catch per-pair exceptions so one failure doesn't
     abort the batch.
   - Optional `def main()` block for ad-hoc single fits.
2. Register it in `runners/run_training.MODEL_REGISTRY`.
3. (Optional) add an entry to `retrain_all._TASKS` with an
   appropriate TTL and data check.

**Do not** shell out to a subprocess to run another module —
import and call. The driver, the batch runner, and the ad-hoc CLI
all share the same library function.

## Troubleshooting

**`model_unavailable` from the MCP tool** — the artifact is
missing. Run the matching driver:

```bash
python -m src.enrich_knowledge.runners.run_training --model anomaly
```

**`percentile_unavailable`** — the OHLCV CSV is missing. Run a
backfill:

```bash
python -m src.enrich_knowledge.runners.run_training --skip-backfill --model retrain_all   # no-op, just to load settings
# or directly:
python scripts/enrich/train_anomaly.py --symbol BTCUSDT --interval 4h
```

**`WARNING: fewer than 500 pooled samples — refusing to fit`** —
the OHLCV CSV is too short. Either backfill more history (raise
the relevant `DEFAULT_ROWS` in
`src/enrich_knowledge/ml_training/backfill_ohlcv.py`) or drop the
problematic timeframe from `ML_TIMEFRAMES`.

**XGBoost training looks stuck** — check whether you're running a
backtest script that hasn't been forced to `n_jobs=1`. The
production trainers default to `n_jobs=1` already.
