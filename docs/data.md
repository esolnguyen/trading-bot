# Data on disk

Five directories under the repo root hold every persistent
artifact this project produces. None of it is committed to git
(see `.gitignore`); all of it is restartable from the bot's two
write-path runners.

```
data/                # OHLCV CSVs + per-symbol position snapshots
models/              # Trained ML artifacts (joblib, json caches)
chroma_db/           # ChromaDB SQLite + collection segments
history/             # Markdown analyses written by the /trade skill
logs/                # Ingestion + training logs
```

The MCP servers under `src/mcp_servers/` only **read** these. The
write path lives in `src/enrich_knowledge/`.

---

## OHLCV CSVs — `data/ohlcv/`

One CSV per `(symbol × timeframe)` pair:

```
data/ohlcv/btcusdt_1m.csv
data/ohlcv/btcusdt_5m.csv
data/ohlcv/btcusdt_15m.csv
data/ohlcv/btcusdt_1h.csv
data/ohlcv/btcusdt_4h.csv
data/ohlcv/btcusdt_1d.csv
data/ohlcv/ethusdt_<tf>.csv
data/ohlcv/solusdt_<tf>.csv
data/ohlcv/bnbusdt_<tf>.csv
```

**Schema** (header row, then ms-epoch rows ordered ascending):

```csv
timestamp,open,high,low,close,volume
1704067200000,42283.4,42301.7,42268.1,42292.5,143.182
```

**Default row counts** (`src/enrich_knowledge/ml_training/backfill_ohlcv.py`):

| Interval | Rows   | History         |
|----------|--------|-----------------|
| `1m`     | 43 200 | ~30 days (Binance REST keeps ~30 d of 1m data) |
| `5m`     | 17 280 | ~60 days        |
| `15m`    | 70 000 | ~2 years        |
| `1h`     | 17 520 | ~2 years        |
| `4h`     | 16 500 | full history since 2017 |
| `1d`     | 2 800  | full history since 2017 |

`run_training` performs an **incremental append** on every
invocation: it reads the last timestamp in the CSV and pulls only
candles newer than that. Running daily is cheap.

---

## Trained models — `models/`

One subdirectory per timeframe, plus a couple of single-file
artifacts at the top level.

```
models/
├── 15m/
│   ├── isolation_forest_btcusdt.joblib       (B4)
│   ├── isolation_forest_ethusdt.joblib       (B4)
│   ├── xgboost_direction.joblib              (B2 — pooled)
│   └── backtest_*.csv                        (walk-forward traces)
├── 1h/
│   ├── isolation_forest_<symbol>.joblib
│   ├── xgboost_direction_<symbol>.joblib     (legacy per-symbol)
│   ├── xgboost_direction.joblib              (B2 — pooled, current)
│   └── backtest_*.csv
├── 4h/   (same shape as 1h)
├── 1d/
│   ├── isolation_forest_<symbol>.joblib
│   ├── xgboost_direction_<symbol>.joblib
│   ├── regime_classifier_<symbol>.joblib     (A4)
│   ├── key_levels_<symbol>_cache.json        (A3)
│   └── backtest_*.csv
└── outcome_predictor.joblib                  (B3 — single global model)
```

**Per-timeframe is intentional** — `retrain_all` globs each path
for staleness checks, and timeframes can be retrained in parallel
without stomping on each other.

**Joblib bundle shape** for the direction classifier:

```python
{
    "model": CalibratedClassifierCV(...),
    "feature_cols": ["rsi_14", "macd_line", ..., "symbol_id"],
    "symbol_map": {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2, "BNBUSDT": 3},
    "max_horizon": 24,
    "barrier_k": 1.5,
    "recency_lambda": 1.0,
}
```

The MCP service (`src/mcp_servers/ml_mcp/services/model_store.py`)
loads these lazily on first call. Missing artifact → tool returns
`{"success": false, "error": "model_unavailable", ...}`.

For the meaning of each model see [docs/ml-models.md](ml-models.md);
for how to refit them see [docs/training.md](training.md).

---

## Position snapshots — `data/position_<symbol>.json`

Per-symbol JSON snapshots written by the trading bot when a
position changes. Survives process restarts so cooldown / open-leg
state is preserved across `systemctl restart`.

```json
{
  "symbol": "ETHUSDT",
  "side": "LONG",
  "entry_price": 2291.5,
  "qty": 0.05,
  "stop_loss_pct": 1.5,
  "take_profit_pct": 3.0,
  "opened_at": "2026-04-12T14:00:00Z"
}
```

These files are managed entirely by the trading bot — neither the
MCP servers nor the trainers touch them.

---

## ChromaDB — `chroma_db/`

ChromaDB persistent client at `CHROMA_PATH` (default `./chroma_db`).
Three collections, all written by `src/enrich_knowledge/runners/run_ingestion.py`:

| Collection      | Source                                            | Cadence | Document id (natural key)                          |
|-----------------|---------------------------------------------------|---------|----------------------------------------------------|
| `news`          | CryptoCompare news API (filtered + deduped)       | 15 min  | `cryptocompare:<article_id>`                       |
| `macro`         | Fear & Greed, CoinGecko global, DefiLlama TVL, OHLCV narrative | 15–60 min | `<job_id>:<bucket_timestamp>`              |
| `trade_memory`  | Written by the trading loop when a trade closes   | event   | `<symbol>:<closed_at_ms>`                          |

**Embedding function.** Both readers and writers call
`build_default_embedding_function()` so they resolve to the same
physical collection suffix.

**Metadata keys** the MCP RAG tools rely on:

```
news:        {published_at, source, symbol, sentiment_score?}
macro:       {timestamp, source, metric}
trade_memory:{timestamp, symbol, action, reasoning, outcome_pnl}
```

`enrich_knowledge.runners.run_ingestion` stores timestamps
inconsistently (`news` uses unix-epoch strings, `macro` uses
ISO-8601, `trade_memory` has `0` sentinels). The MCP service
normalises both formats — clients see a uniform `freshness_seconds`
+ `as_iso` pair on every retrieval.

---

## Analysis archive — `history/`

Every `/trade` invocation that produces an analysis writes a
markdown file under `history/`:

```
history/2026-04-19T08-57-12.md
history/2026-04-19T11-23-00.md
```

Filename format: `YYYY-MM-DDTHH-MM-SS.md` (UTC ISO-8601 with
colons replaced by `-` so the name is filesystem-safe).

**Frontmatter.**

```yaml
---
symbol: BTCUSDT
timeframe: [4h, 1h]
direction: long
leverage: 5x
confidence: 4
generated_at: 2026-04-19T08:57:12Z
user_prompt: "What's going on with BTC at this 4h close?"
---
```

**Required sections** (enforced by the skill):

1. **Verdict** — one-paragraph TL;DR mirroring the chat answer.
2. **Setup table** — entry / SL / TP1 / TP2 / R:R when a setup is
   proposed.
3. **Evidence** — tool calls + numbers behind the verdict (price,
   funding, OI, multi-TF reads, indicators, structure, ML, news /
   macro, backtest).
4. **Confidence + invalidation** — the single condition that kills
   the thesis.
5. **Caveats** — unavailable models, stale RAG, low-sample
   backtests, anything that should discount the read.

The chat reply ends with one line stating the saved path. The full
analysis is **not** pasted back into chat — the user opens the file.

---

## Logs — `logs/`

Two logfiles when the long-running services are under systemd:

```
logs/ingestion.log      # APScheduler tick output from run_ingestion
logs/train.log          # run_training output (cron / timer)
```

Neither is rotated — pair systemd with `journald` or wrap the
`ExecStart` with `logrotate` if you keep these around long-term.

---

## Reset / restart guarantees

Every directory above is restartable from the two runners:

| Resource          | Rebuild via                                                |
|-------------------|------------------------------------------------------------|
| `data/ohlcv/*.csv`| `run_training --model all` (does an incremental backfill)  |
| `models/`         | `run_training --model all`                                 |
| `chroma_db/`      | `run_ingestion` (will repopulate over next 24 h)           |
| `history/`        | (read-only journal; never auto-deleted)                    |
| `logs/`           | (deletable — they're append-only)                          |

If you need a clean slate:

```bash
rm -rf data/ohlcv models chroma_db
python -m src.enrich_knowledge.runners.run_training --model all
python -m src.enrich_knowledge.runners.run_ingestion --dry-run    # confirm jobs first
python -m src.enrich_knowledge.runners.run_ingestion              # then start it
```

`history/` should not be deleted — it's the journal of past
analyses and is read by the operator (and occasionally by the
`retrieve_memory` MCP tool).
