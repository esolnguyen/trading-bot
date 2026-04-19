# enrich_knowledge

Write-path package. Runs continuously to keep ChromaDB (news / macro /
trade_memory) and the ML model files on disk in sync with live data.
The MCP servers (`src/mcp_servers/`) read from the same ChromaDB path
and model files — nothing in this package is read by online code.

## Folder layout

```
src/enrich_knowledge/
├── config/                  # Per-concern BaseSettings + composite
│   ├── ingestion.py         # CRYPTOCOMPARE_API_KEY, RAG_* knobs, SENTIMENT_SCORING_ENABLED
│   ├── ml_training.py       # ML_TIMEFRAME, TRAINING_SYMBOLS, CANDLE_LIMIT
│   ├── schedule.py          # ScheduleEntry dataclass (cadence + jitter)
│   └── __init__.py          # EnrichKnowledgeSettings.load() — single entry
├── rag_ingestion/
│   ├── sources/             # Network only — return raw upstream shapes
│   ├── transforms/          # Pure, no I/O — raw → write-ready Record
│   ├── writers/             # Idempotent ChromaStore upserts
│   └── jobs/                # Per-source JOB_ID + SCHEDULE + async run(settings)
├── ml_training/             # Model drivers (subprocess bridge to scripts/)
└── runners/
    ├── run_ingestion.py     # Long-running APScheduler loop (RAG)
    └── run_training.py      # One-shot CLI (ML)
```

## Two runners

### 1. Ingestion scheduler

```bash
python -m src.enrich_knowledge.runners.run_ingestion                 # foreground
python -m src.enrich_knowledge.runners.run_ingestion --dry-run       # plan-only
python -m src.enrich_knowledge.runners.run_ingestion --log-level DEBUG
```

Walks `rag_ingestion.jobs.SCHEDULES`, registers each entry with
APScheduler, dispatches via `REGISTRY[job_id]` on every tick. A bad
source fails its own cycle — the loop keeps running. Ctrl-C (or
SIGTERM from systemd) triggers a clean shutdown and closes the cached
`BinanceFeed`.

Current jobs (cadences in seconds):

| Job ID                   | Interval | Source                        |
|--------------------------|----------|-------------------------------|
| `macro.alternative_me`   | 1800     | Fear & Greed index            |
| `macro.coingecko`        | 1800     | Global market-cap change 24h  |
| `macro.defillama`        | 900      | Total DeFi TVL                |
| `macro.ohlcv_history`    | 3600     | BTC/ETH snapshot narrative    |
| `news.cryptocompare`     | 900      | Crypto news (filtered + dedup)|

### 2. Training CLI

```bash
python -m src.enrich_knowledge.runners.run_training --model all --dry-run
python -m src.enrich_knowledge.runners.run_training --model anomaly
python -m src.enrich_knowledge.runners.run_training --model retrain_all
```

Batch driver — loads settings, dispatches to the requested model(s),
exits. Never owns a scheduler; put cadence in cron or a systemd timer.

`MODEL_REGISTRY` keys: `anomaly`, `direction`, `key_levels`,
`regime`, `retrain_all`. The last is a staleness-triggered
orchestrator safe for weekly cron — skips models whose on-disk
artifact is still fresh or whose training data is insufficient.

Every invocation first backfills OHLCV (incremental append) for
every `(symbol × timeframe)` pair in `MLTrainingSettings`; pass
`--skip-backfill` when the CSVs are known fresh. Artifacts land in
`models/<timeframe>/<family>_<symbol>.joblib` so timeframes can be
retrained in parallel without stomping on each other.

## Adding a new ingestion source

Three files, one line in an aggregator.

1. **Source** — `rag_ingestion/sources/<name>.py`: `async def fetch(...)` returning a frozen dataclass. Network only; no dedup, no filtering.
2. **Transform** — add a builder to `rag_ingestion/transforms/<collection>.py` that returns a `*Record` with the natural-key `document_id`. Pure function, no I/O.
3. **Job** — `rag_ingestion/jobs/<name>.py` exposing:
   - `JOB_ID` — stable string id
   - `SCHEDULE = ScheduleEntry(job_id=JOB_ID, interval_seconds=..., jitter_seconds=...)`
   - `async def run(settings: EnrichKnowledgeSettings) -> None:` — calls source → transform → writer
4. **Register** — add the module to `rag_ingestion/jobs/__init__.py`'s `_JOBS` tuple.

The writer is usually `chroma_macro.write_records` or
`chroma_news.write_records` — both idempotent via the record's
`document_id`. Calling twice with the same id is a no-op.

## Adding a new ML training driver

1. Create `ml_training/<name>.py` with:
   - `def fit(*, symbol, timeframe, ...) -> None` — the library entrypoint
     that loads the CSV, fits the model, and writes the artifact.
   - `def train(settings, dry_run=False) -> None` — iterates the
     configured `(symbol, timeframe)` pairs and calls `fit()` in-process.
     Catches per-pair exceptions so one failure does not abort the batch.
   - Optional `def main()` + `if __name__ == "__main__":` block for
     ad-hoc single-model refits (`python -m src.enrich_knowledge.ml_training.<name>`).
2. Register it in `runners/run_training.MODEL_REGISTRY`.

Do not shell out to a subprocess — call the fit function directly.
The driver, the batch runner, and the ad-hoc CLI all share the same
library function.

## Trade memory (called from the trading loop)

Not a scheduled job — the trading loop writes directly when a trade
closes.

```python
from src.enrich_knowledge.rag_ingestion.transforms.trade_memory import (
    trade_memory_record,
)
from src.enrich_knowledge.rag_ingestion.writers.trade_memory import write_record

record = trade_memory_record(
    symbol="BTCUSDT",
    action="CLOSE_BUY",
    reasoning="TP1 hit, trailing stop active",
    timestamp="2026-04-18T10:00:00Z",
    quantity=0.01,
    price=42_500.0,
    order_type="LIMIT",
    pnl_usdt=12.5,
    dry_run=False,
)
write_record(storage_settings, record, max_entries=500)
```

The builder enforces the metadata shape the MCP RAG
`retrieve_memory` tool expects. Never hand-assemble the metadata
dict — that's the whole point of the builder.

## Configuration

`EnrichKnowledgeSettings.load()` composes four pydantic-settings
blocks from `.env` / environment:

- `StorageSettings` — `CHROMA_PATH` (default `./chroma_db`)
- `BinanceSettings` — `BINANCE_API_KEY`, `BINANCE_API_SECRET`, etc. (OHLCV job only)
- `IngestionSettings` — upstream API keys + `SENTIMENT_SCORING_ENABLED`
- `MLTrainingSettings` — `ML_TIMEFRAME`, `TRAINING_SYMBOLS`, `CANDLE_LIMIT`

Schedule cadences are code, not env vars — change them in the
relevant `jobs/<name>.py` and commit.

## Invariants

1. **Sources do network only.** Never transform, never write.
2. **Transforms are pure.** Take raw data, return a `*Record`. No I/O,
   no clocks beyond an injectable `now=` parameter.
3. **Writers are idempotent.** Run twice, second run writes zero new
   rows. Dedup by `document_id` (natural key) for macro/news;
   trade_memory appends and prunes by count.
4. **Jobs isolate failures.** One bad upstream never takes the
   scheduler down.
5. **No trading-loop imports.** enrich_knowledge must not reach into
   `src.trading_bot.*`. Cross the boundary by accepting typed
   arguments (see `trade_memory_record`) so the trading loop is the
   caller, never the callee.

## Running under systemd

Ingestion as a long-running service; retraining as a weekly timer.
Split them — don't put retraining in the ingestion loop.

```ini
# /etc/systemd/system/bot-ingestion.service
[Service]
ExecStart=/usr/bin/python -m src.enrich_knowledge.runners.run_ingestion
WorkingDirectory=/home/tnguyen/source/personal/bot
Restart=on-failure
```

```ini
# /etc/systemd/system/bot-retrain.service + .timer (weekly)
[Service]
Type=oneshot
ExecStart=/usr/bin/python -m src.enrich_knowledge.runners.run_training --model retrain_all
WorkingDirectory=/home/tnguyen/source/personal/bot
```
