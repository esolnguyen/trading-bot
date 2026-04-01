# Cold Start Plan

## Problem

On startup, two categories of expensive resources are loaded eagerly:

### 1. ML model files (joblib)
All four ML services call `model_store.load()` inside `__init__`, which deserializes
the joblib bundle into memory immediately:

| Service | Model file |
|---------|-----------|
| `CycleClassifier` | `models/regime_classifier_{timeframe}.joblib` |
| `DirectionClassifier` | `models/xgboost_direction_{timeframe}.joblib` |
| `AnomalyDetector` | `models/isolation_forest_{timeframe}.joblib` |
| `OutcomePredictor` | `models/outcome_predictor.joblib` |

### 2. ChromaDB + embedding model
`ChromaStore.__init__` eagerly:
- Opens `chromadb.PersistentClient` (DB connection + WAL recovery)
- Loads `SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")` (downloads
  ~80 MB model, loads into memory via sentence-transformers/torch)
- Creates/opens all 3 collections

**Already lazy (no change needed):**
- `SentimentScorer` — already defers FinBERT to first `.score()` call
- `HistoricalPercentileScorer` — reads CSV on first `.score()` call, not in `__init__`
- `KeyLevelDetector` — no model file, pure in-memory computation

---

## Approach: Per-Service Lazy Initialization

Use the same pattern `SentimentScorer` already follows:
- `__init__` stores only configuration (path, timeframe, thresholds) — no I/O
- A private `_ensure_loaded()` method loads on first use and caches the result
- Every public method calls `_ensure_loaded()` before accessing `self._bundle`

No settings flag is needed — lazy loading is always correct. Services already handle
`_bundle is None` gracefully (return `None` / fallback), so removing the eager load
cannot break any existing behaviour.

---

## Changes Required

### ML Services (same pattern for all four)

**`CycleClassifier`, `DirectionClassifier`, `AnomalyDetector`, `OutcomePredictor`**

Current `__init__`:
```python
def __init__(self, timeframe: str = "4h") -> None:
    self._bundle = load(f"regime_classifier_{timeframe}")  # ← eager
```

New `__init__` + `_ensure_loaded`:
```python
def __init__(self, timeframe: str = "4h") -> None:
    self._timeframe = timeframe
    self._model_name = f"regime_classifier_{timeframe}"
    self._bundle: dict[str, Any] | None = None
    self._loaded = False

def _ensure_loaded(self) -> None:
    if not self._loaded:
        self._bundle = load(self._model_name)
        self._loaded = True
```

Every public method that accesses `self._bundle` adds one line at the top:
```python
self._ensure_loaded()
```

The `if self._bundle is None: return None` guard already present in each method
continues to handle the "model file absent" case without any change.

### ChromaStore

Split `__init__` into two phases:

**Phase 1 — `__init__` (cheap, always runs):**
- Validate + create the directory
- Store `self.path`, `self._embedding_function_factory` (a callable or `None`)
- Set `self._client = None`, `self._embedding_function = None`, `self.collections = {}`

**Phase 2 — `_ensure_connected()` (deferred to first use):**
```python
def _ensure_connected(self) -> None:
    if self._client is not None:
        return
    import chromadb
    self._client = chromadb.PersistentClient(path=str(self.path))
    self._embedding_function = (
        self._embedding_function_factory()
        if self._embedding_function_factory
        else build_default_embedding_function()
    )
    self._collection_suffix = self._build_collection_suffix()
    self.collections = {
        name: self._client.get_or_create_collection(
            name=self._physical_collection_name(name),
            embedding_function=self._embedding_function,
        )
        for name in self.COLLECTIONS
    }
```

Every public method (`count`, `exists`, `add_document`, `query`, `get_raw_client`)
adds `self._ensure_connected()` at the top.

The `_validate_path()` call stays in `__init__` — it only checks writability, which
is cheap and provides an early error before the bot starts its trading loop.

---

## What Does NOT Change

- `app.py` — `build_runtime()` is unchanged; objects are still constructed at startup,
  they just no longer perform I/O during construction
- The `_bundle is None` / `return None` fallback paths in every ML service — unchanged
- `SentimentScorer` — already lazy, no change
- `HistoricalPercentileScorer` — already lazy, no change
- All tests — the mock/absent-model path already works; lazy loading preserves it

---

## Expected Impact

| Resource | Before | After |
|----------|--------|-------|
| 4× joblib deserialization | at `build_runtime()` | at first trading cycle |
| Chroma PersistentClient (WAL) | at `build_runtime()` | at first RAG/ingestion call |
| SentenceTransformer model (~80 MB) | at `build_runtime()` | at first `add_document`/`query` |
| Bot ready to accept config / health checks | after all loads | immediately |

First trading cycle will be slightly slower as models hydrate — this is the
expected cold-start trade-off.
