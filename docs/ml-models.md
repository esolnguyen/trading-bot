# Machine learning models

The bot ships seven ML models. Six of them write artifacts to
`models/`; one is a pure statistical computation. Every model is
exposed as a `bot-ml` MCP tool — when the artifact is missing, the
tool returns `model_unavailable` rather than crashing.

This document covers, for each model:

1. What it predicts and why.
2. Features and label rule.
3. Validation evidence (walk-forward backtest results, when
   available).
4. Where the artifact lives and which tool consumes it.

To train any of these, see [docs/training.md](training.md).

## Model catalogue

| Code | Name                      | Algorithm                  | Tool                | Artifact                                            |
|------|---------------------------|----------------------------|---------------------|-----------------------------------------------------|
| B2   | Direction classifier      | XGBoost binary, calibrated | `predict_direction` | `models/<tf>/xgboost_direction.joblib`              |
| B3   | Outcome predictor         | Logistic regression        | `predict_outcome`   | `models/outcome_predictor.joblib`                   |
| B4   | Anomaly detector          | IsolationForest            | `detect_anomaly`    | `models/<tf>/isolation_forest_<symbol>.joblib`      |
| A4   | Cycle / regime classifier | Random Forest              | `classify_cycle`    | `models/1d/regime_classifier_<symbol>.joblib`       |
| A3   | Key-level detector        | DBSCAN clustering          | `get_key_levels`    | `models/1d/key_levels_<symbol>_cache.json`          |
| A1   | Historical percentile     | Pure statistical           | `percentile_rank`   | (no artifact — reads OHLCV CSV directly)            |
| B1   | Sentiment scorer          | Pre-trained FinBERT        | `score_sentiment`   | (downloaded weights via `transformers`)             |

The `B*` models inform direction; the `A*` models describe market
context. The trading agent and the `/trade` skill cross-check both
groups before committing to a thesis.

---

## B2 — Direction classifier

XGBoost binary classifier, asymmetric label. Implements the López
de Prado *Advances in Financial Machine Learning* (2018) stack.

**Question it answers.** "Is there a bullish thrust within the next
N bars?" — *not* "long or short?". The output is a calibrated
probability that the upper barrier (`close[t] + 1.5 × ATR`) is
touched before the lower barrier within `MAX_HORIZON` bars.

**Features (15).**

```
rsi_14, macd_line, macd_signal, macd_hist,
ema_spread, ema_20_dist, ema_50_dist,
atr_pct, adx, bb_pos, obv_slope,
vol_ratio, choppiness, cci_14,
symbol_id        # categorical, lets the pooled model specialise
```

**Label.** Triple-barrier on ±`BARRIER_K × ATR` (default 1.5×) over
`MAX_HORIZON` bars (default 24 — 4 days on 4h, 24h on 1h). Bars
where neither barrier is touched (timeout) are dropped, not
zero-labelled.

**Training stack.**

- **Cross-symbol pooling** — every fit sees BTC + ETH + SOL + BNB
  samples together. The model learns regime-conditional structure
  rather than memorising one symbol.
- **Sample-uniqueness weighting** (AFML ch. 4) — bars whose forward
  barrier window overlaps few neighbours get more weight; serially
  overlapping bars get less. Corrects the IID assumption XGBoost
  silently makes.
- **Purged k-fold + embargo** (AFML ch. 7) — drop train rows whose
  barrier-touch time falls inside the validation fold. Then embargo
  the last `MAX_HORIZON` train rows.
- **Recency weighting** — exponential decay (`λ=1.0`) on top of
  uniqueness for the deployment fit. The oldest bar in a 720-day
  window carries `exp(-1) ≈ 0.37×` the weight of the newest. CV
  folds use uniqueness only — applying recency to early folds would
  starve them of data.
- **Isotonic calibration** so `predict_proba` returns numbers that
  match the trade-skill prompt's interpretation.

**Walk-forward results (180-day window, 7-day refit).**

| Variant       | Pooled AUC | Per-fold AUC ± std | Brier | Acc   | Base  | Edge   |
|---------------|------------|--------------------|-------|-------|-------|--------|
| btc 4h 180d   | 0.528      | 0.553 ± 0.160       | 0.301 | 0.519 | 0.461 | +1.8 % |
| **btc 1h 180d** | **0.549** | 0.556 ± **0.083**   | **0.261** | **0.544** | 0.399 | **+3.4 %** |
| btc 4h 90d    | 0.524      | 0.557 ± 0.156       | 0.322 | 0.521 | 0.462 | +1.5 % |
| btc 4h 360d   | 0.528      | 0.556 ± 0.157       | 0.282 | 0.519 | 0.457 | +1.7 % |
| eth 4h 180d   | 0.541      | 0.556 ± 0.161       | 0.299 | 0.527 | 0.470 | +2.6 % |

**Findings.**

1. Real edge is small — pooled AUC ~0.53. Existing in-sample CV AUC
   of 0.6+ overstates by ~7 points; that gap is regime-transition
   error the in-sample fold scheme cannot see.
2. **1h beats 4h decisively.** Fold std drops from 0.160 → 0.083,
   Brier drops from 0.301 → 0.261, pooled AUC is highest. The
   6-bar target on 1h gives macro noise less time to overwhelm
   signal.
3. Lookback length doesn't matter on 4h — 90d / 180d / 360d all sit
   within 0.005 AUC. The bottleneck is feature design, not history.
4. Default 0.5 threshold underperforms the majority baseline. The
   model has ranking power but the cutoff is mis-calibrated;
   production needs per-`(symbol, timeframe)` threshold tuning.

**Counterfactual — regression vs classification.**

|                 | Cls 4h | Reg 4h | Cls 1h | Reg 1h |
|-----------------|--------|--------|--------|--------|
| Pooled AUC      | 0.528  | 0.522  | 0.549  | 0.539  |
| Fold AUC mean   | 0.553  | 0.546  | 0.556  | 0.542  |
| R² (regression) | —      | −0.048 | —      | −0.086 |

R² < 0 means the regressor is worse than predicting the constant
mean return. The classifier wins on every metric AND its output
drops cleanly into a risk gate — calibrated probabilities, Kelly
sizing, ensemble with `OutcomePredictor`.

---

## B3 — Outcome predictor

Logistic regression, used as a **hit-rate prior** on a proposed
trade entry. Inference lives in
`src/mcp_servers/ml_mcp/services/outcome_predictor.py` and consumes
exactly six bucketed features:

```
rsi, adx, atr_pct, trend, vol_state, bb_pos
```

`trend` / `vol_state` / `bb_pos` are categorical (`BULLISH=1`,
`HIGH=1`, `UPPER=1`). Training mirrors that bucketing — change one
side, you must change the other.

**Label.** Same triple-barrier as B2 (`±BARRIER_K × ATR`,
`MAX_HORIZON`). Trained on a long entry — TP hit before SL.

**Training stack.** Identical to B2 — pooled across configured
symbols and timeframes, sample-uniqueness weights, purged folds,
recency decay. Single artifact `models/outcome_predictor.joblib`
serves every symbol.

The walk-forward backtest for B3 needs a labelled trade-history
CSV; until that exists, B3's edge is unmeasured. The MCP tool
returns `model_unavailable` cleanly when the artifact is missing.

---

## B4 — Anomaly detector

IsolationForest. Treated as a **hard gate**: when the flag fires,
the trading cycle is paused — the LLM is never consulted. Right
metric is **lift** = `P(big move | flagged) / P(big move | any candle)`.
Lift > 1 means the flag carries information beyond chance.

**Features (3).**

```
vol_ratio       # current volume / 20-bar average
price_vel       # 3-bar price velocity (% change)
high_low_rng    # candle range / midpoint
```

Microstructure baselines drift — venue / fee / HFT changes shift
"normal" behaviour. Default training window is 180 days
(`DEFAULT_LOOKBACK_DAYS = 180`).

**Walk-forward results.**

|                       | BTC 1h | BTC 4h | ETH 1h |
|-----------------------|--------|--------|--------|
| Flag rate             | 0.90 % | 1.51 % | 0.74 % |
| Base rate (big move)  | 27.5 % | **77.9 %** | 61.1 % |
| Precision (big\|flag) | 88.1 % | 94.1 % | 90.3 % |
| Recall (flag\|big)    | 2.9 %  | 1.8 %  | 1.1 %  |
| **Lift**              | **3.20** | 1.21 | 1.48   |
| Verdict               | predictive | marginal | marginal |

**Findings.**

1. **BTC 1h is the only setup where the gate genuinely earns its
   keep** — lift 3.20, precision 88 % on a 27.5 % base rate.
2. **BTC 4h is near-placebo** at the current threshold (5 % over
   24h would be a more honest bar than 2 % over 24h).
3. The detector also catches **data glitches** — bad ticks like a
   2020-10-30 row with `fwd_runup = 48×` get flagged. In production
   that is a feature: pause trading on a corrupted feed.

---

## A4 — Cycle / regime classifier

Random Forest predicting `BULL_TRENDING` / `BULL_CORRECTION` /
`BEAR_TRENDING` / `ACCUMULATION`. Trained on **forward-realized
labels**, not the rule the inference path used to consume — that
older approach put the rule's features inside the training set and
the RF had nothing to learn beyond memorisation.

**Features (11).**

```
ema50_dist, ema100_dist, ema200_dist,
ema50_slope, ema200_slope,
high_52w_dist, low_52w_dist,
adx_14, realized_vol,
hh_count, ll_count
```

**Label.** Walk forward `FORWARD_HORIZON_DAYS = 60` calendar days
from each bar and assign the regime based on the *actual* forward
return + drawdown + runup that followed. Bars with mixed/noisy
forward windows are dropped.

**Walk-forward results (365d window, 30d refit, 30d forward
horizon).**

| Metric                    | BTC   | ETH   |
|---------------------------|-------|-------|
| Agreement with rule       | 0.919 | 0.932 |
| Median transition lag     | 0d    | 0d    |
| p90 transition lag        | 26d   | 6d    |
| Rule flips / 30d          | 1.89  | 2.46  |
| Pred flips / 30d (smooths)| 1.55  | 2.25  |

**Economic validity** (forward 30d return per predicted regime):

BTC labels separate forward returns in the expected direction
(BULL median +1.3 % vs BEAR median −0.7 %). ETH is **inverted** —
`BEAR_TRENDING` produces *higher* forward returns than
`BULL_TRENDING` on the 30-day horizon. The rule was calibrated for
BTC's trend-following structure; ETH on this horizon mean-reverts.

**Fix shipped.**
`src/mcp_servers/ml_mcp/services/cycle_classifier.py` now exposes
`REGIME_INSTRUCTIONS_OVERRIDES`, a per-symbol overlay over the BTC
defaults. ETHUSDT gets mean-reversion-aware prompt text:
`BEAR_TRENDING` reads "OVERSOLD-BIASED, look for LONGs at support"
instead of the BTC "do not open LONG positions".

---

## A3 — Key-level detector

DBSCAN clustering on (highs, lows, midpoints) of the last 365 daily
candles. **Not** a trained model — a clustering cache rebuilt
weekly. The MCP tool ranks levels by proximity to the current
price.

**Walk-forward validation** — does forward 30-day reversal cluster
near identified levels more than random candles do?

| Variant            | Symbol | Near zone | P(rev near) | P(base) | Lift  | Verdict             |
|--------------------|--------|-----------|-------------|---------|-------|---------------------|
| Default eps 0.3 %  | BTC    | 2.0 %     | 0.466       | 0.491   | 0.95  | no edge             |
| Default eps 0.3 %  | ETH    | 2.0 %     | 0.403       | 0.413   | 0.97  | no edge             |
| Tight eps 0.5 %    | BTC    | 0.5 %     | 0.123       | 0.147   | 0.84  | worse than random   |
| **Tight eps 0.5 %**| **ETH**| **0.5 %** | **0.153**   | **0.093** | **1.64** | **predictive**   |
| Major-rev only     | BTC    | 2.0 %     | 0.470       | 0.491   | 0.96  | no edge             |
| Major-rev only     | ETH    | 2.0 %     | 0.426       | 0.413   | 1.03  | marginal            |

**Fix shipped.**
`src/enrich_knowledge/ml_training/key_levels.py` bumped `eps_pct`
default from `0.003` to `0.005` (function default + CLI default).
The cache files for both symbols regenerated on the next run.

**Why BTC produces no edge.** Likely deeper liquidity (BTC is more
likely to *break* a level than bounce off it) and stronger trends
(reversals occur at fresh extremes rather than historical
clusters). Volume profile / liquidity-zone detection may be a
better methodology for BTC microstructure — open question.

---

## A1 — Historical percentile

Pure statistical computation, no trained artifact. Reads
`data/ohlcv/<symbol>_<tf>.csv`, computes RSI / 30d-momentum /
ATR-pct / vol-ratio, returns the percentile of each at the latest
candle vs. the recent distribution.

Correctness is by definition — there's nothing to validate. The MCP
tool surfaces "unavailable" only when the OHLCV CSV is missing.

---

## B1 — Sentiment scorer

Pre-trained FinBERT (English finance-domain BERT). No
project-specific training. Returns `{label, confidence}` for an
arbitrary string.

Validation requires sentiment-labelled crypto news, which we do not
have. The model has not been measured against forward returns yet —
plausible follow-up work, not a current dependency.

If `transformers` / `torch` is missing on the host, the tool
returns `model_unavailable`.

---

## How the trading agent uses these

The agent's system prompt (`src/trading_bot/agent/prompts.py`)
expects each model to be an **independent voice** rather than an
override. The cross-check rule:

> A recommendation is only "strong" when price structure (TA), ML,
> and context (RAG) agree. If two of three conflict, lean HOLD.

This mirrors `TRADING_MIN_CONVICTION = 6` — below that conviction,
the decision is gated to HOLD regardless of the ReAct graph's
output.

`predict_direction` and `predict_outcome` are the highest-value
inputs for entry timing. `detect_anomaly` is the kill switch.
`classify_cycle` and `get_key_levels` define the regime/structure
context the LLM reasons in.

## Recommended follow-ups (open)

- **Direction classifier**: per-`(symbol, timeframe)` threshold
  tuning. Default 0.5 cutoff loses to the majority baseline on
  every variant tested.
- **Anomaly detector**: scale `abs_threshold` and `horizon` by
  timeframe — 2 % over 4 candles fits 1h; 5 % over 6 candles fits
  4h.
- **Key levels**: investigate volume profile / liquidity-zone
  detection for BTC microstructure.
- **Outcome predictor**: build a labelled trade-history CSV so we
  can measure B3's actual lift.
- **Regime classifier**: test a 90d forward horizon to see whether
  ETH's inversion is a 30d-specific artifact.
