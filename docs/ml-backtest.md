# ML model walk-forward backtest

End-to-end audit of the ML models used by the trading pipeline. For each
model with an artifact on disk, we measure honest out-of-sample
predictive value via walk-forward validation, then compare to a
sensible baseline. The goal is to know what each model is actually
worth before relying on it as either an LLM context input or a hard
gate.

## Methodology — why walk-forward

The existing trainers use `TimeSeriesSplit(n_splits=5)` inside the
training window for cross-validation. That fold scheme is
expanding-origin within a fixed 180-day window — so train and test
always come from the **same regime**. It overstates accuracy because
the model never has to generalise across a regime transition.

The walk-forward backtest mimics how the model is actually deployed:

```
fit on candles t-W .. t      → predict t+1 .. t+S
fit on candles t-W+S .. t+S  → predict t+S+1 .. t+2S
…
```

Refit every `S` candles on a rolling `W`-day window, score the next
`S` candles out-of-sample, then aggregate metrics over the entire
history. Embargo the last `LOOKAHEAD` rows of each train slice so
target labels do not peek into the test window.

All backtest scripts live in `src/enrich_knowledge/ml_training/`:

- `backtest_direction.py` — XGBoost direction classifier (B2)
- `backtest_direction_regression.py` — counterfactual: regress return
  with same features, same window
- `backtest_anomaly.py` — IsolationForest anomaly detector (B4)
- `backtest_key_levels.py` — DBSCAN key-level detector (A3)
- `backtest_regime.py` — RF cycle classifier (A4) — economic validity
  of rule labels

## Engineering note: XGBoost `n_jobs=1`

First run on BTC 4h was projected to take ~8 hours for 321 folds (~90s
per fit). A 1-fit smoke test revealed the cause:

| Setting | Time per fit (1074 × 23 features) |
|---|---|
| XGBoost default `n_jobs=-1` (all cores) | **57.2s** |
| `n_jobs=1` | **0.22s** |

XGBoost's threading overhead dominates on small training sets — it
spends nearly all of its time in coordination, almost none on the
actual splits. **All backtest scripts force `n_jobs=1`**, cutting the
4h run from 8 hours to ~70 seconds. Production trainers may benefit
from the same change for these per-symbol/per-timeframe fits.

## B2 — Direction classifier

XGBoost binary: `P(close[t+6] / close[t] - 1 > 0.002)`. Asymmetric — it
answers "is there a bullish thrust?" not "long or short?".

### Walk-forward results (180d window, 7d refit)

| Variant | Pooled AUC | Per-fold AUC ± std | Brier | Acc | Base | Edge |
|---|---|---|---|---|---|---|
| btc 4h 180d | 0.528 | 0.553 ± 0.160 | 0.301 | 0.519 | 0.461 | +1.8% |
| **btc 1h 180d** | **0.549** | 0.556 ± **0.083** | **0.261** | **0.544** | 0.399 | **+3.4%** |
| btc 4h 90d | 0.524 | 0.557 ± 0.156 | 0.322 | 0.521 | 0.462 | +1.5% |
| btc 4h 360d | 0.528 | 0.556 ± 0.157 | 0.282 | 0.519 | 0.457 | +1.7% |
| eth 4h 180d | 0.541 | 0.556 ± 0.161 | 0.299 | 0.527 | 0.470 | +2.6% |

### Findings

1. **Real edge is small** — pooled AUC ~0.53. Existing in-sample CV
   AUC of 0.6+ overstates performance by ~7 points; that gap is
   regime-transition error the in-sample fold scheme cannot see.
2. **1h beats 4h decisively**. Fold std drops from 0.160 to 0.083 (the
   model is far more stable across regimes), Brier drops from 0.301
   to 0.261 (calibration recovers from "worse than naive prior" to
   "near-naive"), and pooled AUC is highest. The 6-bar target is
   shorter on 1h (6h ahead vs 24h on 4h), so macro noise has less time
   to overwhelm the indicator signal.
3. **Lookback length doesn't matter on 4h**. 90d, 180d, and 360d all
   produce pooled AUC within 0.005 of each other. The bottleneck is
   not how much history we feed but what features the model sees.
4. **Accuracy at threshold 0.5 < majority baseline** (0.519 vs 0.539).
   The model has ranking power (AUC > 0.5) but the default 0.5
   threshold is mis-calibrated — you'd be better off flipping a coin
   weighted by the base rate. Production needs threshold tuning per
   symbol/timeframe.

### Counterfactual — regression vs classification

To test the conjecture "regression on returns is dominated by noise",
we ran the same setup with `XGBRegressor` predicting raw 6-bar return.

|  | Classifier 4h | Regressor 4h | Classifier 1h | Regressor 1h |
|---|---|---|---|---|
| Pooled AUC | **0.528** | 0.522 | **0.549** | 0.539 |
| Fold AUC mean | 0.553 | 0.546 | 0.556 | 0.542 |
| Fold AUC std | 0.160 | 0.161 | 0.083 | 0.088 |
| R² (regression-native) | — | **−0.048** | — | **−0.086** |

R² < 0 means the regressor is **worse than predicting the constant
mean return**. MSE on this signal-to-noise ratio is essentially
unlearnable — the optimiser cannot find traction. The classifier wins
by ~1pp pooled AUC, modest but consistent. With a tree model the gap
is small (trees split on features, not gradients of loss), but the
regressor's R² confirms that its native loss function is unhelpful
even when the ranking accidentally turns out OK.

The decisive practical advantage of the classifier remains: its
output is a calibratable probability that drops cleanly into the risk
gate (Kelly sizing, threshold tuning, ensemble with `OutcomePredictor`).
A predicted return of "0.4%" requires an arbitrary threshold to act
on — you end up rebuilding a classifier on top.

## B4 — Anomaly detector

IsolationForest on `[vol_ratio, price_vel, high_low_rng]`. The flag is
a **hard gate**: when set, the trading cycle is paused — the LLM is
never consulted. Right metric is **lift** = P(big move | flagged) /
P(big move | any candle). Lift > 1 means the flag carries information
beyond chance.

Big move = max forward run-up OR max forward drawdown over `horizon`
candles exceeds 2%.

### Walk-forward results

|  | BTC 1h | BTC 4h | ETH 1h |
|---|---|---|---|
| Flag rate | 0.90% | 1.51% | 0.74% |
| Base rate | 27.5% | **77.9%** | 61.1% |
| Precision (big\|flag) | 88.1% | 94.1% | 90.3% |
| Recall (flag\|big) | 2.9% | 1.8% | 1.1% |
| **Lift** | **3.20** | 1.21 | 1.48 |
| Verdict | predictive | marginal | marginal |

### Findings

1. **BTC 1h is the only setup where the gate genuinely earns its
   keep**: lift 3.20, precision 88% on a base rate of 27.5%. Recall is
   tiny (2.9%) but recall is not the goal of a circuit breaker — the
   detector should fire rarely, with high precision.
2. **BTC 4h is near-placebo** at the current threshold. Base rate is
   already 78% (most 4h candles are followed by some 2% move within
   24h), so flagging adds only 16pp of precision. Threshold/horizon
   need to be calibrated per timeframe — 2% over 24h is a meaningless
   bar for BTC. Suggest 5% over 24h on 4h.
3. **The detector also catches data glitches**: bad ticks (e.g. a
   2020-10-30 row with `fwd_runup = 48x`) get flagged. In production
   that is a feature, not a bug — pause trading on a corrupted feed.
4. **Average flagged-runup metrics are skewed by the glitches**. Use
   the lift number for a fair read; magnitude conditioning (median)
   tells the same story without the outliers.

## A3 — Key-level detector

DBSCAN on (highs, lows, midpoints) of the last 365 days. Not a real
ML model — a clustering cache rebuilt periodically. Right test:
**do forward 30-day reversal points cluster near identified levels
more than random candles do?** Lift = P(reversal near level) /
P(any candle near level).

### Walk-forward results

| Variant | Symbol | Near | Span | P(rev near) | P(base) | Lift | Verdict |
|---|---|---|---|---|---|---|---|
| Default | BTC | 2.0% | 5 | 0.466 | 0.491 | 0.95 | no edge |
| Default | ETH | 2.0% | 5 | 0.403 | 0.413 | 0.97 | no edge |
| Tight (eps 0.5%) | BTC | 0.5% | 5 | 0.123 | 0.147 | 0.84 | worse than random |
| **Tight (eps 0.5%)** | **ETH** | **0.5%** | 5 | **0.153** | **0.093** | **1.64** | **predictive** |
| Major-rev only | BTC | 2.0% | 11 | 0.470 | 0.491 | 0.96 | no edge |
| Major-rev only | ETH | 2.0% | 11 | 0.426 | 0.413 | 1.03 | marginal |

### Findings

1. **Default eps_pct=0.003 packs levels too densely**. With ~15 levels
   in a 365-day price range and a 2% touch zone, ~half of all daily
   candles are "near" some level — there is no remaining room to
   distinguish reversals from non-reversals. Lift collapses to 1.
2. **Tightening to eps=0.005 + 0.5% touch zone produces a real ETH
   signal**: lift 1.64 (Z≈5, p<0.0001 on 603 reversals). ETH reversal
   points really do cluster at the identified levels.
3. **BTC produces no edge in any tested config**. Possible reasons:
   deeper liquidity makes BTC more likely to break levels than bounce;
   stronger trends mean reversals tend to occur at fresh extremes
   rather than historical clusters.
4. **Filtering for major reversals (span=11) does not help** —
   methodology robustness is fine, the levels themselves just are not
   informative for BTC at the default config.

## Fixes applied

**`src/enrich_knowledge/ml_training/key_levels.py`** — bumped
`eps_pct` default from `0.003` to `0.005` (function default + CLI
default). The cache files for both symbols were regenerated
immediately after. The inference path (`KeyLevelDetector`) reads
whatever the cache file contains, so the training-time change flows
through automatically on the next refit.

**`src/mcp_servers/ml_mcp/services/cycle_classifier.py`** — added
`REGIME_INSTRUCTIONS_OVERRIDES`, a per-symbol overlay over the BTC
defaults. ETHUSDT now gets mean-reversion-aware text for all four
labels: BEAR_TRENDING reads "OVERSOLD-BIASED, look for LONGs at
support" instead of the BTC "do not open LONG positions". The
`regime_system_prompt_suffix(regime, confidence, symbol=None)` method
takes an optional symbol and consults the override map first, falling
back to the BTC default. Existing call sites that omit `symbol`
behave exactly as before (unchanged BTC text).

Production behaviour for the direction classifier and anomaly
detector was not changed; the findings inform what work is worth
doing next, not what should be reverted now.

## Recommended follow-ups (not yet implemented)

- **Direction classifier**: per-(symbol, timeframe) threshold tuning.
  The default 0.5 cutoff loses to the majority baseline on every
  variant tested. Pick the threshold that maximises F1 (or Sharpe on a
  simulated equity curve) on each walk-forward trace.
- **Anomaly detector**: scale `abs_threshold` and `horizon` by
  timeframe. 2% over 4 candles is the right calibration for 1h; 5%
  over 6 candles fits 4h better.
- **Key levels**: investigate why BTC produces no edge while ETH
  responds to tighter clustering. Volume profile / liquidity-zone
  detection may be a better methodology for BTC's microstructure.

## A4 — Cycle / regime classifier

Random Forest trained against the rule-based labels in
`regime.py:label_regime`. RF-vs-rule accuracy is circular by
construction; the questions worth answering are transition lag,
stability, and **economic validity** — does the regime label actually
predict different forward-return distributions?

### Walk-forward results (365d window, 30d refit, 30d forward horizon)

|  | BTC | ETH |
|---|---|---|
| Agreement with rule (circular) | 0.919 | 0.932 |
| Median transition lag | 0d | 0d |
| Mean transition lag | 16.8d | 3.4d |
| p90 transition lag | 26d | 6d |
| Rule flips / 30d | 1.89 | 2.46 |
| Pred flips / 30d (RF smooths) | 1.55 | 2.25 |

### Economic validity — forward 30d return per predicted regime

**BTC** (signal aligns with intent):

| Regime | n | Mean | Median |
|---|---|---|---|
| ACCUMULATION | 37 | +0.064 | +0.079 |
| BEAR_TRENDING | 553 | −0.004 | −0.007 |
| BULL_CORRECTION | 465 | +0.027 | +0.025 |
| BULL_TRENDING | 895 | +0.064 | +0.013 |

BTC labels separate forward returns in the expected direction —
modest spread (BULL median +1.3% vs BEAR median −0.7%) but
directionally correct.

**ETH** (signal is **inverted**):

| Regime | n | Mean | Median |
|---|---|---|---|
| ACCUMULATION | 76 | +0.088 | −0.025 |
| **BEAR_TRENDING** | 741 | **+0.054** | **+0.019** |
| BULL_CORRECTION | 341 | −0.005 | −0.036 |
| BULL_TRENDING | 713 | +0.015 | −0.002 |

ETH `BEAR_TRENDING` gives **higher** forward returns than
`BULL_TRENDING` — both mean (+5.4% vs +1.5%) and median (+1.9% vs
−0.2%). The rule is calibrated for BTC's trend-following structure;
ETH on a 30-day horizon mean-reverts more strongly, so by the time
the rule says "bear trend", ETH is already oversold and recovers.

### Critical implication

`cycle_classifier.py` injects the regime label into the LLM system
prompt with directional guidance:

```
BULL_TRENDING:  "Favour LONG entries on pullbacks"
BEAR_TRENDING:  "Short positions require very strong confirmation"
```

For ETH the empirical forward return is the opposite of what the
prompt implies — the LLM is being told to lean long when forward
returns are flat, and lean cautious when forward returns are
positive. The prompt either needs per-symbol customisation or the
labels need a different source that respects ETH's mean-reversion
structure on this horizon.

### Recommended actions

- **Done**: forked the prompt guidance via
  `REGIME_INSTRUCTIONS_OVERRIDES["ETHUSDT"]` so ETH gets
  mean-reversion-aware text. See "Fixes applied".
- Investigate whether other symbols (SOL, BNB, etc.) follow BTC or
  ETH structure — needs more daily history before adding overrides.
- Investigate the BTC `ACCUMULATION` cell (n=37) — small sample, very
  high mean (+6.4%) — could be a real edge case worth special prompt
  treatment, or could be sample noise.
- Consider testing a 90d forward horizon to see whether ETH's
  inversion is a 30d-specific reversion artifact or a structural
  feature.

## Untested models and why

- **B3 OutcomePredictor** — no `outcome_predictor.joblib` artifact
  exists, no historical trade-outcome CSV. Cannot backtest until we
  train it on labelled trade history.
- **B1 Sentiment scorer** — pre-trained FinBERT, no project-side
  training. Validation requires sentiment-labelled crypto news, which
  we do not have. Could be tested via correlation with forward returns
  if news ingestion timestamps are aligned with OHLCV.
- **A1 Historical percentile** — pure statistical computation with no
  trained model. Correctness is by definition.

## How to run

All backtests share the same shape: rolling-window walk-forward with a
configurable refit cadence. Each writes a per-prediction trace CSV
under `models/<timeframe>/backtest_<family>_<symbol>.csv` for further
analysis.

```bash
# Direction classifier (B2)
python -m src.enrich_knowledge.ml_training.backtest_direction \
    --symbol btcusdt --timeframe 1h

# Direction regression counterfactual
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

Add `--lookback-days` and `--refit-every-days` to all four to sweep
window/cadence. Direction and anomaly need a `--timeframe`; key-levels
runs on daily candles only.
