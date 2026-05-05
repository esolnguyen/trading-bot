# Daily run schedule

Recommended times to run the ML training job and the trading-agent price
analysis. All times in both Vietnam local (UTC+7) and UTC so the schedule
stays portable.

## ML training (`run_training --model all`)

Run once per day. The models use daily + intraday OHLCV features, so the
best slot is just after the daily candle closes.

| Time (Vietnam) | UTC     | Rationale                                                           |
|----------------|---------|---------------------------------------------------------------------|
| **07:15**      | 00:15   | Daily 1d candle just closed; 15-min buffer past funding settlement. |

Cron (system local time on WSL):

```bash
15 0 * * * cd ~/source/personal/bot && python -m src.enrich_knowledge.runners.run_training --model all >> logs/train.log 2>&1
```

Second-best slot if 07:15 is inconvenient: **23:15 Vietnam / 16:15 UTC**
(post-funding, pre-US close). Daily candle is still 7h from closing, so
1d features will be slightly stale.

## Trading-agent price analysis

Crypto is 24/7 but liquidity and signal quality vary. Pick by purpose.

| Time (Vietnam) | UTC         | Purpose                                                         |
|----------------|-------------|-----------------------------------------------------------------|
| **20:30-22:00**| 13:30-15:00 | **Primary.** EU+US overlap, highest volume, cleanest structure. |
| 07:15          | 00:15       | Morning bias: post daily-close, post-funding.                   |
| 15:15          | 08:15       | Intraday refresh: post 08:00 UTC funding, pre-EU open.          |
| 23:15          | 16:15       | Confirm/fade EU move: post 16:00 UTC funding, mid US session.   |

If running once a day: **21:00 Vietnam**.
If running twice: add **07:15 Vietnam** for morning bias.

## Times to avoid

- **Funding settlements**: 07:00, 15:00, 23:00 Vietnam (00:00 / 08:00 /
  16:00 UTC) plus or minus 5 minutes. Brief volatility spike; structure
  reads are noisy.
- **Weekends**: volume drops, fakeouts rise, backtest stats degrade.
- **Macro releases**: plus or minus 30 minutes around FOMC / CPI / NFP.
  Price moves override TA; wait for the dust to settle.

## Cadence by bot layer

Depends on which layer of the bot we're talking about:

| Layer | Cadence | Why |
|---|---|---|
| Price / liquidation guard | every 5-15 s | At 50x a 1% move is ~50% PnL; you need tick-level stops |
| 1H signal refresh (analyze_signal + patterns) | every 1H candle close (+ a safety recheck at +55 min) | The setup above is 1H-based; no new structural info arrives mid-candle |
| ML direction + anomaly | every 1H | Models were trained on 1H features; sub-hourly re-inference just adds noise |
| MTF alignment (analyze_multi_tf) | every 4H close | That's when the 4H flips |
| Cycle / key-levels | every 1D | Daily-trained, doesn't move intraday |
| RAG news / macro | every 15-30 min | ETF-flow / headline catalysts are the main intraday risk |
| Funding check | every 8H (just before funding) | To decide if you want to hold through |

Minimum useful full re-analysis: 1 hour. Anything faster re-runs models on
unchanged features and burns API budget. Anything slower than 1H means
you're late to 4H flips.

For this specific ETH setup: the trigger you're watching is a 1H close
through $2,346 (Setup A) or a tag of $2,100 (Setup B) — a bot that polls
price every 30 s and runs the full analysis on each 1H close is
sufficient. No need to poll the whole stack every minute.
