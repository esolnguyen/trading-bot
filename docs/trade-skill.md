# Using the `/trade` skill in Claude

The `/trade` skill turns the five `bot-*` MCP servers into a
hands-on crypto analyst you drive from natural language. It lives
at `~/.claude/skills/trade/SKILL.md` and runs in both **Claude
Code** (CLI / VS Code / JetBrains) and **Claude Desktop**.

The skill is **analysis-only**. It cannot place orders. The trading
bot owns execution and is gated to `dry_run`.

## What the skill does

When you invoke it, the model:

1. Pulls live price + funding + open interest from `bot-binance`.
2. Reads multi-timeframe structure via `bot-analysis`.
3. Cross-checks ML voices via `bot-ml` (direction, anomaly, cycle,
   key levels, percentile, outcome).
4. Retrieves recent news / macro / trade memory via `bot-rag`.
5. Optionally applies a playbook (SMC, Ichimoku, harmonic, etc.)
   from `bot-skills`.
6. Returns a **verdict** with a 1–5 confidence score.
7. **Writes the full analysis** to `history/<UTC datetime>.md` so
   you have a journal of every call.

The cross-check rule the skill enforces:

> A recommendation is only "strong" when **price structure**, **ML**,
> and **context (RAG)** agree. If two of three conflict, it leans
> HOLD and surfaces the conflict.

This mirrors the trading bot's `TRADING_MIN_CONVICTION = 6` gate.

## Prerequisites

1. The five MCP servers registered with your Claude client. From
   the repo root:

   ```bash
   bash scripts/install_mcp_servers.sh
   ```

   That registers `bot-binance`, `bot-analysis`, `bot-ml`,
   `bot-rag`, `bot-skills` under user scope. Verify:

   ```bash
   claude mcp list
   ```

2. ML models trained at least once (otherwise `bot-ml` tools return
   `model_unavailable` — the skill keeps going, just notes it):

   ```bash
   python -m src.enrich_knowledge.runners.run_training --model all
   ```

3. RAG ingestion running (otherwise `bot-rag` returns empty — the
   skill calls `ingestion_status` to surface the cause):

   ```bash
   python -m src.enrich_knowledge.runners.run_ingestion
   ```

4. **Filesystem MCP** rooted at the repo path
   (`/home/tnguyen/source/personal/bot/`) so the skill can write
   `history/` files. Claude Code already CWDs into the repo when
   you run it from there. In Desktop you must configure the
   filesystem MCP root.

## Invoking the skill

In Claude Code or Desktop, type a `/trade` message:

```
/trade BTC 4h, what's the read?
/trade is ETH oversold here, give me a swing setup
/trade should I long SOL into the next funding window?
/trade backtest the analyze_signal rule on BNB 1h, last 6 months
```

The skill is also auto-invoked when the request matches its
description (research a coin, validate a setup, estimate a
strategy's historical edge). You don't need the literal `/trade`
prefix in chat — though it's the cleanest trigger.

## What you get back

A concise, numeric, sourced reply. Format:

```
Verdict: LONG (3/5 conviction)

Confluence:
- Price structure (4h):  ...   tool=analyze_multi_tf
- ML direction:          ...   tool=predict_direction (P=0.58)
- Macro context:         ...   tool=retrieve_macro

Setup table:
| Field   | Value          |
|---------|----------------|
| Entry   | $42 250        |
| SL      | $41 600 (1.5%) |
| TP1     | $43 100 (2.0%) |
| TP2     | $43 900 (3.9%) |
| R:R     | 1 : 2.6        |
| Confidence | 3/5         |

Invalidation: 4h close below $41 600.
Caveats: outcome predictor unavailable; ETHUSDT regime model
inverted on 30d horizon — applies symbol-specific override.

Saved analysis to history/2026-04-19T08-57-12.md
```

The saved markdown file mirrors the verdict and adds the full
evidence section (every tool call + number).

## Confidence score (1–5)

| Score | Meaning                                                          |
|-------|------------------------------------------------------------------|
| 1     | Avoid / no-trade.                                                |
| 2     | Weak — most signals neutral or conflicted.                       |
| 3     | Tradeable with caveats (one of TA / ML / context disagrees).     |
| 4     | Solid confluence (all three agree on direction).                 |
| 5     | Rare full-stack alignment + supportive backtest + clean macro.   |

The score is derived mechanically from the cross-check rule. It is
deducted for unavailable models, low backtest sample sizes,
counter-trend setups, or stale RAG.

## Backtesting a rule

For "should I keep using this signal?" rather than "is this
setup live?", ask the skill to backtest:

```
/trade backtest the analyze_signal rule on BTC 1h, last 6 months
/trade what's the equity curve for this oversold-RSI rule on SOL?
```

The skill calls `bot-analysis.backtest_signal` with default fees
(`fee_bps = 10`, `slippage_bps = 5`) and reports total return,
CAGR, annualised Sharpe, max drawdown, win rate, trade count,
turnover, and a downsampled equity curve. The skill **will not**
lower fees to flatter a setup.

The backtest is single-symbol, close-to-close, doesn't replay
historical funding, and uses flat-bps costs — treat the numbers as
indicative, not a prod PnL claim.

## What the skill won't do

- **No non-crypto markets.** Equities, futures, forex, and options
  are out of scope.
- **No live execution.** It deliberately does not call
  `cancel_order` or any order-placement tool. The trading bot
  handles execution and is itself gated to `dry_run`.
- **No fabrication.** If a tool fails or a model is unavailable,
  the skill says so — it does not paper over missing data.
- **No multi-symbol screens.** The toolchain is single-symbol per
  call. For "top-momentum USDT pairs" the skill must explicitly
  loop and surface the cost.
- **No `WebFetch` / `WebSearch`** unless you explicitly ask for
  general web research. News comes from `bot-rag.retrieve_news`.

## How it interacts with the trading bot

The skill is a **read replica** of the trading bot's signal stack:

- Same MCP servers (the bot spawns them via stdio; the skill
  talks to them via the standard MCP protocol).
- Same playbooks (`bot-skills` reads the SKILL.md files at
  `src/mcp_servers/skills_mcp/skills/`, which the bot's in-process
  loader also reads).
- Same model artifacts and same RAG corpus.

The bot decides on a fixed cadence (`TRADING_DECISION_INTERVAL_SECONDS`,
default 900 s); the skill is on-demand. They never block each
other — both call MCP servers as clients.

If a trade closes, the bot writes a `trade_memory` record. The
next `/trade` call on the same setup retrieves that memory via
`bot-rag.retrieve_memory` and includes it in the analysis.

## Tweaking the skill

The skill is plain markdown at `~/.claude/skills/trade/SKILL.md`.
Editing it changes the prompt the model receives — no rebuild
needed. Useful tweaks:

- Adjust the confidence rubric.
- Add or remove the playbooks the skill is allowed to invoke.
- Change the required sections in the saved history file.

If you change the available MCP tools (add a new one, remove an
existing one), update the **Tool reference** section so the model
knows what's wired in.

## Troubleshooting

**`model_unavailable` everywhere from `bot-ml`** — you haven't
trained the models yet:

```bash
python -m src.enrich_knowledge.runners.run_training --model all
```

**Empty results from `bot-rag`** — ChromaDB is empty. Check
freshness with the skill (it'll call `ingestion_status` for you)
or directly:

```bash
python -m src.enrich_knowledge.runners.run_ingestion --dry-run
python -m src.enrich_knowledge.runners.run_ingestion
```

**Skill fails to write to `history/`** — the filesystem MCP isn't
rooted at the repo path. In Claude Code, run from the repo root.
In Desktop, edit the filesystem MCP config to root at
`/home/tnguyen/source/personal/bot/`.

**Skill calls `cancel_order`** — it shouldn't. The skill
self-restricts; if you see this happening, the SKILL.md was edited
and the "Tool discipline" section needs to put `cancel_order` back
on the do-not-call list.

**One symbol's regime read looks wrong** — the regime classifier
uses BTC-calibrated forward-return labels by default, which invert
on ETH at the 30 d horizon. The fix in
`src/mcp_servers/ml_mcp/services/cycle_classifier.py`
(`REGIME_INSTRUCTIONS_OVERRIDES`) already overrides ETHUSDT prompt
text — extend the dict for any other symbol that exhibits
mean-reversion structure on the chosen horizon.
