"""Trading loop orchestration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Optional

from src.core.config import Settings
from src.domain.trading import Action, TradeDecision
from src.interfaces.notifiers import ConsoleNotifier, LoggerNotifier
from src.domain.analysis import Signal


class TradingLoop:
    """Online trading loop skeleton for orchestration and shutdown handling."""

    def __init__(
        self,
        aggregator: Any,
        tech_analyzer: Any,
        pattern_analyzer: Any,
        chart_gen: Any,
        retriever: Any,
        builder: Any,
        llm: Any,
        risk: Any,
        executor: Any,
        memory: Any,
        persistence: Any,
        settings: Settings,
        *,
        console_notifier: ConsoleNotifier | None = None,
        logger_notifier: LoggerNotifier | None = None,
        logger: logging.Logger | None = None,
        sleep_func: Any = asyncio.sleep,
        trading_strategy: Optional[Any] = None,
        brain_service: Optional[Any] = None,
        memory_service: Optional[Any] = None,
        statistics_service: Optional[Any] = None,
        discord_notifier: Optional[Any] = None,
        # ML services (all optional)
        anomaly_detector: Optional[Any] = None,
        percentile_scorer: Optional[Any] = None,
        direction_classifier: Optional[Any] = None,
        key_level_detector: Optional[Any] = None,
        cycle_classifier: Optional[Any] = None,
        multi_tf_analyzer: Optional[Any] = None,
        ohlcv_writer: Optional[Any] = None,
        # Per-symbol extensions (fix #2 — multi-symbol coverage)
        ohlcv_writers: Optional[dict[str, Any]] = None,
        per_symbol_scorers: Optional[dict[str, Any]] = None,
        # Deterministic signal scorer (replaces LLM when enabled)
        signal_scorer: Optional[Any] = None,
    ) -> None:
        self.aggregator = aggregator
        self.tech_analyzer = tech_analyzer
        self.pattern_analyzer = pattern_analyzer
        self.chart_gen = chart_gen
        self.retriever = retriever
        self.builder = builder
        self.llm = llm
        self.risk = risk
        self.executor = executor
        self.memory = memory
        self.persistence = persistence
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)
        self.console_notifier = console_notifier or ConsoleNotifier()
        self.logger_notifier = logger_notifier or LoggerNotifier(self.logger)
        self._sleep = sleep_func
        self._stop_event = asyncio.Event()
        self._cycle = 0
        # In-memory position state, seeded from persistence so restarts are safe.
        self._open_positions: dict[str, dict[str, Any]] = {}
        for sym in settings.trading_symbols:
            saved = persistence.load_position(sym)
            if saved:
                self._open_positions[sym] = saved
        # Optional advanced services — None means legacy behaviour is preserved
        self.trading_strategy = trading_strategy
        self.brain_service = brain_service
        self.memory_service = memory_service
        self.statistics_service = statistics_service
        self.discord_notifier = discord_notifier
        # ML services
        self._anomaly_detector = anomaly_detector
        self._percentile_scorer = percentile_scorer
        self._direction_classifier = direction_classifier
        self._key_level_detector = key_level_detector
        self._cycle_classifier = cycle_classifier
        self._multi_tf_analyzer = multi_tf_analyzer
        self._ohlcv_writer = ohlcv_writer
        # Per-symbol writers and scorers (fix #2)
        self._ohlcv_writers: dict[str, Any] = ohlcv_writers or {}
        self._per_symbol_scorers: dict[str, Any] = per_symbol_scorers or {}
        self._regime_refresh_date: str = ""
        self._current_regime: str | None = None
        self.signal_scorer = signal_scorer
        # Per-symbol re-entry cooldown: maps symbol → cycle number when position was last closed
        self._last_close_cycle: dict[str, int] = {}
        # Kill-switch state
        self._consecutive_losses: int = 0
        self._daily_loss_pct: float = 0.0
        self._daily_reset_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Lock shared between main cycle and position monitor to prevent double-close races
        self._position_lock: asyncio.Lock = asyncio.Lock()

    def stop(self) -> None:
        self._stop_event.set()

    async def trigger_immediate_cycle(self) -> None:
        """Fix #12: Wake the loop and run a cycle immediately, bypassing the normal sleep."""
        self._webhook_trigger = True

    async def run(self) -> None:
        """Run the trading loop until stopped."""
        self.logger.info(
            "trading loop started — timeframe=%s symbols=%s interval=%ss dry_run=%s",
            self.settings.timeframe,
            ",".join(self.settings.trading_symbols),
            self.settings.effective_bot_interval(),
            self.settings.bot_dry_run,
        )
        self._webhook_trigger: bool = False
        _retry_delay = 30
        # Initialize executor: applies leverage and pre-warms exchange filters
        if hasattr(self.executor, "initialize"):
            await self.executor.initialize()
        try:
            while not self._stop_event.is_set():
                try:
                    await self.run_cycle_once()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    self.logger.exception(
                        "Cycle error — retrying in %ss", _retry_delay
                    )
                    await self._sleep(_retry_delay)
                    continue
                if self._stop_event.is_set():
                    break
                # Fix #12: check for a webhook-triggered immediate cycle before sleeping.
                if getattr(self, "_webhook_trigger", False):
                    self._webhook_trigger = False
                    self.logger.info("Webhook signal received — skipping sleep for immediate cycle")
                    continue
                await self._sleep(self.settings.effective_bot_interval())
        except asyncio.CancelledError:
            raise
        finally:
            self.logger.info("trading loop stopped")

    async def run_cycle_once(self) -> dict[str, Any]:
        """Execute one dry/live trading decision cycle."""
        self._cycle += 1
        timestamp_iso = datetime.now(timezone.utc).isoformat()
        self.logger.info(
            "=== Cycle %d start [tf=%s] symbols=%s ===",
            self._cycle,
            self.settings.timeframe,
            ",".join(self.settings.trading_symbols),
        )

        # Reset daily loss counter at UTC midnight
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_reset_date = today
            self._daily_loss_pct = 0.0
            self.logger.info("Daily loss counter reset for %s", today)

        # Kill-switch check — halt if we've hit loss limits
        if hasattr(self.risk, "check_kill_switches"):
            halt, halt_reason = self.risk.check_kill_switches(
                consecutive_losses=self._consecutive_losses,
                daily_loss_pct=self._daily_loss_pct,
            )
            if halt:
                self.logger.warning("KILL SWITCH ACTIVE: %s — skipping cycle %d", halt_reason, self._cycle)
                return {"timestamp_iso": timestamp_iso, "halted": True, "halt_reason": halt_reason}

        snapshots = await self._collect_snapshots()

        # B4: Anomaly detection — skip cycle if any symbol shows abnormal conditions
        if self._anomaly_detector is not None and snapshots:
            try:
                for snap in snapshots.values():
                    if self._anomaly_detector.is_anomaly(snap, None):
                        self.logger.warning("Anomaly detected for %s — skipping cycle %d", snap.symbol, self._cycle)
                        return {"timestamp_iso": timestamp_iso, "halted": True, "halt_reason": "anomaly"}
            except Exception:  # noqa: BLE001
                self.logger.debug("anomaly_detector failed", exc_info=True)

        # E: Append latest closed candle to OHLCV CSV for every traded symbol (fix #2)
        if snapshots:
            for sym, snap in snapshots.items():
                writer = self._ohlcv_writers.get(sym.upper()) or self._ohlcv_writer
                if writer is not None:
                    try:
                        writer.append(snap.candles)
                    except Exception:  # noqa: BLE001
                        self.logger.debug("ohlcv_writer.append failed for %s", sym, exc_info=True)

        # A4: Refresh macro regime daily and inject into system prompt.
        # Also reload KeyLevelDetector cache so S/R levels stay current (fix #4).
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._regime_refresh_date:
            if self._cycle_classifier is not None:
                try:
                    self._refresh_regime()
                except Exception:  # noqa: BLE001
                    self.logger.debug("regime refresh failed", exc_info=True)
            if self._key_level_detector is not None:
                try:
                    self._key_level_detector.reload()
                    self.logger.debug("KeyLevelDetector cache reloaded")
                except Exception:  # noqa: BLE001
                    self.logger.debug("key_level_detector.reload failed", exc_info=True)
            self._regime_refresh_date = today

        # Detect positions closed by exchange TP/SL since last cycle.
        await self._reconcile_open_positions()

        # Position lifecycle check (only when trading_strategy is wired)
        first_price = next(iter(snapshots.values())).price if snapshots else 0.0
        if self.trading_strategy is not None:
            try:
                # Use the price for the strategy's symbol, not just the first snapshot
                _strat_sym = getattr(self.trading_strategy, "symbol", None)
                _strat_price = (
                    snapshots[_strat_sym].price
                    if _strat_sym and _strat_sym in snapshots
                    else first_price
                )
                async with self._position_lock:
                    # Capture position PnL before check so we can track loss on close
                    _pre_pos = getattr(self.trading_strategy, "current_position", None)
                    _pre_pnl_pct = _pre_pos.calculate_pnl(_strat_price) if _pre_pos else None
                    _pre_quote = _pre_pos.quote_amount if _pre_pos else 0.0

                    close_reason = await self.trading_strategy.check_position(_strat_price)

                    # Update kill-switch counters when a position closes
                    if close_reason in ("stop_loss", "trailing_stop", "take_profit", "partial_tp1"):
                        self._update_loss_tracking(close_reason, _pre_pnl_pct, _pre_quote)

                    # Fix #6: keep _open_positions dict in sync with TradingStrategy state
                    self._sync_position_from_strategy()
            except Exception:  # noqa: BLE001
                self.logger.exception("trading_strategy.check_position failed")

        choppiness_threshold = getattr(self.settings, "choppiness_threshold", 61.8)
        rsi_strong_buy, rsi_buy, rsi_sell, rsi_strong_sell = (
            self.settings.effective_rsi_thresholds()
        )
        analyses = {
            symbol: self.tech_analyzer.analyze(
                symbol, snapshot, choppiness_threshold,
                rsi_strong_buy, rsi_buy, rsi_sell, rsi_strong_sell,
            )
            for symbol, snapshot in snapshots.items()
        }

        # Multi-timeframe: fetch 4h candles and compute HTF signals.
        # Fix #9: only pay the Binance API cost when HTF confirmation is actually enabled.
        htf_enabled = getattr(self.settings, "htf_confirmation_enabled", False)
        htf_analyses = await self._collect_htf_analyses(
            snapshots, choppiness_threshold,
            rsi_strong_buy, rsi_buy, rsi_sell, rsi_strong_sell,
        ) if htf_enabled else None

        patterns = await self._collect_patterns(snapshots, analyses)
        if self.signal_scorer is not None:
            rag_per_symbol: dict[str, str] = {}
        else:
            single_call = getattr(self.settings, "single_symbol_decision", True) and len(snapshots) > 1
            rag_per_symbol = self._build_rag_context_per_symbol(snapshots, analyses, single_call_mode=single_call)

        # Gather optional extra context sections
        position_context: Optional[str] = None
        memory_context: Optional[str] = None
        brain_context: Optional[str] = None
        dynamic_thresholds: Optional[dict[str, Any]] = None

        if self._open_positions:
            position_context = self._format_position_context()
        elif self.trading_strategy is not None:
            try:
                position_context = self.trading_strategy.get_position_context()
            except Exception:  # noqa: BLE001
                self.logger.debug("trading_strategy.get_position_context failed", exc_info=True)

        if self.memory_service is not None:
            try:
                memory_context = self.memory_service.get_memory_context()
            except Exception:  # noqa: BLE001
                self.logger.debug("memory_service.get_memory_context failed", exc_info=True)

        if self.brain_service is not None:
            try:
                first_symbol, first_snap = next(iter(snapshots.items()))
                market_conditions = self.tech_analyzer.get_market_conditions(first_symbol, first_snap)
                # Filter to only the kwargs accepted by get_context()
                _brain_keys = {
                    "trend_direction", "adx", "volatility_level", "rsi_level",
                    "macd_signal", "volume_state", "bb_position",
                    "is_weekend", "market_sentiment", "order_book_bias",
                }
                brain_context = self.brain_service.get_context(
                    **{k: v for k, v in market_conditions.items() if k in _brain_keys}
                )
                dynamic_thresholds = self.brain_service.get_dynamic_thresholds()
            except Exception:  # noqa: BLE001
                self.logger.debug("brain_service context failed", exc_info=True)

        # Append HTF signal summary to dynamic_thresholds context
        if htf_analyses:
            if dynamic_thresholds is None:
                dynamic_thresholds = {}
            for sym, htf_analysis in htf_analyses.items():
                htf_tf = getattr(self.settings, "htf_timeframe", "4h")
                dynamic_thresholds[f"{sym}_{htf_tf}_signal"] = htf_analysis.signal.value
                dynamic_thresholds[f"{sym}_{htf_tf}_reasoning"] = htf_analysis.reasoning

        ml_context = await self._build_ml_context(snapshots, analyses)

        # Per-symbol decisions run in parallel — each symbol gets its own focused prompt.
        decisions = await self._decide_per_symbol(
            snapshots, analyses, patterns, rag_per_symbol, htf_analyses,
            position_context=position_context,
            memory_context=memory_context,
            brain_context=brain_context,
            dynamic_thresholds=dynamic_thresholds,
            ml_context=ml_context,
        )

        balance = await self._build_balance_context(snapshots)

        # Execute decisions sequentially (shared USDT balance).
        outcomes: dict[str, tuple[Any, Any]] = {}
        last_outcome = None
        for symbol, decision in decisions.items():
            # Fill in quantity for close actions from the tracked position
            if decision.action in (Action.CLOSE_LONG, Action.CLOSE_SHORT, Action.CLOSE) and decision.quantity <= 0:
                pos = self._open_positions.get(symbol)
                if pos and pos.get("size", 0) > 0:
                    decision = TradeDecision(
                        symbol=decision.symbol,
                        action=decision.action,
                        quantity=pos["size"],
                        order_type=decision.order_type,
                        price=decision.price,
                        reasoning=decision.reasoning,
                        confidence=decision.confidence,
                        timestamp=decision.timestamp,
                        source=decision.source,
                    )
                    self.logger.info(
                        "Filled close quantity for %s from tracked position: %.8f",
                        symbol, pos["size"],
                    )
            risk_result = self.risk.validate(decision, balance, self._open_positions)
            outcome = await self.executor.execute(risk_result.decision, risk_result.dry_run)

            # Compute PnL for close actions while the position entry data is still available
            if (
                not outcome.dry_run
                and outcome.order_id
                and outcome.decision.action in (Action.CLOSE_LONG, Action.CLOSE_SHORT, Action.CLOSE)
            ):
                outcome = self._attach_pnl(outcome, snapshots)

            self.memory.record(outcome)
            self._check_slippage(outcome, snapshots)
            if not outcome.dry_run and outcome.order_id:
                await self._update_open_positions(outcome)
                # Deduct the executed notional from the local balance so the
                # next symbol's risk validation sees the reduced USDT instead
                # of the stale pre-cycle snapshot.
                _exec_price = outcome.executed_price or balance.get("prices", {}).get(symbol, 0.0)
                if _exec_price > 0 and isinstance(balance.get("USDT"), dict):
                    _notional = outcome.decision.quantity * _exec_price
                    balance["USDT"]["free"] = max(0.0, balance["USDT"]["free"] - _notional)
            outcomes[symbol] = (outcome, risk_result)
            last_outcome = outcome

            self.persistence.append_trade(outcome, timestamp_iso)
            self.persistence.append_cycle_log(
                timestamp_iso=timestamp_iso,
                cycle=self._cycle,
                symbol=symbol,
                analysis=analyses.get(symbol) or next(iter(analyses.values())),
                patterns=patterns.get(symbol) or next(iter(patterns.values())),
                rag_docs_retrieved=rag_per_symbol.get(symbol, "").count("["),
                llm_decision=outcome.decision.action.value,
                llm_usage=getattr(self.llm, "last_usage", None),
                llm_prompt=getattr(self.llm, "last_prompt", None),
                llm_raw_response=getattr(self.llm, "last_raw_response", None),
                decision_source=outcome.decision.source,
                decision_reasoning=outcome.decision.reasoning,
                llm_error=getattr(self.llm, "last_error", None),
                risk_outcome=risk_result.status,
                order_id=outcome.order_id,
                dry_run=outcome.dry_run,
            )

        symbol_signals = [(s.replace("USDT", ""), analyses[s].signal.value) for s in analyses]
        # Use the most significant non-HOLD decision for the console summary, else last.
        summary_outcome = next(
            (o for o, _ in outcomes.values() if o.decision.action.value != "HOLD"),
            last_outcome,
        )
        self.console_notifier.notify_cycle(
            cycle=self._cycle,
            timestamp_iso=timestamp_iso,
            symbol_signals=symbol_signals,
            final_decision=summary_outcome.decision.action.value if summary_outcome else "HOLD",
        )
        decisions_log = " | ".join(
            f"{sym}={o.decision.action.value}({r.status})"
            for sym, (o, r) in outcomes.items()
        )
        dry_run_flag = last_outcome.dry_run if last_outcome else self.settings.bot_dry_run
        self.logger_notifier.notify(
            f"cycle={self._cycle} {decisions_log} dry_run={dry_run_flag}"
        )

        # Optional Discord notification — send for any non-HOLD decision.
        if self.discord_notifier is not None:
            for symbol, (outcome, _) in outcomes.items():
                if outcome.decision.action.value == "HOLD":
                    continue
                try:
                    await self.discord_notifier.send_trading_decision(
                        symbol=symbol,
                        decision=outcome.decision.action.value,
                        price=snapshots[symbol].price if symbol in snapshots else first_price,
                        reasoning=outcome.decision.reasoning,
                        dry_run=outcome.dry_run,
                    )
                except Exception:  # noqa: BLE001
                    self.logger.debug("discord_notifier.send_trading_decision failed", exc_info=True)

        return {
            "timestamp_iso": timestamp_iso,
            "snapshots": snapshots,
            "analyses": analyses,
            "patterns": patterns,
            "rag_per_symbol": rag_per_symbol,
            "decisions": decisions,
            "outcomes": outcomes,
        }

    async def _collect_snapshots(self) -> dict[str, Any]:
        symbols = getattr(self.settings, "trading_symbols", None) or ["BTCUSDT", "ETHUSDT"]
        timeframe = getattr(self.settings, "timeframe", "1h")
        limit = getattr(self.settings, "candle_limit", 200)
        snapshots = await asyncio.gather(*[
            self.aggregator.snapshot(s, timeframe=timeframe, limit=limit) for s in symbols
        ])
        return dict(zip(symbols, snapshots))

    async def _collect_patterns(self, snapshots: dict[str, Any], analyses: dict[str, Any]) -> dict[str, Any]:
        results = {}
        chart_tf = getattr(self.settings, "ai_chart_timeframe", "5m")
        chart_limit = getattr(self.settings, "ai_chart_candle_limit", 120)
        primary_tf = getattr(self.settings, "timeframe", "1h")
        feed = getattr(self.aggregator, "feed", None)

        for symbol, snapshot in snapshots.items():
            pattern_result = self.pattern_analyzer.analyze(
                symbol, snapshot.candles,
                timeframe=getattr(self.settings, "timeframe", "1h"),
            )
            if self.settings.model_supports_vision:
                chart_candles = snapshot.candles
                if feed is not None and chart_tf != primary_tf:
                    try:
                        chart_candles = await feed.get_ohlcv(symbol, timeframe=chart_tf, limit=chart_limit)
                    except Exception:  # noqa: BLE001
                        self.logger.debug("chart candle fetch failed for %s/%s — using primary candles", symbol, chart_tf)
                pattern_result.chart_png_b64 = self.chart_gen.render(symbol, chart_candles, analyses[symbol].indicators)
            results[symbol] = pattern_result
        return results

    def _build_rag_context_per_symbol(
        self,
        snapshots: dict[str, Any],
        analyses: dict[str, Any],
        *,
        single_call_mode: bool = False,
    ) -> dict[str, str]:
        """Build per-symbol RAG context strings.

        When ``single_call_mode`` is True the macro section (global market data)
        is omitted from each symbol's block so it can be attached once to the
        combined prompt by ``_decide_all_symbols_single_call``.  This prevents
        sending the same macro paragraphs N times — one for each symbol.
        """
        result: dict[str, str] = {}
        for symbol in snapshots:
            section = self.retriever.retrieve(
                snapshots[symbol],
                analyses[symbol],
                include_macro=not single_call_mode,
            )
            result[symbol] = section if section and section != "=== NO CONTEXT AVAILABLE ===" else "=== NO CONTEXT AVAILABLE ==="
        return result

    async def _decide_per_symbol(
        self,
        snapshots: dict[str, Any],
        analyses: dict[str, Any],
        patterns: dict[str, Any],
        rag_per_symbol: dict[str, str],
        htf_analyses: dict[str, Any] | None,
        *,
        position_context: Optional[str] = None,
        memory_context: Optional[str] = None,
        brain_context: Optional[str] = None,
        dynamic_thresholds: Optional[dict[str, Any]] = None,
        ml_context: Optional[str] = None,
    ) -> dict[str, TradeDecision]:
        """Run LLM decisions for all symbols.

        Fix #8: when ``single_symbol_decision=True`` (SINGLE_SYMBOL_DECISION=true),
        all symbols are evaluated in a *single* LLM call — the model sees cross-asset
        context and picks the best opportunity, halving API cost for 2-symbol setups.
        When False (default), one call per symbol runs in parallel (legacy behaviour).
        """
        # Signal scorer path — deterministic, no LLM call
        if self.signal_scorer is not None:
            decisions: dict[str, TradeDecision] = {}
            for symbol in snapshots:
                # Determine current position direction for this symbol
                pos_dir: str | None = None
                if self.trading_strategy and getattr(self.trading_strategy, "symbol", None) == symbol:
                    strat_pos = getattr(self.trading_strategy, "current_position", None)
                    if strat_pos:
                        pos_dir = strat_pos.direction
                # Fallback: always check _open_positions if strategy didn't provide direction
                if pos_dir is None and symbol in self._open_positions:
                    raw_dir = self._open_positions[symbol].get("direction", "")
                    pos_dir = "LONG" if raw_dir == "BUY" else ("SHORT" if raw_dir == "SELL" else None)

                htf_sig = None
                if htf_analyses:
                    htf_a = htf_analyses.get(symbol)
                    if htf_a is not None:
                        htf_sig = htf_a.signal

                decision = self.signal_scorer.decide(
                    symbol,
                    snapshots[symbol],
                    analyses[symbol],
                    regime=self._current_regime,
                    htf_signal=htf_sig,
                    position_direction=pos_dir,
                )

                # Re-entry cooldown: suppress new entries if the symbol
                # was closed too recently (prevents rapid-fire flipping).
                cooldown = getattr(self.settings, "reentry_cooldown_cycles", 3)
                if (
                    cooldown > 0
                    and decision.action in (Action.BUY, Action.SELL)
                    and pos_dir is None  # only applies to new entries
                    and symbol in self._last_close_cycle
                    and (self._cycle - self._last_close_cycle[symbol]) < cooldown
                ):
                    remaining = cooldown - (self._cycle - self._last_close_cycle[symbol])
                    self.logger.info(
                        "[cooldown] %s suppressed %s — %d cycle(s) remaining before re-entry",
                        symbol, decision.action.value, remaining,
                    )
                    decision = TradeDecision(
                        symbol=symbol, action=Action.HOLD,
                        confidence=0.0,
                        reasoning=f"cooldown ({remaining} cycles left) | {decision.reasoning}",
                        source="signal_scorer",
                    )

                decisions[symbol] = decision
            return decisions

        # LLM paths below
        if getattr(self.settings, "single_symbol_decision", False) and len(snapshots) > 1:
            return await self._decide_all_symbols_single_call(
                snapshots, analyses, patterns, rag_per_symbol, htf_analyses,
                position_context=position_context,
                memory_context=memory_context,
                brain_context=brain_context,
                dynamic_thresholds=dynamic_thresholds,
                ml_context=ml_context,
            )

        async def _decide_one(symbol: str) -> TradeDecision:
            snap = {symbol: snapshots[symbol]}
            ana = {symbol: analyses[symbol]}
            pat = {symbol: patterns[symbol]}
            rag = rag_per_symbol.get(symbol, "=== NO CONTEXT AVAILABLE ===")
            # Per-symbol HTF thresholds (filter to this symbol only)
            sym_htf = None
            if htf_analyses:
                htf_val = htf_analyses.get(symbol)
                sym_htf = {symbol: htf_val} if htf_val is not None else {}
            system_prompt, user_message = self.builder.build(
                snap, ana, pat, rag,
                position_context=position_context,
                memory_context=memory_context,
                brain_context=brain_context,
                dynamic_thresholds=dynamic_thresholds,
                ml_context=ml_context,
            )
            decision = await self._decide(system_prompt, user_message, sym_htf)
            # Enforce correct symbol — LLM may omit it and the parser defaults to the primary symbol.
            if decision.symbol != symbol:
                decision = TradeDecision(
                    symbol=symbol,
                    action=decision.action,
                    quantity=decision.quantity,
                    order_type=decision.order_type,
                    price=decision.price,
                    reasoning=decision.reasoning,
                    confidence=decision.confidence,
                    timestamp=decision.timestamp,
                    source=decision.source,
                )
            return decision

        results = await asyncio.gather(*[_decide_one(sym) for sym in snapshots])
        return dict(zip(snapshots.keys(), results))

    async def _decide_all_symbols_single_call(
        self,
        snapshots: dict[str, Any],
        analyses: dict[str, Any],
        patterns: dict[str, Any],
        rag_per_symbol: dict[str, str],
        htf_analyses: dict[str, Any] | None,
        *,
        position_context: Optional[str] = None,
        memory_context: Optional[str] = None,
        brain_context: Optional[str] = None,
        dynamic_thresholds: Optional[dict[str, Any]] = None,
        ml_context: Optional[str] = None,
    ) -> dict[str, TradeDecision]:
        """Fix #8: single LLM call for all symbols — pick the best opportunity.

        Per-symbol RAG blocks contain news + trade memory for each symbol.
        Macro context (global market data) is fetched once and appended at the
        end so it is not repeated N times in the combined prompt.
        The returned dict maps every symbol to either the chosen action or HOLD.
        """
        # Per-symbol sections (news + memory, no macro — see _build_rag_context_per_symbol)
        combined_rag = "\n\n".join(
            f"--- {sym} ---\n{rag}"
            for sym, rag in rag_per_symbol.items()
            if rag and rag != "=== NO CONTEXT AVAILABLE ==="
        ) or "=== NO CONTEXT AVAILABLE ==="

        # Append global macro context once instead of repeating it per symbol
        shared_macro = self.retriever.retrieve_macro("global macro crypto market conditions")
        if shared_macro:
            combined_rag = f"{combined_rag}\n\n{shared_macro}"

        system_prompt, user_message = self.builder.build(
            snapshots, analyses, patterns, combined_rag,
            position_context=position_context,
            memory_context=memory_context,
            brain_context=brain_context,
            dynamic_thresholds=dynamic_thresholds,
            ml_context=ml_context,
        )

        decision = await self._decide(system_prompt, user_message, htf_analyses)

        # The LLM chose one symbol — all others default to HOLD
        decisions: dict[str, TradeDecision] = {}
        for sym in snapshots:
            if sym == decision.symbol:
                decisions[sym] = decision
            else:
                decisions[sym] = TradeDecision(
                    symbol=sym,
                    action=Action.HOLD,
                    reasoning=f"single_call:not_chosen (chose {decision.symbol})",
                    source=decision.source,
                )
        return decisions


    async def _decide(self, system_prompt: str, user_message: str, htf_analyses: dict[str, Any] | None = None) -> TradeDecision:
        decision = await self.llm.decide(system_prompt, user_message)
        if decision.action == Action.HOLD:
            return decision
        valid_symbols = set(getattr(self.settings, "trading_symbols", ["BTCUSDT", "ETHUSDT"]))
        if decision.symbol not in valid_symbols:
            fallback = next(iter(valid_symbols))
            return TradeDecision(symbol=fallback, action=Action.HOLD, reasoning="invalid_symbol", source="fallback_hold")

        # HTF confirmation gate: block entries when higher-timeframe trend disagrees
        htf_enabled = getattr(self.settings, "htf_confirmation_enabled", False)
        if htf_enabled and htf_analyses and decision.action in (Action.BUY, Action.SELL):
            htf = htf_analyses.get(decision.symbol)
            if htf is not None:
                htf_signal = htf.signal
                if decision.action == Action.BUY and htf_signal in (Signal.SELL, Signal.STRONG_SELL):
                    self.logger.warning(
                        "HTF confirmation failed: 1h=BUY but %s=%s — overriding to HOLD",
                        getattr(self.settings, "htf_timeframe", "4h"), htf_signal.value,
                    )
                    return TradeDecision(
                        symbol=decision.symbol, action=Action.HOLD,
                        reasoning=f"htf_disagreement:{htf_signal.value}", source="htf_filter",
                    )
                if decision.action == Action.SELL and htf_signal in (Signal.BUY, Signal.STRONG_BUY):
                    self.logger.warning(
                        "HTF confirmation failed: 1h=SELL but %s=%s — overriding to HOLD",
                        getattr(self.settings, "htf_timeframe", "4h"), htf_signal.value,
                    )
                    return TradeDecision(
                        symbol=decision.symbol, action=Action.HOLD,
                        reasoning=f"htf_disagreement:{htf_signal.value}", source="htf_filter",
                    )

        return decision

    async def _build_balance_context(self, snapshots: dict[str, Any]) -> dict[str, Any]:
        if self.settings.bot_dry_run:
            return {"prices": {symbol: snapshot.price for symbol, snapshot in snapshots.items()}}

        balance_payload = await self.aggregator.feed.get_balance()
        balance_data = balance_payload.get("data", {}) if isinstance(balance_payload, dict) else {}
        asset_balances = balance_data.get("balances", [])
        balance_map = {
            asset["asset"]: {"free": float(asset.get("free", 0.0)), "locked": float(asset.get("locked", 0.0))}
            for asset in asset_balances
            if asset.get("asset")
        }
        balance_map["prices"] = {symbol: snapshot.price for symbol, snapshot in snapshots.items()}
        return balance_map

    async def _get_snapshot_price(self, symbol: str) -> float:
        """Fetch the latest price for a symbol from the aggregator."""
        try:
            snap = await self.aggregator.snapshot(symbol, timeframe="1m", limit=1)
            return snap.price if snap else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    async def _update_open_positions(self, outcome: Any) -> None:
        """Sync _open_positions and persistence after a live execution.

        For futures entries we immediately place exchange-native STOP_MARKET and
        TAKE_PROFIT_MARKET bracket orders so positions are protected even if the
        bot crashes.  For a bot-managed close we cancel the surviving bracket
        first, then clear local state.
        """
        sym = outcome.decision.symbol
        action = outcome.decision.action
        existing = self._open_positions.get(sym)
        if action == Action.HOLD:
            return

        if existing:
            existing_dir = existing.get("direction", "")
            is_close = (
                (action == Action.BUY and existing_dir == "SELL")
                or (action == Action.SELL and existing_dir == "BUY")
            )
            if is_close:
                await self.executor.cancel_bracket_orders(
                    sym,
                    existing.get("sl_order_id"),
                    existing.get("tp_order_id"),
                )
                del self._open_positions[sym]
                self._last_close_cycle[sym] = self._cycle
                await self.persistence.async_save_position(sym, None)
                return

        # New entry — place exchange bracket orders for both futures AND spot.
        entry_price = outcome.executed_price or 0.0
        # Fallback: use the latest snapshot price so brackets are never silently skipped
        if entry_price <= 0:
            snap = await self._get_snapshot_price(sym)
            if snap > 0:
                self.logger.warning(
                    "executed_price missing for %s — using snapshot price %.6f for bracket orders",
                    sym, snap,
                )
                entry_price = snap
        sl_order_id: str | None = None
        tp_order_id: str | None = None
        sl_price: float | None = None
        tp_price: float | None = None

        if entry_price > 0 and action in (Action.BUY, Action.SELL):
            is_long = action == Action.BUY

            # Prefer SL/TP from TradingStrategy position (set by RiskManager with ATR math)
            # Fall back to simple percentage defaults
            strategy_pos = (
                getattr(self.trading_strategy, "current_position", None)
                if self.trading_strategy else None
            )
            if strategy_pos and strategy_pos.symbol == sym:
                sl_price = strategy_pos.stop_loss
                tp_price = strategy_pos.take_profit
            else:
                sl_pct = getattr(self.settings, "default_stop_loss_pct", 0.02)
                tp_pct = getattr(self.settings, "default_take_profit_pct", 0.04)
                sl_price = round(entry_price * ((1 - sl_pct) if is_long else (1 + sl_pct)), 2)
                tp_price = round(entry_price * ((1 + tp_pct) if is_long else (1 - tp_pct)), 2)

            sl_order_id, tp_order_id = await self.executor.place_bracket_orders(
                sym, action.value, sl_price, tp_price,
                quantity=outcome.decision.quantity,  # needed for spot limit orders
            )

        pos: dict[str, Any] = {
            "direction": action.value,
            "symbol": sym,
            "size": outcome.decision.quantity,
            "entry_price": entry_price,
            "entry_time": outcome.timestamp,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "sl_order_id": sl_order_id,
            "tp_order_id": tp_order_id,
        }
        self._open_positions[sym] = pos
        await self.persistence.async_save_position(sym, pos)

    async def _reconcile_open_positions(self) -> None:
        """Detect positions closed by exchange TP/SL and clean up local state.

        Only runs for futures; silently skips if the Binance call fails so a
        transient error never clears a still-open position.
        """
        if not self._open_positions:
            return
        if getattr(self.settings, "binance_product", "") != "usdt_futures":
            return

        for sym in list(self._open_positions):
            pos = self._open_positions[sym]
            live_size = await self.executor.get_live_position_size(sym)
            expected_size = pos.get("size") or 0.0
            if live_size < expected_size * 0.05:  # 95%+ gone → exchange closed it
                self.logger.info(
                    "Position %s was closed by exchange (TP/SL or liquidation) — cleaning up",
                    sym,
                )
                await self.executor.cancel_bracket_orders(
                    sym, pos.get("sl_order_id"), pos.get("tp_order_id")
                )
                del self._open_positions[sym]
                self._last_close_cycle[sym] = self._cycle
                await self.persistence.async_save_position(sym, None)

    async def _collect_htf_analyses(
        self,
        snapshots: dict[str, Any],
        choppiness_threshold: float = 61.8,
        rsi_strong_buy: float = 30.0,
        rsi_buy: float = 40.0,
        rsi_sell: float = 60.0,
        rsi_strong_sell: float = 70.0,
    ) -> dict[str, Any]:
        """Fetch higher-timeframe candles and compute HTF signals for each symbol.

        Returns an empty dict if the feed doesn't support it or all requests fail.
        """
        htf_tf = getattr(self.settings, "htf_timeframe", "4h")
        htf_limit = getattr(self.settings, "candle_limit", 200)
        results: dict[str, Any] = {}
        feed = getattr(self.aggregator, "feed", None)
        if feed is None:
            return results

        for symbol, snapshot in snapshots.items():
            try:
                htf_candles = await feed.get_ohlcv(symbol, timeframe=htf_tf, limit=htf_limit)
                if len(htf_candles) < 50:
                    continue
                from src.domain.market import MarketSnapshot as _MS  # noqa: PLC0415
                htf_snap = _MS(
                    symbol=symbol,
                    price=snapshot.price,
                    change_24h_pct=snapshot.change_24h_pct,
                    volume_24h=snapshot.volume_24h,
                    bid=snapshot.bid,
                    ask=snapshot.ask,
                    candles=htf_candles,
                    funding_rate=snapshot.funding_rate,
                    open_interest=snapshot.open_interest,
                )
                results[symbol] = self.tech_analyzer.analyze(
                    symbol, htf_snap, choppiness_threshold,
                    rsi_strong_buy, rsi_buy, rsi_sell, rsi_strong_sell,
                )
            except Exception:  # noqa: BLE001
                self.logger.debug("HTF analysis failed for %s", symbol, exc_info=True)
        return results

    async def _build_ml_context(self, snapshots: dict[str, Any], analyses: dict[str, Any]) -> str | None:
        """Assemble ML-derived context sections for the LLM prompt."""
        parts: list[str] = []
        first_symbol = next(iter(snapshots)) if snapshots else None

        # A1: Historical percentiles — use per-symbol scorer when available (fix #2)
        for sym, snap in snapshots.items():
            scorer = self._per_symbol_scorers.get(sym.upper()) or self._percentile_scorer
            if scorer is None:
                continue
            try:
                analysis = analyses.get(sym)
                if analysis is not None:
                    pct_text = scorer.score(analysis.indicators, snap.price)
                    if pct_text:
                        header = f"### {sym}" if len(snapshots) > 1 else ""
                        parts.append(f"{header}\n{pct_text}".strip())
            except Exception:  # noqa: BLE001
                self.logger.debug("percentile_scorer failed for %s", sym, exc_info=True)

        # A2: Multi-timeframe alignment
        if self._multi_tf_analyzer is not None and first_symbol:
            try:
                current_signal = analyses[first_symbol].signal.value if first_symbol in analyses else "NEUTRAL"
                mtf_text = await self._multi_tf_analyzer.build_summary(first_symbol, current_signal)
                if mtf_text:
                    parts.append(mtf_text)
            except Exception:  # noqa: BLE001
                self.logger.debug("multi_tf_analyzer failed", exc_info=True)

        # A3: Key S/R levels — per symbol
        if self._key_level_detector is not None:
            for sym, snap in snapshots.items():
                try:
                    levels_text = self._key_level_detector.format_context(snap.price, symbol=sym)
                    if levels_text:
                        parts.append(f"### {sym}\n{levels_text}")
                except Exception:  # noqa: BLE001
                    self.logger.debug("key_level_detector failed for %s", sym, exc_info=True)

        # B2: XGBoost direction signal — per symbol
        if self._direction_classifier is not None:
            for sym, snap in snapshots.items():
                try:
                    analysis = analyses.get(sym)
                    if analysis is not None:
                        dir_text = self._direction_classifier.format_context(
                            analysis.indicators, snap.price, symbol=sym
                        )
                        if dir_text:
                            parts.append(f"### {sym}\n{dir_text}")
                except Exception:  # noqa: BLE001
                    self.logger.debug("direction_classifier failed for %s", sym, exc_info=True)

        return "\n\n".join(parts) if parts else None

    def _refresh_regime(self) -> None:
        """Recompute the regime label from daily CSV and update the system prompt suffix."""
        if self._cycle_classifier is None:
            return
        try:
            import csv as _csv  # noqa: PLC0415
            from pathlib import Path as _Path  # noqa: PLC0415
            daily_path = _Path(self.settings.ohlcv_csv_path(self.settings.crypto_pair, "1d"))
            if not daily_path.exists():
                return
            rows: list[dict] = []
            with open(daily_path, newline="") as f:
                for row in _csv.DictReader(f):
                    rows.append(row)
            rows = rows[-200:]  # last ~200 daily candles
            if len(rows) < 100:
                return

            import math as _math  # noqa: PLC0415
            closes = [float(r["close"]) for r in rows]
            highs  = [float(r["high"])  for r in rows]
            lows   = [float(r["low"])   for r in rows]

            def ema(series: list[float], p: int) -> float:
                k = 2.0 / (p + 1)
                v = series[0]
                for x in series[1:]:
                    v = x * k + v * (1 - k)
                return v

            c = closes[-1]
            e50  = ema(closes[-50:],  50)
            e100 = ema(closes[-100:], 100)
            e200 = ema(closes,        200)
            h52w = max(highs[-365:]) if len(highs) >= 365 else max(highs)
            l52w = min(lows[-365:])  if len(lows)  >= 365 else min(lows)

            features = {
                "ema50_dist":   (c - e50)  / e50,
                "ema100_dist":  (c - e100) / e100,
                "ema200_dist":  (c - e200) / e200,
                "ema50_slope":  (ema(closes[-10:], 10) - ema(closes[-15:-5], 10)) / e50,
                "high_52w_dist": (c - h52w) / h52w,
                "low_52w_dist":  (c - l52w) / l52w,
                "adx_14":       0.0,  # simplified — ADX needs more complex calc
                "realized_vol": (sum((closes[i] / closes[i-1] - 1) ** 2 for i in range(-30, 0)) / 30) ** 0.5 * (365 ** 0.5),
                "hh_count":     sum(1 for i in range(-89, 0) if i + 1 < 0 and highs[i] > highs[i-1]),
                "ll_count":     sum(1 for i in range(-89, 0) if i + 1 < 0 and lows[i] < lows[i-1]),
            }

            result = self._cycle_classifier.predict(features)
            if result is not None:
                regime, confidence = result
                self._current_regime = regime
                suffix = self._cycle_classifier.regime_system_prompt_suffix(regime, confidence)
                self.builder.set_regime_suffix(suffix)
                self.logger.info("Macro regime updated: %s (%.0f%%)", regime, confidence * 100)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("_refresh_regime failed: %s", exc)

    async def run_position_monitor(self) -> None:
        """Fast SL monitor that runs independently of the main trading cycle.

        Checks open positions against current price every
        ``position_monitor_interval`` seconds (default 15 s).  When a stop loss
        is hit it immediately:

        1. Cancels surviving exchange bracket orders.
        2. Executes a market close order on Binance (live mode only).
        3. Updates TradingStrategy internal state so the main cycle stays in sync.

        This task runs concurrently with the main loop.  A shared asyncio.Lock
        prevents a race between the monitor and the main cycle both trying to
        close the same position at the same time.
        """
        enabled = getattr(self.settings, "position_monitor_enabled", True)
        if not enabled:
            self.logger.info("Position monitor disabled via settings")
            return

        interval = getattr(self.settings, "position_monitor_interval", 15)
        self.logger.info("Position monitor started (interval=%ds)", interval)
        try:
            while not self._stop_event.is_set():
                await self._sleep(interval)
                if self._stop_event.is_set():
                    break
                try:
                    await self._monitor_check_positions()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    self.logger.debug("Position monitor tick failed", exc_info=True)
        except asyncio.CancelledError:
            raise
        finally:
            self.logger.info("Position monitor stopped")

    async def _monitor_check_positions(self) -> None:
        """Single fast-SL tick: fetch prices and close any position that hit its SL."""
        # Collect symbols that have an open position in either tracking system
        symbols: set[str] = set(self._open_positions.keys())
        if self.trading_strategy:
            pos = getattr(self.trading_strategy, "current_position", None)
            if pos:
                symbols.add(pos.symbol)

        if not symbols:
            return

        feed = getattr(self.aggregator, "feed", None)
        if feed is None:
            return

        for symbol in symbols:
            try:
                ticker = await feed.get_ticker(symbol)
                price: float = float(ticker.get("data", {}).get("last_price", 0.0))
                if price <= 0:
                    continue

                # --- Strategy-path check (has trailing stop, ATR-based SL) ---
                if self.trading_strategy:
                    async with self._position_lock:
                        strat_pos = getattr(self.trading_strategy, "current_position", None)
                        if strat_pos and strat_pos.symbol == symbol:
                            active_sl = strat_pos.trailing_stop_price or strat_pos.stop_loss
                            is_long = strat_pos.direction == "LONG"
                            sl_hit = (price <= active_sl) if is_long else (price >= active_sl)
                            if sl_hit:
                                reason = "trailing_stop" if strat_pos.trailing_stop_price else "stop_loss"
                                self.logger.warning(
                                    "MONITOR [strategy]: SL hit for %s %s @ $%.2f (SL=$%.2f)",
                                    symbol, strat_pos.direction, price, active_sl,
                                )
                                conditions = getattr(
                                    self.trading_strategy, "_build_conditions_from_position",
                                    lambda p: {},
                                )(strat_pos)
                                pre_pnl = strat_pos.calculate_pnl(price)
                                pre_quote = strat_pos.quote_amount
                                await self.trading_strategy.close_position(reason, price, conditions)
                                await self._monitor_execute_close(symbol, price, reason)
                                self._update_loss_tracking(reason, pre_pnl, pre_quote)
                            continue  # strategy path handled; skip simple-dict check below

                # --- Simple executor-path check (_open_positions dict only) ---
                async with self._position_lock:
                    pos_dict = self._open_positions.get(symbol)
                    if pos_dict is None:
                        continue
                    sl_price_dict: Optional[float] = pos_dict.get("sl_price")
                    if sl_price_dict is None:
                        continue
                    direction = pos_dict.get("direction", "BUY")
                    is_long_dict = direction == "BUY"
                    if (price <= sl_price_dict) if is_long_dict else (price >= sl_price_dict):
                        self.logger.warning(
                            "MONITOR [executor]: SL hit for %s %s @ $%.2f (SL=$%.2f)",
                            symbol, direction, price, sl_price_dict,
                        )
                        await self._monitor_execute_close(symbol, price, "stop_loss")

            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                self.logger.debug("Monitor check failed for %s", symbol, exc_info=True)

    async def _monitor_execute_close(self, symbol: str, price: float, reason: str) -> None:
        """Cancel brackets then execute a market close order for a monitor-triggered SL."""
        pos_dict = self._open_positions.get(symbol)

        # 1. Cancel any surviving exchange bracket orders
        if pos_dict:
            await self.executor.cancel_bracket_orders(
                symbol,
                pos_dict.get("sl_order_id"),
                pos_dict.get("tp_order_id"),
            )

        # 2. Execute market close on exchange (live mode only)
        if not self.settings.bot_dry_run and pos_dict:
            direction = pos_dict.get("direction", "BUY")
            size = pos_dict.get("size", 0.0)
            close_action = Action.SELL if direction == "BUY" else Action.BUY
            close_decision = TradeDecision(
                symbol=symbol,
                action=close_action,
                quantity=size,
                order_type="MARKET",
                reasoning=f"monitor_{reason}",
                source="position_monitor",
            )
            try:
                outcome = await self.executor.execute(close_decision, dry_run=False)
                self.logger.info(
                    "MONITOR: market close executed for %s, order_id=%s", symbol, outcome.order_id
                )
                self._check_slippage(outcome, {symbol: type("_S", (), {"price": price})()})
            except Exception:  # noqa: BLE001
                self.logger.exception("MONITOR: failed to execute market close for %s", symbol)
        elif self.settings.bot_dry_run:
            self.logger.info("MONITOR [dry-run]: would close %s @ $%.2f (%s)", symbol, price, reason)

        # 3. Clear local position state
        if symbol in self._open_positions:
            del self._open_positions[symbol]
            await self.persistence.async_save_position(symbol, None)
            self.logger.info("MONITOR: cleared position state for %s (%s)", symbol, reason)

    def _update_loss_tracking(
        self,
        close_reason: str,
        pre_pnl_pct: Optional[float],
        pre_quote_amount: float,
    ) -> None:
        """Update consecutive-loss counter and daily-loss accumulator after a close."""
        if close_reason in ("stop_loss", "trailing_stop"):
            self._consecutive_losses += 1
            # Convert PnL% to capital fraction for daily loss tracking
            if pre_pnl_pct is not None and pre_pnl_pct < 0 and pre_quote_amount > 0:
                demo_capital = getattr(self.settings, "demo_quote_capital", 10000.0)
                loss_usdt = abs(pre_pnl_pct / 100.0) * pre_quote_amount
                self._daily_loss_pct += loss_usdt / demo_capital
            self.logger.info(
                "Loss tracking updated: consecutive=%d, daily_loss=%.2f%%",
                self._consecutive_losses, self._daily_loss_pct * 100,
            )
        elif close_reason == "take_profit":
            self._consecutive_losses = 0
            self.logger.debug("Consecutive loss counter reset after take_profit")

    def _attach_pnl(self, outcome: Any, snapshots: dict[str, Any]) -> Any:
        """Compute and attach PnL to a close outcome using the tracked entry price."""
        from src.domain.trading import TradeOutcome  # noqa: PLC0415

        sym = outcome.decision.symbol
        pos = self._open_positions.get(sym)
        if not pos:
            return outcome
        entry_price = pos.get("entry_price", 0.0)
        if entry_price <= 0:
            return outcome
        exit_price = outcome.executed_price or snapshots.get(sym, None) and snapshots[sym].price or 0.0
        if exit_price <= 0:
            return outcome
        qty = outcome.decision.quantity
        direction = pos.get("direction", "")
        if direction == "BUY":  # was LONG
            pnl = (exit_price - entry_price) * qty
        elif direction == "SELL":  # was SHORT
            pnl = (entry_price - exit_price) * qty
        else:
            return outcome
        self.logger.info(
            "PnL for %s close: entry=%.4f exit=%.4f qty=%.8f pnl=%.4f USDT",
            sym, entry_price, exit_price, qty, pnl,
        )
        return TradeOutcome(
            decision=outcome.decision,
            order_id=outcome.order_id,
            executed_price=exit_price,
            pnl_usdt=pnl,
            dry_run=outcome.dry_run,
            timestamp=outcome.timestamp,
        )

    def _check_slippage(self, outcome: Any, snapshots: dict[str, Any]) -> None:
        """Log a warning if executed price deviates more than max_slippage_pct from snapshot price."""
        executed_price = getattr(outcome, "executed_price", None)
        if executed_price is None or executed_price <= 0:
            return
        symbol = outcome.decision.symbol
        snapshot = snapshots.get(symbol)
        if snapshot is None:
            return
        reference_price = snapshot.price
        if reference_price <= 0:
            return
        slippage = abs(executed_price - reference_price) / reference_price
        max_slip = getattr(self.settings, "max_slippage_pct", 0.005)
        if slippage > max_slip:
            self.logger.warning(
                "High slippage detected for %s: executed=$%.4f vs reference=$%.4f (%.3f%% > %.3f%% limit)",
                symbol, executed_price, reference_price, slippage * 100, max_slip * 100,
            )

    def _format_position_context(self) -> str:
        lines = []
        for sym, pos in self._open_positions.items():
            direction = pos.get("direction", "?")
            size = pos.get("size", 0)
            entry = pos.get("entry_price")
            sl = pos.get("sl_price")
            tp = pos.get("tp_price")
            entry_str = f"${entry:,.2f}" if entry else "?"
            sl_str = f"${sl:,.2f}" if sl else "—"
            tp_str = f"${tp:,.2f}" if tp else "—"
            lines.append(
                f"OPEN {direction} {sym}: qty={size:.6f} entry={entry_str} SL={sl_str} TP={tp_str}"
            )
        return "\n".join(lines) if lines else "CURRENT POSITION: None"

    def _sync_position_from_strategy(self) -> None:
        """Fix #6: Keep _open_positions dict in sync with TradingStrategy.current_position.

        TradingStrategy is the authoritative source of truth for the rich position
        lifecycle (trailing stop, partial TP, etc.). After every position lifecycle
        check we mirror its state back into the simple _open_positions dict so the
        position monitor and executor-path code always sees a consistent picture.
        """
        if self.trading_strategy is None:
            return
        pos = getattr(self.trading_strategy, "current_position", None)
        sym = getattr(self.trading_strategy, "symbol", None)
        if sym is None:
            return

        if pos is None:
            # Strategy says no open position — remove it from the dict if present
            if sym in self._open_positions:
                del self._open_positions[sym]
                self._last_close_cycle[sym] = self._cycle
                self.logger.debug("_sync_position_from_strategy: removed stale dict entry for %s", sym)
        else:
            # Mirror key fields so the monitor path and context formatter stay accurate
            direction_val = "BUY" if pos.direction == "LONG" else "SELL"
            existing = self._open_positions.get(sym, {})
            self._open_positions[sym] = {
                **existing,
                "direction": direction_val,
                "symbol": sym,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "sl_price": pos.trailing_stop_price or pos.stop_loss,
                "tp_price": pos.take_profit,
            }
