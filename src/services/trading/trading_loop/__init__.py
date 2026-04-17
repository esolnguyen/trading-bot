"""Trading loop orchestration.

The concrete `TradingLoop` class composes four mixins:
  - ContextMixin   — snapshot/pattern/RAG/balance/ML/HTF/regime assembly, PnL, slippage
  - DecisionsMixin — signal_scorer / LLM / HTF gate decision paths
  - PositionsMixin — bracket order placement, reconciliation, state sync
  - MonitorMixin   — fast-SL watcher that runs alongside the main cycle

The split keeps each module under the 500-line cap while preserving the public
surface (`run`, `run_cycle_once`, `run_position_monitor`, `stop`,
`trigger_immediate_cycle`).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Optional

from src.core.config import Settings
from src.domain.trading import Action, TradeDecision
from src.interfaces.notifiers import ConsoleNotifier, LoggerNotifier

from .context import ContextMixin
from .decisions import DecisionsMixin
from .monitor import MonitorMixin
from .positions import PositionsMixin


class TradingLoop(
    ContextMixin,
    DecisionsMixin,
    PositionsMixin,
    MonitorMixin,
):
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
            _pos_before = self._open_positions.get(symbol)
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
            self.persistence.append_trade_event(
                outcome,
                timestamp_iso,
                position_before=_pos_before,
                position_after=self._open_positions.get(symbol),
            )
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
