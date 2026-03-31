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
        self._regime_refresh_date: str = ""
        # Kill-switch state
        self._consecutive_losses: int = 0
        self._daily_loss_pct: float = 0.0
        self._daily_reset_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Lock shared between main cycle and position monitor to prevent double-close races
        self._position_lock: asyncio.Lock = asyncio.Lock()

    def stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        """Run the trading loop until stopped."""
        self.logger.info("trading loop started")
        _retry_delay = 30
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
                await self._sleep(self.settings.bot_interval_seconds)
        except asyncio.CancelledError:
            raise
        finally:
            self.logger.info("trading loop stopped")

    async def run_cycle_once(self) -> dict[str, Any]:
        """Execute one dry/live trading decision cycle."""
        self._cycle += 1
        timestamp_iso = datetime.now(timezone.utc).isoformat()

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

        # B4: Anomaly detection — skip cycle entirely if market is abnormal
        if self._anomaly_detector is not None and snapshots:
            first_snap = next(iter(snapshots.values()))
            try:
                if self._anomaly_detector.is_anomaly(first_snap, None):
                    self.logger.warning("Anomaly detected — skipping cycle %d", self._cycle)
                    return {"timestamp_iso": timestamp_iso, "halted": True, "halt_reason": "anomaly"}
            except Exception:  # noqa: BLE001
                self.logger.debug("anomaly_detector failed", exc_info=True)

        # E: Append latest closed candle to OHLCV CSV
        if self._ohlcv_writer is not None and snapshots:
            try:
                first_snap = next(iter(snapshots.values()))
                self._ohlcv_writer.append(first_snap.candles)
            except Exception:  # noqa: BLE001
                self.logger.debug("ohlcv_writer.append failed", exc_info=True)

        # A4: Refresh macro regime daily and inject into system prompt
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._cycle_classifier is not None and today != self._regime_refresh_date:
            try:
                self._refresh_regime()
                self._regime_refresh_date = today
            except Exception:  # noqa: BLE001
                self.logger.debug("regime refresh failed", exc_info=True)

        # Detect positions closed by exchange TP/SL since last cycle.
        await self._reconcile_open_positions()

        # Position lifecycle check (only when trading_strategy is wired)
        first_price = next(iter(snapshots.values())).price if snapshots else 0.0
        if self.trading_strategy is not None:
            try:
                async with self._position_lock:
                    # Capture position PnL before check so we can track loss on close
                    _pre_pos = getattr(self.trading_strategy, "current_position", None)
                    _pre_pnl_pct = _pre_pos.calculate_pnl(first_price) if _pre_pos else None
                    _pre_quote = _pre_pos.quote_amount if _pre_pos else 0.0

                    close_reason = await self.trading_strategy.check_position(first_price)

                    # Update kill-switch counters when a position closes
                    if close_reason in ("stop_loss", "trailing_stop", "take_profit", "partial_tp1"):
                        self._update_loss_tracking(close_reason, _pre_pnl_pct, _pre_quote)
            except Exception:  # noqa: BLE001
                self.logger.exception("trading_strategy.check_position failed")

        choppiness_threshold = getattr(self.settings, "choppiness_threshold", 61.8)
        analyses = {
            symbol: self.tech_analyzer.analyze(symbol, snapshot, choppiness_threshold)
            for symbol, snapshot in snapshots.items()
        }

        # Multi-timeframe: fetch 4h candles and compute HTF signals
        htf_analyses = await self._collect_htf_analyses(snapshots)

        patterns = self._collect_patterns(snapshots, analyses)
        rag_context = self._build_rag_context(snapshots, analyses)

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
                brain_context = self.brain_service.get_brain_context(market_conditions)
                dynamic_thresholds = self.brain_service.get_dynamic_thresholds(market_conditions)
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

        system_prompt, user_message = self.builder.build(
            snapshots,
            analyses,
            patterns,
            rag_context,
            position_context=position_context,
            memory_context=memory_context,
            brain_context=brain_context,
            dynamic_thresholds=dynamic_thresholds,
            ml_context=ml_context,
        )
        decision = await self._decide(system_prompt, user_message, htf_analyses)

        balance = await self._build_balance_context(snapshots)
        risk_result = self.risk.validate(decision, balance, self._open_positions)
        outcome = await self.executor.execute(risk_result.decision, risk_result.dry_run)
        self.memory.record(outcome)

        # Slippage check: warn if executed price deviates significantly from decision price
        self._check_slippage(outcome, snapshots)

        # Keep _open_positions in sync so subsequent cycles see the position.
        if not outcome.dry_run and outcome.order_id:
            await self._update_open_positions(outcome)

        chosen_symbol = outcome.decision.symbol
        chosen_analysis = analyses.get(chosen_symbol) or next(iter(analyses.values()))
        chosen_patterns = patterns.get(chosen_symbol) or next(iter(patterns.values()))
        rag_docs_retrieved = rag_context.count("[")
        self.persistence.append_trade(outcome, timestamp_iso)
        self.persistence.append_cycle_log(
            timestamp_iso=timestamp_iso,
            cycle=self._cycle,
            symbol=chosen_symbol,
            analysis=chosen_analysis,
            patterns=chosen_patterns,
            rag_docs_retrieved=rag_docs_retrieved,
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
        symbol_signals = [(symbol.replace("USDT", ""), analysis.signal.value) for symbol, analysis in analyses.items()]
        self.console_notifier.notify_cycle(
            cycle=self._cycle,
            timestamp_iso=timestamp_iso,
            symbol_signals=symbol_signals,
            final_decision=outcome.decision.action.value,
        )
        self.logger_notifier.notify(
            f"cycle={self._cycle} symbol={chosen_symbol} decision={outcome.decision.action.value} "
            f"risk={risk_result.status} dry_run={outcome.dry_run}"
        )

        # Optional Discord notification
        if self.discord_notifier is not None:
            try:
                await self.discord_notifier.send_trading_decision(
                    symbol=chosen_symbol,
                    decision=outcome.decision.action.value,
                    price=first_price,
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
            "rag_context": rag_context,
            "decision": decision,
            "risk_result": risk_result,
            "outcome": outcome,
        }

    async def _collect_snapshots(self) -> dict[str, Any]:
        symbols = getattr(self.settings, "trading_symbols", None) or ["BTCUSDT", "ETHUSDT"]
        snapshots = await asyncio.gather(*[self.aggregator.snapshot(s) for s in symbols])
        return dict(zip(symbols, snapshots))

    def _collect_patterns(self, snapshots: dict[str, Any], analyses: dict[str, Any]) -> dict[str, Any]:
        results = {}
        for symbol, snapshot in snapshots.items():
            pattern_result = self.pattern_analyzer.analyze(symbol, snapshot.candles)
            if self.settings.model_supports_vision:
                pattern_result.chart_png_b64 = self.chart_gen.render(symbol, snapshot.candles, analyses[symbol].indicators)
            results[symbol] = pattern_result
        return results

    def _build_rag_context(self, snapshots: dict[str, Any], analyses: dict[str, Any]) -> str:
        sections = [
            self.retriever.retrieve(snapshots[symbol], analyses[symbol])
            for symbol in snapshots
        ]
        combined = "\n\n".join(section for section in sections if section and section != "=== NO CONTEXT AVAILABLE ===")
        return combined or "=== NO CONTEXT AVAILABLE ==="

    async def _decide(self, system_prompt: str, user_message: str, htf_analyses: dict[str, Any] | None = None) -> TradeDecision:
        decision = await self.llm.decide(system_prompt, user_message)
        if decision.action == Action.HOLD:
            return decision
        if decision.symbol not in {"BTCUSDT", "ETHUSDT"}:
            return TradeDecision(symbol="BTCUSDT", action=Action.HOLD, reasoning="invalid_symbol", source="fallback_hold")

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
                await self.persistence.async_save_position(sym, None)
                return

        # New entry — place exchange bracket orders for both futures AND spot.
        entry_price = outcome.executed_price or 0.0
        sl_order_id: str | None = None
        tp_order_id: str | None = None
        sl_price: float | None = None
        tp_price: float | None = None
        product = getattr(self.settings, "binance_product", "spot")

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
                await self.persistence.async_save_position(sym, None)

    async def _collect_htf_analyses(self, snapshots: dict[str, Any]) -> dict[str, Any]:
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
                choppiness_threshold = getattr(self.settings, "choppiness_threshold", 61.8)
                results[symbol] = self.tech_analyzer.analyze(symbol, htf_snap, choppiness_threshold)
            except Exception:  # noqa: BLE001
                self.logger.debug("HTF analysis failed for %s", symbol, exc_info=True)
        return results

    async def _build_ml_context(self, snapshots: dict[str, Any], analyses: dict[str, Any]) -> str | None:
        """Assemble ML-derived context sections for the LLM prompt."""
        parts: list[str] = []
        first_symbol = next(iter(snapshots)) if snapshots else None

        # A1: Historical percentiles
        if self._percentile_scorer is not None and first_symbol:
            try:
                snap = snapshots[first_symbol]
                analysis = analyses.get(first_symbol)
                if analysis is not None:
                    pct_text = self._percentile_scorer.score(analysis.indicators, snap.price)
                    if pct_text:
                        parts.append(pct_text)
            except Exception:  # noqa: BLE001
                self.logger.debug("percentile_scorer failed", exc_info=True)

        # A2: Multi-timeframe alignment
        if self._multi_tf_analyzer is not None and first_symbol:
            try:
                current_signal = analyses[first_symbol].signal.value if first_symbol in analyses else "NEUTRAL"
                mtf_text = await self._multi_tf_analyzer.build_summary(first_symbol, current_signal)
                if mtf_text:
                    parts.append(mtf_text)
            except Exception:  # noqa: BLE001
                self.logger.debug("multi_tf_analyzer failed", exc_info=True)

        # A3: Key S/R levels
        if self._key_level_detector is not None and first_symbol:
            try:
                price = snapshots[first_symbol].price
                levels_text = self._key_level_detector.format_context(price)
                if levels_text:
                    parts.append(levels_text)
            except Exception:  # noqa: BLE001
                self.logger.debug("key_level_detector failed", exc_info=True)

        # B2: XGBoost direction signal
        if self._direction_classifier is not None and first_symbol:
            try:
                analysis = analyses.get(first_symbol)
                if analysis is not None:
                    dir_text = self._direction_classifier.format_context(analysis.indicators)
                    if dir_text:
                        parts.append(dir_text)
            except Exception:  # noqa: BLE001
                self.logger.debug("direction_classifier failed", exc_info=True)

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
                suffix = self._cycle_classifier.regime_system_prompt_suffix(regime, confidence)
                self.builder._regime_suffix = suffix
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
