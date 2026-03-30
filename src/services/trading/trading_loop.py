"""Trading loop orchestration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Optional

from src.core.config import Settings
from src.domain.trading import Action, TradeDecision
from src.interfaces.notifiers import ConsoleNotifier, LoggerNotifier


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
        snapshots = await self._collect_snapshots()

        # Detect positions closed by exchange TP/SL since last cycle.
        await self._reconcile_open_positions()

        # Position lifecycle check (only when trading_strategy is wired)
        first_price = next(iter(snapshots.values())).price if snapshots else 0.0
        if self.trading_strategy is not None:
            try:
                await self.trading_strategy.check_position(first_price)
            except Exception:  # noqa: BLE001
                self.logger.exception("trading_strategy.check_position failed")

        analyses = {
            symbol: self.tech_analyzer.analyze(symbol, snapshot)
            for symbol, snapshot in snapshots.items()
        }
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

        system_prompt, user_message = self.builder.build(
            snapshots,
            analyses,
            patterns,
            rag_context,
            position_context=position_context,
            memory_context=memory_context,
            brain_context=brain_context,
            dynamic_thresholds=dynamic_thresholds,
        )
        decision = await self._decide(system_prompt, user_message)

        balance = await self._build_balance_context(snapshots)
        risk_result = self.risk.validate(decision, balance)
        outcome = await self.executor.execute(risk_result.decision, risk_result.dry_run)
        self.memory.record(outcome)

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

    async def _decide(self, system_prompt: str, user_message: str) -> TradeDecision:
        decision = await self.llm.decide(system_prompt, user_message)
        if decision.action == Action.HOLD:
            return decision
        if decision.symbol not in {"BTCUSDT", "ETHUSDT"}:
            return TradeDecision(symbol="BTCUSDT", action=Action.HOLD, reasoning="invalid_symbol", source="fallback_hold")
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

        # New entry (or rare same-direction add) — place bracket orders for futures.
        entry_price = outcome.executed_price or 0.0
        sl_order_id: str | None = None
        tp_order_id: str | None = None
        sl_price: float | None = None
        tp_price: float | None = None

        if getattr(self.settings, "binance_product", "") == "usdt_futures" and entry_price > 0:
            sl_pct = getattr(self.settings, "default_stop_loss_pct", 0.02)
            tp_pct = getattr(self.settings, "default_take_profit_pct", 0.04)
            is_long = action == Action.BUY
            sl_price = round(entry_price * ((1 - sl_pct) if is_long else (1 + sl_pct)), 2)
            tp_price = round(entry_price * ((1 + tp_pct) if is_long else (1 - tp_pct)), 2)
            sl_order_id, tp_order_id = await self.executor.place_bracket_orders(
                sym, action.value, sl_price, tp_price
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
