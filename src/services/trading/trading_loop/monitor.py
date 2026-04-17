"""Position monitor mixin: fast stop-loss watcher that runs alongside the main cycle."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from src.domain.trading import Action, TradeDecision


class MonitorMixin:
    """Provides `run_position_monitor`, `_monitor_check_positions`, `_monitor_execute_close`."""

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
