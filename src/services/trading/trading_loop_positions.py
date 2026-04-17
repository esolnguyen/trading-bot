"""Position state mixin: bracket orders, reconciliation, strategy sync, formatting."""

from __future__ import annotations

from typing import Any

from src.domain.trading import Action


class PositionsMixin:
    """Provides `_update_open_positions`, `_reconcile_open_positions`,
    `_sync_position_from_strategy`, `_format_position_context`.
    """

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
            if sym in self._open_positions:
                del self._open_positions[sym]
                self._last_close_cycle[sym] = self._cycle
                self.logger.debug("_sync_position_from_strategy: removed stale dict entry for %s", sym)
        else:
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
