"""Trading strategy that wraps analysis with position management.

Helpers live in sibling modules:
  * :mod:`.monitor` — trailing stop, partial TP1, SL clamp
  * :mod:`.context` — prompt-facing position/capital context
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TYPE_CHECKING

from src.legacy.contracts.risk_contract import RiskManagerProtocol
from src.legacy.domain.trading.models import Position, TradeRecord
from .context import format_position_context
from .monitor import (
    build_conditions_from_position,
    clamp_sl_update,
    trigger_partial_tp,
    update_trailing_stop,
)

if TYPE_CHECKING:
    from src.mcp_servers.shared.infrastructure.storage.persistence import Persistence


class TradingStrategy:
    """Manages trading positions and decision execution based on AI analysis."""

    def __init__(
        self,
        logger: logging.Logger,
        persistence: "Persistence",
        risk_manager: RiskManagerProtocol,
        symbol: str = "BTCUSDT",
        settings: Any = None,
        brain_service: Optional[Any] = None,
        statistics_service: Optional[Any] = None,
        memory_service: Optional[Any] = None,
        config: Any = None,
        position_extractor: Optional[Any] = None,
        position_factory: Optional[Any] = None,
    ):
        self.logger = logger
        self.persistence = persistence
        self.brain_service = brain_service
        self.statistics_service = statistics_service
        self.memory_service = memory_service
        self.risk_manager = risk_manager
        # Normalise symbol: "BTC/USDT" → "BTCUSDT"
        self.symbol = (
            symbol.replace("/", "").replace(":", "").upper() if symbol else "BTCUSDT"
        )
        self.settings = settings or config
        self.position_factory = position_factory

        raw = self.persistence.load_position(self.symbol)
        if raw and isinstance(raw, dict):
            try:
                self.current_position: Optional[Position] = Position.from_dict(raw)
            except Exception:  # noqa: BLE001
                self.logger.warning(
                    "Could not deserialise saved position for %s — ignoring",
                    self.symbol,
                )
                self.current_position = None
        else:
            self.current_position = None

        if self.current_position:
            self.logger.info(
                "Loaded existing position: %s %s @ $%s",
                self.current_position.direction,
                self.current_position.symbol,
                f"{self.current_position.entry_price:,.2f}",
            )

    async def check_position(self, current_price: float) -> Optional[str]:
        """Check if current position hit SL, TP, trailing stop, or partial TP.

        Returns a close reason string if triggered, else None.
        """
        if not self.current_position:
            return None

        self.current_position.update_metrics(current_price)

        if getattr(self.settings, "trailing_stop_enabled", False):
            update_trailing_stop(
                self.current_position, self.settings, self.logger, current_price
            )

        active_sl = (
            self.current_position.trailing_stop_price or self.current_position.stop_loss
        )
        is_long = self.current_position.direction == "LONG"
        sl_hit = (
            (current_price <= active_sl) if is_long else (current_price >= active_sl)
        )
        if sl_hit:
            reason = (
                "trailing_stop"
                if self.current_position.trailing_stop_price
                else "stop_loss"
            )
            conditions = build_conditions_from_position(self.current_position)
            await self.persistence.async_save_position(
                self.symbol, self.current_position.to_dict()
            )
            await self.close_position(reason, current_price, conditions)
            return reason

        partial_enabled = getattr(self.settings, "partial_tp_enabled", False)
        if (
            partial_enabled
            and self.current_position.tp1_price > 0
            and not self.current_position.partial_tp1_hit
        ):
            tp1_hit = (
                (current_price >= self.current_position.tp1_price)
                if is_long
                else (current_price <= self.current_position.tp1_price)
            )
            if tp1_hit:
                await trigger_partial_tp(
                    self.current_position,
                    self.settings,
                    self.logger,
                    current_price,
                    self.persistence.async_save_trade_decision,
                )
                await self.persistence.async_save_position(
                    self.symbol, self.current_position.to_dict()
                )
                return "partial_tp1"

        if self.current_position.is_target_hit(current_price):
            conditions = build_conditions_from_position(self.current_position)
            await self.persistence.async_save_position(
                self.symbol, self.current_position.to_dict()
            )
            await self.close_position("take_profit", current_price, conditions)
            return "take_profit"

        await self.persistence.async_save_position(
            self.symbol, self.current_position.to_dict()
        )
        return None

    async def close_position(
        self,
        reason: str,
        current_price: float,
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Close the current position and update trading brain."""
        if not self.current_position:
            return

        pnl = self.current_position.calculate_pnl(current_price)

        fee_pct = getattr(self.settings, "transaction_fee_percent", None) or getattr(
            self.settings, "TRANSACTION_FEE_PERCENT", 0.00075
        )
        closing_fee = self.current_position.calculate_closing_fee(
            current_price, fee_pct
        )

        decision = TradeRecord(
            timestamp=datetime.now(timezone.utc),
            symbol=self.current_position.symbol,
            action=f"CLOSE_{self.current_position.direction}",
            confidence=self.current_position.confidence,
            price=current_price,
            stop_loss=self.current_position.stop_loss,
            take_profit=self.current_position.take_profit,
            position_size=self.current_position.size_pct,
            quote_amount=self.current_position.quote_amount,
            quantity=self.current_position.size,
            fee=closing_fee,
            reasoning=f"Position closed: {reason}. P&L: {pnl:+.2f}%. Fee: ${closing_fee:.4f}",
        )

        self.logger.info(
            "Closing %s position (%s) @ $%s, P&L: %s%%, Fee: $%.4f",
            self.current_position.direction,
            reason,
            f"{current_price:,.2f}",
            f"{pnl:+.2f}",
            closing_fee,
        )

        entry_decision = None
        try:
            entry_decision = self.persistence.get_entry_decision_for_position(
                self.symbol
            )
            if entry_decision:
                reasoning_preview = (
                    entry_decision.reasoning[:500]
                    if entry_decision.reasoning
                    else "(no reasoning)"
                )
                self.logger.debug(
                    "Retrieved entry decision with reasoning: %s...", reasoning_preview
                )
            else:
                self.logger.warning(
                    "Could not retrieve entry decision from trade history"
                )
        except Exception as e:
            self.logger.error("Error retrieving entry decision: %s", e)

        try:
            if self.brain_service is None:
                raise AttributeError("brain_service not wired")
            self.brain_service.update_from_closed_trade(
                position=self.current_position,
                close_price=current_price,
                close_reason=reason,
                entry_decision=entry_decision,
                market_conditions=market_conditions,
            )
        except Exception as e:
            self.logger.error("Error updating trading brain: %s", e)

        # Save close decision FIRST so statistics include this trade
        await self.persistence.async_save_trade_decision(decision)

        if self.statistics_service is not None:
            try:
                demo_capital = getattr(
                    self.settings, "demo_quote_capital", None
                ) or getattr(self.settings, "DEMO_QUOTE_CAPITAL", 10000.0)
                self.statistics_service.recalculate(demo_capital)
            except Exception as e:
                self.logger.error("Error recalculating statistics: %s", e)

        await self.persistence.async_save_position(self.symbol, None)
        self.current_position = None

    async def _update_position_parameters(
        self,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> bool:
        """Update position stop loss and take profit."""
        if not self.current_position:
            return False

        updated = False
        new_sl = self.current_position.stop_loss
        new_tp = self.current_position.take_profit

        if stop_loss and stop_loss != self.current_position.stop_loss:
            direction = self.current_position.direction
            entry = self.current_position.entry_price
            old_sl = self.current_position.stop_loss
            stop_loss = clamp_sl_update(direction, entry, stop_loss, self.logger)

            # FULL AI AUTONOMY within clamped bounds
            if direction == "LONG" and stop_loss < old_sl:
                self.logger.info(
                    "AI Widening Stop Loss for LONG: $%.2f -> $%.2f (Risk Increased)",
                    old_sl,
                    stop_loss,
                )
            elif direction == "SHORT" and stop_loss > old_sl:
                self.logger.info(
                    "AI Widening Stop Loss for SHORT: $%.2f -> $%.2f (Risk Increased)",
                    old_sl,
                    stop_loss,
                )
            else:
                self.logger.info("Updated Stop Loss: $%s", f"{stop_loss:,.2f}")
            new_sl = stop_loss
            updated = True

        if take_profit and take_profit != self.current_position.take_profit:
            new_tp = take_profit
            self.logger.info("Updated Take Profit: $%s", f"{take_profit:,.2f}")
            updated = True

        if updated:
            self.current_position = self.position_factory.create_updated_position(
                original_position=self.current_position,
                new_stop_loss=new_sl,
                new_take_profit=new_tp,
            )
            await self.persistence.async_save_position(
                self.symbol, self.current_position.to_dict()
            )

        return updated

    async def reconcile(self, feed: Any) -> None:
        """Verify saved position against live Binance state on startup.

        If the exchange has no open position matching the saved one (e.g. after a
        liquidation or manual close), the stale local state is cleared so the bot
        does not try to manage a ghost position.  Skipped in dry-run mode.
        """
        if getattr(self.settings, "bot_dry_run", True):
            return
        if self.current_position is None:
            return
        try:
            balance_resp = await feed.get_balance()
            if not balance_resp.get("success"):
                self.logger.warning("reconcile: could not fetch balance — skipping")
                return

            data = balance_resp.get("data", {})
            base_asset = self.current_position.symbol.replace("USDT", "").replace(
                "BUSD", ""
            )
            balances = data.get("balances", [])
            asset_entry = next(
                (b for b in balances if b.get("asset") == base_asset), None
            )
            live_qty = float(asset_entry.get("free", 0) if asset_entry else 0)

            if live_qty < self.current_position.size * 0.05:
                self.logger.warning(
                    "reconcile: saved %s position (qty=%.6f) not found on exchange "
                    "(live qty=%.6f) — clearing stale position",
                    self.current_position.symbol,
                    self.current_position.size,
                    live_qty,
                )
                await self.persistence.async_save_position(self.symbol, None)
                self.current_position = None
            else:
                self.logger.info(
                    "reconcile: %s position confirmed on exchange (qty=%.6f)",
                    self.current_position.symbol,
                    live_qty,
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.error("reconcile failed: %s", exc)

    def get_position_context(self, current_price: Optional[float] = None) -> str:
        """Return a formatted capital-and-position summary for prompts."""
        demo_capital = getattr(self.settings, "demo_quote_capital", None) or getattr(
            self.settings, "DEMO_QUOTE_CAPITAL", 10000.0
        )
        capital = (
            self.statistics_service.get_current_capital(demo_capital)
            if self.statistics_service
            else demo_capital
        )
        return format_position_context(self.current_position, capital, current_price)
