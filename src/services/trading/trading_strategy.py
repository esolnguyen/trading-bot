"""Trading strategy that wraps analysis with position management."""

import logging
from datetime import datetime, timezone
from typing import Optional, Any, Dict, TYPE_CHECKING

from src.contracts.risk_contract import RiskManagerProtocol
from src.domain.trading.models import Position, TradeRecord
from src.services.trading.brain_service import TradingBrainService
from src.services.trading.statistics_service import TradingStatisticsService
from src.services.trading.memory_service import TradingMemoryService

if TYPE_CHECKING:
    from src.infrastructure.storage.persistence import Persistence


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
        # Support both new Settings dataclass and legacy config module
        self.settings = settings or config
        self.position_factory = position_factory

        # Load any existing position from disk
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
        """Check if current position hit stop loss, take profit, trailing stop, or partial TP.

        Args:
            current_price: Current market price

        Returns:
            Reason for closing/partially-closing position if triggered, else None
        """
        if not self.current_position:
            return None

        # Update live performance metrics (MAE/MFE)
        self.current_position.update_metrics(current_price)

        # --- Trailing stop update ---
        trailing_enabled = getattr(self.settings, "trailing_stop_enabled", False)
        if trailing_enabled:
            self._update_trailing_stop(current_price)

        # --- Hard stop loss check (includes trailing stop if activated) ---
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
            conditions = self._build_conditions_from_position(self.current_position)
            await self.persistence.async_save_position(
                self.symbol, self.current_position.to_dict()
            )
            await self.close_position(reason, current_price, conditions)
            return reason

        # --- Partial TP1 check ---
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
                await self._trigger_partial_tp(current_price)
                await self.persistence.async_save_position(
                    self.symbol, self.current_position.to_dict()
                )
                return "partial_tp1"

        # --- Full take profit check ---
        if self.current_position.is_target_hit(current_price):
            conditions = self._build_conditions_from_position(self.current_position)
            await self.persistence.async_save_position(
                self.symbol, self.current_position.to_dict()
            )
            await self.close_position("take_profit", current_price, conditions)
            return "take_profit"

        await self.persistence.async_save_position(
            self.symbol, self.current_position.to_dict()
        )
        return None

    def _update_trailing_stop(self, current_price: float) -> None:
        """Ratchet the trailing stop level as price moves in our favour."""
        pos = self.current_position
        if pos is None:
            return
        activation_pct = getattr(self.settings, "trailing_stop_activation_pct", 0.01)
        distance_pct = getattr(self.settings, "trailing_stop_distance_pct", 0.005)
        is_long = pos.direction == "LONG"
        pnl_pct = pos.calculate_pnl(current_price) / 100.0  # convert to decimal

        # Only activate once position is sufficiently in profit
        if pnl_pct < activation_pct:
            return

        if is_long:
            new_trail = current_price * (1.0 - distance_pct)
            # Only move trail upward
            if new_trail > pos.trailing_stop_price:
                pos.trailing_stop_price = new_trail
                self.logger.debug(
                    "Trailing stop moved up to $%.2f for %s", new_trail, pos.symbol
                )
        else:
            new_trail = current_price * (1.0 + distance_pct)
            # Only move trail downward (for shorts, a lower trail is better)
            if pos.trailing_stop_price == 0.0 or new_trail < pos.trailing_stop_price:
                pos.trailing_stop_price = new_trail
                self.logger.debug(
                    "Trailing stop moved down to $%.2f for %s", new_trail, pos.symbol
                )

    async def _trigger_partial_tp(self, current_price: float) -> None:
        """Record partial TP1 hit: reduce position size and adjust trailing stop."""
        pos = self.current_position
        if pos is None:
            return
        closed_size = pos.size * pos.tp1_size_pct
        remaining_size = pos.size - closed_size
        closed_quote = closed_size * current_price
        fee_pct = getattr(self.settings, "transaction_fee_percent", 0.00075)
        fee = closed_quote * fee_pct
        pnl_pct = pos.calculate_pnl(current_price)

        self.logger.info(
            "Partial TP1 triggered for %s @ $%.2f — closing %.1f%% (%.6f units), P&L: %+.2f%%, fee: $%.4f",
            pos.symbol,
            current_price,
            pos.tp1_size_pct * 100,
            closed_size,
            pnl_pct,
            fee,
        )

        # Mark TP1 hit and update remaining position size (in-place on slots dataclass)
        object.__setattr__(pos, "partial_tp1_hit", True)
        object.__setattr__(pos, "size", remaining_size)
        object.__setattr__(pos, "quote_amount", remaining_size * pos.entry_price)

        # Move stop loss to break-even once TP1 is hit (protect profits)
        if pos.direction == "LONG" and pos.entry_price > pos.stop_loss:
            object.__setattr__(pos, "stop_loss", pos.entry_price)
            self.logger.info(
                "Stop loss moved to break-even $%.2f after TP1", pos.entry_price
            )
        elif pos.direction == "SHORT" and pos.entry_price < pos.stop_loss:
            object.__setattr__(pos, "stop_loss", pos.entry_price)
            self.logger.info(
                "Stop loss moved to break-even $%.2f after TP1", pos.entry_price
            )

        # Save partial close record
        decision = TradeRecord(
            timestamp=datetime.now(timezone.utc),
            symbol=pos.symbol,
            action=f"PARTIAL_TP1_{pos.direction}",
            confidence=pos.confidence,
            price=current_price,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            position_size=pos.tp1_size_pct,
            quote_amount=closed_quote,
            quantity=closed_size,
            fee=fee,
            reasoning=f"Partial TP1 hit at ${current_price:,.2f}. P&L: {pnl_pct:+.2f}%. Remaining: {remaining_size:.6f} units.",
        )
        await self.persistence.async_save_trade_decision(decision)

    async def close_position(
        self,
        reason: str,
        current_price: float,
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Close the current position and update trading brain.

        Args:
            reason: Reason for closing (stop_loss, take_profit, signal)
            current_price: Current market price
            market_conditions: Optional market conditions for brain learning
        """
        if not self.current_position:
            return

        pnl = self.current_position.calculate_pnl(current_price)

        # Calculate closing fee
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

        # Retrieve entry decision from trade history for brain learning
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

        # Update trading brain with closed trade insights
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

        # Recalculate performance statistics (Sharpe, Sortino, drawdown, etc.)
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
        """Update position stop loss and take profit.

        Args:
            stop_loss: New stop loss
            take_profit: New take profit

        Returns:
            True if anything was updated
        """
        if not self.current_position:
            return False

        updated = False
        new_sl = self.current_position.stop_loss
        new_tp = self.current_position.take_profit

        if stop_loss and stop_loss != self.current_position.stop_loss:
            direction = self.current_position.direction
            entry = self.current_position.entry_price
            old_sl = self.current_position.stop_loss

            # Clamp SL distance: same bounds enforced at entry (0.5 % – 10 %)
            raw_dist = abs(entry - stop_loss) / entry if entry else 0
            if raw_dist > 0.10:
                self.logger.warning(
                    "AI SL update exceeds 10%% max distance (%.1f%%) — clamping",
                    raw_dist * 100,
                )
                stop_loss = entry * 0.90 if direction == "LONG" else entry * 1.10
            elif raw_dist < 0.005:
                self.logger.warning(
                    "AI SL update below 0.5%% min distance (%.2f%%) — expanding",
                    raw_dist * 100,
                )
                stop_loss = entry * 0.995 if direction == "LONG" else entry * 1.005

            # FULL AI AUTONOMY within clamped bounds
            if direction == "LONG" and stop_loss < old_sl:
                self.logger.info(
                    "AI Widening Stop Loss for LONG: $%.2f -> $%.2f (Risk Increased)",
                    old_sl,
                    stop_loss,
                )
                new_sl = stop_loss
                updated = True
            elif direction == "SHORT" and stop_loss > old_sl:
                self.logger.info(
                    "AI Widening Stop Loss for SHORT: $%.2f -> $%.2f (Risk Increased)",
                    old_sl,
                    stop_loss,
                )
                new_sl = stop_loss
                updated = True
            else:
                new_sl = stop_loss
                self.logger.info("Updated Stop Loss: $%s", f"{stop_loss:,.2f}")
                updated = True

        if take_profit and take_profit != self.current_position.take_profit:
            new_tp = take_profit
            self.logger.info("Updated Take Profit: $%s", f"{take_profit:,.2f}")
            updated = True

        if updated:
            # Create new position with updated values using factory
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
        does not try to manage a ghost position.

        Skipped in dry-run mode.
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
            # Derive base asset from symbol, e.g. "BTCUSDT" → "BTC"
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

    @staticmethod
    def _build_conditions_from_position(position: Position) -> Dict[str, Any]:
        """Reconstruct market conditions from Position's stored entry fields.

        Used when closing via SL/TP hit where no fresh analysis is available.
        """
        rsi = position.rsi_at_entry
        if rsi > 70:
            rsi_level = "OVERBOUGHT"
        elif rsi > 60:
            rsi_level = "STRONG"
        elif rsi < 30:
            rsi_level = "OVERSOLD"
        elif rsi < 40:
            rsi_level = "WEAK"
        else:
            rsi_level = "NEUTRAL"

        return {
            "trend_direction": position.trend_direction_at_entry,
            "adx": position.adx_at_entry,
            "rsi": rsi,
            "rsi_level": rsi_level,
            "volatility": position.volatility_level,
            "macd_signal": position.macd_signal_at_entry,
            "bb_position": position.bb_position_at_entry,
            "volume_state": position.volume_state_at_entry,
            "market_sentiment": position.market_sentiment_at_entry,
        }

    def get_position_context(self, current_price: Optional[float] = None) -> str:
        """Get formatted context about current position for prompts.

        Args:
            current_price: Current market price for P&L calculation

        Returns:
            Formatted position context string with capital status
        """
        demo_capital = getattr(self.settings, "demo_quote_capital", None) or getattr(
            self.settings, "DEMO_QUOTE_CAPITAL", 10000.0
        )
        capital = (
            self.statistics_service.get_current_capital(demo_capital)
            if self.statistics_service
            else demo_capital
        )
        currency = "USDT"
        if not self.current_position:
            return (
                f"## Capital Status\n"
                f"- Total Capital: ${capital:,.2f} {currency}\n"
                f"- Available: ${capital:,.2f} (100%)\n\n"
                f"CURRENT POSITION: None"
            )
        pos = self.current_position
        # Ensure both datetimes are timezone-aware for subtraction
        now = datetime.now(timezone.utc)
        entry_time = pos.entry_time
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        duration = now - entry_time
        hours = duration.total_seconds() / 3600
        allocated = pos.quote_amount
        available = capital - allocated
        allocation_pct = (allocated / capital) * 100 if capital > 0 else 0
        context_lines = [
            "## Capital Status",
            f"- Total Capital: ${capital:,.2f} {currency}",
            f"- Allocated: ${allocated:,.2f} ({allocation_pct:.1f}%)",
            f"- Available: ${available:,.2f} ({100 - allocation_pct:.1f}%)",
            "",
            "## Current Position",
            f"- Direction: {pos.direction}",
            f"- Symbol: {pos.symbol}",
            f"- Entry Price: ${pos.entry_price:,.2f}",
        ]
        if current_price and current_price > 0:
            context_lines.append(f"- Current Price: ${current_price:,.2f}")
        context_lines.extend(
            [
                f"- Stop Loss: ${pos.stop_loss:,.2f}",
                f"- Take Profit: ${pos.take_profit:,.2f}",
                f"- Position Size: {pos.size_pct * 100:.2f}%",
                f"- Quantity: {pos.size:.6f}",
                f"- Entry Fee: ${pos.entry_fee:.4f}",
                f"- Duration: {hours:.1f} hours",
                f"- Confidence: {pos.confidence}",
            ]
        )
        if current_price and current_price > 0:
            pnl_pct = pos.calculate_pnl(current_price)
            pnl_quote = (
                (current_price - pos.entry_price) * pos.size
                if pos.direction == "LONG"
                else (pos.entry_price - current_price) * pos.size
            )
            context_lines.append(
                f"- Unrealized P&L: {pnl_pct:+.2f}% (${pnl_quote:+,.2f} {currency})"
            )
        return "\n".join(context_lines)
