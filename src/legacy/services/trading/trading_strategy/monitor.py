"""Position-monitor helpers: trailing stop, partial TP1, condition snapshots."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict

from src.legacy.domain.trading.models import Position, TradeRecord


def update_trailing_stop(
    pos: Position, settings: Any, logger: logging.Logger, current_price: float
) -> None:
    """Ratchet the trailing stop level as price moves in our favour (mutates pos)."""
    activation_pct = getattr(settings, "trailing_stop_activation_pct", 0.01)
    distance_pct = getattr(settings, "trailing_stop_distance_pct", 0.005)
    is_long = pos.direction == "LONG"
    pnl_pct = pos.calculate_pnl(current_price) / 100.0

    if pnl_pct < activation_pct:
        return

    if is_long:
        new_trail = current_price * (1.0 - distance_pct)
        if new_trail > pos.trailing_stop_price:
            pos.trailing_stop_price = new_trail
            logger.debug(
                "Trailing stop moved up to $%.2f for %s", new_trail, pos.symbol
            )
    else:
        new_trail = current_price * (1.0 + distance_pct)
        if pos.trailing_stop_price == 0.0 or new_trail < pos.trailing_stop_price:
            pos.trailing_stop_price = new_trail
            logger.debug(
                "Trailing stop moved down to $%.2f for %s", new_trail, pos.symbol
            )


async def trigger_partial_tp(
    pos: Position,
    settings: Any,
    logger: logging.Logger,
    current_price: float,
    save_trade_decision: Callable[[TradeRecord], Awaitable[None]],
) -> None:
    """Record partial TP1 hit: reduce position size and adjust trailing stop."""
    closed_size = pos.size * pos.tp1_size_pct
    remaining_size = pos.size - closed_size
    closed_quote = closed_size * current_price
    fee_pct = getattr(settings, "transaction_fee_percent", 0.00075)
    fee = closed_quote * fee_pct
    pnl_pct = pos.calculate_pnl(current_price)

    logger.info(
        "Partial TP1 triggered for %s @ $%.2f — closing %.1f%% (%.6f units), P&L: %+.2f%%, fee: $%.4f",
        pos.symbol,
        current_price,
        pos.tp1_size_pct * 100,
        closed_size,
        pnl_pct,
        fee,
    )

    object.__setattr__(pos, "partial_tp1_hit", True)
    object.__setattr__(pos, "size", remaining_size)
    object.__setattr__(pos, "quote_amount", remaining_size * pos.entry_price)

    # Move stop loss to break-even once TP1 is hit (protect profits)
    if pos.direction == "LONG" and pos.entry_price > pos.stop_loss:
        object.__setattr__(pos, "stop_loss", pos.entry_price)
        logger.info("Stop loss moved to break-even $%.2f after TP1", pos.entry_price)
    elif pos.direction == "SHORT" and pos.entry_price < pos.stop_loss:
        object.__setattr__(pos, "stop_loss", pos.entry_price)
        logger.info("Stop loss moved to break-even $%.2f after TP1", pos.entry_price)

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
        reasoning=(
            f"Partial TP1 hit at ${current_price:,.2f}. "
            f"P&L: {pnl_pct:+.2f}%. Remaining: {remaining_size:.6f} units."
        ),
    )
    await save_trade_decision(decision)


def build_conditions_from_position(position: Position) -> Dict[str, Any]:
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


def clamp_sl_update(
    direction: str,
    entry_price: float,
    new_sl: float,
    logger: logging.Logger,
) -> float:
    """Clamp an AI-proposed SL update to the 0.5 %–10 % entry-distance band."""
    raw_dist = abs(entry_price - new_sl) / entry_price if entry_price else 0
    if raw_dist > 0.10:
        logger.warning(
            "AI SL update exceeds 10%% max distance (%.1f%%) — clamping",
            raw_dist * 100,
        )
        return entry_price * 0.90 if direction == "LONG" else entry_price * 1.10
    if raw_dist < 0.005:
        logger.warning(
            "AI SL update below 0.5%% min distance (%.2f%%) — expanding",
            raw_dist * 100,
        )
        return entry_price * 0.995 if direction == "LONG" else entry_price * 1.005
    return new_sl
