"""LIMIT-order polling and price extraction helpers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.legacy.domain.trading import Action, TradeOutcome

logger = logging.getLogger(__name__)


def extract_price(data: dict[str, Any]) -> float | None:
    """Pull the fill price from a Binance order response.

    Binance MARKET orders return ``"price": "0.00000000"``; we fall back to
    ``avgPrice``, then to ``cumQuote / executedQty``, then to the first fill.
    Returns ``None`` when no positive value is present.
    """
    for key in ("price", "avgPrice"):
        raw = data.get(key)
        if raw is not None:
            try:
                val = float(raw)
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass
    cum_quote = data.get("cumQuote") or data.get("cumBase")
    exec_qty = data.get("executedQty")
    if cum_quote is not None and exec_qty is not None:
        try:
            cq, eq = float(cum_quote), float(exec_qty)
            if cq > 0 and eq > 0:
                return cq / eq
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    for fill in data.get("fills") or []:
        raw = fill.get("price")
        if raw is not None:
            try:
                val = float(raw)
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass
    return None


async def await_limit_fill(
    outcome: TradeOutcome,
    client: Any,
    *,
    timeout_seconds: int = 300,
    poll_interval: int = 30,
) -> TradeOutcome:
    """Poll until a LIMIT order fills or the timeout elapses, then cancel.

    On fill, returns a new TradeOutcome with the filled price.
    On terminal non-fill statuses or timeout, returns an outcome whose decision
    is coerced to HOLD so downstream position bookkeeping is a no-op.
    """
    elapsed = 0
    symbol = outcome.decision.symbol
    if not outcome.order_id:
        logger.warning(
            "await_limit_fill called with empty order_id — returning outcome unchanged"
        )
        return outcome
    order_id_int = int(outcome.order_id)

    while elapsed < timeout_seconds:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        try:
            status_data = await asyncio.to_thread(
                client.query_order,
                symbol=symbol,
                orderId=order_id_int,
            )
            status = status_data.get("status", "")
            if status == "FILLED":
                filled_price = extract_price(status_data)
                logger.info("LIMIT order %s filled @ %s", order_id_int, filled_price)
                return TradeOutcome(
                    decision=outcome.decision,
                    order_id=outcome.order_id,
                    executed_price=filled_price,
                    pnl_usdt=None,
                    dry_run=False,
                    timestamp=outcome.timestamp,
                )
            if status in ("CANCELED", "EXPIRED", "REJECTED"):
                logger.warning(
                    "LIMIT order %s ended with status %s", order_id_int, status
                )
                return TradeOutcome(
                    decision=outcome.decision.__class__(
                        symbol=outcome.decision.symbol,
                        action=Action.HOLD,
                        reasoning=f"limit_order_{status.lower()}",
                    ),
                    order_id=outcome.order_id,
                    dry_run=False,
                    timestamp=outcome.timestamp,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error polling LIMIT order %s: %s", order_id_int, exc)

    logger.warning(
        "LIMIT order %s timed out after %ss — cancelling", order_id_int, timeout_seconds
    )
    try:
        await asyncio.to_thread(
            client.cancel_order,
            symbol=symbol,
            orderId=order_id_int,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to cancel timed-out LIMIT order %s: %s", order_id_int, exc)
    return TradeOutcome(
        decision=outcome.decision.__class__(
            symbol=outcome.decision.symbol,
            action=Action.HOLD,
            reasoning="limit_order_timeout",
        ),
        order_id=outcome.order_id,
        dry_run=False,
        timestamp=outcome.timestamp,
    )
