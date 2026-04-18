"""Exchange-native bracket (SL/TP) order placement and cancellation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.mcp_servers.shared.infrastructure.binance.client import is_demo_fapi

from .filters import ExchangeFilters

logger = logging.getLogger(__name__)


async def place_bracket_orders(
    client: Any,
    settings: Any,
    filters: ExchangeFilters,
    symbol: str,
    entry_side: str,
    sl_price: float,
    tp_price: float,
    quantity: float = 0.0,
) -> tuple[str | None, str | None]:
    """Place exchange-native SL and TP orders to protect an open position.

    Places STOP_MARKET + TAKE_PROFIT_MARKET with ``closePosition=true``
    (no quantity needed).

    On ``demo-fapi.binance.com`` conditional order types are rejected by
    ``/fapi/v1/order`` with ``-4120``. Bracket placement is skipped there
    and the position monitor in ``trading_loop.run_position_monitor``
    enforces the stored ``sl_price``/``tp_price`` in software (15 s poll).

    Returns ``(sl_order_id, tp_order_id)``; either may be None if the
    individual placement fails — the error is logged and the position monitor
    will fall back to software SL management.
    """
    close_side = "SELL" if entry_side == "BUY" else "BUY"

    if is_demo_fapi(settings):
        logger.info(
            "Skipping exchange bracket orders for %s on demo-fapi "
            "(conditional types unsupported) — software SL/TP active via "
            "position monitor: sl=%.6f tp=%.6f",
            symbol,
            sl_price,
            tp_price,
        )
        return None, None

    sl_stop_str = await filters.format_price(symbol, sl_price, client)
    tp_stop_str = await filters.format_price(symbol, tp_price, client)

    sl_order_id = await _place_futures_bracket(
        client,
        filters,
        symbol,
        close_side,
        "STOP_MARKET",
        sl_stop_str,
        quantity,
        "SL",
    )
    tp_order_id = await _place_futures_bracket(
        client,
        filters,
        symbol,
        close_side,
        "TAKE_PROFIT_MARKET",
        tp_stop_str,
        quantity,
        "TP",
    )
    return sl_order_id, tp_order_id


async def cancel_bracket_orders(
    client: Any,
    symbol: str,
    sl_order_id: str | None,
    tp_order_id: str | None,
) -> None:
    """Cancel surviving SL/TP bracket orders.

    Called before a bot-managed close or when reconciliation detects that the
    exchange already filled one side.  Silently ignores already-filled orders.
    """
    if not sl_order_id and not tp_order_id:
        return
    for oid, label in ((sl_order_id, "SL"), (tp_order_id, "TP")):
        if not oid:
            continue
        try:
            await asyncio.to_thread(
                client.cancel_order, symbol=symbol, orderId=int(oid)
            )
            logger.info("Cancelled %s bracket order %s for %s", label, oid, symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not cancel %s bracket %s for %s: %s", label, oid, symbol, exc
            )


async def _place_futures_bracket(
    client: Any,
    filters: ExchangeFilters,
    symbol: str,
    side: str,
    order_type: str,
    stop_str: str,
    quantity: float,
    label: str,
) -> str | None:
    """Place a single futures bracket order (SL or TP) on Binance UM.

    Strategy:
      1. STOP_MARKET / TAKE_PROFIT_MARKET with ``closePosition=true`` and
         ``workingType=MARK_PRICE`` (avoids transient -4120 from contract
         price wicks on testnet).
      2. On -4061 (position-side mismatch — hedge mode is on), retry with
         explicit ``positionSide`` derived from the entry side.
      3. On -4131 / "reduceOnly Order is rejected" (closePosition not
         accepted on this endpoint), retry once with ``reduceOnly=true``
         + quantity.
      4. Anything else: log the full context loudly so the next live run
         surfaces a real diagnostic instead of silently dropping the SL.
    """
    qty_str = (
        await filters.format_quantity(symbol, quantity, client)
        if quantity > 0
        else None
    )
    position_side = "LONG" if side == "SELL" else "SHORT"

    base_params: dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "stopPrice": stop_str,
        "workingType": "MARK_PRICE",
        "priceProtect": "TRUE",
    }

    async def _try(
        params: dict[str, Any], variant: str
    ) -> tuple[str | None, str | None]:
        try:
            data = await asyncio.to_thread(client.new_order, **params)
            oid = str(data.get("orderId", "")) or None
            logger.info(
                "Futures %s bracket placed (%s): %s type=%s stopPrice=%s order=%s",
                label,
                variant,
                symbol,
                order_type,
                stop_str,
                oid,
            )
            return oid, None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

    params = {**base_params, "closePosition": "true"}
    oid, err1 = await _try(params, "closePosition")
    if oid:
        return oid

    if "-4061" in (err1 or "") or "position side" in (err1 or "").lower():
        params = {**base_params, "closePosition": "true", "positionSide": position_side}
        oid, err2 = await _try(params, f"closePosition+positionSide={position_side}")
        if oid:
            return oid
        err1 = err2 or err1

    if qty_str is not None:
        params = {
            **base_params,
            "quantity": qty_str,
            "reduceOnly": "true",
        }
        oid, err3 = await _try(params, "reduceOnly")
        if oid:
            return oid
        if "-4061" in (err3 or "") or "position side" in (err3 or "").lower():
            params["positionSide"] = position_side
            params.pop("reduceOnly", None)
            oid, err4 = await _try(params, f"reduceOnly+positionSide={position_side}")
            if oid:
                return oid
            err1 = err4 or err3 or err1
        else:
            err1 = err3 or err1

    logger.warning(
        "Failed to place futures %s bracket for %s — software SL/TP active "
        "via position monitor (15 s poll). "
        "side=%s type=%s stopPrice=%s qty=%s last_error=%s",
        label,
        symbol,
        side,
        order_type,
        stop_str,
        qty_str,
        err1,
    )
    return None
