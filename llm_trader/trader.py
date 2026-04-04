"""Execute futures trades on Binance based on LLM decisions."""

from __future__ import annotations

import logging
import sys
import os
from typing import Any

# Allow running this module from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.infrastructure.binance.rest_client import BinanceRestClient  # noqa: E402

from .config import LLMTraderConfig
from .llm_decision import Action, LLMDecision
from .models import OpenPosition

logger = logging.getLogger(__name__)

# Binance order types used for SL/TP
_STOP_MARKET = "STOP_MARKET"
_TP_MARKET = "TAKE_PROFIT_MARKET"


def _count_decimals(value_str: str) -> int:
    """Count meaningful decimal places in a string like '0.00100000'."""
    if "." not in value_str:
        return 0
    stripped = value_str.rstrip("0").rstrip(".")
    if "." not in stripped:
        return 0
    return len(stripped.split(".")[1])


class FuturesTrader:
    """Places market orders + SL/TP stop orders on Binance USDT-M futures."""

    def __init__(self, cfg: LLMTraderConfig) -> None:
        self._cfg = cfg
        self._client = BinanceRestClient(
            api_key=cfg.binance_api_key,
            api_secret=cfg.binance_api_secret,
            product="usdt_futures",
            testnet=cfg.binance_testnet,
            base_url=cfg.binance_base_url,
        )
        # Cached per-symbol precision: (price_decimals, qty_decimals)
        self._precision: dict[str, tuple[int, int]] = {}

    def close(self) -> None:
        self._client.close_connection()

    # ── Public API ────────────────────────────────────────────────────────────

    def setup_leverage(self, symbol: str, leverage: int) -> None:
        if self._cfg.dry_run:
            logger.info("[DRY-RUN] Would set leverage=%d for %s", leverage, symbol)
            return
        result = self._client.change_leverage(symbol=symbol, leverage=leverage)
        logger.info("Leverage set: %s", result)

    def get_open_position(self, symbol: str) -> OpenPosition:
        """Return a snapshot of the current position + active SL/TP orders."""
        side = "NONE"
        entry_price = liquidation_price = unrealized_pnl = qty = 0.0
        try:
            positions = self._client.get_open_positions(symbol)
            for p in positions:
                amt = float(p.get("positionAmt", 0))
                if amt == 0:
                    continue
                side = "LONG" if amt > 0 else "SHORT"
                qty = abs(amt)
                entry_price = float(p.get("entryPrice", 0))
                liquidation_price = float(p.get("liquidationPrice", 0))
                unrealized_pnl = float(p.get("unRealizedProfit", 0))
                break
        except Exception as exc:
            logger.warning("Could not fetch position for %s: %s", symbol, exc)

        current_sl: float | None = None
        current_tp: float | None = None
        try:
            open_orders = self._client.get_open_orders(symbol)
            for o in open_orders:
                otype = o.get("type", "")
                stop_price = float(o.get("stopPrice", 0))
                if otype == _STOP_MARKET and stop_price > 0:
                    current_sl = stop_price
                elif otype == _TP_MARKET and stop_price > 0:
                    current_tp = stop_price
        except Exception as exc:
            logger.warning("Could not fetch open orders for %s: %s", symbol, exc)

        return OpenPosition(
            side=side,
            entry_price=entry_price,
            quantity=qty,
            unrealized_pnl=unrealized_pnl,
            liquidation_price=liquidation_price,
            current_sl=current_sl,
            current_tp=current_tp,
        )

    def get_usdt_balance(self) -> float:
        """Return available USDT balance from the futures account."""
        try:
            account = self._client.get_account()
            for asset in account.get("assets", []):
                if asset.get("asset") == "USDT":
                    return float(asset.get("walletBalance", 0))
        except Exception as exc:
            logger.warning("Could not fetch account balance: %s", exc)
        return 0.0

    def execute(
        self,
        symbol: str,
        decision: LLMDecision,
        position: OpenPosition,
        current_price: float,
    ) -> float | None:
        """Translate an LLM decision into Binance futures orders.

        Returns estimated realized P&L if a position was closed, else None.
        """
        logger.info(
            "[%s] LLM → %s  sl=%s  tp=%s  confidence=%d/10  reason=%s",
            symbol, decision.action.value,
            decision.sl, decision.tp,
            decision.confidence, decision.reason,
        )

        # ── HOLD ──────────────────────────────────────────────────────────────
        if decision.action is Action.HOLD:
            if position.is_open and (decision.sl or decision.tp):
                is_long = position.side == "LONG"
                sl, tp = self._validate_sl_tp(is_long, current_price, decision.sl, decision.tp)
                logger.info("[%s] HOLD — updating SL/TP on existing %s position", symbol, position.side)
                self._cancel_sl_tp_orders(symbol)
                self._place_sl_tp(symbol, position.side, sl, tp)
            else:
                logger.info("[%s] HOLD — no changes.", symbol)
            return None

        # ── BUY / SELL ────────────────────────────────────────────────────────
        is_long = decision.action is Action.BUY
        sl, tp = self._validate_sl_tp(is_long, current_price, decision.sl, decision.tp)

        # Cancel existing SL/TP before touching the position
        self._cancel_sl_tp_orders(symbol)

        realized_pnl: float | None = None

        if decision.action is Action.BUY:
            if position.side == "LONG":
                # Already long — reaffirming; just refresh SL/TP.
                logger.info("[%s] Already LONG — skipping entry, updating SL/TP only.", symbol)
                self._place_sl_tp(symbol, "LONG", sl, tp)
                return None

            new_qty = self._calc_quantity(symbol, current_price)
            if new_qty is None:
                logger.error("[%s] Cannot compute quantity — aborting cycle.", symbol)
                return None

            if position.side == "SHORT":
                # Atomic flip: single order closes short + opens long
                flip_qty = position.quantity + new_qty
                realized_pnl = (position.entry_price - current_price) * position.quantity
                logger.info(
                    "[%s] Flipping SHORT→LONG: close %.6f + open %.6f = %.6f",
                    symbol, position.quantity, new_qty, flip_qty,
                )
                self._market_order(symbol, "BUY", flip_qty)
            else:
                self._market_order(symbol, "BUY", new_qty)

            self._place_sl_tp(symbol, "LONG", sl, tp)

        elif decision.action is Action.SELL:
            if position.side == "SHORT":
                # Already short — reaffirming; just refresh SL/TP.
                logger.info("[%s] Already SHORT — skipping entry, updating SL/TP only.", symbol)
                self._place_sl_tp(symbol, "SHORT", sl, tp)
                return None

            new_qty = self._calc_quantity(symbol, current_price)
            if new_qty is None:
                logger.error("[%s] Cannot compute quantity — aborting cycle.", symbol)
                return None

            if position.side == "LONG":
                # Atomic flip: single order closes long + opens short
                flip_qty = position.quantity + new_qty
                realized_pnl = (current_price - position.entry_price) * position.quantity
                logger.info(
                    "[%s] Flipping LONG→SHORT: close %.6f + open %.6f = %.6f",
                    symbol, position.quantity, new_qty, flip_qty,
                )
                self._market_order(symbol, "SELL", flip_qty)
            else:
                self._market_order(symbol, "SELL", new_qty)

            self._place_sl_tp(symbol, "SHORT", sl, tp)

        return realized_pnl

    # ── SL / TP validation ───────────────────────────────────────────────────

    def _validate_sl_tp(
        self,
        is_long: bool,
        current_price: float,
        sl: float | None,
        tp: float | None,
    ) -> tuple[float, float]:
        """Validate and fix SL/TP so they are on the correct side of price.

        Returns (validated_sl, validated_tp).  Applies:
        - Side check: SL on loss side, TP on profit side
        - Minimum distance: 0.1% (avoids immediate trigger from spread)
        - Maximum distance: 10% SL, 20% TP (prevents unreasonable levels)
        - Fallback defaults: 1.5% SL, 3% TP when values are missing/invalid
        """
        default_sl_pct = 0.015
        default_tp_pct = 0.030
        min_sl_pct = 0.001
        max_sl_pct = 0.10
        min_tp_pct = 0.002
        max_tp_pct = 0.20

        # Detect swapped SL/TP and fix
        if sl is not None and tp is not None:
            if is_long and sl > current_price and tp < current_price:
                sl, tp = tp, sl
                logger.warning("SL/TP were swapped for LONG — corrected")
            elif not is_long and sl < current_price and tp > current_price:
                sl, tp = tp, sl
                logger.warning("SL/TP were swapped for SHORT — corrected")

        if is_long:
            # SL below price, TP above price
            if sl is None or sl >= current_price:
                sl = current_price * (1 - default_sl_pct)
                logger.warning("SL invalid for LONG → default %.2f", sl)
            if tp is None or tp <= current_price:
                tp = current_price * (1 + default_tp_pct)
                logger.warning("TP invalid for LONG → default %.2f", tp)

            # Clamp SL distance
            sl_dist = (current_price - sl) / current_price
            if sl_dist < min_sl_pct:
                sl = current_price * (1 - min_sl_pct)
            elif sl_dist > max_sl_pct:
                sl = current_price * (1 - max_sl_pct)

            # Clamp TP distance
            tp_dist = (tp - current_price) / current_price
            if tp_dist < min_tp_pct:
                tp = current_price * (1 + min_tp_pct)
            elif tp_dist > max_tp_pct:
                tp = current_price * (1 + max_tp_pct)
        else:
            # SHORT: SL above price, TP below price
            if sl is None or sl <= current_price:
                sl = current_price * (1 + default_sl_pct)
                logger.warning("SL invalid for SHORT → default %.2f", sl)
            if tp is None or tp >= current_price:
                tp = current_price * (1 - default_tp_pct)
                logger.warning("TP invalid for SHORT → default %.2f", tp)

            sl_dist = (sl - current_price) / current_price
            if sl_dist < min_sl_pct:
                sl = current_price * (1 + min_sl_pct)
            elif sl_dist > max_sl_pct:
                sl = current_price * (1 + max_sl_pct)

            tp_dist = (current_price - tp) / current_price
            if tp_dist < min_tp_pct:
                tp = current_price * (1 - min_tp_pct)
            elif tp_dist > max_tp_pct:
                tp = current_price * (1 - max_tp_pct)

        return sl, tp

    # ── SL / TP placement ────────────────────────────────────────────────────

    def _place_sl_tp(
        self,
        symbol: str,
        position_side: str,
        sl: float,
        tp: float,
    ) -> None:
        """Place STOP_MARKET (SL) and TAKE_PROFIT_MARKET (TP) close-only orders."""
        close_side = "SELL" if position_side == "LONG" else "BUY"
        self._stop_order(symbol, close_side, _STOP_MARKET, sl, label="SL")
        self._stop_order(symbol, close_side, _TP_MARKET, tp, label="TP")

    def _stop_order(
        self, symbol: str, side: str, order_type: str, stop_price: float, label: str,
    ) -> None:
        rounded_price = self._round_price(symbol, stop_price)
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "stopPrice": rounded_price,
            "closePosition": "true",
            "timeInForce": "GTE_GTC",
        }
        if self._cfg.dry_run:
            logger.info("[DRY-RUN] %s order: %s @ %s", label, side, rounded_price)
            return
        try:
            result = self._client.create_order(**params)
            logger.info("%s order placed — id=%s price=%s", label, result.get("orderId"), rounded_price)
        except Exception as exc:
            logger.error("Failed to place %s order at %s: %s", label, rounded_price, exc)

    def _cancel_sl_tp_orders(self, symbol: str) -> None:
        """Cancel all open orders for the symbol (clears stale SL/TP)."""
        if self._cfg.dry_run:
            logger.info("[DRY-RUN] Would cancel all open orders for %s", symbol)
            return
        try:
            self._client.cancel_all_open_orders(symbol)
            logger.info("Cancelled all open orders for %s", symbol)
        except Exception as exc:
            logger.warning("Could not cancel open orders for %s: %s", symbol, exc)

    # ── Market order ─────────────────────────────────────────────────────────

    def _market_order(self, symbol: str, side: str, quantity: float) -> str | None:
        """Place a market order with explicit quantity (already computed)."""
        rounded_qty = self._round_qty(symbol, quantity)
        if float(rounded_qty) <= 0:
            logger.error("Computed quantity rounds to zero for %s — skipping order.", symbol)
            return None

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": rounded_qty,
        }

        if self._cfg.dry_run:
            logger.info("[DRY-RUN] %s %s qty=%s", side, symbol, rounded_qty)
            return f"DRY-RUN:{side}:{rounded_qty}"

        result = self._client.create_order(**params)
        order_id = result.get("orderId", "?")
        logger.info("Order placed — id=%s side=%s qty=%s", order_id, side, rounded_qty)
        return str(order_id)

    # ── Precision helpers ────────────────────────────────────────────────────

    def _ensure_precision(self, symbol: str) -> None:
        """Fetch and cache tick_size / step_size from exchange info."""
        if symbol in self._precision:
            return
        try:
            info = self._client.get_exchange_info(symbol)
            for sym_info in info.get("symbols", []):
                if sym_info["symbol"] == symbol:
                    price_dec = 2
                    qty_dec = 3
                    for f in sym_info.get("filters", []):
                        if f["filterType"] == "PRICE_FILTER":
                            price_dec = _count_decimals(f.get("tickSize", "0.01"))
                        elif f["filterType"] == "LOT_SIZE":
                            qty_dec = _count_decimals(f.get("stepSize", "0.001"))
                    self._precision[symbol] = (price_dec, qty_dec)
                    logger.debug("Precision for %s: price=%d qty=%d", symbol, price_dec, qty_dec)
                    return
        except Exception as exc:
            logger.warning("Could not fetch exchange info for %s: %s — using defaults", symbol, exc)
        self._precision[symbol] = (2, 3)

    def _round_price(self, symbol: str, price: float) -> str:
        self._ensure_precision(symbol)
        decimals = self._precision[symbol][0]
        return f"{price:.{decimals}f}"

    def _round_qty(self, symbol: str, qty: float) -> str:
        self._ensure_precision(symbol)
        decimals = self._precision[symbol][1]
        return f"{qty:.{decimals}f}"

    # ── Quantity calculation ─────────────────────────────────────────────────

    def _calc_quantity(self, symbol: str, current_price: float | None = None) -> float | None:
        """Compute order quantity from config. Returns None on failure (no fallback)."""
        price = current_price
        if price is None or price <= 0:
            try:
                ticker = self._client.get_ticker(symbol)
                price = float(ticker.get("lastPrice") or ticker.get("price") or 0)
            except Exception as exc:
                logger.error("Cannot fetch price for %s: %s", symbol, exc)
                return None
        if price <= 0:
            logger.error("Invalid price for %s: %s", symbol, price)
            return None

        raw_qty = (self._cfg.order_usdt * self._cfg.leverage) / price
        rounded = float(self._round_qty(symbol, raw_qty))
        if rounded <= 0:
            logger.error("Quantity rounds to zero for %s (raw=%.10f)", symbol, raw_qty)
            return None
        return rounded
