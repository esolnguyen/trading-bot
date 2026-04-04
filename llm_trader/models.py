"""Shared data models used across llm_trader modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpenPosition:
    """Snapshot of the current futures position for a single symbol."""
    side: str             # "LONG", "SHORT", or "NONE"
    entry_price: float    # 0.0 when no position
    quantity: float       # absolute value (contracts)
    unrealized_pnl: float
    liquidation_price: float
    current_sl: float | None  # price of the active STOP_MARKET order, if any
    current_tp: float | None  # price of the active TAKE_PROFIT_MARKET order, if any

    @property
    def is_open(self) -> bool:
        return self.side != "NONE" and self.quantity > 0
