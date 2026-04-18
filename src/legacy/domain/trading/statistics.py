"""TradingStatistics domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

from src.mcp_servers.shared.utils.data_utils import SerializableMixin


@dataclass(slots=True)
class TradingStatistics(SerializableMixin):
    """All-time cumulative trading statistics.

    Updated after every closed trade to provide the AI with
    performance context for position sizing decisions.
    """

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    total_pnl_quote: float = 0.0
    initial_capital: float = 10000.0
    current_capital: float = 10000.0
    avg_trade_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict with ISO-format datetime."""
        d = super().to_dict()
        if isinstance(d.get("last_updated"), datetime):
            d["last_updated"] = d["last_updated"].isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingStatistics":
        """Deserialize from dict, parsing ISO-format datetime."""
        if "last_updated" in data and isinstance(data["last_updated"], str):
            try:
                data = {
                    **data,
                    "last_updated": datetime.fromisoformat(data["last_updated"]),
                }
            except ValueError:
                pass
        return super().from_dict(data)  # type: ignore[return-value]


__all__ = ["TradingStatistics"]
