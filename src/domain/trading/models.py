"""Trading domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from src.shared.data_utils import SerializableMixin


class Action(str, Enum):
    """Trading action emitted by the decision layer."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    UPDATE = "UPDATE"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


@dataclass(slots=True)
class TradeDecision:
    """Validated trade decision from the LLM or fallback layer (executor path)."""

    symbol: str
    action: Action
    quantity: float = 0.0
    order_type: str = "MARKET"
    price: float | None = None
    reasoning: str = ""
    confidence: float = 0.0
    timestamp: int = 0
    source: str = "azure"


@dataclass(slots=True)
class TradeOutcome:
    """Execution outcome for a trade decision."""

    decision: TradeDecision
    order_id: str | None = None
    executed_price: float | None = None
    pnl_usdt: float | None = None
    dry_run: bool = False
    timestamp: int = 0


@dataclass(slots=True)
class RiskValidationResult:
    """Risk-manager output containing the final decision and execution mode."""

    decision: TradeDecision
    dry_run: bool = False
    status: str = "passed"


# ---------------------------------------------------------------------------
# Position lifecycle models (LLM_trader-style trading with SL/TP)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Position(SerializableMixin):
    """Represents an active trading position with SL/TP and brain-learning metadata."""

    entry_price: float
    stop_loss: float
    take_profit: float
    size: float  # Quantity in base currency (e.g., BTC)
    entry_time: datetime
    confidence: str  # HIGH, MEDIUM, LOW
    direction: str   # LONG, SHORT
    symbol: str
    # Confluence factors at entry for brain factor-performance learning
    confluence_factors: tuple = field(default_factory=tuple)  # type: ignore[assignment]
    # Transaction details
    entry_fee: float = 0.0
    quote_amount: float = 0.0
    size_pct: float = 0.0
    # Market conditions at entry for brain learning
    atr_at_entry: float = 0.0
    volatility_level: str = "MEDIUM"    # HIGH, MEDIUM, LOW
    sl_distance_pct: float = 0.0
    tp_distance_pct: float = 0.0
    rr_ratio_at_entry: float = 0.0
    adx_at_entry: float = 0.0
    rsi_at_entry: float = 50.0
    trend_direction_at_entry: str = "NEUTRAL"      # BULLISH/BEARISH/NEUTRAL
    macd_signal_at_entry: str = "NEUTRAL"
    bb_position_at_entry: str = "MIDDLE"           # UPPER/MIDDLE/LOWER
    volume_state_at_entry: str = "NORMAL"          # ACCUMULATION/NORMAL/DISTRIBUTION
    market_sentiment_at_entry: str = "NEUTRAL"
    # Performance metrics
    max_drawdown_pct: float = 0.0   # Max adverse excursion (MAE)
    max_profit_pct: float = 0.0     # Max favorable excursion (MFE)

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.direction == "LONG":
            return ((current_price - self.entry_price) / self.entry_price) * 100
        return ((self.entry_price - current_price) / self.entry_price) * 100

    def update_metrics(self, current_price: float) -> None:
        """Update live performance metrics (MAE/MFE)."""
        pnl = self.calculate_pnl(current_price)
        if pnl < 0 and pnl < self.max_drawdown_pct:
            self.max_drawdown_pct = pnl
        if pnl > 0 and pnl > self.max_profit_pct:
            self.max_profit_pct = pnl

    def calculate_closing_fee(self, close_price: float, fee_percent: float) -> float:
        """Calculate the transaction fee for closing this position."""
        return close_price * self.size * fee_percent

    def is_stop_hit(self, current_price: float) -> bool:
        """Check if stop loss is hit."""
        if self.direction == "LONG":
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def is_target_hit(self, current_price: float) -> bool:
        """Check if take profit is hit."""
        if self.direction == "LONG":
            return current_price >= self.take_profit
        return current_price <= self.take_profit


@dataclass(slots=True)
class TradeRecord(SerializableMixin):
    """LLM-trader-style trade record used for position tracking and memory.

    Distinct from TradeDecision which is used by the executor/risk path.
    """

    timestamp: datetime
    symbol: str
    action: str   # BUY, SELL, HOLD, CLOSE, CLOSE_LONG, CLOSE_SHORT, UPDATE
    confidence: str  # HIGH, MEDIUM, LOW
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: float = 0.0   # AI's suggested percentage of capital (0.0-1.0)
    quote_amount: float = 0.0    # Invested quote currency amount (e.g. USDT)
    quantity: float = 0.0        # Actual quantity in base currency (e.g., BTC)
    fee: float = 0.0             # Transaction fee in quote currency
    reasoning: str = ""


@dataclass(slots=True)
class RiskAssessment(SerializableMixin):
    """Calculated risk parameters for a trade entry."""

    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    size_pct: float
    quote_amount: float
    entry_fee: float
    sl_distance_pct: float
    tp_distance_pct: float
    rr_ratio: float
    volatility_level: str


@dataclass(slots=True)
class ClosedTradeResult(SerializableMixin):
    """Result of a closed trade for statistics calculation."""

    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_quote: float
    quantity: float
    direction: str  # LONG, SHORT


@dataclass(slots=True)
class TradingMemory(SerializableMixin):
    """Rolling memory of recent trading decisions for context injection."""

    decisions: List[TradeRecord] = field(default_factory=list)
    max_decisions: int = 10

    def add_decision(self, decision: TradeRecord) -> None:
        """Add a decision to memory, maintaining max size."""
        self.decisions.append(decision)
        if len(self.decisions) > self.max_decisions:
            self.decisions.pop(0)

    def get_recent_decisions(self, n: int = 5) -> List[TradeRecord]:
        """Return the n most recent decisions."""
        return self.decisions[-n:]

    def get_context_summary(self, full_history: Optional[List[TradeRecord]] = None) -> str:
        """Generate a concise summary for prompt injection."""
        if not self.decisions:
            return "No previous trading decisions."

        def _ensure_utc(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        recent_source = full_history if full_history else self.decisions
        recent = recent_source[-5:]
        lines: List[str] = []
        if recent:
            lines.append("## Recent Trading History (Last 5 Decisions):")

        history_to_analyze = full_history if full_history else self.decisions
        history_to_analyze = sorted(history_to_analyze, key=lambda x: _ensure_utc(x.timestamp))

        total_pnl_quote = 0.0
        total_pnl_pct = 0.0
        closed_trades = 0
        winning_trades = 0
        open_position: Optional[TradeRecord] = None

        for decision in history_to_analyze:
            if decision.action in ("BUY", "SELL"):
                open_position = decision
            elif decision.action in ("CLOSE", "CLOSE_LONG", "CLOSE_SHORT") and open_position:
                if open_position.action == "BUY":
                    pnl_pct = ((decision.price - open_position.price) / open_position.price) * 100
                    pnl_quote = (decision.price - open_position.price) * open_position.quantity
                else:
                    pnl_pct = ((open_position.price - decision.price) / open_position.price) * 100
                    pnl_quote = (open_position.price - decision.price) * open_position.quantity
                total_pnl_quote += pnl_quote
                total_pnl_pct += pnl_pct
                closed_trades += 1
                if pnl_pct > 0:
                    winning_trades += 1
                open_position = None

        for decision in recent:
            time_str = decision.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"- [{time_str}] {decision.action} @ ${decision.price:,.2f} "
                f"(Conf: {decision.confidence}) - {decision.reasoning}"
            )

        if closed_trades > 0:
            avg_pnl_pct = total_pnl_pct / closed_trades
            win_rate = (winning_trades / closed_trades) * 100
            lines.append("")
            lines.append(f"## Overall Performance ({closed_trades} Total Closed Trades):")
            lines.append(f"- Total P&L: ${total_pnl_quote:+,.2f} ({total_pnl_pct:+.2f}%)")
            lines.append(f"- Average P&L per Trade: {avg_pnl_pct:+.2f}%")
            lines.append(f"- Win Rate: {win_rate:.1f}% ({winning_trades}/{closed_trades} trades)")

        return "\n".join(lines)

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dicts for JSON serialization."""
        return [d.to_dict() for d in self.decisions]

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]], max_decisions: int = 10) -> "TradingMemory":
        """Create TradingMemory from list of dictionaries."""
        memory = cls(max_decisions=max_decisions)
        for item in data:
            memory.decisions.append(TradeRecord.from_dict(item))
        return memory
