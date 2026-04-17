"""Position-context formatter used by the prompt builders."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.domain.trading.models import Position


def format_position_context(
    position: Optional[Position],
    capital: float,
    current_price: Optional[float] = None,
    currency: str = "USDT",
) -> str:
    """Return a Markdown-formatted capital + position summary for prompts."""
    if not position:
        return (
            f"## Capital Status\n"
            f"- Total Capital: ${capital:,.2f} {currency}\n"
            f"- Available: ${capital:,.2f} (100%)\n\n"
            f"CURRENT POSITION: None"
        )

    now = datetime.now(timezone.utc)
    entry_time = position.entry_time
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)
    duration = now - entry_time
    hours = duration.total_seconds() / 3600
    allocated = position.quote_amount
    available = capital - allocated
    allocation_pct = (allocated / capital) * 100 if capital > 0 else 0

    lines = [
        "## Capital Status",
        f"- Total Capital: ${capital:,.2f} {currency}",
        f"- Allocated: ${allocated:,.2f} ({allocation_pct:.1f}%)",
        f"- Available: ${available:,.2f} ({100 - allocation_pct:.1f}%)",
        "",
        "## Current Position",
        f"- Direction: {position.direction}",
        f"- Symbol: {position.symbol}",
        f"- Entry Price: ${position.entry_price:,.2f}",
    ]
    if current_price and current_price > 0:
        lines.append(f"- Current Price: ${current_price:,.2f}")
    lines.extend([
        f"- Stop Loss: ${position.stop_loss:,.2f}",
        f"- Take Profit: ${position.take_profit:,.2f}",
        f"- Position Size: {position.size_pct * 100:.2f}%",
        f"- Quantity: {position.size:.6f}",
        f"- Entry Fee: ${position.entry_fee:.4f}",
        f"- Duration: {hours:.1f} hours",
        f"- Confidence: {position.confidence}",
    ])
    if current_price and current_price > 0:
        pnl_pct = position.calculate_pnl(current_price)
        pnl_quote = (
            (current_price - position.entry_price) * position.size
            if position.direction == "LONG"
            else (position.entry_price - current_price) * position.size
        )
        lines.append(
            f"- Unrealized P&L: {pnl_pct:+.2f}% (${pnl_quote:+,.2f} {currency})"
        )
    return "\n".join(lines)
