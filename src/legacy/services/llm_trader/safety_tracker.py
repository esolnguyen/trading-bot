"""Safety tracker for the LLM trader — kill switch on consecutive losses / daily drawdown."""

from __future__ import annotations

from datetime import date


class SafetyTracker:
    """Track consecutive losses and daily P&L for kill-switch decisions."""

    def __init__(self, max_consecutive_losses: int, max_daily_loss_pct: float) -> None:
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_loss_pct = max_daily_loss_pct
        self.consecutive_losses: int = 0
        self._day: date | None = None
        self._day_start_balance: float | None = None

    def update_daily_baseline(self, balance: float) -> None:
        today = date.today()
        if self._day != today:
            self._day = today
            self._day_start_balance = balance
            self.consecutive_losses = 0  # reset streak on new day

    def check(self, current_balance: float) -> tuple[bool, str]:
        """Return (should_halt, reason)."""
        if self.consecutive_losses >= self.max_consecutive_losses:
            return True, (
                f"consecutive losses: {self.consecutive_losses}/{self.max_consecutive_losses}"
            )
        if self._day_start_balance and self._day_start_balance > 0:
            daily_pnl_pct = (current_balance - self._day_start_balance) / self._day_start_balance
            if daily_pnl_pct <= -self.max_daily_loss_pct:
                return True, f"daily loss {daily_pnl_pct:.2%} (max -{self.max_daily_loss_pct:.1%})"
        return False, ""

    def record_close(self, pnl: float) -> None:
        if pnl < 0:
            self.consecutive_losses += 1
        elif pnl > 0:
            self.consecutive_losses = 0
        # pnl == 0: breakeven, don't change streak
