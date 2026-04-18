"""Console notifier with colored cycle output."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text


class ConsoleNotifier:
    """Emit concise colored cycle summaries to the console."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def notify_cycle(
        self,
        *,
        cycle: int,
        timestamp_iso: str,
        symbol_signals: list[tuple[str, str]],
        final_decision: str,
    ) -> None:
        text = Text()
        text.append(f"Cycle {cycle} ", style="bold cyan")
        text.append(f"{timestamp_iso} ", style="white")
        for symbol, signal in symbol_signals:
            text.append(f"{symbol}:{signal} ", style=self._signal_style(signal))
        text.append(f"Decision:{final_decision}", style=self._signal_style(final_decision))
        self.console.print(text)

    def notify_position_opened(
        self,
        *,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        size_pct: float,
        confidence: str,
    ) -> None:
        text = Text()
        text.append("POSITION OPENED  ", style="bold green" if direction == "LONG" else "bold red")
        text.append(f"{symbol} {direction} ", style="bold white")
        text.append(f"@ ${entry_price:,.4f}  ", style="white")
        text.append(f"SL:${stop_loss:,.4f} ", style="bold red")
        text.append(f"TP:${take_profit:,.4f} ", style="bold green")
        text.append(f"Size:{size_pct:.1%} Conf:{confidence}", style="bold cyan")
        self.console.print(text)

    def notify_position_closed(
        self,
        *,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        reason: str = "manual",
    ) -> None:
        text = Text()
        pnl_style = "bold green" if pnl_pct >= 0 else "bold red"
        text.append("POSITION CLOSED  ", style=pnl_style)
        text.append(f"{symbol} {direction} ", style="bold white")
        text.append(f"entry:${entry_price:,.4f} → exit:${exit_price:,.4f}  ", style="white")
        text.append(f"PnL:{pnl_pct:+.2f}%  ", style=pnl_style)
        text.append(f"Reason:{reason}", style="bold yellow")
        self.console.print(text)

    @staticmethod
    def _signal_style(value: str) -> str:
        if "BUY" in value:
            return "bold green"
        if "SELL" in value:
            return "bold red"
        return "bold yellow"
