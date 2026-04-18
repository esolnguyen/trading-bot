"""Performance metrics for backtest output.

Slim set of calculations a single-symbol LLM-in-the-loop backtest consumes
(no benchmarks, no per-symbol breakdowns, no annualisation tables).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Trade:
    """A completed round-trip simulated trade."""

    direction: int          # 1 long, -1 short
    entry_price: float
    exit_price: float
    entry_ts: str
    exit_ts: str
    size: float
    leverage: float
    pnl: float              # realised P&L in USDT (after commissions)
    pnl_pct: float          # P&L as % of margin
    exit_reason: str        # signal | sl | tp | liquidation | end
    holding_bars: int
    commission: float


def _stdev(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def trade_stats(trades: list[Trade]) -> dict[str, Any]:
    """Win rate, profit factor, max consecutive loss, etc."""
    if not trades:
        return {
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0,
            "profit_factor": 0.0,
            "max_consecutive_loss": 0,
            "avg_holding_bars": 0.0,
        }
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]

    win_rate = len(wins) / len(trades)
    avg_win = _mean(wins)
    avg_loss = abs(_mean(losses)) or 1e-10
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 1e-10 else 0.0

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) or 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else 0.0

    max_consec = cur = 0
    for t in trades:
        if t.pnl < 0:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0

    hold = [t.holding_bars for t in trades if t.holding_bars > 0]

    return {
        "win_rate": round(win_rate, 4),
        "profit_loss_ratio": round(profit_loss_ratio, 4),
        "profit_factor": round(profit_factor, 4),
        "max_consecutive_loss": max_consec,
        "avg_holding_bars": round(_mean(hold), 1),
    }


def calc_metrics(
    equity_curve: list[float],
    trades: list[Trade],
    initial_cash: float,
    bars_per_year: int,
) -> dict[str, Any]:
    """Full performance summary for a single equity curve.

    Args:
        equity_curve: Equity at each bar of the backtest.
        trades: Completed round-trip trades.
        initial_cash: Starting capital.
        bars_per_year: For annualisation. Crypto 15m ≈ 365*96 = 35040.
    """
    if not equity_curve:
        return {
            "final_value": initial_cash, "total_return": 0.0,
            "annual_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
            "calmar": 0.0, "sortino": 0.0, "trade_count": 0,
            "win_rate": 0.0, "profit_loss_ratio": 0.0,
            "profit_factor": 0.0, "max_consecutive_loss": 0,
            "avg_holding_bars": 0.0,
        }

    n = len(equity_curve)
    final = equity_curve[-1]
    total_ret = final / initial_cash - 1
    ann_ret = (1 + total_ret) ** (bars_per_year / max(n, 1)) - 1

    rets = [
        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        if equity_curve[i - 1] > 0
        else 0.0
        for i in range(1, n)
    ]
    vol = _stdev(rets)
    sharpe = (_mean(rets) / (vol + 1e-10)) * math.sqrt(bars_per_year) if vol > 0 else 0.0

    downside = [r for r in rets if r < 0]
    dvol = _stdev(downside) if len(downside) > 1 else 1e-10
    sortino = (_mean(rets) / (dvol + 1e-10)) * math.sqrt(bars_per_year) if dvol > 0 else 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    out = {
        "final_value": round(final, 4),
        "total_return": round(total_ret, 6),
        "annual_return": round(ann_ret, 6),
        "max_drawdown": round(max_dd, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "trade_count": len(trades),
    }
    out.update(trade_stats(trades))
    return out
