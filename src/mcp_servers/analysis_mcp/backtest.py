"""Pure backtest simulator for a bar-by-bar directional signal stream.

No I/O. No feed calls. Takes candles + per-bar signals + cost assumptions,
returns metrics and trade list. Kept separate from ``service.py`` so it is
trivially unit-testable on synthetic candles.
"""

from dataclasses import dataclass, field
from math import sqrt
from typing import Literal

from src.mcp_servers.shared.domain.market import OHLCVCandle


Direction = Literal["long_short", "long_only"]


_BARS_PER_YEAR: dict[str, float] = {
    "1m": 525_600.0,
    "5m": 105_120.0,
    "15m": 35_040.0,
    "30m": 17_520.0,
    "1h": 8_760.0,
    "2h": 4_380.0,
    "4h": 2_190.0,
    "6h": 1_460.0,
    "8h": 1_095.0,
    "12h": 730.0,
    "1d": 365.0,
}


_SIGNAL_TO_POS: dict[str, int] = {
    "STRONG_BUY": 1,
    "BUY": 1,
    "NEUTRAL": 0,
    "SELL": -1,
    "STRONG_SELL": -1,
}


@dataclass(slots=True)
class BacktestTrade:
    entry_ts: int
    exit_ts: int
    side: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    pnl_pct: float


@dataclass(slots=True)
class BacktestMetrics:
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    turnover: float
    time_in_market_pct: float
    equity_curve: list[float] = field(default_factory=list)


@dataclass(slots=True)
class BacktestResult:
    metrics: BacktestMetrics
    trades: list[BacktestTrade]
    bars_evaluated: int


def signal_to_position(signal: str, direction: Direction) -> int:
    """Map a Signal enum value to a target position in {-1, 0, 1}."""
    pos = _SIGNAL_TO_POS.get(signal, 0)
    if direction == "long_only" and pos < 0:
        return 0
    return pos


def run_backtest(
    candles: list[OHLCVCandle],
    signals: list[str],
    *,
    timeframe: str,
    fee_bps: float,
    slippage_bps: float,
    direction: Direction,
) -> BacktestResult:
    """Simulate close-to-close PnL for ``signals`` over ``candles``.

    ``signals[i]`` is the directional call produced at the close of
    ``candles[i]`` and is held through the next bar. The first bar of
    the returned equity curve is always 1.0 (no exposure yet).

    Fees and slippage are charged on every position change, proportional
    to ``|dpos|`` in basis points of notional. A flip from +1 to -1 pays
    the cost twice.
    """
    if len(candles) != len(signals):
        raise ValueError(
            f"candles/signals length mismatch: {len(candles)} vs {len(signals)}"
        )
    if len(candles) < 2:
        raise ValueError("need at least 2 candles to compute bar returns")

    cost_per_unit = (fee_bps + slippage_bps) / 10_000.0

    equity = 1.0
    curve: list[float] = [1.0]
    bar_returns: list[float] = []
    turnover = 0.0
    bars_in_market = 0

    trades: list[BacktestTrade] = []
    open_entry_ts: int | None = None
    open_entry_price: float | None = None
    open_side: Literal["LONG", "SHORT"] | None = None

    prev_pos = 0

    for i in range(1, len(candles)):
        # Signal was generated at the close of bar i-1, applied over
        # [close_{i-1}, close_i]. That aligns the pnl with the
        # information set available when the decision was made.
        target_pos = signal_to_position(signals[i - 1], direction)

        prev_close = candles[i - 1].close
        this_close = candles[i].close
        gross_ret = (this_close - prev_close) / prev_close if prev_close else 0.0

        dpos = target_pos - prev_pos
        cost = cost_per_unit * abs(dpos)
        net_ret = target_pos * gross_ret - cost

        equity *= 1.0 + net_ret
        curve.append(equity)
        bar_returns.append(net_ret)
        turnover += abs(dpos)
        if target_pos != 0:
            bars_in_market += 1

        # Trade bookkeeping: a trade spans from opening a non-zero
        # position to the bar we flip it flat (or reverse).
        if dpos != 0:
            if prev_pos != 0 and open_entry_price is not None:
                trades.append(
                    BacktestTrade(
                        entry_ts=open_entry_ts or 0,
                        exit_ts=candles[i - 1].timestamp,
                        side=open_side or "LONG",
                        entry_price=open_entry_price,
                        exit_price=prev_close,
                        pnl_pct=(
                            (prev_close - open_entry_price) / open_entry_price
                            if open_side == "LONG"
                            else (open_entry_price - prev_close) / open_entry_price
                        )
                        * 100.0
                        - cost_per_unit * 2 * 100.0,
                    )
                )
                open_entry_ts = None
                open_entry_price = None
                open_side = None
            if target_pos != 0:
                open_entry_ts = candles[i - 1].timestamp
                open_entry_price = prev_close
                open_side = "LONG" if target_pos > 0 else "SHORT"

        prev_pos = target_pos

    # Close any position still open at the end of the series.
    if prev_pos != 0 and open_entry_price is not None:
        final_close = candles[-1].close
        trades.append(
            BacktestTrade(
                entry_ts=open_entry_ts or 0,
                exit_ts=candles[-1].timestamp,
                side=open_side or "LONG",
                entry_price=open_entry_price,
                exit_price=final_close,
                pnl_pct=(
                    (final_close - open_entry_price) / open_entry_price
                    if open_side == "LONG"
                    else (open_entry_price - final_close) / open_entry_price
                )
                * 100.0
                - cost_per_unit * 2 * 100.0,
            )
        )

    metrics = _compute_metrics(
        equity_curve=curve,
        bar_returns=bar_returns,
        trades=trades,
        turnover=turnover,
        bars_in_market=bars_in_market,
        bars_evaluated=len(candles) - 1,
        timeframe=timeframe,
    )
    return BacktestResult(
        metrics=metrics, trades=trades, bars_evaluated=len(candles) - 1
    )


def _compute_metrics(
    *,
    equity_curve: list[float],
    bar_returns: list[float],
    trades: list[BacktestTrade],
    turnover: float,
    bars_in_market: int,
    bars_evaluated: int,
    timeframe: str,
) -> BacktestMetrics:
    final_equity = equity_curve[-1]
    total_return_pct = (final_equity - 1.0) * 100.0

    bars_per_year = _BARS_PER_YEAR.get(timeframe, 8_760.0)
    years = bars_evaluated / bars_per_year if bars_evaluated else 0.0
    if years > 0 and final_equity > 0:
        try:
            cagr_pct = (final_equity ** (1.0 / years) - 1.0) * 100.0
        except OverflowError:
            cagr_pct = float("inf") if final_equity > 1.0 else -100.0
    else:
        cagr_pct = 0.0

    # Sharpe: annualized, zero risk-free rate. Uses bar-level sample std.
    if len(bar_returns) > 1:
        mean_r = sum(bar_returns) / len(bar_returns)
        var_r = sum((r - mean_r) ** 2 for r in bar_returns) / (len(bar_returns) - 1)
        std_r = sqrt(var_r)
        sharpe = (mean_r / std_r) * sqrt(bars_per_year) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown: largest peak-to-trough decline on the equity curve.
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (eq - peak) / peak
            if dd < max_dd:
                max_dd = dd
    max_drawdown_pct = max_dd * 100.0

    wins = sum(1 for t in trades if t.pnl_pct > 0)
    win_rate_pct = (wins / len(trades) * 100.0) if trades else 0.0

    time_in_market_pct = (
        (bars_in_market / bars_evaluated * 100.0) if bars_evaluated else 0.0
    )

    return BacktestMetrics(
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        sharpe=sharpe,
        max_drawdown_pct=max_drawdown_pct,
        win_rate_pct=win_rate_pct,
        num_trades=len(trades),
        turnover=turnover,
        time_in_market_pct=time_in_market_pct,
        equity_curve=equity_curve,
    )
