"""Tests for the pure backtest simulator.

Exercises the math on synthetic candles so we can assert PnL direction,
fee accounting, drawdown, and trade bookkeeping without touching any
feed.
"""

from __future__ import annotations

import math

import pytest

from src.mcp_servers.analysis_mcp.backtest import (
    run_backtest,
    signal_to_position,
)
from src.mcp_servers.shared.domain.market import OHLCVCandle


def _candles(prices: list[float], start_ts_ms: int = 0, step_ms: int = 3_600_000):
    return [
        OHLCVCandle(
            timestamp=start_ts_ms + i * step_ms,
            open=p,
            high=p,
            low=p,
            close=p,
            volume=1.0,
        )
        for i, p in enumerate(prices)
    ]


def test_signal_to_position_maps_correctly():
    assert signal_to_position("STRONG_BUY", "long_short") == 1
    assert signal_to_position("BUY", "long_short") == 1
    assert signal_to_position("NEUTRAL", "long_short") == 0
    assert signal_to_position("SELL", "long_short") == -1
    assert signal_to_position("STRONG_SELL", "long_short") == -1


def test_long_only_clamps_shorts_to_zero():
    assert signal_to_position("SELL", "long_only") == 0
    assert signal_to_position("STRONG_SELL", "long_only") == 0
    assert signal_to_position("BUY", "long_only") == 1


def test_constant_uptrend_long_wins():
    candles = _candles([100, 110, 121, 133.1])  # 10% per bar
    signals = ["BUY"] * 4
    r = run_backtest(
        candles,
        signals,
        timeframe="1h",
        fee_bps=0.0,
        slippage_bps=0.0,
        direction="long_short",
    )
    assert r.metrics.total_return_pct == pytest.approx(33.1, rel=1e-6)
    assert r.metrics.num_trades == 1
    assert r.trades[0].side == "LONG"
    assert r.metrics.max_drawdown_pct == 0.0


def test_constant_downtrend_short_wins():
    candles = _candles([100, 90, 81, 72.9])
    signals = ["SELL"] * 4
    r = run_backtest(
        candles,
        signals,
        timeframe="1h",
        fee_bps=0.0,
        slippage_bps=0.0,
        direction="long_short",
    )
    assert r.metrics.total_return_pct > 0.0
    assert r.trades[0].side == "SHORT"


def test_long_only_sits_out_downtrend():
    candles = _candles([100, 90, 81])
    signals = ["SELL"] * 3
    r = run_backtest(
        candles,
        signals,
        timeframe="1h",
        fee_bps=0.0,
        slippage_bps=0.0,
        direction="long_only",
    )
    assert r.metrics.total_return_pct == 0.0
    assert r.metrics.num_trades == 0
    assert r.metrics.time_in_market_pct == 0.0


def test_fees_reduce_return():
    candles = _candles([100, 110])
    signals = ["BUY", "BUY"]
    free = run_backtest(
        candles,
        signals,
        timeframe="1h",
        fee_bps=0.0,
        slippage_bps=0.0,
        direction="long_short",
    )
    costly = run_backtest(
        candles,
        signals,
        timeframe="1h",
        fee_bps=100.0,
        slippage_bps=0.0,
        direction="long_short",
    )
    assert costly.metrics.total_return_pct < free.metrics.total_return_pct


def test_whipsaw_pays_turnover_twice():
    # Flat prices but the signal flips every bar — pure cost drag.
    candles = _candles([100, 100, 100, 100])
    signals = ["BUY", "SELL", "BUY", "SELL"]
    r = run_backtest(
        candles,
        signals,
        timeframe="1h",
        fee_bps=10.0,
        slippage_bps=0.0,
        direction="long_short",
    )
    assert r.metrics.total_return_pct < 0.0
    # First bar opens a position (dpos=1). Then 2 flips from +1 to -1 and back
    # (dpos=2 each). Final turnover = 1 + 2 + 2 = 5.
    assert r.metrics.turnover == pytest.approx(5.0)


def test_max_drawdown_reflects_peak_to_trough():
    # Long through +10%, then -20%. Peak=1.10, trough=0.88, dd=-20%.
    candles = _candles([100, 110, 88])
    signals = ["BUY", "BUY", "BUY"]
    r = run_backtest(
        candles,
        signals,
        timeframe="1h",
        fee_bps=0.0,
        slippage_bps=0.0,
        direction="long_short",
    )
    assert r.metrics.max_drawdown_pct == pytest.approx(-20.0, rel=1e-6)


def test_mismatched_lengths_raises():
    candles = _candles([100, 101, 102])
    with pytest.raises(ValueError, match="length mismatch"):
        run_backtest(
            candles,
            ["BUY", "BUY"],
            timeframe="1h",
            fee_bps=0.0,
            slippage_bps=0.0,
            direction="long_short",
        )


def test_trade_closed_at_final_bar():
    # A position opened late and never flipped should still be booked.
    candles = _candles([100, 100, 110, 120])
    signals = ["NEUTRAL", "BUY", "BUY", "BUY"]
    r = run_backtest(
        candles,
        signals,
        timeframe="1h",
        fee_bps=0.0,
        slippage_bps=0.0,
        direction="long_short",
    )
    assert r.metrics.num_trades == 1
    t = r.trades[0]
    assert t.side == "LONG"
    assert t.entry_price == 100.0
    assert t.exit_price == 120.0
    assert t.pnl_pct == pytest.approx(20.0, rel=1e-6)


def test_sharpe_is_zero_when_no_variance():
    candles = _candles([100, 100, 100])
    signals = ["NEUTRAL", "NEUTRAL", "NEUTRAL"]
    r = run_backtest(
        candles,
        signals,
        timeframe="1h",
        fee_bps=0.0,
        slippage_bps=0.0,
        direction="long_short",
    )
    assert r.metrics.sharpe == 0.0
    assert not math.isnan(r.metrics.sharpe)
