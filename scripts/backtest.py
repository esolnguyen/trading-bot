#!/usr/bin/env python3
"""Simple event-driven backtester.

Replays a local OHLCV CSV through the full indicator + pattern + ML pipeline,
simulates LLM decisions via a configurable rule-based mock, and reports P&L.

Usage:
    python scripts/backtest.py --csv data/ohlcv/btcusdt_15m.csv --symbol BTCUSDT
    python scripts/backtest.py --csv data/ohlcv/ethusdt_15m.csv --symbol ETHUSDT \\
        --sl 0.015 --tp 0.03 --confidence 60

The mock LLM uses the same signals that IndicatorCalculator + TechnicalAnalyzer
produce (RSI, MACD, EMA, Choppiness).  To inject a real LLM, subclass
``MockLLM`` and override ``decide()``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ohlcv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "timestamp": float(row["timestamp"]),
                "open":   float(row["open"]),
                "high":   float(row["high"]),
                "low":    float(row["low"]),
                "close":  float(row["close"]),
                "volume": float(row["volume"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Lightweight indicator calculation (mirrors IndicatorCalculator)
# ---------------------------------------------------------------------------

def _ema(closes: list[float], period: int) -> float:
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
    return ema


def _rsi(closes: list[float], period: int = 14) -> float:
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    avg_g  = sum(gains[-period:]) / period
    avg_l  = sum(losses[-period:]) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - 100.0 / (1.0 + rs)


def _choppiness(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    import math
    window_h = highs[-period:]
    window_l = lows[-period:]
    tr_sum = sum(
        max(window_h[i] - window_l[i], abs(window_h[i] - closes[-period + i - 1]))
        for i in range(period)
    )
    hl_range = max(window_h) - min(window_l)
    if hl_range == 0 or tr_sum == 0:
        return 50.0
    return 100.0 * math.log10(tr_sum / hl_range) / math.log10(period)


def _macd_hist(closes: list[float]) -> float:
    fast = _ema(closes, 12)
    slow = _ema(closes, 26)
    line = fast - slow
    signal = _ema(closes[-35:], 9)
    return line - signal


# ---------------------------------------------------------------------------
# Mock LLM — rule-based signal generator
# ---------------------------------------------------------------------------

class MockLLM:
    """Deterministic signal generator using indicator thresholds.

    Subclass and override ``decide()`` to plug in a real LLM for walk-forward
    testing.
    """

    def __init__(
        self,
        symbol: str,
        sl_pct: float = 0.02,
        tp_pct: float = 0.04,
        chop_threshold: float = 61.8,
        min_confidence: float = 0.0,
    ) -> None:
        self.symbol = symbol
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.chop_threshold = chop_threshold
        self.min_confidence = min_confidence

    def decide(
        self,
        closes: list[float],
        highs: list[float],
        lows: list[float],
    ) -> tuple[str, float]:
        """Return (action, confidence) where action in BUY/SELL/HOLD."""
        if len(closes) < 55:
            return "HOLD", 0.0

        rsi    = _rsi(closes[-20:])
        macd_h = _macd_hist(closes[-40:])
        ema20  = _ema(closes, 20)
        ema50  = _ema(closes, 50)
        chop   = _choppiness(highs, lows, closes)
        price  = closes[-1]

        if chop > self.chop_threshold:
            return "HOLD", 0.0

        bullish = sum([
            rsi < 40,
            macd_h > 0,
            price > ema20,
            ema20 > ema50,
        ])
        bearish = sum([
            rsi > 60,
            macd_h < 0,
            price < ema20,
            ema20 < ema50,
        ])

        if bullish >= 3:
            confidence = 50.0 + bullish * 10.0
            return "BUY", min(confidence, 95.0)
        if bearish >= 3:
            confidence = 50.0 + bearish * 10.0
            return "SELL", min(confidence, 95.0)
        return "HOLD", 0.0


# ---------------------------------------------------------------------------
# Position tracker
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    direction: str      # BUY or SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_bar: int
    size: float = 1.0   # normalised size (1 unit = 1 USDT at entry)


@dataclass
class ClosedTrade:
    direction: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_usdt: float
    bars_held: int
    close_reason: str


# ---------------------------------------------------------------------------
# Backtester engine
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl_pct: float = 0.0
    total_pnl_usdt: float = 0.0
    max_drawdown_pct: float = 0.0
    trades: list[ClosedTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades else 0.0

    @property
    def avg_pnl_pct(self) -> float:
        return self.total_pnl_pct / self.total_trades if self.total_trades else 0.0

    def print_summary(self) -> None:
        print("\n" + "=" * 55)
        print("BACKTEST RESULTS")
        print("=" * 55)
        print(f"  Total trades   : {self.total_trades}")
        print(f"  Win rate       : {self.win_rate:.1%}")
        print(f"  Total P&L      : {self.total_pnl_usdt:+.2f} USDT  ({self.total_pnl_pct:+.2f}%)")
        print(f"  Avg P&L/trade  : {self.avg_pnl_pct:+.2f}%")
        print(f"  Max drawdown   : {self.max_drawdown_pct:.2f}%")
        print("=" * 55)


def run_backtest(
    rows: list[dict[str, float]],
    llm: MockLLM,
    warmup: int = 60,
    fee_pct: float = 0.00075,
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    result = BacktestResult()
    position: OpenPosition | None = None
    equity = initial_capital
    peak_equity = initial_capital

    closes  = [r["close"]  for r in rows]
    highs   = [r["high"]   for r in rows]
    lows    = [r["low"]    for r in rows]

    for i in range(warmup, len(rows)):
        price  = closes[i]
        equity_before_close = equity

        # Check stop-loss / take-profit
        if position is not None:
            close_reason: str | None = None
            if position.direction == "BUY":
                if price <= position.stop_loss:
                    close_reason = "stop_loss"
                elif price >= position.take_profit:
                    close_reason = "take_profit"
            else:  # SELL
                if price >= position.stop_loss:
                    close_reason = "stop_loss"
                elif price <= position.take_profit:
                    close_reason = "take_profit"

            if close_reason:
                if position.direction == "BUY":
                    pnl_pct = (price - position.entry_price) / position.entry_price * 100
                else:
                    pnl_pct = (position.entry_price - price) / position.entry_price * 100
                pnl_usdt = equity * (pnl_pct / 100.0)
                fee = equity * fee_pct * 2  # entry + exit fee
                pnl_usdt -= fee
                equity += pnl_usdt
                result.trades.append(ClosedTrade(
                    direction=position.direction,
                    entry_price=position.entry_price,
                    exit_price=price,
                    pnl_pct=pnl_pct,
                    pnl_usdt=pnl_usdt,
                    bars_held=i - position.entry_bar,
                    close_reason=close_reason,
                ))
                result.total_trades += 1
                result.total_pnl_pct += pnl_pct
                result.total_pnl_usdt += pnl_usdt
                if pnl_pct > 0:
                    result.winning_trades += 1
                position = None

        # Drawdown tracking
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity * 100
        if drawdown > result.max_drawdown_pct:
            result.max_drawdown_pct = drawdown

        result.equity_curve.append(equity)

        # Enter new position only when flat
        if position is None:
            action, confidence = llm.decide(closes[:i + 1], highs[:i + 1], lows[:i + 1])
            if action in ("BUY", "SELL") and confidence >= llm.min_confidence:
                if action == "BUY":
                    sl = price * (1 - llm.sl_pct)
                    tp = price * (1 + llm.tp_pct)
                else:
                    sl = price * (1 + llm.sl_pct)
                    tp = price * (1 - llm.tp_pct)
                position = OpenPosition(
                    direction=action,
                    entry_price=price,
                    stop_loss=sl,
                    take_profit=tp,
                    entry_bar=i,
                )

    # Close any open position at last price
    if position is not None:
        price = closes[-1]
        if position.direction == "BUY":
            pnl_pct = (price - position.entry_price) / position.entry_price * 100
        else:
            pnl_pct = (position.entry_price - price) / position.entry_price * 100
        pnl_usdt = equity * (pnl_pct / 100.0)
        equity += pnl_usdt
        result.trades.append(ClosedTrade(
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=price,
            pnl_pct=pnl_pct,
            pnl_usdt=pnl_usdt,
            bars_held=len(rows) - position.entry_bar,
            close_reason="end_of_data",
        ))
        result.total_trades += 1
        result.total_pnl_pct += pnl_pct
        result.total_pnl_usdt += pnl_usdt
        if pnl_pct > 0:
            result.winning_trades += 1

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Simple rule-based backtester")
    parser.add_argument("--csv",    required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol name (display only)")
    parser.add_argument("--sl",     type=float, default=0.02,  help="Stop-loss %% (default 0.02)")
    parser.add_argument("--tp",     type=float, default=0.04,  help="Take-profit %% (default 0.04)")
    parser.add_argument("--chop",   type=float, default=61.8,  help="Choppiness filter (default 61.8)")
    parser.add_argument("--confidence", type=float, default=0.0, help="Min confidence to enter (default 0)")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Starting capital USDT")
    parser.add_argument("--fee",    type=float, default=0.00075, help="Taker fee %% per side")
    parser.add_argument("--warmup", type=int,   default=60,    help="Warm-up candles before trading")
    parser.add_argument("--verbose", action="store_true", help="Print each trade")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_ohlcv(csv_path)
    print(f"Loaded {len(rows)} candles from {csv_path.name}")

    llm = MockLLM(
        symbol=args.symbol,
        sl_pct=args.sl,
        tp_pct=args.tp,
        chop_threshold=args.chop,
        min_confidence=args.confidence,
    )

    result = run_backtest(
        rows,
        llm,
        warmup=args.warmup,
        fee_pct=args.fee,
        initial_capital=args.capital,
    )

    if args.verbose:
        print("\nTrade log:")
        for t in result.trades:
            print(
                f"  {t.direction:4s} entry={t.entry_price:.2f} exit={t.exit_price:.2f} "
                f"pnl={t.pnl_pct:+.2f}% ({t.pnl_usdt:+.2f} USDT) [{t.close_reason}] "
                f"{t.bars_held}bars"
            )

    result.print_summary()


if __name__ == "__main__":
    main()
