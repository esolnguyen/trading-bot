"""LLM-in-the-loop backtest harness.

Walks historical Binance candles, periodically asks the LLM for a
BUY/SELL/HOLD decision via ``LLMDecisionEngine``, and simulates the
resulting trade with realistic crypto perpetual mechanics:

  * Maker/Taker fees, slippage on entry and exit
  * Stop-loss / take-profit triggered intra-bar by ``high``/``low``
  * Funding fee deducted at 00:00 / 08:00 / 16:00 UTC
  * Forced liquidation when (margin + unrealised) ≤ maintenance margin

Usage:

    python -m src.services.backtest.backtester --symbol BTCUSDT --days 30
    python -m src.services.backtest.backtester --symbol ETHUSDT --start 2025-01-01 --end 2025-02-01 --mock
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from src.mcp_servers.config.llm_trader_settings import LLMTraderConfig
from src.legacy.domain.trading.backtest_metrics import Trade, calc_metrics
from src.legacy.domain.trading.models import Action, OpenPosition
from src.mcp_servers.shared.infrastructure.binance.candles import Candle
from src.mcp_servers.shared.infrastructure.binance.client import (
    build_client,
    close_client,
)
from src.mcp_servers.shared.infrastructure.binance.historical_data import fetch_history
from src.legacy.services.llm_trader.llm_decision import LLMDecision, LLMDecisionEngine

logger = logging.getLogger(__name__)

DecideFn = Callable[[str, list[Candle], Optional[OpenPosition]], LLMDecision]


# ── Tiered maintenance margin ─────────────────────────────────────────────────

_TIER_TABLE: list[tuple[float, float]] = [
    (100_000, 0.004),
    (500_000, 0.006),
    (1_000_000, 0.01),
    (5_000_000, 0.02),
    (10_000_000, 0.05),
    (float("inf"), 0.10),
]


def _maintenance_rate(notional_usd: float) -> float:
    for tier_max, rate in _TIER_TABLE:
        if notional_usd <= tier_max:
            return rate
    return _TIER_TABLE[-1][1]


_FUNDING_HOURS = {0, 8, 16}


# ── Internal mutable position state ──────────────────────────────────────────


@dataclasses.dataclass
class _PosState:
    direction: int  # 1 long, -1 short
    entry_price: float  # post-slippage
    entry_ts: str
    size: float  # contracts (raw quantity)
    leverage: float
    margin: float  # locked collateral
    entry_commission: float
    sl: float | None
    tp: float | None
    entry_bar_idx: int


# ── Bar-level mechanics ───────────────────────────────────────────────────────


def _slip(price: float, direction: int, rate: float) -> float:
    return price * (1 + direction * rate)


def _commission(size: float, price: float, rate: float) -> float:
    return size * price * rate


def _bars_per_year(interval: str) -> int:
    """Crypto trades 24/7, so 365 days × bars-per-day."""
    per_day = {
        "1m": 1440,
        "3m": 480,
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "2h": 12,
        "4h": 6,
        "6h": 4,
        "12h": 2,
        "1d": 1,
    }.get(interval, 96)
    return 365 * per_day


def _parse_iso(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# ── Backtester ────────────────────────────────────────────────────────────────


class Backtester:
    """Drives an LLM decision function across historical candles."""

    def __init__(
        self,
        cfg: LLMTraderConfig,
        symbol: str,
        candles: list[Candle],
        decide_fn: DecideFn,
        warmup: int = 100,
    ) -> None:
        if len(candles) <= warmup:
            raise ValueError(f"Need more than {warmup} candles, got {len(candles)}")
        self.cfg = cfg
        self.symbol = symbol
        self.candles = candles
        self.decide_fn = decide_fn
        self.warmup = warmup

        self.cash: float = cfg.backtest_initial_cash
        self.equity_curve: list[float] = []
        self.trades: list[Trade] = []
        self.pos: _PosState | None = None
        self._funding_done: set[tuple[str, int]] = set()
        self._last_decision: LLMDecision | None = None

    # ── PnL helpers ──

    def _unrealized(self, mark: float) -> float:
        if self.pos is None:
            return 0.0
        return self.pos.direction * self.pos.size * (mark - self.pos.entry_price)

    def _equity(self, mark: float) -> float:
        if self.pos is None:
            return self.cash
        return self.cash + self.pos.margin + self._unrealized(mark)

    # ── Open / close ──

    def _snapshot_position(self, mark: float) -> OpenPosition:
        if self.pos is None:
            return OpenPosition(
                side="NONE",
                entry_price=0.0,
                quantity=0.0,
                unrealized_pnl=0.0,
                liquidation_price=0.0,
                current_sl=None,
                current_tp=None,
            )
        return OpenPosition(
            side="LONG" if self.pos.direction == 1 else "SHORT",
            entry_price=self.pos.entry_price,
            quantity=self.pos.size,
            unrealized_pnl=self._unrealized(mark),
            liquidation_price=0.0,
            current_sl=self.pos.sl,
            current_tp=self.pos.tp,
        )

    def _open(
        self,
        direction: int,
        bar: Candle,
        ts: str,
        bar_idx: int,
        sl: float | None,
        tp: float | None,
    ) -> None:
        slipped = _slip(bar.open, direction, self.cfg.backtest_slippage)
        leverage = float(self.cfg.leverage or 1)
        notional = self.cfg.order_usdt * leverage
        size = notional / slipped
        if size <= 0:
            return
        margin = size * slipped / leverage
        comm = _commission(size, slipped, self.cfg.backtest_taker_fee)
        if margin + comm > self.cash:
            available = max(self.cash - comm, 0.0)
            if available <= 0:
                return
            size = (available * leverage) / slipped
            margin = size * slipped / leverage
            comm = _commission(size, slipped, self.cfg.backtest_taker_fee)
        self.cash -= margin + comm
        self.pos = _PosState(
            direction=direction,
            entry_price=slipped,
            entry_ts=ts,
            size=size,
            leverage=leverage,
            margin=margin,
            entry_commission=comm,
            sl=sl,
            tp=tp,
            entry_bar_idx=bar_idx,
        )

    def _close(
        self,
        exit_price: float,
        exit_ts: str,
        reason: str,
        bar_idx: int,
    ) -> None:
        if self.pos is None:
            return
        slipped = _slip(exit_price, -self.pos.direction, self.cfg.backtest_slippage)
        pnl_gross = (
            self.pos.direction * self.pos.size * (slipped - self.pos.entry_price)
        )
        exit_comm = _commission(self.pos.size, slipped, self.cfg.backtest_maker_fee)
        pnl = pnl_gross - exit_comm
        pnl_pct = pnl / self.pos.margin * 100 if self.pos.margin > 1e-9 else 0.0
        self.cash += self.pos.margin + pnl
        self.trades.append(
            Trade(
                direction=self.pos.direction,
                entry_price=self.pos.entry_price,
                exit_price=slipped,
                entry_ts=self.pos.entry_ts,
                exit_ts=exit_ts,
                size=self.pos.size,
                leverage=self.pos.leverage,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason=reason,
                holding_bars=max(bar_idx - self.pos.entry_bar_idx, 0),
                commission=self.pos.entry_commission + exit_comm,
            )
        )
        self.pos = None

    # ── Bar hooks: SL/TP, funding, liquidation ──

    def _check_sl_tp(self, bar: Candle, ts: str, bar_idx: int) -> bool:
        """Return True if position was closed by SL/TP this bar."""
        if self.pos is None:
            return False
        sl, tp, d = self.pos.sl, self.pos.tp, self.pos.direction
        sl_hit = sl is not None and (
            (d == 1 and bar.low <= sl) or (d == -1 and bar.high >= sl)
        )
        tp_hit = tp is not None and (
            (d == 1 and bar.high >= tp) or (d == -1 and bar.low <= tp)
        )
        if sl_hit and tp_hit:
            self._close(sl, ts, "sl", bar_idx)
            return True
        if sl_hit:
            self._close(sl, ts, "sl", bar_idx)
            return True
        if tp_hit:
            self._close(tp, ts, "tp", bar_idx)
            return True
        return False

    def _apply_funding(self, bar: Candle, ts: str) -> None:
        if self.pos is None:
            return
        dt = _parse_iso(ts)
        if dt.hour not in _FUNDING_HOURS:
            return
        key = (dt.strftime("%Y-%m-%d"), dt.hour)
        if key in self._funding_done:
            return
        self._funding_done.add(key)
        notional = self.pos.size * bar.close
        fee = notional * self.cfg.backtest_funding_rate * self.pos.direction
        self.cash -= fee

    def _check_liquidation(self, bar: Candle, ts: str, bar_idx: int) -> None:
        if self.pos is None or self.pos.leverage <= 1.0:
            return
        unrealized = self._unrealized(bar.close)
        notional = self.pos.size * bar.close
        maint = notional * _maintenance_rate(notional)
        if self.pos.margin + unrealized <= maint:
            liq_price = _slip(
                bar.close, -self.pos.direction, self.cfg.backtest_slippage
            )
            self._close(liq_price, ts, "liquidation", bar_idx)

    # ── Decision handling ──

    def _apply_decision(
        self,
        decision: LLMDecision,
        bar: Candle,
        ts: str,
        bar_idx: int,
    ) -> None:
        target_dir = (
            1
            if decision.action is Action.BUY
            else -1 if decision.action is Action.SELL else 0
        )

        if decision.confidence < self.cfg.min_confidence and target_dir != 0:
            target_dir = 0  # forced HOLD

        if self.pos is not None:
            if target_dir == 0:
                if decision.sl is not None:
                    self.pos.sl = decision.sl
                if decision.tp is not None:
                    self.pos.tp = decision.tp
                return
            if target_dir != self.pos.direction:
                self._close(bar.open, ts, "signal", bar_idx)
            else:
                if decision.sl is not None:
                    self.pos.sl = decision.sl
                if decision.tp is not None:
                    self.pos.tp = decision.tp
                return

        if target_dir != 0:
            self._open(target_dir, bar, ts, bar_idx, decision.sl, decision.tp)

    # ── Main loop ──

    def run(self) -> dict:
        eval_every = max(self.cfg.backtest_eval_every_n_bars, 1)
        window_size = self.cfg.kline_limit

        for i in range(self.warmup, len(self.candles)):
            bar = self.candles[i]
            ts = bar.timestamp

            self._apply_funding(bar, ts)
            self._check_liquidation(bar, ts, i)
            closed = self._check_sl_tp(bar, ts, i)

            if (i - self.warmup) % eval_every == 0:
                window = self.candles[max(0, i - window_size + 1) : i + 1]
                snap = self._snapshot_position(bar.close)
                try:
                    decision = self.decide_fn(self.symbol, window, snap)
                    self._last_decision = decision
                    self._apply_decision(decision, bar, ts, i)
                except Exception as exc:
                    logger.warning("decide_fn failed at %s: %s", ts, exc)

            self.equity_curve.append(self._equity(bar.close))

            if closed:
                continue

        if self.pos is not None:
            last = self.candles[-1]
            self._close(last.close, last.timestamp, "end", len(self.candles) - 1)
            if self.equity_curve:
                self.equity_curve[-1] = self._equity(last.close)

        return calc_metrics(
            self.equity_curve,
            self.trades,
            self.cfg.backtest_initial_cash,
            _bars_per_year(self.cfg.kline_interval),
        )


# ── Mock decide function ──────────────────────────────────────────────────────


def _mock_decide(
    symbol: str,
    candles: list[Candle],
    position: OpenPosition | None,
) -> LLMDecision:
    """Trivial momentum baseline: long if last close > SMA(20), else short."""
    if len(candles) < 20:
        return LLMDecision(Action.HOLD, None, None, 5, "warmup", "")
    sma = sum(c.close for c in candles[-20:]) / 20
    last = candles[-1].close
    atr = (max(c.high for c in candles[-14:]) - min(c.low for c in candles[-14:])) / 14
    if last > sma:
        return LLMDecision(
            Action.BUY,
            sl=last - 2 * atr,
            tp=last + 3 * atr,
            confidence=6,
            reason="last>sma20",
            raw_response="",
        )
    return LLMDecision(
        Action.SELL,
        sl=last + 2 * atr,
        tp=last - 3 * atr,
        confidence=6,
        reason="last<sma20",
        raw_response="",
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM-in-the-loop backtest for llm_trader",
    )
    p.add_argument("--symbol", default=None)
    p.add_argument(
        "--interval", default=None, help="Kline interval (default from config)"
    )
    p.add_argument("--start", default=None, help="YYYY-MM-DD UTC")
    p.add_argument("--end", default=None, help="YYYY-MM-DD UTC")
    p.add_argument(
        "--days",
        type=int,
        default=None,
        help="Lookback window in days (mutually exclusive with --start/--end)",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic SMA-momentum baseline instead of the LLM",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Call the LLM every N bars (default from config)",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        stream=sys.stdout,
    )

    cfg = LLMTraderConfig()
    overrides: dict = {}
    if args.symbol:
        overrides["symbol"] = args.symbol
    if args.interval:
        overrides["kline_interval"] = args.interval
    if args.eval_every is not None:
        overrides["backtest_eval_every_n_bars"] = args.eval_every
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    symbol = cfg.symbol.upper()

    if args.days is not None and (args.start or args.end):
        sys.exit("--days is mutually exclusive with --start/--end")
    if args.days is not None:
        end_dt = datetime.now(tz=timezone.utc)
        start_dt = end_dt - timedelta(days=args.days)
    else:
        if not args.start or not args.end:
            sys.exit("Provide either --days N or both --start and --end")
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    logger.info(
        "Backtest %s %s  %s → %s",
        symbol,
        cfg.kline_interval,
        start_dt.date(),
        end_dt.date(),
    )

    market = build_client(cfg)
    try:
        candles = fetch_history(market, symbol, cfg.kline_interval, start_ms, end_ms)
    finally:
        close_client(market)

    if len(candles) < 200:
        sys.exit(f"Not enough candles fetched: {len(candles)}")

    if args.mock:
        decide_fn: DecideFn = _mock_decide
        logger.info("Using mock SMA-momentum decide_fn (no LLM calls)")
    else:
        engine = LLMDecisionEngine(cfg)
        decide_fn = engine.decide

    bt = Backtester(cfg, symbol, candles, decide_fn)
    metrics = bt.run()

    print(
        json.dumps(
            {
                "symbol": symbol,
                "interval": cfg.kline_interval,
                "bars": len(candles),
                "trades": len(bt.trades),
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
