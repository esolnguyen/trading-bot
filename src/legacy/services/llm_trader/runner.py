"""LLM trader — main entry point.

Usage:
    python -m src.services.llm_trader.runner                   # uses defaults from .env
    python -m src.services.llm_trader.runner --symbol ETHUSDT --leverage 5
    python -m src.services.llm_trader.runner --symbol BTCUSDT --dry-run
    python -m src.services.llm_trader.runner --once            # single evaluation, then exit
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from typing import Any

from src.mcp_servers.config.llm_trader_settings import LLMTraderConfig
from src.legacy.domain.trading.models import Action, OpenPosition
from src.mcp_servers.shared.infrastructure.binance.candles import fetch_latest_candles
from src.mcp_servers.shared.infrastructure.binance.client import (
    build_client,
    close_client,
)
from src.legacy.services.llm_trader.futures_trader import FuturesTrader
from src.legacy.services.llm_trader.llm_decision import LLMDecision, LLMDecisionEngine
from src.legacy.services.llm_trader.safety_tracker import SafetyTracker


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-driven futures trader")
    p.add_argument("--symbol", default=None, help="Trading symbol, e.g. BTCUSDT")
    p.add_argument(
        "--leverage", type=int, default=None, help="Futures leverage (1-125)"
    )
    p.add_argument(
        "--order-usdt",
        type=float,
        default=None,
        dest="order_usdt",
        help="Position size in USDT (before leverage)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Log orders but do not submit to Binance",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Opposite of --dry-run: actually submit orders",
    )
    p.add_argument(
        "--once", action="store_true", help="Run a single evaluation cycle and exit"
    )
    p.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Loop interval in seconds (default: 900 = 15 min)",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        stream=sys.stdout,
    )


# ── Cycle ─────────────────────────────────────────────────────────────────────


def _detect_inter_cycle_close(
    last_position: OpenPosition | None,
    current_position: OpenPosition,
    current_price: float,
    tracker: SafetyTracker,
    logger_: logging.Logger,
) -> None:
    """Detect and record P&L for positions closed by SL/TP between cycles."""
    if last_position is None or not last_position.is_open:
        return
    if current_position.is_open and current_position.side == last_position.side:
        return  # same position still open

    if last_position.side == "LONG":
        est_pnl = (current_price - last_position.entry_price) * last_position.quantity
    else:
        est_pnl = (last_position.entry_price - current_price) * last_position.quantity

    logger_.info(
        "Position closed between cycles (%s entry=%.2f) — estimated P&L: %+.4f USDT",
        last_position.side,
        last_position.entry_price,
        est_pnl,
    )
    tracker.record_close(est_pnl)


def _run_cycle(
    symbol: str,
    cfg: LLMTraderConfig,
    market: Any,
    engine: LLMDecisionEngine,
    trader: FuturesTrader,
    tracker: SafetyTracker,
    last_position: OpenPosition | None,
) -> OpenPosition | None:
    """Run one evaluation cycle. Returns the position state after execution."""
    logger_ = logging.getLogger("llm_trader")

    # 1. Fetch candles
    logger_.info(
        "── Fetching %d × %s candles for %s …",
        cfg.kline_limit,
        cfg.kline_interval,
        symbol,
    )
    candles = fetch_latest_candles(market, symbol, cfg.kline_interval, cfg.kline_limit)
    if not candles:
        logger_.error("No candles returned — skipping cycle.")
        return last_position

    current_price = candles[-1].close
    logger_.info(
        "Received %d closed candles (latest close: %s, price: %.2f)",
        len(candles),
        candles[-1].timestamp,
        current_price,
    )

    # 2. Fetch position + detect inter-cycle closes
    position = trader.get_open_position(symbol)
    logger_.info(
        "Position: side=%s  entry=%.4f  qty=%.4f  pnl=%+.4f  sl=%s  tp=%s",
        position.side,
        position.entry_price,
        position.quantity,
        position.unrealized_pnl,
        position.current_sl,
        position.current_tp,
    )
    _detect_inter_cycle_close(last_position, position, current_price, tracker, logger_)

    # 3. Kill switch check
    balance = trader.get_usdt_balance()
    if balance > 0:
        tracker.update_daily_baseline(balance)
        halted, reason = tracker.check(balance)
        if halted:
            logger_.warning("╔══ KILL SWITCH ══╗  %s  — skipping cycle", reason)
            return position

    # 4. LLM decision
    logger_.info("── Asking LLM for trading decision …")
    decision = engine.decide(symbol, candles, position)
    logger_.info(
        "LLM decision → action=%s  sl=%s  tp=%s  confidence=%d/10  reason=%s",
        decision.action.value,
        decision.sl,
        decision.tp,
        decision.confidence,
        decision.reason,
    )

    # 5. Confidence gate
    if decision.action is not Action.HOLD and decision.confidence < cfg.min_confidence:
        logger_.info(
            "Confidence %d < minimum %d → forcing HOLD",
            decision.confidence,
            cfg.min_confidence,
        )
        decision = LLMDecision(
            action=Action.HOLD,
            sl=decision.sl,
            tp=decision.tp,
            confidence=decision.confidence,
            reason=f"low_confidence ({decision.confidence}<{cfg.min_confidence}): {decision.reason}",
            raw_response=decision.raw_response,
        )

    # 6. Execute
    logger_.info("── Executing …")
    realized_pnl = trader.execute(symbol, decision, position, current_price)
    if realized_pnl is not None:
        logger_.info("Position closed — estimated P&L: %+.4f USDT", realized_pnl)
        tracker.record_close(realized_pnl)

    # 7. Return post-execution position for next cycle comparison
    return trader.get_open_position(symbol)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)
    logger_ = logging.getLogger("llm_trader")

    cfg = LLMTraderConfig()
    overrides: dict = {}
    if args.symbol:
        overrides["symbol"] = args.symbol
    if args.leverage is not None:
        overrides["leverage"] = args.leverage
    if args.order_usdt is not None:
        overrides["order_usdt"] = args.order_usdt
    if args.interval is not None:
        overrides["interval_seconds"] = args.interval
    if args.live:
        overrides["dry_run"] = False
    elif args.dry_run:
        overrides["dry_run"] = True
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    symbol = cfg.symbol.upper()

    logger_.info("=" * 60)
    logger_.info("LLM Trader starting")
    logger_.info("  Symbol         : %s", symbol)
    logger_.info("  Leverage       : %dx", cfg.leverage)
    logger_.info("  Order          : $%.2f USDT", cfg.order_usdt)
    logger_.info("  Dry-run        : %s", cfg.dry_run)
    logger_.info("  Interval       : %ds", cfg.interval_seconds)
    logger_.info("  Kline limit    : %d", cfg.kline_limit)
    logger_.info("  Min confidence : %d/10", cfg.min_confidence)
    logger_.info("  Max losses     : %d consecutive", cfg.max_consecutive_losses)
    logger_.info("  Max daily loss : %.1f%%", cfg.max_daily_loss_pct * 100)
    logger_.info("  LLM            : %s @ %s", cfg.azure_deployment, cfg.azure_endpoint)
    logger_.info("=" * 60)

    market = build_client(cfg)
    engine = LLMDecisionEngine(cfg)
    trader = FuturesTrader(cfg)
    tracker = SafetyTracker(cfg.max_consecutive_losses, cfg.max_daily_loss_pct)

    try:
        trader.setup_leverage(symbol, cfg.leverage)
        last_position: OpenPosition | None = None

        if args.once:
            _run_cycle(symbol, cfg, market, engine, trader, tracker, last_position)
            return

        while True:
            try:
                last_position = _run_cycle(
                    symbol,
                    cfg,
                    market,
                    engine,
                    trader,
                    tracker,
                    last_position,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                logger_.exception("Cycle error: %s", exc)

            logger_.info("Sleeping %ds until next evaluation …", cfg.interval_seconds)
            time.sleep(cfg.interval_seconds)

    except KeyboardInterrupt:
        logger_.info("Interrupted by user.")
    finally:
        close_client(market)
        trader.close()
        logger_.info("LLM Trader stopped.")


if __name__ == "__main__":
    main()
