"""Acceptance tests for S2 shared contracts."""

from __future__ import annotations

import json

from src.mcp_servers.shared.domain.analysis import (
    IndicatorSet,
    PatternResult,
    Signal,
    TechnicalAnalysis,
)
from src.mcp_servers.shared.domain.market import MarketSnapshot, OHLCVCandle
from src.legacy.domain.trading import Action, TradeDecision, TradeOutcome


def test_contracts_import_cleanly() -> None:
    candle = OHLCVCandle(
        timestamp=1, open=1.0, high=2.0, low=0.5, close=1.5, volume=100.0
    )
    snapshot = MarketSnapshot(
        symbol="BTCUSDT",
        price=60000.0,
        change_24h_pct=2.5,
        volume_24h=1000000.0,
        bid=59999.0,
        ask=60001.0,
        candles=[candle],
        timestamp=1,
    )
    indicators = IndicatorSet(
        rsi_14=45.0,
        macd_line=1.0,
        macd_signal=0.5,
        macd_hist=0.5,
        bb_upper=61000.0,
        bb_mid=60000.0,
        bb_lower=59000.0,
        ema_20=59800.0,
        ema_50=59200.0,
        volume_sma_20=50000.0,
    )
    analysis = TechnicalAnalysis(
        symbol="BTCUSDT",
        signal=Signal.NEUTRAL,
        indicators=indicators,
        reasoning="Stable market",
    )
    patterns = PatternResult(symbol="BTCUSDT")
    decision = TradeDecision(symbol="BTCUSDT", action=Action.HOLD)
    outcome = TradeOutcome(decision=decision)

    assert snapshot.candles[0] == candle
    assert analysis.signal is Signal.NEUTRAL
    assert patterns.patterns == []
    assert outcome.decision.action is Action.HOLD


def test_signal_and_action_are_json_serializable() -> None:
    payload = json.dumps({"signal": Signal.BUY, "action": Action.SELL})
    parsed = json.loads(payload)

    assert parsed == {"signal": "BUY", "action": "SELL"}
    assert Signal(parsed["signal"]) is Signal.BUY
    assert Action(parsed["action"]) is Action.SELL


def test_hold_trade_decision_defaults_quantity_and_price() -> None:
    decision = TradeDecision(symbol="ETHUSDT", action=Action.HOLD, reasoning="No edge")

    assert decision.quantity == 0.0
    assert decision.price is None
    assert decision.order_type == "MARKET"
    assert decision.reasoning == "No edge"
