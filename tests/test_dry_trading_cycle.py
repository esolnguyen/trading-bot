"""Integration-style test for one dry trading cycle."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import Settings
from src.domain.analysis import IndicatorSet, PatternResult, Signal, TechnicalAnalysis
from src.domain.market import MarketSnapshot, OHLCVCandle
from src.domain.trading import Action, RiskValidationResult, TradeDecision, TradeOutcome
from src.infrastructure.storage import Persistence
from src.services.trading import TradingLoop


def candle() -> OHLCVCandle:
    return OHLCVCandle(timestamp=1, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)


def snapshot(symbol: str, price: float) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        price=price,
        change_24h_pct=1.0,
        volume_24h=1000.0,
        bid=price - 1,
        ask=price + 1,
        candles=[candle()] * 60,
        timestamp=1,
    )


class FakeAggregator:
    def __init__(self) -> None:
        self.feed = self

    async def snapshot(self, symbol: str) -> MarketSnapshot:
        return snapshot(symbol, 64000.0 if symbol == "BTCUSDT" else 3200.0)

    async def get_balance(self) -> dict:
        return {
            "success": True,
            "data": {
                "balances": [
                    {"asset": "USDT", "free": "1000", "locked": "0"},
                ]
            },
        }


class FakeTechnicalAnalyzer:
    def analyze(self, symbol: str, snapshot: MarketSnapshot) -> TechnicalAnalysis:
        return TechnicalAnalysis(
            symbol=symbol,
            signal=Signal.BUY if symbol == "BTCUSDT" else Signal.NEUTRAL,
            indicators=IndicatorSet(
                rsi_14=35.0,
                macd_line=1.0,
                macd_signal=0.8,
                macd_hist=0.2,
                bb_upper=65000.0,
                bb_mid=63000.0,
                bb_lower=61000.0,
                ema_20=63500.0,
                ema_50=62000.0,
                volume_sma_20=1000.0,
            ),
            reasoning=f"{symbol} looks constructive.",
        )


class FakePatternAnalyzer:
    def analyze(self, symbol: str, candles: list[OHLCVCandle]) -> PatternResult:
        return PatternResult(symbol=symbol, patterns=["double_bottom"] if symbol == "BTCUSDT" else [])


class FakeChartGenerator:
    def render(self, symbol: str, candles: list[OHLCVCandle], indicators: IndicatorSet) -> str:
        return "AAAABBBB"


class FakeRetriever:
    def retrieve(self, snapshot: MarketSnapshot, analysis: TechnicalAnalysis) -> str:
        return f"=== RECENT NEWS (BTC/ETH) ===\n[1] {snapshot.symbol} context"


class FakeBuilder:
    def build(self, snapshots, analyses, patterns, rag_context, **kwargs):
        return "system", "user"


class FakeLLM:
    def __init__(self) -> None:
        self.last_usage = {"prompt_tokens": 90, "completion_tokens": 20, "total_tokens": 110}
        self.last_error = None
        self.last_prompt = {"system_prompt": "system", "user_message": "user", "chart_included": False}
        self.last_raw_response = '{"action":"BUY","symbol":"BTCUSDT"}'

    async def decide(self, system_prompt: str, user_message: str, chart_b64: str | None = None) -> TradeDecision:
        return TradeDecision(
            symbol="BTCUSDT",
            action=Action.BUY,
            quantity=0.001,
            order_type="MARKET",
            price=None,
            reasoning="Momentum aligned",
            confidence=0.8,
            timestamp=1700000000,
            source="grok",
        )


class FakeRisk:
    def validate(self, decision: TradeDecision, balance: dict) -> RiskValidationResult:
        return RiskValidationResult(decision=decision, dry_run=True, status="passed")


class FakeExecutor:
    async def execute(self, decision: TradeDecision, dry_run: bool) -> TradeOutcome:
        assert dry_run is True
        return TradeOutcome(
            decision=decision,
            order_id=None,
            executed_price=None,
            pnl_usdt=None,
            dry_run=True,
            timestamp=1700000001,
        )


class FakeMemory:
    def __init__(self) -> None:
        self.recorded = []

    def record(self, outcome: TradeOutcome) -> None:
        self.recorded.append(outcome)


class FakeConsoleNotifier:
    def __init__(self) -> None:
        self.calls = []

    def notify_cycle(self, **kwargs) -> None:
        self.calls.append(kwargs)


class FakeLoggerNotifier:
    def __init__(self) -> None:
        self.messages = []

    def notify(self, message: str) -> None:
        self.messages.append(message)


@pytest.mark.asyncio
async def test_dry_trading_cycle_logs_without_real_trade(tmp_path: Path) -> None:
    persistence = Persistence(tmp_path / "logs")
    memory = FakeMemory()
    console = FakeConsoleNotifier()
    logger_notifier = FakeLoggerNotifier()
    settings = Settings(
        azure_endpoint="https://example-resource.openai.azure.com",
        azure_api_key="azure-key-1234",
        azure_deployment="grok-prod",
        binance_api_key="binance-key-1234",
        binance_api_secret="binance-secret-1234",
        cryptocompare_api_key="cc-key-1234",
        bot_mode="dry_run",
        bot_interval_seconds=60,
    )
    loop = TradingLoop(
        FakeAggregator(),
        FakeTechnicalAnalyzer(),
        FakePatternAnalyzer(),
        FakeChartGenerator(),
        FakeRetriever(),
        FakeBuilder(),
        FakeLLM(),
        FakeRisk(),
        FakeExecutor(),
        memory,
        persistence,
        settings,
        console_notifier=console,
        logger_notifier=logger_notifier,
    )

    result = await loop.run_cycle_once()

    assert result["outcome"].dry_run is True
    assert persistence.trades_csv_path.exists()
    assert persistence.bot_log_path.exists()
    assert "BTCUSDT" in persistence.trades_csv_path.read_text(encoding="utf-8")
    assert '"dry_run": true' in persistence.bot_log_path.read_text(encoding="utf-8").lower()
    assert '"total_tokens": 110' in persistence.bot_log_path.read_text(encoding="utf-8")
    assert '"decision_source": "grok"' in persistence.bot_log_path.read_text(encoding="utf-8")
    assert '"llm_raw_response": "{\\"action\\":\\"BUY\\",\\"symbol\\":\\"BTCUSDT\\"}"' in persistence.bot_log_path.read_text(encoding="utf-8")
    assert len(memory.recorded) == 1
    assert len(console.calls) == 1
    assert logger_notifier.messages
