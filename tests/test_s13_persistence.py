"""Acceptance tests for S13 persistence and notifiers."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

from rich.console import Console

from src.mcp_servers.shared.domain.analysis import (
    IndicatorSet,
    PatternResult,
    Signal,
    TechnicalAnalysis,
)
from src.legacy.domain.trading import Action, TradeDecision, TradeOutcome
from src.mcp_servers.rag_mcp.storage import Persistence
from src.legacy.interfaces.notifiers import ConsoleNotifier, LoggerNotifier


def sample_outcome() -> TradeOutcome:
    return TradeOutcome(
        decision=TradeDecision(
            symbol="BTCUSDT",
            action=Action.BUY,
            quantity=0.001,
            order_type="MARKET",
            price=None,
            reasoning="Momentum aligned",
            confidence=0.8,
            timestamp=1700000000,
            source="grok",
        ),
        order_id="12345",
        executed_price=64000.0,
        pnl_usdt=12.5,
        dry_run=True,
        timestamp=1700000001,
    )


def sample_analysis() -> TechnicalAnalysis:
    return TechnicalAnalysis(
        symbol="BTCUSDT",
        signal=Signal.BUY,
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
        reasoning="Momentum aligned",
    )


def sample_patterns() -> PatternResult:
    return PatternResult(
        symbol="BTCUSDT",
        patterns=["double_bottom"],
        support=62000.0,
        resistance=65000.0,
    )


def test_logs_directory_created_automatically(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    assert not log_dir.exists()

    persistence = Persistence(log_dir)

    assert persistence.log_dir.exists()
    assert persistence.log_dir.is_dir()


def test_csv_header_written_only_once(tmp_path: Path) -> None:
    persistence = Persistence(tmp_path / "logs")
    outcome = sample_outcome()

    persistence.append_trade(outcome, "2026-03-25T00:00:00Z")
    persistence.append_trade(outcome, "2026-03-25T00:05:00Z")

    rows = persistence.trades_csv_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("timestamp_iso,timeframe,symbol,action")
    assert len(rows) == 3
    assert "source" in rows[0]


def test_each_trade_cycle_appends_one_json_line(tmp_path: Path) -> None:
    persistence = Persistence(tmp_path / "logs")

    persistence.append_cycle_log(
        timestamp_iso="2026-03-25T00:00:00Z",
        cycle=1,
        symbol="BTCUSDT",
        analysis=sample_analysis(),
        patterns=sample_patterns(),
        rag_docs_retrieved=4,
        llm_decision="BUY",
        llm_usage={"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168},
        llm_prompt={
            "system_prompt": "system",
            "user_message": "user",
            "chart_included": False,
        },
        llm_raw_response='{"action":"BUY"}',
        decision_source="grok",
        decision_reasoning="Momentum aligned",
        llm_error=None,
        risk_outcome="passed",
        order_id="12345",
        dry_run=True,
    )
    persistence.append_cycle_log(
        timestamp_iso="2026-03-25T00:05:00Z",
        cycle=2,
        symbol="BTCUSDT",
        analysis=sample_analysis(),
        patterns=sample_patterns(),
        rag_docs_retrieved=5,
        llm_decision="HOLD",
        llm_usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        llm_prompt={
            "system_prompt": "system",
            "user_message": "user",
            "chart_included": False,
        },
        llm_raw_response=None,
        decision_source="fallback_hold",
        decision_reasoning="request_failed",
        llm_error="401 Unauthorized",
        risk_outcome="blocked",
        order_id=None,
        dry_run=True,
    )

    lines = persistence.bot_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first_payload = json.loads(lines[0])
    assert first_payload["cycle"] == 1
    assert first_payload["dry_run"] is True
    assert first_payload["llm_usage"]["total_tokens"] == 168
    assert first_payload["llm_prompt"]["system_prompt"] == "system"
    assert first_payload["llm_raw_response"] == '{"action":"BUY"}'
    second_payload = json.loads(lines[1])
    assert second_payload["decision_source"] == "fallback_hold"
    assert second_payload["decision_reasoning"] == "request_failed"
    assert second_payload["llm_error"] == "401 Unauthorized"


def test_console_output_shows_cycle_timestamp_signals_and_decision() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, color_system="standard")
    notifier = ConsoleNotifier(console)

    notifier.notify_cycle(
        cycle=3,
        timestamp_iso="2026-03-25T00:10:00Z",
        symbol_signals=[("BTC", "BUY"), ("ETH", "NEUTRAL")],
        final_decision="BUY",
    )

    rendered = buffer.getvalue()
    assert "Cycle 3" in rendered
    assert "2026-03-25T00:10:00Z" in rendered
    assert "BTC:BUY" in rendered
    assert "Decision:BUY" in rendered


def test_logger_notifier_writes_message(caplog) -> None:
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test.logger.notifier")
    notifier = LoggerNotifier(logger)

    notifier.notify("trade cycle completed")

    assert "trade cycle completed" in caplog.text
