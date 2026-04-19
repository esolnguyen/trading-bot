"""Agent-layer contracts — decision schema and tool bundle shape.

No network, no LLM calls. These tests only check the declarative
surface that the rest of the bot relies on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.trading_bot.agent import MCP_SERVER_LAUNCHERS, TradingDecision


EXPECTED_MCP_SERVERS = {"ml", "binance", "analysis", "rag", "skills"}


def test_mcp_launchers_cover_every_server() -> None:
    assert set(MCP_SERVER_LAUNCHERS) == EXPECTED_MCP_SERVERS


def test_mcp_launcher_scripts_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for name, launcher in MCP_SERVER_LAUNCHERS.items():
        path = repo_root / launcher
        assert path.is_file(), f"{name} launcher missing: {path}"


def test_decision_valid_hold() -> None:
    d = TradingDecision(
        symbol="BTCUSDT",
        side="HOLD",
        conviction=3,
        rationale="choppy range, waiting for breakout",
    )
    assert d.stop_loss_pct is None
    assert d.take_profit_pct is None


def test_decision_valid_long() -> None:
    d = TradingDecision(
        symbol="BTCUSDT",
        side="LONG",
        conviction=8,
        rationale="EMA stack aligned + FVG reclaim",
        stop_loss_pct=1.5,
        take_profit_pct=3.0,
    )
    assert d.side == "LONG"
    assert d.stop_loss_pct == 1.5


@pytest.mark.parametrize("conviction", [0, 11, -1, 100])
def test_decision_rejects_out_of_range_conviction(conviction: int) -> None:
    with pytest.raises(ValidationError):
        TradingDecision(
            symbol="BTCUSDT",
            side="HOLD",
            conviction=conviction,
            rationale="x",
        )


def test_decision_rejects_unknown_side() -> None:
    with pytest.raises(ValidationError):
        TradingDecision(
            symbol="BTCUSDT",
            side="MAYBE",  # type: ignore[arg-type]
            conviction=5,
            rationale="x",
        )
