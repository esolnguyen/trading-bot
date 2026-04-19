"""Base helpers shared by every MCP server."""

from __future__ import annotations

from src.mcp_servers.base import tool_error


def test_tool_error_shape() -> None:
    payload = tool_error("boom")
    assert payload == {"success": False, "error": "boom"}


def test_tool_error_merges_extra() -> None:
    payload = tool_error("stale", freshness_seconds=42.0, symbol="BTCUSDT")
    assert payload["success"] is False
    assert payload["error"] == "stale"
    assert payload["freshness_seconds"] == 42.0
    assert payload["symbol"] == "BTCUSDT"
