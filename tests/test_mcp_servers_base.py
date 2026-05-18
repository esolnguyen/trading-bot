"""tool_error envelope used by every MCP handler on failure."""

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


def test_tool_error_with_no_extras_omits_keys() -> None:
    payload = tool_error("oops")
    assert set(payload) == {"success", "error"}


def test_tool_error_preserves_nested_payload() -> None:
    extra = {"a": 1, "b": [1, 2, 3]}
    payload = tool_error("bad", details=extra)
    assert payload["details"] is extra
    assert payload["details"]["b"] == [1, 2, 3]


def test_tool_error_extra_can_override_default_keys() -> None:
    # Callers that pass ``success=True`` or ``error=...`` deliberately win —
    # tool_error doesn't try to defend its own keys.
    payload = tool_error("original", success=True, error="rewritten")
    assert payload["success"] is True
    assert payload["error"] == "rewritten"


def test_tool_error_empty_message_still_returns_envelope() -> None:
    payload = tool_error("")
    assert payload == {"success": False, "error": ""}
