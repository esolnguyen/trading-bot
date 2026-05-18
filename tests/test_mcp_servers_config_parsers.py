"""Primitive env-var parsers and the BOT_MODE legacy fallback."""

from __future__ import annotations

import pytest

from src.mcp_servers.config.parsers import (
    VALID_BOT_MODES,
    VALID_PROVIDERS,
    parse_bool,
    parse_bot_mode,
    parse_float,
    parse_int,
    parse_list,
    require_str,
)


class TestParseBool:
    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "On"])
    def test_truthy_variants(self, raw: str) -> None:
        assert parse_bool(raw, default=False) is True

    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "Off"])
    def test_falsy_variants(self, raw: str) -> None:
        assert parse_bool(raw, default=True) is False

    def test_none_returns_default(self) -> None:
        assert parse_bool(None, default=True) is True
        assert parse_bool(None, default=False) is False

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_bool("maybe", default=False)


class TestParseInt:
    def test_parses_value(self) -> None:
        assert parse_int("42", default=0) == 42

    def test_none_returns_default(self) -> None:
        assert parse_int(None, default=7) == 7

    def test_empty_string_returns_default(self) -> None:
        assert parse_int("   ", default=11) == 11

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_int("not-a-number", default=0)


class TestParseFloat:
    def test_parses_value(self) -> None:
        assert parse_float("3.14", default=0.0) == pytest.approx(3.14)

    def test_none_returns_default(self) -> None:
        assert parse_float(None, default=1.5) == 1.5

    def test_empty_returns_default(self) -> None:
        assert parse_float("", default=2.5) == 2.5


class TestParseList:
    def test_splits_and_strips(self) -> None:
        assert parse_list("a, b ,c", default=[]) == ["a", "b", "c"]

    def test_drops_empty_tokens(self) -> None:
        assert parse_list("a,,b,  ,c", default=[]) == ["a", "b", "c"]

    def test_none_returns_default(self) -> None:
        default = ["x"]
        result = parse_list(None, default=default)
        assert result == ["x"]
        # parse_list returns the same default reference when raw is None —
        # callers that mutate should pass a fresh list.
        assert result is default

    def test_empty_string_returns_default(self) -> None:
        assert parse_list("   ", default=["y"]) == ["y"]


class TestParseBotMode:
    @pytest.mark.parametrize("mode", sorted(VALID_BOT_MODES))
    def test_accepts_valid_modes(self, mode: str) -> None:
        assert parse_bot_mode(mode) == mode

    def test_normalizes_case_and_dashes(self) -> None:
        assert parse_bot_mode("DRY-RUN") == "dry_run"
        assert parse_bot_mode("  Live  ") == "live"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_bot_mode("paper")

    def test_legacy_fallback_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BOT_ENABLED", raising=False)
        monkeypatch.delenv("BOT_DRY_RUN", raising=False)
        assert parse_bot_mode(None) == "off"

    def test_legacy_fallback_enabled_dry_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BOT_ENABLED", "true")
        monkeypatch.setenv("BOT_DRY_RUN", "true")
        assert parse_bot_mode("") == "dry_run"

    def test_legacy_fallback_enabled_live(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BOT_ENABLED", "1")
        monkeypatch.setenv("BOT_DRY_RUN", "0")
        assert parse_bot_mode(None) == "live"


class TestRequireStr:
    def test_returns_stripped_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_VAR", "  hello  ")
        assert require_str("MY_VAR") == "hello"

    def test_missing_raises_with_env_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MY_VAR", raising=False)
        with pytest.raises(ValueError, match="MY_VAR"):
            require_str("MY_VAR")

    def test_blank_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_VAR", "   ")
        with pytest.raises(ValueError, match="MY_VAR"):
            require_str("MY_VAR")


def test_whitelist_constants_are_frozen() -> None:
    # Frozensets are immutable — guards against accidental .add() mutation.
    assert isinstance(VALID_PROVIDERS, frozenset)
    assert isinstance(VALID_BOT_MODES, frozenset)
    assert "azure" in VALID_PROVIDERS
    assert {"off", "dry_run", "live"} <= VALID_BOT_MODES
