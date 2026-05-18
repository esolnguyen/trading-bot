"""Settings validation — required fields, provider-specific creds, ranges."""

from __future__ import annotations

from dataclasses import MISSING

import pytest

from src.mcp_servers.config.settings import Settings
from src.mcp_servers.config.validation import (
    validate_ranges,
    validate_required_fields,
)


def _base_kwargs() -> dict[str, object]:
    """Override defaults so REQUIRED_FIELDS + provider creds pass validation."""
    return {
        "binance_api_key": "k",
        "binance_api_secret": "s",
        "cryptocompare_api_key": "cc",
        "provider": "azure",
        "azure_endpoint": "https://x.openai.azure.com",
        "azure_api_key": "azk",
        "azure_deployment": "gpt-4o",
    }


def _make(**overrides: object) -> Settings:
    """Build a Settings instance from declared defaults, bypassing __post_init__.

    The validators are intentionally exercised individually so each invariant
    can be tested in isolation.
    """
    settings = object.__new__(Settings)
    for name, fld in Settings.__dataclass_fields__.items():  # type: ignore[attr-defined]
        if fld.default is not MISSING:
            setattr(settings, name, fld.default)
        elif fld.default_factory is not MISSING:  # type: ignore[misc]
            setattr(settings, name, fld.default_factory())  # type: ignore[misc]
        else:
            setattr(settings, name, "")
    for key, value in {**_base_kwargs(), **overrides}.items():
        setattr(settings, key, value)
    return settings


class TestRequiredFields:
    def test_happy_path(self) -> None:
        validate_required_fields(_make())

    @pytest.mark.parametrize(
        "field", ["binance_api_key", "binance_api_secret", "cryptocompare_api_key"]
    )
    def test_missing_required_raises(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            validate_required_fields(_make(**{field: ""}))

    def test_azure_endpoint_required_when_provider_azure(self) -> None:
        with pytest.raises(ValueError, match="AZURE_ENDPOINT"):
            validate_required_fields(_make(provider="azure", azure_endpoint=""))

    def test_azure_creds_not_required_when_other_provider(self) -> None:
        validate_required_fields(
            _make(
                provider="openrouter",
                azure_endpoint="",
                azure_api_key="",
                azure_deployment="",
                openrouter_api_key="or-key",
            )
        )

    def test_openrouter_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            validate_required_fields(
                _make(
                    provider="openrouter",
                    azure_endpoint="",
                    azure_api_key="",
                    azure_deployment="",
                    openrouter_api_key="",
                )
            )

    def test_googleai_requires_studio_key(self) -> None:
        with pytest.raises(ValueError, match="GOOGLE_STUDIO_API_KEY"):
            validate_required_fields(
                _make(
                    provider="googleai",
                    azure_endpoint="",
                    azure_api_key="",
                    azure_deployment="",
                    google_studio_api_key="",
                )
            )


class TestRanges:
    def test_happy_path(self) -> None:
        validate_ranges(_make())

    def test_unknown_provider_rejected(self) -> None:
        with pytest.raises(ValueError, match="provider"):
            validate_ranges(_make(provider="nonsense"))

    def test_unknown_bot_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="bot_mode"):
            validate_ranges(_make(bot_mode="paper"))

    @pytest.mark.parametrize("leverage", [0, 126, -3])
    def test_leverage_out_of_range_rejected(self, leverage: int) -> None:
        with pytest.raises(ValueError, match="futures_leverage"):
            validate_ranges(_make(futures_leverage=leverage))

    def test_unknown_ml_timeframe_rejected(self) -> None:
        with pytest.raises(ValueError, match="ml_timeframe"):
            validate_ranges(_make(ml_timeframe="7m"))

    def test_unknown_binance_product_rejected(self) -> None:
        with pytest.raises(ValueError, match="binance_product"):
            validate_ranges(_make(binance_product="margin"))

    @pytest.mark.parametrize("value", [0.0, -1.0, 10_001.0])
    def test_max_order_usdt_bounds(self, value: float) -> None:
        with pytest.raises(ValueError, match="max_order_usdt"):
            validate_ranges(_make(max_order_usdt=value))

    def test_negative_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match="bot_interval_seconds"):
            validate_ranges(_make(bot_interval_seconds=-1))

    def test_zero_interval_allowed(self) -> None:
        # 0 means "auto-align to candle duration" — must be valid.
        validate_ranges(_make(bot_interval_seconds=0))


class TestSettingsHelpers:
    def test_bot_enabled_property(self) -> None:
        assert _make(bot_mode="off").bot_enabled is False
        assert _make(bot_mode="dry_run").bot_enabled is True
        assert _make(bot_mode="live").bot_enabled is True

    def test_bot_dry_run_property(self) -> None:
        assert _make(bot_mode="off").bot_dry_run is True
        assert _make(bot_mode="dry_run").bot_dry_run is True
        assert _make(bot_mode="live").bot_dry_run is False

    @pytest.mark.parametrize(
        "timeframe,expected",
        [("1m", 60), ("5m", 300), ("1h", 3600), ("4h", 14400), ("1d", 86400)],
    )
    def test_effective_interval_auto_align(self, timeframe: str, expected: int) -> None:
        s = _make(timeframe=timeframe, bot_interval_seconds=0)
        assert s.effective_bot_interval() == expected

    def test_effective_interval_explicit_override(self) -> None:
        s = _make(timeframe="1h", bot_interval_seconds=42)
        assert s.effective_bot_interval() == 42

    def test_effective_interval_unknown_tf_falls_back(self) -> None:
        s = _make(timeframe="7m", bot_interval_seconds=0)
        assert s.effective_bot_interval() == 300

    def test_effective_rsi_thresholds_short_tf_relaxed(self) -> None:
        s_1h = _make(timeframe="1h")
        s_5m = _make(timeframe="5m")
        sb_1h, b_1h, sl_1h, ss_1h = s_1h.effective_rsi_thresholds()
        sb_5m, b_5m, sl_5m, ss_5m = s_5m.effective_rsi_thresholds()
        # Short timeframes push buy/strong-buy higher and sell/strong-sell lower
        # so signals fire more readily.
        assert sb_5m > sb_1h
        assert b_5m > b_1h
        assert sl_5m < sl_1h
        assert ss_5m < ss_1h

    def test_ohlcv_csv_path_normalizes_symbol(self) -> None:
        s = _make(data_dir="data")
        assert s.ohlcv_csv_path("BTC/USDT", "4h") == "data/ohlcv/btcusdt_4h.csv"

    def test_repr_redacts_secrets(self) -> None:
        s = _make(
            binance_api_secret="supersecretvalue", azure_api_key="anothersecret"
        )
        rendered = repr(s)
        assert "supersecretvalue" not in rendered
        assert "anothersecret" not in rendered
        assert "***" in rendered

    def test_repr_redacts_short_secret_to_stars(self) -> None:
        # ≤4 chars renders as all-stars (no leak of first/last two chars).
        rendered = repr(_make(binance_api_secret="ab"))
        assert "'**'" in rendered

    def test_get_model_config_non_google(self) -> None:
        s = _make()
        cfg = s.get_model_config("some-model")
        assert "thinking_level" not in cfg
        assert cfg["temperature"] == s.model_temperature

    def test_get_model_config_google_branch(self) -> None:
        s = _make(google_studio_model="gemini-x")
        cfg = s.get_model_config("gemini-x")
        assert "thinking_level" in cfg
        assert cfg["temperature"] == s.google_temperature

    def test_get_model_config_applies_overrides(self) -> None:
        s = _make()
        cfg = s.get_model_config("anything", overrides={"temperature": 0.1})
        assert cfg["temperature"] == 0.1
