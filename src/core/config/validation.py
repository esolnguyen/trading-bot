"""Post-init validation for the Settings dataclass."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.config.parsers import (
    VALID_BOT_MODES,
    VALID_PROVIDERS,
    VALID_TRADING_ENGINES,
)

if TYPE_CHECKING:
    from src.core.config.settings import Settings


_PROVIDER_REQUIRED_KEYS: dict[str, str] = {
    "openrouter": "openrouter_api_key",
    "googleai": "google_studio_api_key",
    "blockrun": "blockrun_wallet_key",
}

_AZURE_REQUIRED_FIELDS = ("azure_endpoint", "azure_api_key", "azure_deployment")


def validate_required_fields(settings: "Settings") -> None:
    """Ensure REQUIRED_FIELDS and provider-specific credentials are populated."""
    for field_name in settings.REQUIRED_FIELDS:
        value = getattr(settings, field_name)
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError(field_name)

    if settings.provider == "azure":
        for azure_field in _AZURE_REQUIRED_FIELDS:
            value = getattr(settings, azure_field)
            if not isinstance(value, str) or value.strip() == "":
                raise ValueError(azure_field.upper())

    required_key = _PROVIDER_REQUIRED_KEYS.get(settings.provider)
    if required_key:
        value = getattr(settings, required_key, "")
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError(required_key.upper())


def validate_ranges(settings: "Settings") -> None:
    """Enforce whitelist and numeric-range invariants."""
    if settings.provider not in VALID_PROVIDERS:
        raise ValueError("provider")

    if settings.bot_mode not in VALID_BOT_MODES:
        raise ValueError(
            f"bot_mode must be one of: off, dry_run, live (got {settings.bot_mode!r})"
        )

    if settings.trading_engine not in VALID_TRADING_ENGINES:
        raise ValueError(
            "trading_engine must be one of: scorer, llm_skills, llm_enriched "
            f"(got {settings.trading_engine!r})"
        )

    if not (1 <= settings.futures_leverage <= 125):
        raise ValueError("futures_leverage must be between 1 and 125")

    if settings.ml_timeframe not in {"1m", "5m", "15m", "30m", "1h", "4h"}:
        raise ValueError("ml_timeframe must be one of: 1m, 5m, 15m, 30m, 1h, 4h")

    if settings.binance_product not in {"spot", "usdt_futures"}:
        raise ValueError("binance_product")

    if settings.max_order_usdt <= 0 or settings.max_order_usdt > 10000:
        raise ValueError("max_order_usdt")

    if settings.bot_interval_seconds < 0:
        raise ValueError(
            "bot_interval_seconds must be >= 0 (0 = auto-align to candle)"
        )
