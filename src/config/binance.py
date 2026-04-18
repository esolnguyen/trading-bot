"""Binance API credentials + endpoint selection."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BinanceSettings(BaseSettings):
    """Binance UM-futures credentials + endpoint config.

    Field names retain the ``binance_`` prefix so any caller that already
    reads ``settings.binance_api_key`` (e.g. the ``_HasBinanceCreds``
    Protocol in ``shared/infrastructure/binance/client.py``) works without
    modification. Env vars keep their canonical ``BINANCE_*`` names via
    explicit field-level aliases (no ``env_prefix`` = no double-prefixing).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    binance_api_key: str = Field(default="", alias="BINANCE_API_KEY")
    binance_api_secret: str = Field(default="", alias="BINANCE_API_SECRET")
    binance_base_url: str = Field(default="", alias="BINANCE_BASE_URL")
    binance_testnet: bool = Field(default=True, alias="BINANCE_TESTNET")
    binance_product: Literal["spot", "usdt_futures"] = Field(
        default="usdt_futures", alias="BINANCE_PRODUCT"
    )
