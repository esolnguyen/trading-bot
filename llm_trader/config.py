"""Configuration for the LLM trader — reads from the project .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


@dataclass
class LLMTraderConfig:
    # ── Azure OpenAI ──────────────────────────────────────────────────────────
    azure_endpoint: str = field(default_factory=lambda: os.environ["AZURE_ENDPOINT"])
    azure_api_key: str = field(default_factory=lambda: os.environ["AZURE_API_KEY"])
    azure_deployment: str = field(default_factory=lambda: os.environ["AZURE_DEPLOYMENT"])
    azure_api_version: str = field(
        default_factory=lambda: os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
    )

    # ── Binance ───────────────────────────────────────────────────────────────
    binance_api_key: str = field(default_factory=lambda: os.environ["BINANCE_API_KEY"])
    binance_api_secret: str = field(default_factory=lambda: os.environ["BINANCE_API_SECRET"])
    binance_testnet: bool = field(
        default_factory=lambda: os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    )
    binance_base_url: str = field(default_factory=lambda: os.getenv("BINANCE_BASE_URL", ""))

    # ── Trading ───────────────────────────────────────────────────────────────
    symbol: str = field(default_factory=lambda: os.getenv("LLM_TRADER_SYMBOL", "BTCUSDT"))
    leverage: int = field(
        default_factory=lambda: int(os.getenv("LLM_TRADER_LEVERAGE", os.getenv("FUTURES_LEVERAGE", "1")))
    )
    order_usdt: float = field(
        default_factory=lambda: float(os.getenv("LLM_TRADER_ORDER_USDT", os.getenv("MAX_ORDER_USDT", "50")))
    )
    dry_run: bool = field(
        default_factory=lambda: os.getenv("LLM_TRADER_DRY_RUN", os.getenv("BOT_DRY_RUN", "true")).lower() == "true"
    )

    # ── Chart settings ────────────────────────────────────────────────────────
    kline_interval: str = "15m"
    kline_limit: int = field(
        default_factory=lambda: int(os.getenv("LLM_TRADER_KLINE_LIMIT", "200"))
    )

    # ── Safety ────────────────────────────────────────────────────────────────
    min_confidence: int = field(
        default_factory=lambda: int(os.getenv("LLM_TRADER_MIN_CONFIDENCE", "3"))
    )
    max_consecutive_losses: int = field(
        default_factory=lambda: int(os.getenv("LLM_TRADER_MAX_CONSECUTIVE_LOSSES", "3"))
    )
    max_daily_loss_pct: float = field(
        default_factory=lambda: float(os.getenv("LLM_TRADER_MAX_DAILY_LOSS_PCT", "0.05"))
    )

    # ── Loop ──────────────────────────────────────────────────────────────────
    # How often to re-evaluate (seconds). Default: every 15 minutes.
    interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("LLM_TRADER_INTERVAL_SECONDS", "900"))
    )
