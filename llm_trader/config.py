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

    # ── Skills (prompt-injected playbooks) ────────────────────────────────────
    # Comma-separated list of skill names to inject into the system prompt.
    # See llm_trader/skills/ for available names.
    skills: list[str] = field(
        default_factory=lambda: [
            s.strip()
            for s in os.getenv(
                "LLM_TRADER_SKILLS",
                "candlestick,technical-basic,smc,crypto-derivatives,perp-funding-basis",
            ).split(",")
            if s.strip()
        ]
    )

    # ── Backtest ──────────────────────────────────────────────────────────────
    backtest_initial_cash: float = field(
        default_factory=lambda: float(os.getenv("LLM_TRADER_BT_CASH", "10000"))
    )
    backtest_taker_fee: float = field(
        default_factory=lambda: float(os.getenv("LLM_TRADER_BT_TAKER_FEE", "0.0005"))
    )
    backtest_maker_fee: float = field(
        default_factory=lambda: float(os.getenv("LLM_TRADER_BT_MAKER_FEE", "0.0002"))
    )
    backtest_slippage: float = field(
        default_factory=lambda: float(os.getenv("LLM_TRADER_BT_SLIPPAGE", "0.0005"))
    )
    backtest_funding_rate: float = field(
        default_factory=lambda: float(os.getenv("LLM_TRADER_BT_FUNDING_RATE", "0.0001"))
    )
    # How often (in bars) to call the LLM during backtest. 1 = every bar
    # (slow + expensive), 16 = every ~4h on 15m candles. Between calls the
    # last decision's SL/TP still apply.
    backtest_eval_every_n_bars: int = field(
        default_factory=lambda: int(os.getenv("LLM_TRADER_BT_EVAL_EVERY", "16"))
    )
