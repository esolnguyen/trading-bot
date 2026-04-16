"""Typed runtime settings for the trading bot.

Env parsing, validation, and loader logic live in sibling modules:
- ``parsers.py`` — primitive ``parse_*`` helpers and whitelist constants.
- ``validation.py`` — ``validate_required_fields`` / ``validate_ranges``.
- ``loader.py`` — ``load_settings_from_env`` that populates the dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict

from src.core.config.loader import load_settings_from_env
from src.core.config.validation import validate_ranges, validate_required_fields


@dataclass(repr=False)
class Settings:
    """Single source of truth for runtime configuration."""

    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "binance_api_key",
        "binance_api_secret",
        "cryptocompare_api_key",
    )

    # Azure OpenAI (used when provider == "azure")
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_deployment: str = ""
    azure_api_version: str = "2024-08-01-preview"

    # Binance
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_product: str = "spot"
    binance_base_url: str = ""
    binance_testnet: bool = True

    # Bot behaviour
    # ``bot_interval_seconds``: 0 = auto-align to the candle duration so the
    # loop fires exactly once per closed candle (prevents 2–3× LLM calls on
    # short timeframes).
    bot_interval_seconds: int = 0
    max_order_usdt: float = 50.0
    # ``bot_mode``: "off" (master kill — always HOLD), "dry_run" (simulate),
    # "live" (real orders). Replaces legacy BOT_ENABLED / BOT_DRY_RUN.
    bot_mode: str = "off"
    model_supports_vision: bool = False

    @property
    def bot_enabled(self) -> bool:
        """True when bot_mode allows decisions through (dry_run or live)."""
        return self.bot_mode in {"dry_run", "live"}

    @property
    def bot_dry_run(self) -> bool:
        """True when orders should be simulated instead of placed."""
        return self.bot_mode != "live"

    @property
    def use_signal_scorer(self) -> bool:
        """Legacy alias: True when the scorer engine is active."""
        return self.trading_engine == "scorer"

    _CANDLE_SECONDS: dict[str, int] = field(default_factory=lambda: {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
        "8h": 28800, "12h": 43200, "1d": 86400,
    })

    def effective_bot_interval(self) -> int:
        """Return the actual sleep interval in seconds.

        If ``bot_interval_seconds`` is 0 (auto), use the candle duration for
        the current trading timeframe so the loop fires exactly once per candle.
        """
        if self.bot_interval_seconds > 0:
            return self.bot_interval_seconds
        return self._CANDLE_SECONDS.get(self.timeframe, 300)

    # RAG / ChromaDB
    cryptocompare_api_key: str = ""
    coingecko_api_key: str = ""
    chroma_path: str = "./chroma_db"
    news_interval: int = 900
    macro_interval: int = 1800
    ohlcv_interval: int = 3600

    # AI provider selection
    provider: str = "azure"

    google_studio_api_key: str = ""
    google_studio_paid_api_key: str = ""
    google_studio_model: str = "gemini-2.5-flash"

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_base_model: str = "google/gemini-2.5-pro"
    openrouter_fallback_model: str = "deepseek/deepseek-r1:free"

    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "local-model"
    lm_studio_streaming: bool = True

    blockrun_wallet_key: str = ""
    blockrun_base_url: str = "https://blockrun.ai/api"
    blockrun_model: str = "openai/gpt-4o"

    # Model tuning (non-Google)
    model_temperature: float = 0.7
    model_max_tokens: int = 8192
    model_top_p: float = 0.9
    model_top_k: int = 40
    model_freq_penalty: float = 0.0
    model_pres_penalty: float = 0.0

    # Google-specific tuning
    google_max_tokens: int = 32768
    google_temperature: float = 0.7
    google_top_p: float = 0.9
    google_top_k: int = 40
    google_thinking_level: str = "high"
    google_code_execution: bool = False

    # Trading parameters
    timeframe: str = "1h"
    candle_limit: int = 200
    ai_chart_candle_limit: int = 120
    ai_chart_timeframe: str = "5m"
    demo_quote_capital: float = 10000.0
    transaction_fee_percent: float = 0.00075
    default_stop_loss_pct: float = 0.02
    default_take_profit_pct: float = 0.04
    default_position_size: float = 0.02
    include_coin_description: bool = False

    # Discord
    discord_bot_enabled: bool = False
    bot_token_discord: str = ""
    guild_id_discord: str = ""
    main_channel_id: str = ""
    temporary_channel_id_discord: str = ""
    admin_user_ids: list[int] = field(default_factory=list)
    file_message_expiry: int = 604800  # 7 days in seconds

    # Trading symbols & order management
    trading_symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    limit_order_timeout_seconds: int = 300
    trade_memory_max_entries: int = 500

    # Risk guards
    max_daily_loss_pct: float = 0.05
    max_consecutive_losses: int = 3
    min_confidence_threshold: float = 0.0

    # Trailing stop
    trailing_stop_enabled: bool = False
    trailing_stop_activation_pct: float = 0.01
    trailing_stop_distance_pct: float = 0.005

    # Partial take-profit
    partial_tp_enabled: bool = False
    partial_tp1_atr_multiplier: float = 2.0
    partial_tp1_size_pct: float = 0.5

    # Regime detection
    choppiness_threshold: float = 61.8

    # Signal RSI thresholds — auto-relaxed for short timeframes (≤15m) by
    # effective_rsi_thresholds(); override via env vars to fix at any value.
    signal_rsi_strong_buy: float = 30.0
    signal_rsi_buy: float = 40.0
    signal_rsi_sell: float = 60.0
    signal_rsi_strong_sell: float = 70.0

    def effective_rsi_thresholds(self) -> tuple[float, float, float, float]:
        """Return (strong_buy, buy, sell, strong_sell) adjusted for the trading timeframe.

        Shorter timeframes rarely reach extreme RSI levels, so thresholds are
        relaxed to avoid chronic NEUTRAL signals.
        """
        _tf_minutes: dict[str, int] = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360,
            "8h": 480, "12h": 720, "1d": 1440,
        }
        minutes = _tf_minutes.get(self.timeframe, 60)

        sb = self.signal_rsi_strong_buy
        b = self.signal_rsi_buy
        s = self.signal_rsi_sell
        ss = self.signal_rsi_strong_sell

        if minutes <= 5:
            return (sb + 8, b + 7, s - 7, ss - 8)
        if minutes <= 15:
            return (sb + 5, b + 5, s - 5, ss - 5)
        if minutes <= 30:
            return (sb + 3, b + 3, s - 3, ss - 3)
        return (sb, b, s, ss)

    # Multi-timeframe confirmation
    htf_timeframe: str = "4h"
    htf_confirmation_enabled: bool = False

    # When True, all trading symbols are evaluated in a single LLM call.
    single_symbol_decision: bool = True

    # Execution validation
    max_slippage_pct: float = 0.005

    # Fast position monitor
    position_monitor_enabled: bool = True
    position_monitor_interval: int = 15

    # Spot bracket orders
    spot_sl_limit_offset_pct: float = 0.01
    spot_tp_limit_offset_pct: float = 0.002

    # Futures leverage
    futures_leverage: int = 1

    # Trading engine — which decision path drives the main loop.
    trading_engine: str = "llm_enriched"
    trader_skills: list[str] = field(default_factory=lambda: [
        "candlestick", "technical-basic", "smc",
        "crypto-derivatives", "perp-funding-basis",
    ])

    scoring_entry_threshold: float = 0.30
    scoring_exit_threshold: float = 0.20
    scoring_w_signal: float = 0.25
    scoring_w_direction: float = 0.25
    scoring_w_trend: float = 0.15
    scoring_w_momentum: float = 0.15
    scoring_w_volume: float = 0.10
    scoring_w_key_levels: float = 0.10
    scoring_choppiness_penalty: float = 0.3
    reentry_cooldown_cycles: int = 3

    # ML / OHLCV
    ml_timeframe: str = "4h"

    # Debug / directories
    logger_debug: bool = False
    debug_save_charts: bool = False
    debug_chart_save_path: str = "test_images"
    log_dir: str = "logs"
    data_dir: str = "data"

    # RAG tuning
    rag_update_interval_hours: int = 4
    rag_categories_update_interval_hours: int = 24
    rag_coingecko_update_interval_hours: int = 24
    rag_defillama_update_interval_hours: float = 0.25
    rag_news_limit: int = 5
    rag_article_max_tokens: int = 256
    rag_density_penalty_threshold: int = 300
    rag_density_boost_threshold: int = 1000
    rag_density_penalty_multiplier: float = 0.5
    rag_density_boost_multiplier: float = 1.2
    rag_cooccurrence_multiplier: float = 1.5
    rag_retrieval_news: int = 3
    rag_retrieval_macro: int = 1
    rag_retrieval_memory: int = 2

    def __post_init__(self) -> None:
        validate_required_fields(self)
        validate_ranges(self)

    def __repr__(self) -> str:
        redacted = {
            "azure_api_key": self._redact(self.azure_api_key),
            "binance_api_secret": self._redact(self.binance_api_secret),
            "cryptocompare_api_key": self._redact(self.cryptocompare_api_key),
            "google_studio_api_key": self._redact(self.google_studio_api_key),
            "openrouter_api_key": self._redact(self.openrouter_api_key),
            "blockrun_wallet_key": self._redact(self.blockrun_wallet_key),
        }
        return (
            "Settings("
            f"provider={self.provider!r}, "
            f"azure_endpoint={self.azure_endpoint!r}, "
            f"azure_api_key={redacted['azure_api_key']!r}, "
            f"azure_deployment={self.azure_deployment!r}, "
            f"binance_api_key={self._redact(self.binance_api_key)!r}, "
            f"binance_api_secret={redacted['binance_api_secret']!r}, "
            f"binance_product={self.binance_product!r}, "
            f"binance_testnet={self.binance_testnet!r}, "
            f"bot_interval_seconds={self.bot_interval_seconds!r}, "
            f"max_order_usdt={self.max_order_usdt!r}, "
            f"bot_mode={self.bot_mode!r}, "
            f"timeframe={self.timeframe!r}, "
            f"discord_bot_enabled={self.discord_bot_enabled!r}"
            ")"
        )

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "Settings":
        """Load settings from the environment and optional .env file."""
        return load_settings_from_env(cls, env_file=env_file)

    def ohlcv_csv_path(self, symbol: str, interval: str) -> str:
        """Return the canonical path for an OHLCV CSV file.

        e.g. ohlcv_csv_path("BTC/USDT", "4h") → "data/ohlcv/btcusdt_4h.csv"
        """
        sym = symbol.replace("/", "").lower()
        return f"{self.data_dir}/ohlcv/{sym}_{interval}.csv"

    def get_model_config(
        self, model_name: str, overrides: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Return model config dict appropriate for the given model name."""
        if model_name == self.google_studio_model:
            base: Dict[str, Any] = {
                "temperature": self.google_temperature,
                "top_p": self.google_top_p,
                "top_k": self.google_top_k,
                "max_tokens": self.google_max_tokens,
                "thinking_level": self.google_thinking_level,
                "google_code_execution": self.google_code_execution,
            }
        else:
            base = {
                "temperature": self.model_temperature,
                "top_p": self.model_top_p,
                "top_k": self.model_top_k,
                "freq_penalty": self.model_freq_penalty,
                "pres_penalty": self.model_pres_penalty,
                "max_tokens": self.model_max_tokens,
            }
        if overrides:
            base.update(overrides)
        return {k: v for k, v in base.items() if v is not None}

    @staticmethod
    def _redact(value: str) -> str:
        if not value:
            return "***"
        if len(value) <= 4:
            return "*" * len(value)
        return f"{value[:2]}***{value[-2:]}"


__all__ = ["Settings"]
