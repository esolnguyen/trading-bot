"""Typed runtime settings for the trading bot."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, ClassVar, Dict

from dotenv import load_dotenv


_VALID_PROVIDERS = frozenset({"azure", "googleai", "openrouter", "local", "blockrun", "all"})


def _parse_bool(raw_value: str | None, *, default: bool) -> bool:
    """Parse a boolean environment variable with a safe default."""
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"Invalid boolean value: {raw_value}")


def _parse_int(raw_value: str | None, *, default: int) -> int:
    """Parse an integer environment variable with a default."""
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


def _parse_float(raw_value: str | None, *, default: float) -> float:
    """Parse a float environment variable with a default."""
    if raw_value is None or raw_value.strip() == "":
        return default
    return float(raw_value)


def _parse_list(raw_value: str | None, *, default: list[str]) -> list[str]:
    """Parse a comma-separated environment variable into a list of strings."""
    if raw_value is None or raw_value.strip() == "":
        return default
    return [item.strip() for item in raw_value.split(",") if item.strip()]


@dataclass(repr=False)
class Settings:
    """Single source of truth for runtime configuration."""

    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "binance_api_key",
        "binance_api_secret",
        "cryptocompare_api_key",
    )

    # ------------------------------------------------------------------ #
    # Azure OpenAI (used when provider == "azure")
    # ------------------------------------------------------------------ #
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_deployment: str = ""
    azure_api_version: str = "2024-08-01-preview"

    # ------------------------------------------------------------------ #
    # Binance
    # ------------------------------------------------------------------ #
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_product: str = "spot"
    binance_base_url: str = ""
    binance_testnet: bool = True

    # ------------------------------------------------------------------ #
    # Bot behaviour
    # ------------------------------------------------------------------ #
    # How often the main trading loop wakes and runs a cycle (seconds).
    # 0 = auto-align: sleep exactly one candle duration so the bot runs once
    # per closed candle (prevents 2-3× redundant LLM calls on short TFs).
    # Set BOT_INTERVAL_SECONDS=0 in .env to enable auto-alignment.
    bot_interval_seconds: int = 0
    max_order_usdt: float = 50.0
    bot_enabled: bool = False
    bot_dry_run: bool = True
    model_supports_vision: bool = False

    # Seconds per candle for each supported timeframe (used by auto-alignment).
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

    # ------------------------------------------------------------------ #
    # RAG / ChromaDB
    # ------------------------------------------------------------------ #
    cryptocompare_api_key: str = ""
    coingecko_api_key: str = ""
    chroma_path: str = "./chroma_db"
    news_interval: int = 900
    macro_interval: int = 1800
    ohlcv_interval: int = 3600

    # ------------------------------------------------------------------ #
    # AI provider selection
    # ------------------------------------------------------------------ #
    provider: str = "azure"

    # Google AI
    google_studio_api_key: str = ""
    google_studio_paid_api_key: str = ""
    google_studio_model: str = "gemini-2.5-flash"

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_base_model: str = "google/gemini-2.5-pro"
    openrouter_fallback_model: str = "deepseek/deepseek-r1:free"

    # LM Studio (local)
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "local-model"
    lm_studio_streaming: bool = True

    # BlockRun
    blockrun_wallet_key: str = ""
    blockrun_base_url: str = "https://blockrun.ai/api"
    blockrun_model: str = "openai/gpt-4o"

    # ------------------------------------------------------------------ #
    # Model tuning (non-Google)
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Trading parameters
    # ------------------------------------------------------------------ #
    crypto_pair: str = "BTC/USDT"
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

    # ------------------------------------------------------------------ #
    # Discord
    # ------------------------------------------------------------------ #
    discord_bot_enabled: bool = False
    bot_token_discord: str = ""
    guild_id_discord: str = ""
    main_channel_id: str = ""
    temporary_channel_id_discord: str = ""
    admin_user_ids: list[int] = field(default_factory=list)
    file_message_expiry: int = 604800  # 7 days in seconds

    # ------------------------------------------------------------------ #
    # Trading symbols & order management
    # ------------------------------------------------------------------ #
    trading_symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    limit_order_timeout_seconds: int = 300
    trade_memory_max_entries: int = 500

    # ------------------------------------------------------------------ #
    # Risk guards
    # ------------------------------------------------------------------ #
    max_daily_loss_pct: float = 0.05          # halt trading after losing this % of capital in one day
    max_consecutive_losses: int = 3           # halt trading after this many losses in a row
    min_confidence_threshold: float = 0.0    # skip LLM decisions below this confidence (0-100); 0 = disabled

    # ------------------------------------------------------------------ #
    # Trailing stop
    # ------------------------------------------------------------------ #
    trailing_stop_enabled: bool = False
    trailing_stop_activation_pct: float = 0.01   # profit % required to activate trailing stop
    trailing_stop_distance_pct: float = 0.005    # trail distance from best price

    # ------------------------------------------------------------------ #
    # Partial take-profit
    # ------------------------------------------------------------------ #
    partial_tp_enabled: bool = False
    partial_tp1_atr_multiplier: float = 2.0  # TP1 = entry ± (ATR × this); full TP uses 4x
    partial_tp1_size_pct: float = 0.5        # fraction of position closed at TP1

    # ------------------------------------------------------------------ #
    # Regime detection
    # ------------------------------------------------------------------ #
    choppiness_threshold: float = 61.8       # skip entry signals when choppiness > this

    # ------------------------------------------------------------------ #
    # Signal RSI thresholds — auto-relaxed for short timeframes (≤15m) by
    # _effective_rsi_thresholds(); override via env vars to fix at any value.
    signal_rsi_strong_buy: float = 30.0     # RSI below this → STRONG_BUY candidate
    signal_rsi_buy: float = 40.0            # RSI below this → BUY candidate
    signal_rsi_sell: float = 60.0           # RSI above this → SELL candidate
    signal_rsi_strong_sell: float = 70.0    # RSI above this → STRONG_SELL candidate

    def effective_rsi_thresholds(self) -> tuple[float, float, float, float]:
        """Return (strong_buy, buy, sell, strong_sell) adjusted for the trading timeframe.

        On shorter timeframes RSI oscillates more rapidly and rarely reaches
        extreme levels, so the thresholds are relaxed to avoid chronic NEUTRAL
        signals.  Override any threshold via env var to suppress auto-adjustment.
        """
        # Minutes per candle for the current timeframe
        _tf_minutes: dict[str, int] = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360,
            "8h": 480, "12h": 720, "1d": 1440,
        }
        minutes = _tf_minutes.get(self.timeframe, 60)

        sb = self.signal_rsi_strong_buy
        b  = self.signal_rsi_buy
        s  = self.signal_rsi_sell
        ss = self.signal_rsi_strong_sell

        if minutes <= 5:        # 1m / 5m: very relaxed
            return (sb + 8, b + 7, s - 7, ss - 8)
        if minutes <= 15:       # 15m: moderately relaxed
            return (sb + 5, b + 5, s - 5, ss - 5)
        if minutes <= 30:       # 30m: slightly relaxed
            return (sb + 3, b + 3, s - 3, ss - 3)
        return (sb, b, s, ss)   # 1h+: use configured values as-is

    # ------------------------------------------------------------------ #
    # Multi-timeframe confirmation
    # ------------------------------------------------------------------ #
    htf_timeframe: str = "4h"
    htf_confirmation_enabled: bool = False   # require HTF trend agreement before entry

    # When True, all trading symbols are evaluated in a single LLM call instead
    # of one call per symbol. The LLM picks the best opportunity across symbols.
    # Default True: halves LLM cost for 2-symbol trading and provides cross-asset context.
    # Set SINGLE_SYMBOL_DECISION=false to revert to per-symbol parallel calls.
    single_symbol_decision: bool = True

    # ------------------------------------------------------------------ #
    # Execution validation
    # ------------------------------------------------------------------ #
    max_slippage_pct: float = 0.005          # warn if executed price deviates > 0.5% from decision price

    # ------------------------------------------------------------------ #
    # Fast position monitor
    # ------------------------------------------------------------------ #
    position_monitor_enabled: bool = True
    position_monitor_interval: int = 15     # seconds between fast SL price checks (independent of main cycle)

    # ------------------------------------------------------------------ #
    # Spot bracket orders
    # ------------------------------------------------------------------ #
    spot_sl_limit_offset_pct: float = 0.01  # STOP_LOSS_LIMIT price = stopPrice × (1 - this); gives 1% fill room
    spot_tp_limit_offset_pct: float = 0.002 # TAKE_PROFIT_LIMIT price = stopPrice × (1 - this) for sells

    # ------------------------------------------------------------------ #
    # Futures leverage
    # ------------------------------------------------------------------ #
    futures_leverage: int = 1   # applied to all trading_symbols on startup (futures only)

    # ------------------------------------------------------------------ #
    # Signal Scorer (deterministic engine — replaces LLM when enabled)
    # ------------------------------------------------------------------ #
    use_signal_scorer: bool = False          # True = scorer, False = LLM
    scoring_entry_threshold: float = 0.30    # minimum |score| to open a position
    scoring_exit_threshold: float = 0.20     # minimum |score| to close on reversal
    # Component weights (should roughly sum to 1.0)
    scoring_w_signal: float = 0.25           # technical analyzer signal
    scoring_w_direction: float = 0.25        # XGBoost P(bullish)
    scoring_w_trend: float = 0.15            # ADX + EMA alignment
    scoring_w_momentum: float = 0.15         # RSI + MACD histogram
    scoring_w_volume: float = 0.10           # vol ratio + OBV slope
    scoring_w_key_levels: float = 0.10       # S/R proximity
    # Multipliers
    scoring_choppiness_penalty: float = 0.3  # multiply score by this when choppy
    # Re-entry cooldown: skip N cycles after a position closes before opening
    # a new one on the same symbol (prevents rapid-fire flipping)
    reentry_cooldown_cycles: int = 3

    # ------------------------------------------------------------------ #
    # ML / OHLCV
    # ------------------------------------------------------------------ #
    ml_timeframe: str = "4h"   # candle resolution used for ML training & scoring: 15m | 1h | 4h

    # ------------------------------------------------------------------ #
    # Debug / directories
    # ------------------------------------------------------------------ #
    logger_debug: bool = False
    debug_save_charts: bool = False
    debug_chart_save_path: str = "test_images"
    log_dir: str = "logs"
    data_dir: str = "data"

    # ------------------------------------------------------------------ #
    # RAG tuning
    # ------------------------------------------------------------------ #
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
    # How many documents to pull from each Chroma collection at decision time.
    # Lower numbers = fewer tokens in the prompt. Tune to your cost tolerance.
    rag_retrieval_news: int = 3      # was hardcoded 5
    rag_retrieval_macro: int = 1     # was hardcoded 3; macro is global, 1 doc is usually enough
    rag_retrieval_memory: int = 2    # was hardcoded 3

    def __post_init__(self) -> None:
        self._validate_required_fields()
        self._validate_ranges()

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
            f"bot_enabled={self.bot_enabled!r}, "
            f"bot_dry_run={self.bot_dry_run!r}, "
            f"crypto_pair={self.crypto_pair!r}, "
            f"timeframe={self.timeframe!r}, "
            f"discord_bot_enabled={self.discord_bot_enabled!r}"
            ")"
        )

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "Settings":
        """Load settings from the environment and optional .env file."""
        load_dotenv(dotenv_path=env_file, override=False)

        admin_ids = _parse_list(os.getenv("ADMIN_USER_IDS"), default=[])
        parsed_admin_ids: list[int] = []
        for raw_id in admin_ids:
            try:
                parsed_admin_ids.append(int(raw_id))
            except ValueError:
                pass

        return cls(
            # Azure OpenAI
            azure_endpoint=os.getenv("AZURE_ENDPOINT", "").strip(),
            azure_api_key=os.getenv("AZURE_API_KEY", "").strip(),
            azure_deployment=os.getenv("AZURE_DEPLOYMENT", "").strip(),
            azure_api_version=os.getenv("AZURE_API_VERSION", "2024-08-01-preview").strip(),
            # Binance
            binance_api_key=cls._require_str("BINANCE_API_KEY"),
            binance_api_secret=cls._require_str("BINANCE_API_SECRET"),
            binance_product=os.getenv("BINANCE_PRODUCT", "spot").strip().lower(),
            binance_base_url=os.getenv("BINANCE_BASE_URL", "").strip(),
            binance_testnet=_parse_bool(os.getenv("BINANCE_TESTNET"), default=True),
            # Bot
            bot_interval_seconds=_parse_int(os.getenv("BOT_INTERVAL_SECONDS"), default=0),
            max_order_usdt=_parse_float(os.getenv("MAX_ORDER_USDT"), default=50.0),
            bot_enabled=_parse_bool(os.getenv("BOT_ENABLED"), default=False),
            bot_dry_run=_parse_bool(os.getenv("BOT_DRY_RUN"), default=True),
            model_supports_vision=_parse_bool(os.getenv("MODEL_SUPPORTS_VISION"), default=False),
            # RAG / ChromaDB
            cryptocompare_api_key=cls._require_str("CRYPTOCOMPARE_API_KEY"),
            coingecko_api_key=os.getenv("COINGECKO_API_KEY", "").strip(),
            chroma_path=os.getenv("CHROMA_PATH", "./chroma_db"),
            news_interval=_parse_int(os.getenv("NEWS_INTERVAL"), default=900),
            macro_interval=_parse_int(os.getenv("MACRO_INTERVAL"), default=1800),
            ohlcv_interval=_parse_int(os.getenv("OHLCV_INTERVAL"), default=3600),
            # AI provider selection
            provider=os.getenv("PROVIDER", "azure").strip().lower(),
            # Google
            google_studio_api_key=os.getenv("GOOGLE_STUDIO_API_KEY", "").strip(),
            google_studio_paid_api_key=os.getenv("GOOGLE_STUDIO_PAID_API_KEY", "").strip(),
            google_studio_model=os.getenv("GOOGLE_STUDIO_MODEL", "gemini-2.5-flash").strip(),
            # OpenRouter
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip(),
            openrouter_base_model=os.getenv("OPENROUTER_BASE_MODEL", "google/gemini-2.5-pro").strip(),
            openrouter_fallback_model=os.getenv("OPENROUTER_FALLBACK_MODEL", "deepseek/deepseek-r1:free").strip(),
            # LM Studio
            lm_studio_base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1").strip(),
            lm_studio_model=os.getenv("LM_STUDIO_MODEL", "local-model").strip(),
            lm_studio_streaming=_parse_bool(os.getenv("LM_STUDIO_STREAMING"), default=True),
            # BlockRun
            blockrun_wallet_key=os.getenv("BLOCKRUN_WALLET_KEY", "").strip(),
            blockrun_base_url=os.getenv("BLOCKRUN_BASE_URL", "https://blockrun.ai/api").strip(),
            blockrun_model=os.getenv("BLOCKRUN_MODEL", "openai/gpt-4o").strip(),
            # Model tuning
            model_temperature=_parse_float(os.getenv("MODEL_TEMPERATURE"), default=0.7),
            model_max_tokens=_parse_int(os.getenv("MODEL_MAX_TOKENS"), default=8192),
            model_top_p=_parse_float(os.getenv("MODEL_TOP_P"), default=0.9),
            model_top_k=_parse_int(os.getenv("MODEL_TOP_K"), default=40),
            model_freq_penalty=_parse_float(os.getenv("MODEL_FREQ_PENALTY"), default=0.0),
            model_pres_penalty=_parse_float(os.getenv("MODEL_PRES_PENALTY"), default=0.0),
            google_max_tokens=_parse_int(os.getenv("GOOGLE_MAX_TOKENS"), default=32768),
            google_temperature=_parse_float(os.getenv("GOOGLE_TEMPERATURE"), default=0.7),
            google_top_p=_parse_float(os.getenv("GOOGLE_TOP_P"), default=0.9),
            google_top_k=_parse_int(os.getenv("GOOGLE_TOP_K"), default=40),
            google_thinking_level=os.getenv("GOOGLE_THINKING_LEVEL", "high").strip(),
            google_code_execution=_parse_bool(os.getenv("GOOGLE_CODE_EXECUTION"), default=False),
            # Trading parameters
            crypto_pair=os.getenv("CRYPTO_PAIR", "BTC/USDT").strip(),
            timeframe=os.getenv("TIMEFRAME", "1h").strip(),
            candle_limit=_parse_int(os.getenv("CANDLE_LIMIT"), default=200),
            ai_chart_candle_limit=_parse_int(os.getenv("AI_CHART_CANDLE_LIMIT"), default=120),
            ai_chart_timeframe=os.getenv("AI_CHART_TIMEFRAME", "5m").strip(),
            demo_quote_capital=_parse_float(os.getenv("DEMO_QUOTE_CAPITAL"), default=10000.0),
            transaction_fee_percent=_parse_float(os.getenv("TRANSACTION_FEE_PERCENT"), default=0.00075),
            default_stop_loss_pct=_parse_float(os.getenv("DEFAULT_STOP_LOSS_PCT"), default=0.02),
            default_take_profit_pct=_parse_float(os.getenv("DEFAULT_TAKE_PROFIT_PCT"), default=0.04),
            default_position_size=_parse_float(os.getenv("DEFAULT_POSITION_SIZE"), default=0.02),
            include_coin_description=_parse_bool(os.getenv("INCLUDE_COIN_DESCRIPTION"), default=False),
            # Discord
            discord_bot_enabled=_parse_bool(os.getenv("DISCORD_BOT_ENABLED"), default=False),
            bot_token_discord=os.getenv("BOT_TOKEN_DISCORD", "").strip(),
            guild_id_discord=os.getenv("GUILD_ID_DISCORD", "").strip(),
            main_channel_id=os.getenv("MAIN_CHANNEL_ID", "").strip(),
            temporary_channel_id_discord=os.getenv("TEMPORARY_CHANNEL_ID_DISCORD", "").strip(),
            admin_user_ids=parsed_admin_ids,
            file_message_expiry=_parse_int(os.getenv("FILE_MESSAGE_EXPIRY_HOURS"), default=168) * 3600,
            # Trading symbols & order management
            trading_symbols=_parse_list(os.getenv("TRADING_SYMBOLS"), default=["BTCUSDT", "ETHUSDT"]),
            limit_order_timeout_seconds=_parse_int(os.getenv("LIMIT_ORDER_TIMEOUT_SECONDS"), default=300),
            trade_memory_max_entries=_parse_int(os.getenv("TRADE_MEMORY_MAX_ENTRIES"), default=500),
            # Risk guards
            max_daily_loss_pct=_parse_float(os.getenv("MAX_DAILY_LOSS_PCT"), default=0.05),
            max_consecutive_losses=_parse_int(os.getenv("MAX_CONSECUTIVE_LOSSES"), default=3),
            min_confidence_threshold=_parse_float(os.getenv("MIN_CONFIDENCE_THRESHOLD"), default=0.0),
            # Trailing stop
            trailing_stop_enabled=_parse_bool(os.getenv("TRAILING_STOP_ENABLED"), default=False),
            trailing_stop_activation_pct=_parse_float(os.getenv("TRAILING_STOP_ACTIVATION_PCT"), default=0.01),
            trailing_stop_distance_pct=_parse_float(os.getenv("TRAILING_STOP_DISTANCE_PCT"), default=0.005),
            # Partial TP
            partial_tp_enabled=_parse_bool(os.getenv("PARTIAL_TP_ENABLED"), default=False),
            partial_tp1_atr_multiplier=_parse_float(os.getenv("PARTIAL_TP1_ATR_MULTIPLIER"), default=2.0),
            partial_tp1_size_pct=_parse_float(os.getenv("PARTIAL_TP1_SIZE_PCT"), default=0.5),
            # Regime detection
            choppiness_threshold=_parse_float(os.getenv("CHOPPINESS_THRESHOLD"), default=61.8),
            # Signal RSI thresholds
            signal_rsi_strong_buy=_parse_float(os.getenv("SIGNAL_RSI_STRONG_BUY"), default=30.0),
            signal_rsi_buy=_parse_float(os.getenv("SIGNAL_RSI_BUY"), default=40.0),
            signal_rsi_sell=_parse_float(os.getenv("SIGNAL_RSI_SELL"), default=60.0),
            signal_rsi_strong_sell=_parse_float(os.getenv("SIGNAL_RSI_STRONG_SELL"), default=70.0),
            # Multi-timeframe
            htf_timeframe=os.getenv("HTF_TIMEFRAME", "4h").strip(),
            htf_confirmation_enabled=_parse_bool(os.getenv("HTF_CONFIRMATION_ENABLED"), default=False),
            single_symbol_decision=_parse_bool(os.getenv("SINGLE_SYMBOL_DECISION"), default=True),
            # Execution validation
            max_slippage_pct=_parse_float(os.getenv("MAX_SLIPPAGE_PCT"), default=0.005),
            # Fast position monitor
            position_monitor_enabled=_parse_bool(os.getenv("POSITION_MONITOR_ENABLED"), default=True),
            position_monitor_interval=_parse_int(os.getenv("POSITION_MONITOR_INTERVAL"), default=15),
            # Spot bracket orders
            spot_sl_limit_offset_pct=_parse_float(os.getenv("SPOT_SL_LIMIT_OFFSET_PCT"), default=0.01),
            spot_tp_limit_offset_pct=_parse_float(os.getenv("SPOT_TP_LIMIT_OFFSET_PCT"), default=0.002),
            # Futures leverage
            futures_leverage=_parse_int(os.getenv("FUTURES_LEVERAGE"), default=1),
            # Signal Scorer
            use_signal_scorer=_parse_bool(os.getenv("USE_SIGNAL_SCORER"), default=False),
            scoring_entry_threshold=_parse_float(os.getenv("SCORING_ENTRY_THRESHOLD"), default=0.30),
            scoring_exit_threshold=_parse_float(os.getenv("SCORING_EXIT_THRESHOLD"), default=0.20),
            scoring_w_signal=_parse_float(os.getenv("SCORING_W_SIGNAL"), default=0.25),
            scoring_w_direction=_parse_float(os.getenv("SCORING_W_DIRECTION"), default=0.25),
            scoring_w_trend=_parse_float(os.getenv("SCORING_W_TREND"), default=0.15),
            scoring_w_momentum=_parse_float(os.getenv("SCORING_W_MOMENTUM"), default=0.15),
            scoring_w_volume=_parse_float(os.getenv("SCORING_W_VOLUME"), default=0.10),
            scoring_w_key_levels=_parse_float(os.getenv("SCORING_W_KEY_LEVELS"), default=0.10),
            scoring_choppiness_penalty=_parse_float(os.getenv("SCORING_CHOPPINESS_PENALTY"), default=0.3),
            reentry_cooldown_cycles=int(os.getenv("REENTRY_COOLDOWN_CYCLES", "3")),
            # ML / OHLCV
            ml_timeframe=os.getenv("ML_TIMEFRAME", "4h").strip().lower(),
            # Debug / directories
            logger_debug=_parse_bool(os.getenv("LOGGER_DEBUG"), default=False),
            debug_save_charts=_parse_bool(os.getenv("DEBUG_SAVE_CHARTS"), default=False),
            debug_chart_save_path=os.getenv("DEBUG_CHART_SAVE_PATH", "test_images").strip(),
            log_dir=os.getenv("LOG_DIR", "logs").strip(),
            data_dir=os.getenv("DATA_DIR", "data").strip(),
            # RAG tuning
            rag_update_interval_hours=_parse_int(os.getenv("RAG_UPDATE_INTERVAL_HOURS"), default=4),
            rag_categories_update_interval_hours=_parse_int(os.getenv("RAG_CATEGORIES_UPDATE_INTERVAL_HOURS"), default=24),
            rag_coingecko_update_interval_hours=_parse_int(os.getenv("RAG_COINGECKO_UPDATE_INTERVAL_HOURS"), default=24),
            rag_defillama_update_interval_hours=_parse_float(os.getenv("RAG_DEFILLAMA_UPDATE_INTERVAL_HOURS"), default=0.25),
            rag_news_limit=_parse_int(os.getenv("RAG_NEWS_LIMIT"), default=5),
            rag_article_max_tokens=_parse_int(os.getenv("RAG_ARTICLE_MAX_TOKENS"), default=256),
            rag_density_penalty_threshold=_parse_int(os.getenv("RAG_DENSITY_PENALTY_THRESHOLD"), default=300),
            rag_density_boost_threshold=_parse_int(os.getenv("RAG_DENSITY_BOOST_THRESHOLD"), default=1000),
            rag_density_penalty_multiplier=_parse_float(os.getenv("RAG_DENSITY_PENALTY_MULTIPLIER"), default=0.5),
            rag_density_boost_multiplier=_parse_float(os.getenv("RAG_DENSITY_BOOST_MULTIPLIER"), default=1.2),
            rag_cooccurrence_multiplier=_parse_float(os.getenv("RAG_COOCCURRENCE_MULTIPLIER"), default=1.5),
            # RAG retrieval counts
            rag_retrieval_news=_parse_int(os.getenv("RAG_RETRIEVAL_NEWS"), default=3),
            rag_retrieval_macro=_parse_int(os.getenv("RAG_RETRIEVAL_MACRO"), default=1),
            rag_retrieval_memory=_parse_int(os.getenv("RAG_RETRIEVAL_MEMORY"), default=2),
        )

    def ohlcv_csv_path(self, symbol: str, interval: str) -> str:
        """Return the canonical path for an OHLCV CSV file.

        e.g. ohlcv_csv_path("BTC/USDT", "4h") → "data/ohlcv/btcusdt_4h.csv"
        """
        sym = symbol.replace("/", "").lower()
        return f"{self.data_dir}/ohlcv/{sym}_{interval}.csv"

    def get_model_config(self, model_name: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
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
    def _require_str(env_name: str) -> str:
        value = os.getenv(env_name)
        if value is None or value.strip() == "":
            raise ValueError(env_name)
        return value.strip()

    @staticmethod
    def _redact(value: str) -> str:
        if not value:
            return "***"
        if len(value) <= 4:
            return "*" * len(value)
        return f"{value[:2]}***{value[-2:]}"

    def _validate_required_fields(self) -> None:
        for field_name in self.REQUIRED_FIELDS:
            value = getattr(self, field_name)
            if not isinstance(value, str) or value.strip() == "":
                raise ValueError(field_name)

        # Provider-conditional credential validation
        if self.provider == "azure":
            for azure_field in ("azure_endpoint", "azure_api_key", "azure_deployment"):
                value = getattr(self, azure_field)
                if not isinstance(value, str) or value.strip() == "":
                    raise ValueError(azure_field.upper())

        _provider_keys: dict[str, str] = {
            "openrouter": "openrouter_api_key",
            "googleai": "google_studio_api_key",
            "blockrun": "blockrun_wallet_key",
        }
        required_key = _provider_keys.get(self.provider)
        if required_key:
            value = getattr(self, required_key, "")
            if not isinstance(value, str) or value.strip() == "":
                raise ValueError(required_key.upper())

    def _validate_ranges(self) -> None:
        if self.provider not in _VALID_PROVIDERS:
            raise ValueError("provider")

        if not (1 <= self.futures_leverage <= 125):
            raise ValueError("futures_leverage must be between 1 and 125")

        if self.ml_timeframe not in {"1m", "5m", "15m", "30m", "1h", "4h"}:
            raise ValueError("ml_timeframe must be one of: 1m, 5m, 15m, 30m, 1h, 4h")

        if self.binance_product not in {"spot", "usdt_futures"}:
            raise ValueError("binance_product")

        if self.max_order_usdt <= 0 or self.max_order_usdt > 10000:
            raise ValueError("max_order_usdt")

        if self.bot_interval_seconds < 0:
            raise ValueError("bot_interval_seconds must be >= 0 (0 = auto-align to candle)")


__all__ = ["Settings"]
