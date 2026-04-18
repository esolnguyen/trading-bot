"""Build a Settings instance from environment variables."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from src.mcp_servers.config.parsers import (
    parse_bool,
    parse_bot_mode,
    parse_float,
    parse_int,
    parse_list,
    parse_trading_engine,
    require_str,
)

if TYPE_CHECKING:
    from src.mcp_servers.config.settings import Settings


_DEFAULT_TRADER_SKILLS = [
    "candlestick",
    "technical-basic",
    "smc",
    "crypto-derivatives",
    "perp-funding-basis",
]


def load_settings_from_env(
    cls: type["Settings"], env_file: str | None = None
) -> "Settings":
    """Construct a ``Settings`` instance from os.environ + optional ``.env``."""
    load_dotenv(dotenv_path=env_file, override=False)

    bot_mode = parse_bot_mode(os.getenv("BOT_MODE"))
    trading_engine = parse_trading_engine(os.getenv("TRADING_ENGINE"))
    trader_skills = parse_list(
        os.getenv("TRADER_SKILLS") or os.getenv("LLM_TRADER_SKILLS"),
        default=list(_DEFAULT_TRADER_SKILLS),
    )

    admin_ids = parse_list(os.getenv("ADMIN_USER_IDS"), default=[])
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
        binance_api_key=require_str("BINANCE_API_KEY"),
        binance_api_secret=require_str("BINANCE_API_SECRET"),
        binance_product=os.getenv("BINANCE_PRODUCT", "usdt_futures").strip().lower(),
        binance_base_url=os.getenv("BINANCE_BASE_URL", "").strip(),
        binance_testnet=parse_bool(os.getenv("BINANCE_TESTNET"), default=True),
        # Bot
        bot_interval_seconds=parse_int(os.getenv("BOT_INTERVAL_SECONDS"), default=0),
        max_order_usdt=parse_float(os.getenv("MAX_ORDER_USDT"), default=50.0),
        bot_mode=bot_mode,
        model_supports_vision=parse_bool(
            os.getenv("MODEL_SUPPORTS_VISION"), default=False
        ),
        # RAG / ChromaDB
        cryptocompare_api_key=require_str("CRYPTOCOMPARE_API_KEY"),
        coingecko_api_key=os.getenv("COINGECKO_API_KEY", "").strip(),
        chroma_path=os.getenv("CHROMA_PATH", "./chroma_db"),
        news_interval=parse_int(os.getenv("NEWS_INTERVAL"), default=900),
        macro_interval=parse_int(os.getenv("MACRO_INTERVAL"), default=1800),
        ohlcv_interval=parse_int(os.getenv("OHLCV_INTERVAL"), default=3600),
        # AI provider selection
        provider=os.getenv("PROVIDER", "azure").strip().lower(),
        # Google
        google_studio_api_key=os.getenv("GOOGLE_STUDIO_API_KEY", "").strip(),
        google_studio_paid_api_key=os.getenv("GOOGLE_STUDIO_PAID_API_KEY", "").strip(),
        google_studio_model=os.getenv(
            "GOOGLE_STUDIO_MODEL", "gemini-2.5-flash"
        ).strip(),
        # OpenRouter
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).strip(),
        openrouter_base_model=os.getenv(
            "OPENROUTER_BASE_MODEL", "google/gemini-2.5-pro"
        ).strip(),
        openrouter_fallback_model=os.getenv(
            "OPENROUTER_FALLBACK_MODEL", "deepseek/deepseek-r1:free"
        ).strip(),
        # LM Studio
        lm_studio_base_url=os.getenv(
            "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
        ).strip(),
        lm_studio_model=os.getenv("LM_STUDIO_MODEL", "local-model").strip(),
        lm_studio_streaming=parse_bool(os.getenv("LM_STUDIO_STREAMING"), default=True),
        # BlockRun
        blockrun_wallet_key=os.getenv("BLOCKRUN_WALLET_KEY", "").strip(),
        blockrun_base_url=os.getenv(
            "BLOCKRUN_BASE_URL", "https://blockrun.ai/api"
        ).strip(),
        blockrun_model=os.getenv("BLOCKRUN_MODEL", "openai/gpt-4o").strip(),
        # Model tuning
        model_temperature=parse_float(os.getenv("MODEL_TEMPERATURE"), default=0.7),
        model_max_tokens=parse_int(os.getenv("MODEL_MAX_TOKENS"), default=8192),
        model_top_p=parse_float(os.getenv("MODEL_TOP_P"), default=0.9),
        model_top_k=parse_int(os.getenv("MODEL_TOP_K"), default=40),
        model_freq_penalty=parse_float(os.getenv("MODEL_FREQ_PENALTY"), default=0.0),
        model_pres_penalty=parse_float(os.getenv("MODEL_PRES_PENALTY"), default=0.0),
        google_max_tokens=parse_int(os.getenv("GOOGLE_MAX_TOKENS"), default=32768),
        google_temperature=parse_float(os.getenv("GOOGLE_TEMPERATURE"), default=0.7),
        google_top_p=parse_float(os.getenv("GOOGLE_TOP_P"), default=0.9),
        google_top_k=parse_int(os.getenv("GOOGLE_TOP_K"), default=40),
        google_thinking_level=os.getenv("GOOGLE_THINKING_LEVEL", "high").strip(),
        google_code_execution=parse_bool(
            os.getenv("GOOGLE_CODE_EXECUTION"), default=False
        ),
        # Trading parameters
        timeframe=os.getenv("TIMEFRAME", "1h").strip(),
        candle_limit=parse_int(os.getenv("CANDLE_LIMIT"), default=200),
        ai_chart_candle_limit=parse_int(
            os.getenv("AI_CHART_CANDLE_LIMIT"), default=120
        ),
        ai_chart_timeframe=os.getenv("AI_CHART_TIMEFRAME", "5m").strip(),
        demo_quote_capital=parse_float(
            os.getenv("DEMO_QUOTE_CAPITAL"), default=10000.0
        ),
        transaction_fee_percent=parse_float(
            os.getenv("TRANSACTION_FEE_PERCENT"), default=0.00075
        ),
        default_stop_loss_pct=parse_float(
            os.getenv("DEFAULT_STOP_LOSS_PCT"), default=0.02
        ),
        default_take_profit_pct=parse_float(
            os.getenv("DEFAULT_TAKE_PROFIT_PCT"), default=0.04
        ),
        default_position_size=parse_float(
            os.getenv("DEFAULT_POSITION_SIZE"), default=0.02
        ),
        include_coin_description=parse_bool(
            os.getenv("INCLUDE_COIN_DESCRIPTION"), default=False
        ),
        # Discord
        discord_bot_enabled=parse_bool(os.getenv("DISCORD_BOT_ENABLED"), default=False),
        bot_token_discord=os.getenv("BOT_TOKEN_DISCORD", "").strip(),
        guild_id_discord=os.getenv("GUILD_ID_DISCORD", "").strip(),
        main_channel_id=os.getenv("MAIN_CHANNEL_ID", "").strip(),
        temporary_channel_id_discord=os.getenv(
            "TEMPORARY_CHANNEL_ID_DISCORD", ""
        ).strip(),
        admin_user_ids=parsed_admin_ids,
        file_message_expiry=parse_int(
            os.getenv("FILE_MESSAGE_EXPIRY_HOURS"), default=168
        )
        * 3600,
        # Trading symbols & order management
        trading_symbols=parse_list(
            os.getenv("TRADING_SYMBOLS"), default=["BTCUSDT", "ETHUSDT"]
        ),
        limit_order_timeout_seconds=parse_int(
            os.getenv("LIMIT_ORDER_TIMEOUT_SECONDS"), default=300
        ),
        trade_memory_max_entries=parse_int(
            os.getenv("TRADE_MEMORY_MAX_ENTRIES"), default=500
        ),
        # Risk guards
        max_daily_loss_pct=parse_float(os.getenv("MAX_DAILY_LOSS_PCT"), default=0.05),
        max_consecutive_losses=parse_int(
            os.getenv("MAX_CONSECUTIVE_LOSSES"), default=3
        ),
        min_confidence_threshold=parse_float(
            os.getenv("MIN_CONFIDENCE_THRESHOLD"), default=0.0
        ),
        # Trailing stop
        trailing_stop_enabled=parse_bool(
            os.getenv("TRAILING_STOP_ENABLED"), default=False
        ),
        trailing_stop_activation_pct=parse_float(
            os.getenv("TRAILING_STOP_ACTIVATION_PCT"), default=0.01
        ),
        trailing_stop_distance_pct=parse_float(
            os.getenv("TRAILING_STOP_DISTANCE_PCT"), default=0.005
        ),
        # Partial TP
        partial_tp_enabled=parse_bool(os.getenv("PARTIAL_TP_ENABLED"), default=False),
        partial_tp1_atr_multiplier=parse_float(
            os.getenv("PARTIAL_TP1_ATR_MULTIPLIER"), default=2.0
        ),
        partial_tp1_size_pct=parse_float(
            os.getenv("PARTIAL_TP1_SIZE_PCT"), default=0.5
        ),
        # Regime detection
        choppiness_threshold=parse_float(
            os.getenv("CHOPPINESS_THRESHOLD"), default=61.8
        ),
        # Signal RSI thresholds
        signal_rsi_strong_buy=parse_float(
            os.getenv("SIGNAL_RSI_STRONG_BUY"), default=30.0
        ),
        signal_rsi_buy=parse_float(os.getenv("SIGNAL_RSI_BUY"), default=40.0),
        signal_rsi_sell=parse_float(os.getenv("SIGNAL_RSI_SELL"), default=60.0),
        signal_rsi_strong_sell=parse_float(
            os.getenv("SIGNAL_RSI_STRONG_SELL"), default=70.0
        ),
        # Multi-timeframe
        htf_timeframe=os.getenv("HTF_TIMEFRAME", "4h").strip(),
        htf_confirmation_enabled=parse_bool(
            os.getenv("HTF_CONFIRMATION_ENABLED"), default=False
        ),
        single_symbol_decision=parse_bool(
            os.getenv("SINGLE_SYMBOL_DECISION"), default=True
        ),
        # Execution validation
        max_slippage_pct=parse_float(os.getenv("MAX_SLIPPAGE_PCT"), default=0.005),
        # Fast position monitor
        position_monitor_enabled=parse_bool(
            os.getenv("POSITION_MONITOR_ENABLED"), default=True
        ),
        position_monitor_interval=parse_int(
            os.getenv("POSITION_MONITOR_INTERVAL"), default=15
        ),
        # Spot bracket orders
        spot_sl_limit_offset_pct=parse_float(
            os.getenv("SPOT_SL_LIMIT_OFFSET_PCT"), default=0.01
        ),
        spot_tp_limit_offset_pct=parse_float(
            os.getenv("SPOT_TP_LIMIT_OFFSET_PCT"), default=0.002
        ),
        # Futures leverage
        futures_leverage=parse_int(os.getenv("FUTURES_LEVERAGE"), default=1),
        # Trading engine
        trading_engine=trading_engine,
        trader_skills=trader_skills,
        scoring_entry_threshold=parse_float(
            os.getenv("SCORING_ENTRY_THRESHOLD"), default=0.30
        ),
        scoring_exit_threshold=parse_float(
            os.getenv("SCORING_EXIT_THRESHOLD"), default=0.20
        ),
        scoring_w_signal=parse_float(os.getenv("SCORING_W_SIGNAL"), default=0.25),
        scoring_w_direction=parse_float(os.getenv("SCORING_W_DIRECTION"), default=0.25),
        scoring_w_trend=parse_float(os.getenv("SCORING_W_TREND"), default=0.15),
        scoring_w_momentum=parse_float(os.getenv("SCORING_W_MOMENTUM"), default=0.15),
        scoring_w_volume=parse_float(os.getenv("SCORING_W_VOLUME"), default=0.10),
        scoring_w_key_levels=parse_float(
            os.getenv("SCORING_W_KEY_LEVELS"), default=0.10
        ),
        scoring_choppiness_penalty=parse_float(
            os.getenv("SCORING_CHOPPINESS_PENALTY"), default=0.3
        ),
        reentry_cooldown_cycles=int(os.getenv("REENTRY_COOLDOWN_CYCLES", "3")),
        # ML / OHLCV
        ml_timeframe=os.getenv("ML_TIMEFRAME", "4h").strip().lower(),
        # Debug / directories
        logger_debug=parse_bool(os.getenv("LOGGER_DEBUG"), default=False),
        debug_save_charts=parse_bool(os.getenv("DEBUG_SAVE_CHARTS"), default=False),
        debug_chart_save_path=os.getenv("DEBUG_CHART_SAVE_PATH", "test_images").strip(),
        log_dir=os.getenv("LOG_DIR", "logs").strip(),
        data_dir=os.getenv("DATA_DIR", "data").strip(),
        # RAG tuning
        rag_update_interval_hours=parse_int(
            os.getenv("RAG_UPDATE_INTERVAL_HOURS"), default=4
        ),
        rag_categories_update_interval_hours=parse_int(
            os.getenv("RAG_CATEGORIES_UPDATE_INTERVAL_HOURS"), default=24
        ),
        rag_coingecko_update_interval_hours=parse_int(
            os.getenv("RAG_COINGECKO_UPDATE_INTERVAL_HOURS"), default=24
        ),
        rag_defillama_update_interval_hours=parse_float(
            os.getenv("RAG_DEFILLAMA_UPDATE_INTERVAL_HOURS"), default=0.25
        ),
        rag_news_limit=parse_int(os.getenv("RAG_NEWS_LIMIT"), default=5),
        rag_article_max_tokens=parse_int(
            os.getenv("RAG_ARTICLE_MAX_TOKENS"), default=256
        ),
        rag_density_penalty_threshold=parse_int(
            os.getenv("RAG_DENSITY_PENALTY_THRESHOLD"), default=300
        ),
        rag_density_boost_threshold=parse_int(
            os.getenv("RAG_DENSITY_BOOST_THRESHOLD"), default=1000
        ),
        rag_density_penalty_multiplier=parse_float(
            os.getenv("RAG_DENSITY_PENALTY_MULTIPLIER"), default=0.5
        ),
        rag_density_boost_multiplier=parse_float(
            os.getenv("RAG_DENSITY_BOOST_MULTIPLIER"), default=1.2
        ),
        rag_cooccurrence_multiplier=parse_float(
            os.getenv("RAG_COOCCURRENCE_MULTIPLIER"), default=1.5
        ),
        # RAG retrieval counts
        rag_retrieval_news=parse_int(os.getenv("RAG_RETRIEVAL_NEWS"), default=3),
        rag_retrieval_macro=parse_int(os.getenv("RAG_RETRIEVAL_MACRO"), default=1),
        rag_retrieval_memory=parse_int(os.getenv("RAG_RETRIEVAL_MEMORY"), default=2),
    )
