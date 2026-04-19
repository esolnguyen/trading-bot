"""Trading-bot settings.

Owns *how the agent runs*: symbol universe, decision cadence, LLM
choice, bot mode. Order-placement gating happens here — with no
Phase 7 execution MCP in the tree, ``BOT_MODE=live`` is refused at
startup so the loop cannot accidentally graduate to real orders.
"""

from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BotMode = Literal["off", "dry_run", "live"]


class TradingBotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    trading_symbols: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["BTCUSDT"],
        alias="TRADING_SYMBOLS",
    )
    primary_timeframe: str = Field(default="15m", alias="TRADING_TIMEFRAME")
    decision_interval_seconds: int = Field(
        default=900, alias="TRADING_DECISION_INTERVAL_SECONDS"
    )
    max_order_usdt: float = Field(default=50.0, alias="MAX_ORDER_USDT")
    leverage: int = Field(default=10, alias="FUTURES_LEVERAGE")
    min_conviction: int = Field(default=6, alias="TRADING_MIN_CONVICTION")
    bot_mode: BotMode = Field(default="dry_run", alias="BOT_MODE")

    llm_model: str = Field(default="claude-sonnet-4-6", alias="LLM_MODEL")
    llm_max_iterations: int = Field(default=12, alias="LLM_MAX_ITERATIONS")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    @field_validator("bot_mode", mode="before")
    @classmethod
    def _normalize_mode(cls, v: str) -> str:
        return str(v).strip().lower()

    @field_validator("trading_symbols", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        return v

    def assert_runnable(self) -> None:
        """Refuse to start under conditions the codebase can't honour yet."""
        if self.bot_mode == "live":
            raise RuntimeError(
                "BOT_MODE=live is not supported — the order-execution MCP "
                "(Phase 7) has not shipped. Use BOT_MODE=dry_run."
            )
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is empty. Set it in .env before starting."
            )
        if not self.trading_symbols:
            raise RuntimeError("TRADING_SYMBOLS is empty.")
