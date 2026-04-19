"""ML training settings.

Drives *how* models are fit (timeframes, symbol universe, candle
depth). Hyperparameters specific to one model (e.g. anomaly
contamination) live on the model class itself, not here — this file
stays shallow enough to review at a glance.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLTrainingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # Timeframes to pull and train across. `run_training` backfills the
    # CSV for each entry before dispatching per-timeframe model fits.
    # 1d is included so regime / key_levels drivers find their fixed
    # daily CSV without a separate config knob.
    ml_timeframes: list[str] = Field(
        default_factory=lambda: ["15m", "1h", "4h", "1d"],
        alias="ML_TIMEFRAMES",
    )
    candle_limit: int = Field(default=200, alias="CANDLE_LIMIT")

    # Training symbol universe — CSV parsed into a list. Can drift from
    # TraderSettings.trading_symbols intentionally: ingestion usually
    # wants a broader universe than the bot actually trades.
    training_symbols: list[str] = Field(
        default_factory=lambda: ["BTCUSDT", "ETHUSDT"],
        alias="TRAINING_SYMBOLS",
    )
