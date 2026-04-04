"""Timeframe validation and conversion utilities."""

from __future__ import annotations

import re
from typing import Optional, Tuple


class TimeframeValidator:
    """Validates and manages timeframe configurations."""

    SUPPORTED_TIMEFRAMES = [
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "1w",
    ]

    MINUTES_IN_HOUR = 60
    MINUTES_IN_DAY = 1440
    MINUTES_IN_WEEK = 10080

    MS_IN_SECOND = 1000
    MS_IN_MINUTE = 60 * 1000
    MS_IN_HOUR = 60 * 60 * 1000
    MS_IN_DAY = 24 * 60 * 60 * 1000

    # Unix Epoch (1970-01-01) was Thursday; Monday (1970-01-05) is +4 days.
    MONDAY_ALIGNMENT_OFFSET_DAYS = 4

    TIMEFRAME_MINUTES: dict[str, int] = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": MINUTES_IN_HOUR,
        "2h": 2 * MINUTES_IN_HOUR,
        "4h": 4 * MINUTES_IN_HOUR,
        "6h": 6 * MINUTES_IN_HOUR,
        "8h": 8 * MINUTES_IN_HOUR,
        "12h": 12 * MINUTES_IN_HOUR,
        "1d": MINUTES_IN_DAY,
        "1w": MINUTES_IN_WEEK,
    }

    CRYPTOCOMPARE_FORMAT: dict[str, str] = {
        "1h": "hour",
        "2h": "hour",
        "4h": "hour",
        "6h": "hour",
        "8h": "hour",
        "12h": "hour",
        "1d": "day",
    }

    CCXT_STANDARD_TIMEFRAMES = [
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "1w",
    ]

    @classmethod
    def validate(cls, timeframe: str) -> bool:
        return timeframe in cls.SUPPORTED_TIMEFRAMES

    @classmethod
    def to_minutes(cls, timeframe: str) -> int:
        if timeframe not in cls.TIMEFRAME_MINUTES:
            raise ValueError(f"Unrecognized timeframe: {timeframe}")
        return cls.TIMEFRAME_MINUTES[timeframe]

    @classmethod
    def parse_period_to_minutes(cls, period: str) -> int:
        match = re.match(r"^(\d+)([hd])$", period.lower())
        if not match:
            raise ValueError(f"Invalid period format: {period}")
        value = int(match.group(1))
        unit = match.group(2)
        if unit == "h":
            return value * cls.MINUTES_IN_HOUR
        if unit == "d":
            return value * cls.MINUTES_IN_DAY
        raise ValueError(f"Invalid period unit: {unit}")

    @classmethod
    def calculate_period_candles(cls, base_timeframe: str, target_period: str) -> int:
        return cls.parse_period_to_minutes(target_period) // cls.to_minutes(
            base_timeframe
        )

    @classmethod
    def to_cryptocompare_format(cls, timeframe: str) -> Tuple[str, int]:
        if timeframe not in cls.CRYPTOCOMPARE_FORMAT:
            raise ValueError(
                f"Timeframe {timeframe} not supported by CryptoCompare API."
            )
        endpoint = cls.CRYPTOCOMPARE_FORMAT[timeframe]
        if "h" in timeframe:
            multiplier = int(timeframe.replace("h", ""))
        elif "d" in timeframe:
            multiplier = int(timeframe.replace("d", ""))
        else:
            multiplier = 1
        return endpoint, multiplier

    @classmethod
    def is_ccxt_compatible(
        cls, timeframe: str, exchange_name: Optional[str] = None
    ) -> bool:
        return timeframe in cls.CCXT_STANDARD_TIMEFRAMES

    @classmethod
    def get_candle_limit_for_days(cls, timeframe: str, target_days: int = 30) -> int:
        return (target_days * cls.MINUTES_IN_DAY) // cls.to_minutes(timeframe)

    @classmethod
    def validate_and_normalize(cls, timeframe: str) -> str:
        normalized = timeframe.lower()
        if not cls.validate(normalized):
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {', '.join(cls.SUPPORTED_TIMEFRAMES)}"
            )
        return normalized

    @classmethod
    def _get_alignment_offset(cls, timeframe: str) -> int:
        if timeframe == "1w":
            return cls.MONDAY_ALIGNMENT_OFFSET_DAYS * cls.MS_IN_DAY
        return 0

    @classmethod
    def calculate_next_candle_time(cls, current_time_ms: int, timeframe: str) -> int:
        interval_ms = cls.to_minutes(timeframe) * cls.MS_IN_MINUTE
        offset = cls._get_alignment_offset(timeframe)
        aligned_time = current_time_ms - offset
        next_index = (aligned_time // interval_ms) + 1
        return next_index * interval_ms + offset

    @classmethod
    def calculate_wait_duration(
        cls, current_time_ms: int, timeframe: str, buffer_seconds: int = 5
    ) -> float:
        next_candle_ms = cls.calculate_next_candle_time(current_time_ms, timeframe)
        delay_ms = next_candle_ms - current_time_ms + buffer_seconds * cls.MS_IN_SECOND
        return max(0.0, delay_ms / cls.MS_IN_SECOND)

    @classmethod
    def is_same_candle(cls, time1_ms: int, time2_ms: int, timeframe: str) -> bool:
        interval_ms = cls.to_minutes(timeframe) * cls.MS_IN_MINUTE
        offset = cls._get_alignment_offset(timeframe)
        candle1 = (time1_ms - offset) // interval_ms
        candle2 = (time2_ms - offset) // interval_ms
        return candle1 == candle2


__all__ = ["TimeframeValidator"]
