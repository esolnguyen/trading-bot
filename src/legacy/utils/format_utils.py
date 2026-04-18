"""Formatting utilities for prices, percentages, and timestamps."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from src.mcp_servers.shared.utils.data_utils import get_indicator_value

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False


SCIENTIFIC_NOTATION_THRESHOLD = 1e-7
CRYPTO_DUST_THRESHOLD = 1e-5
MICRO_VALUE_THRESHOLD = 1e-4
MILLI_VALUE_THRESHOLD = 1e-3
CENT_VALUE_THRESHOLD = 1e-2
DIME_VALUE_THRESHOLD = 0.1
FULL_PRECISION_THRESHOLD = 10.0
CLEAN_NUMBER_CHARS = ("$", "€", "£", "%", ",")


def timestamps_from_ms_array(timestamps_ms: Any) -> List[datetime]:
    """Convert array of millisecond timestamps to list of datetime objects."""
    if _HAS_PANDAS and _HAS_NUMPY:
        return (
            pd.to_datetime(timestamps_ms, unit="ms", utc=True).to_pydatetime().tolist()
        )
    return [datetime.utcfromtimestamp(ts / 1000) for ts in timestamps_ms]


class FormatUtils:
    """Centralized formatter for technical analysis data and values."""

    def __init__(self, default_precision: int = 8) -> None:
        self.default_precision = default_precision

    def parse_value(self, value: Any, default: Any = None) -> float:
        """Parse various numeric formats into a clean float."""
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return default
        clean = value.strip()
        for char in CLEAN_NUMBER_CHARS:
            clean = clean.replace(char, "")
        try:
            return float(clean)
        except ValueError:
            return default

    def fmt(self, val: float | None, precision: int | None = None) -> str:
        """Format a value with precision based on magnitude."""
        if val is None:
            return "N/A"
        eff = precision if precision is not None else self.default_precision
        if _HAS_NUMPY and np.isnan(val):
            return "N/A"
        abs_val = abs(val)
        if 0 < abs_val < SCIENTIFIC_NOTATION_THRESHOLD:
            return f"{val:.{eff}e}"
        if abs_val < CRYPTO_DUST_THRESHOLD:
            return f"{val:.8f}"
        if abs_val < MICRO_VALUE_THRESHOLD:
            return f"{val:.7f}"
        if abs_val < MILLI_VALUE_THRESHOLD:
            return f"{val:.6f}"
        if abs_val < CENT_VALUE_THRESHOLD:
            return f"{val:.5f}"
        if abs_val < DIME_VALUE_THRESHOLD:
            return f"{val:.4f}"
        if abs_val < FULL_PRECISION_THRESHOLD:
            return f"{val:.{eff}f}"
        return f"{val:.2f}"

    def fmt_ta(
        self, td: dict, key: str, precision: int | None = None, default: str = "N/A"
    ) -> str:
        """Format a technical-analysis indicator value from a data dict."""
        eff = precision if precision is not None else self.default_precision
        val = get_indicator_value(td, key)
        if isinstance(val, (int, float)):
            if _HAS_NUMPY and np.isnan(val):
                return default
            return self.fmt(val, eff)
        return default

    def format_timestamp(self, timestamp_ms: Any) -> str:
        try:
            dt = datetime.fromtimestamp(timestamp_ms / 1000)
            return dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError, OSError):
            return "N/A"

    def format_current_time(self, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        return datetime.now().strftime(format_str)

    def format_timestamp_seconds(
        self, timestamp_sec: float, format_str: str = "%Y-%m-%d"
    ) -> str:
        try:
            return datetime.fromtimestamp(timestamp_sec).strftime(format_str)
        except (ValueError, TypeError, OSError):
            return "N/A"

    def format_date_from_timestamp(self, timestamp_sec: float) -> str:
        return self.format_timestamp_seconds(timestamp_sec, "%Y-%m-%d")

    def timestamp_from_iso(self, iso_str: str) -> float:
        try:
            if iso_str.endswith("Z"):
                iso_str = iso_str[:-1] + "+00:00"
            return datetime.fromisoformat(iso_str).timestamp()
        except (ValueError, TypeError, AttributeError):
            return 0.0

    def parse_timestamp(self, timestamp_field: Any) -> float:
        if timestamp_field is None:
            return 0.0
        if isinstance(timestamp_field, (int, float)):
            return float(timestamp_field)
        if isinstance(timestamp_field, str):
            if timestamp_field.isdigit():
                return float(timestamp_field)
            return self.timestamp_from_iso(timestamp_field)
        return 0.0

    def parse_timestamp_ms(self, timestamp_ms: float) -> Optional[datetime]:
        try:
            return datetime.fromtimestamp(timestamp_ms / 1000)
        except (ValueError, TypeError, OSError):
            return None

    def get_supertrend_direction_string(self, direction: Any) -> str:
        if direction > 0:
            return "Bullish"
        if direction < 0:
            return "Bearish"
        return "Neutral"

    def format_bollinger_interpretation(self, td: dict) -> str:
        bb_position = get_indicator_value(td, "bb_position")
        if isinstance(bb_position, (int, float)):
            if bb_position > 0.8:
                return " [Near upper band - possible overbought]"
            if bb_position < 0.2:
                return " [Near lower band - possible oversold]"
            return " [Within normal range]"
        return ""

    def format_cmf_interpretation(self, td: dict) -> str:
        cmf_val = get_indicator_value(td, "cmf")
        if isinstance(cmf_val, (int, float)):
            if cmf_val > 0.1:
                return " [Accumulation phase]"
            if cmf_val < -0.1:
                return " [Distribution phase]"
            return " [Neutral]"
        return ""


__all__ = ["FormatUtils", "timestamps_from_ms_array"]
