"""HistoricalPercentileScorer: CSV cache, mtime invalidation, score format."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.mcp_servers.ml_mcp.services.historical_percentile import (
    HistoricalPercentileScorer,
)


@dataclass
class _StubIndicators:
    rsi_14: float = 50.0
    adx: float = 25.0
    atr: float = 100.0
    volume_sma_20: float = 1000.0


def _write_csv(path: Path, n: int, *, base_price: float = 100.0) -> None:
    rng = np.random.default_rng(42)
    closes = base_price + np.cumsum(rng.normal(0, 1.0, size=n))
    highs = closes + np.abs(rng.normal(0, 0.5, size=n))
    lows = closes - np.abs(rng.normal(0, 0.5, size=n))
    volumes = np.abs(rng.normal(1000, 100, size=n))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["close", "high", "low", "volume"])
        writer.writeheader()
        for i in range(n):
            writer.writerow(
                {
                    "close": closes[i],
                    "high": highs[i],
                    "low": lows[i],
                    "volume": volumes[i],
                }
            )


class TestLoadArrays:
    def test_missing_file_returns_none_score(self, tmp_path: Path) -> None:
        scorer = HistoricalPercentileScorer(
            csv_path=str(tmp_path / "missing.csv"), timeframe="4h"
        )
        assert scorer.score(_StubIndicators(), current_price=100.0) is None

    def test_too_few_rows_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "small.csv"
        _write_csv(path, n=100)  # < 500 threshold
        scorer = HistoricalPercentileScorer(csv_path=str(path), timeframe="4h")
        assert scorer.score(_StubIndicators(), current_price=100.0) is None


class TestScoreOutput:
    def test_score_contains_header_and_sections(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        _write_csv(path, n=700)
        scorer = HistoricalPercentileScorer(csv_path=str(path), timeframe="4h")
        out = scorer.score(_StubIndicators(rsi_14=55.0, adx=30.0, atr=2.0), current_price=100.0)
        assert out is not None
        lines = out.splitlines()
        assert lines[0].startswith("## Historical Context")
        joined = "\n".join(lines)
        assert "RSI(14)" in joined
        assert "ADX(14)" in joined
        assert "Vol SMA(20)" in joined

    def test_score_includes_6m_range_line(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        _write_csv(path, n=700)
        scorer = HistoricalPercentileScorer(csv_path=str(path), timeframe="4h")
        out = scorer.score(_StubIndicators(), current_price=100.0)
        assert out is not None
        assert "Price vs 6M high" in out


class TestCacheInvalidation:
    def test_cache_used_on_repeated_calls(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        _write_csv(path, n=700)
        scorer = HistoricalPercentileScorer(csv_path=str(path), timeframe="4h")
        scorer.score(_StubIndicators(), current_price=100.0)
        first_mtime = scorer._cache_mtime
        scorer.score(_StubIndicators(), current_price=100.0)
        assert scorer._cache_mtime == first_mtime

    def test_invalidate_cache_forces_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        _write_csv(path, n=700)
        scorer = HistoricalPercentileScorer(csv_path=str(path), timeframe="4h")
        scorer.score(_StubIndicators(), current_price=100.0)
        scorer.invalidate_cache()
        assert scorer._cache_mtime == 0.0

    def test_mtime_change_triggers_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        _write_csv(path, n=700)
        scorer = HistoricalPercentileScorer(csv_path=str(path), timeframe="4h")
        scorer.score(_StubIndicators(), current_price=100.0)
        # Bump mtime by 5s and rewrite.
        time.sleep(0.01)
        _write_csv(path, n=700, base_price=200.0)
        new_mtime = path.stat().st_mtime
        # Force OS mtime far enough forward that the equality check fails.
        import os

        os.utime(path, (new_mtime + 10, new_mtime + 10))
        scorer.score(_StubIndicators(), current_price=200.0)
        assert scorer._cache_mtime == new_mtime + 10


class TestTimeframeMapping:
    @pytest.mark.parametrize(
        "tf,expected_window",
        [("15m", 17_280), ("1h", 4_320), ("4h", 1_080)],
    )
    def test_window_for_known_timeframes(self, tf: str, expected_window: int) -> None:
        scorer = HistoricalPercentileScorer(csv_path="x.csv", timeframe=tf)
        assert scorer._window == expected_window

    def test_unknown_timeframe_falls_back_to_4h(self) -> None:
        scorer = HistoricalPercentileScorer(csv_path="x.csv", timeframe="7m")
        assert scorer._window == 1_080
