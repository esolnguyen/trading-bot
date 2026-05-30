"""Train/serve feature parity.

Each ML model computes its inputs twice — offline in the trainer and live
in the serving path. These tests pin the two to the *same* numbers so a
future edit to one side can't silently reintroduce train/serve skew. They
import only the pure-numpy/pandas ``src.features`` kernel, so they run
without the binance/xgboost extras.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.features.microstructure import (
    ANOMALY_FEATURE_COLS,
    anomaly_features_frame,
    anomaly_features_from_candles,
)
from src.features.obv import normalized_slope, obv_series, rolling_normalized_obv_slope
from src.features.outcome import (
    bb_position_label,
    bucket_bb_pos,
    bucket_trend,
    bucket_vol_state,
    outcome_row_from_conditions,
    trend_label,
    vol_state_label,
)
from src.features.regime import (
    REGIME_FEATURE_COLS,
    compute_regime_features,
    regime_features_from_candles,
)


@dataclass
class _Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _synthetic_ohlcv(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, n)
    close = 30_000 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    vol = rng.lognormal(10, 0.5, n)
    return pd.DataFrame(
        {
            "timestamp": np.arange(n) * 86_400_000,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


def _candles(df: pd.DataFrame) -> list[_Candle]:
    return [
        _Candle(int(r.timestamp), r.open, r.high, r.low, r.close, r.volume)
        for r in df.itertuples()
    ]


class TestObvSlopeParity:
    def test_rolling_last_equals_live_window(self) -> None:
        df = _synthetic_ohlcv()
        closes = df["close"].to_numpy()
        volumes = df["volume"].to_numpy()

        rolling = rolling_normalized_obv_slope(closes, volumes, period=20)
        # Live path: normalized_slope over the trailing window of OBV.
        obv = obv_series(closes, volumes)
        live = normalized_slope(obv[-20:])

        assert rolling[-1] == pytest.approx(live)

    def test_matches_indicator_calculator(self) -> None:
        # The live indicator calculator must reproduce the shared definition.
        ic = pytest.importorskip(
            "src.mcp_servers.shared.services.indicator_calculator",
            reason="IndicatorCalculator pulls optional deps",
        )
        df = _synthetic_ohlcv()
        closes = list(df["close"])
        volumes = list(df["volume"])
        obv = obv_series(closes, volumes)
        expected = normalized_slope(obv[-20:])
        assert ic.IndicatorCalculator._obv_slope(closes, volumes, 20) == pytest.approx(
            expected
        )


class TestAnomalyParity:
    def test_frame_last_row_equals_live_row(self) -> None:
        df = _synthetic_ohlcv()
        feat = anomaly_features_frame(df.copy())
        live = anomaly_features_from_candles(_candles(df))
        for col in ANOMALY_FEATURE_COLS:
            assert live[col] == pytest.approx(float(feat[col].iloc[-1])), col


class TestRegimeParity:
    def test_from_candles_equals_compute_last_row(self) -> None:
        df = _synthetic_ohlcv()
        computed = compute_regime_features(df.copy(), timeframe="1d")
        live = regime_features_from_candles(_candles(df), timeframe="1d")
        for col in REGIME_FEATURE_COLS:
            ref = computed[col].iloc[-1]
            ref = 0.0 if pd.isna(ref) else float(ref)
            assert live[col] == pytest.approx(ref), col

    def test_adx_is_not_hardcoded_zero(self) -> None:
        # Regression guard for the old `adx_14 = 0.0` serving bug.
        df = _synthetic_ohlcv()
        live = regime_features_from_candles(_candles(df), timeframe="1d")
        assert live["adx_14"] != 0.0


class TestOutcomeBucketParity:
    def test_trainer_bucket_matches_live_label_mapping(self) -> None:
        # Trainer produces the numeric value directly; the live path goes
        # continuous → label → numeric. Both must agree on every regime.
        cases = [
            # (ema_spread, adx, vol_ratio, bb_pos)
            (50.0, 30.0, 2.0, 0.9),
            (-50.0, 30.0, 1.0, 0.1),
            (50.0, 10.0, 0.5, 0.5),   # weak ADX ⇒ neutral trend
            (0.0, 30.0, 1.5, 0.66),   # boundary values
        ]
        for ema_spread, adx, vol_ratio, bb_pos in cases:
            trend_num = float(bucket_trend(ema_spread, adx))
            vol_num = float(bucket_vol_state(vol_ratio))
            bb_num = float(bucket_bb_pos(bb_pos))

            conditions = {
                "trend_direction": trend_label(ema_spread, adx),
                "volume_state": vol_state_label(vol_ratio),
                "bb_position": bb_position_label(bb_pos),
            }
            row = outcome_row_from_conditions(
                conditions, ["trend", "vol_state", "bb_pos"]
            )
            assert row == pytest.approx([trend_num, vol_num, bb_num])
