"""Catalog dispatch + JSON-safe coercion of indicator outputs."""

from __future__ import annotations

import numpy as np
import pytest

from src.mcp_servers.analysis_mcp import indicator_catalog
from src.mcp_servers.analysis_mcp.indicator_catalog import (
    _to_json_list,
    available,
    run_indicator,
)
from src.mcp_servers.analysis_mcp.indicators import TechnicalIndicators


def _ohlcv_array(n: int = 80, *, seed: int = 0) -> np.ndarray:
    """Build a 5-column OHLCV ndarray for TechnicalIndicators.get_data()."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, size=n))
    high = close + np.abs(rng.normal(0, 0.3, size=n))
    low = close - np.abs(rng.normal(0, 0.3, size=n))
    open_ = close + rng.normal(0, 0.2, size=n)
    volume = np.abs(rng.normal(1000, 200, size=n))
    return np.column_stack([open_, high, low, close, volume])


@pytest.fixture
def ti() -> TechnicalIndicators:
    instance = TechnicalIndicators()
    instance.get_data(_ohlcv_array(120))
    return instance


class TestCatalogShape:
    @pytest.mark.parametrize(
        "category", ["momentum", "trend", "volatility", "volume", "structure", "statistical"]
    )
    def test_categories_expose_at_least_one_indicator(self, category: str) -> None:
        assert len(available(category)) > 0

    def test_available_returns_sorted(self) -> None:
        names = available("momentum")
        assert names == sorted(names)

    def test_unknown_category_returns_empty(self) -> None:
        assert available("nonsense") == []


class TestRunIndicator:
    def test_dispatches_single_series(self, ti: TechnicalIndicators) -> None:
        series, params = run_indicator(ti, "momentum", "rsi", None)
        assert len(series) == 1
        label, values = series[0]
        assert label == "rsi"
        assert len(values) == 120
        assert params == {"length": 14}

    def test_dispatches_multi_series_macd(self, ti: TechnicalIndicators) -> None:
        series, _ = run_indicator(ti, "momentum", "macd", None)
        labels = [label for label, _ in series]
        assert labels == ["macd", "signal", "hist"]

    def test_unknown_indicator_raises(self, ti: TechnicalIndicators) -> None:
        with pytest.raises(ValueError, match="unknown indicator"):
            run_indicator(ti, "momentum", "nope", None)

    def test_unknown_category_raises(self, ti: TechnicalIndicators) -> None:
        with pytest.raises(ValueError, match="unknown indicator"):
            run_indicator(ti, "fake", "rsi", None)

    def test_caller_params_override_defaults(self, ti: TechnicalIndicators) -> None:
        _, merged = run_indicator(ti, "momentum", "rsi", {"length": 21})
        assert merged == {"length": 21}

    def test_int_default_coerces_string_param(self, ti: TechnicalIndicators) -> None:
        # Defaults typed as int force coercion so handlers can pass JSON strings.
        _, merged = run_indicator(ti, "momentum", "rsi", {"length": "21"})
        assert merged["length"] == 21
        assert isinstance(merged["length"], int)

    def test_unknown_param_passes_through(self, ti: TechnicalIndicators) -> None:
        with pytest.raises(TypeError):
            run_indicator(ti, "momentum", "rsi", {"bogus": 1})


class TestToJsonList:
    def test_finite_values_pass_through(self) -> None:
        assert _to_json_list(np.array([1.0, 2.5, -3.0])) == [1.0, 2.5, -3.0]

    def test_nan_becomes_none(self) -> None:
        out = _to_json_list(np.array([1.0, np.nan, 2.0]))
        assert out == [1.0, None, 2.0]

    def test_inf_becomes_none(self) -> None:
        out = _to_json_list(np.array([np.inf, -np.inf, 0.0]))
        assert out == [None, None, 0.0]

    def test_none_returns_empty_list(self) -> None:
        assert _to_json_list(None) == []

    def test_2d_arrays_get_flattened(self) -> None:
        out = _to_json_list(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert out == [1.0, 2.0, 3.0, 4.0]


def test_catalog_dict_covers_every_documented_category() -> None:
    # The handlers iterate over these — guard against accidental drops.
    expected = {"momentum", "trend", "volatility", "volume", "structure", "statistical"}
    assert set(indicator_catalog._CATALOG) == expected
