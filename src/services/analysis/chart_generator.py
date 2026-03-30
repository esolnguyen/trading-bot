"""Render market charts for visual model input."""

from __future__ import annotations

import base64
from io import BytesIO
import os
import tempfile

from src.domain.analysis import IndicatorSet
from src.domain.market import OHLCVCandle


class ChartGenerator:
    """Render a candlestick-style chart as base64 PNG."""

    def render(self, symbol: str, candles: list[OHLCVCandle], indicators: IndicatorSet) -> str:
        if not candles:
            raise ValueError("At least one candle is required")

        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

        try:
            return self._render_with_mplfinance(symbol, candles, indicators)
        except ModuleNotFoundError:
            return self._render_with_matplotlib(symbol, candles, indicators)

    def _render_with_mplfinance(self, symbol: str, candles: list[OHLCVCandle], indicators: IndicatorSet) -> str:
        import pandas as pd
        import mplfinance as mpf

        dataframe = pd.DataFrame(
            {
                "Open": [candle.open for candle in candles],
                "High": [candle.high for candle in candles],
                "Low": [candle.low for candle in candles],
                "Close": [candle.close for candle in candles],
                "Volume": [candle.volume for candle in candles],
            },
            index=pd.to_datetime([candle.timestamp for candle in candles], unit="ms"),
        )

        close_values = dataframe["Close"].tolist()
        volume_values = dataframe["Volume"].tolist()
        rsi_series = [indicators.rsi_14] * len(close_values)
        macd_hist_series = [indicators.macd_hist] * len(close_values)

        add_plots = [
            mpf.make_addplot(rsi_series, panel=2, color="deepskyblue", ylabel="RSI"),
            mpf.make_addplot(macd_hist_series, type="bar", panel=3, color="lightgreen", ylabel="MACD"),
        ]

        figure, _axes = mpf.plot(
            dataframe,
            type="candle",
            style="nightclouds",
            volume=True,
            addplot=add_plots,
            figsize=(12, 8),
            panel_ratios=(4, 1, 1, 1),
            title=symbol,
            returnfig=True,
        )
        return self._figure_to_base64_png(figure)

    def _render_with_matplotlib(self, symbol: str, candles: list[OHLCVCandle], indicators: IndicatorSet) -> str:
        import matplotlib.pyplot as plt

        closes = [candle.close for candle in candles]
        volumes = [candle.volume for candle in candles]
        indices = list(range(len(candles)))
        colors = ["#49d17d" if candle.close >= candle.open else "#ff5c7a" for candle in candles]

        figure, axes = plt.subplots(
            4,
            1,
            figsize=(12, 8),
            dpi=100,
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1, 1, 1]},
        )
        figure.patch.set_facecolor("#10161f")

        for axis in axes:
            axis.set_facecolor("#10161f")
            axis.tick_params(colors="#d8dee9")
            for spine in axis.spines.values():
                spine.set_color("#32404f")

        main_axis, volume_axis, rsi_axis, macd_axis = axes
        main_axis.vlines(indices, [c.low for c in candles], [c.high for c in candles], color=colors, linewidth=1)
        for index, candle in enumerate(candles):
            body_low = min(candle.open, candle.close)
            body_height = abs(candle.close - candle.open) or 0.01
            main_axis.bar(index, body_height, bottom=body_low, color=colors[index], width=0.6)

        main_axis.plot(indices, closes, color="#f6c177", linewidth=1.0, alpha=0.8)
        main_axis.set_title(symbol, color="#ffffff")

        volume_axis.bar(indices, volumes, color=colors, width=0.8)
        rsi_axis.plot(indices, [indicators.rsi_14] * len(indices), color="#5ad1ff", linewidth=1.5)
        rsi_axis.axhline(70, color="#ff5c7a", linestyle="--", linewidth=0.8)
        rsi_axis.axhline(30, color="#49d17d", linestyle="--", linewidth=0.8)
        rsi_axis.set_ylim(0, 100)

        macd_axis.bar(indices, [indicators.macd_hist] * len(indices), color="#8bd450", width=0.8)
        macd_axis.axhline(0, color="#d8dee9", linewidth=0.8)

        figure.tight_layout()
        return self._figure_to_base64_png(figure)

    @staticmethod
    def _figure_to_base64_png(figure) -> str:
        import matplotlib.pyplot as plt

        buffer = BytesIO()
        figure.savefig(buffer, format="png", facecolor=figure.get_facecolor(), bbox_inches="tight")
        plt.close(figure)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("ascii")
