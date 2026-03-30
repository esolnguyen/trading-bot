"""Directional technical analysis built on shared indicators."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.domain.analysis import Signal, TechnicalAnalysis
from src.domain.market import MarketSnapshot
from src.services.analysis.indicator_calculator import IndicatorCalculator


class TechnicalAnalyzer:
    """Produce a directional signal and human-readable reasoning."""

    def __init__(
        self,
        calculator: IndicatorCalculator,
        advanced_calculator: Optional[Any] = None,
    ) -> None:
        self.calculator = calculator
        self.advanced_calculator = advanced_calculator

    def analyze(self, symbol: str, snapshot: MarketSnapshot) -> TechnicalAnalysis:
        indicators = self.calculator.compute(snapshot.candles)
        signal = self._select_signal(snapshot.price, indicators)
        band_relation = self._band_relation(snapshot.price, indicators)
        reasoning = (
            f"RSI={indicators.rsi_14:.2f}; "
            f"MACD_hist={indicators.macd_hist:.4f}; "
            f"price is {band_relation}."
        )
        return TechnicalAnalysis(
            symbol=symbol,
            signal=signal,
            indicators=indicators,
            reasoning=reasoning,
        )

    def get_market_conditions(self, symbol: str, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """Return a summarised market-conditions dict for downstream services.

        Uses the advanced_calculator when available; falls back to the basic
        IndicatorCalculator otherwise.

        Keys returned:
            trend_direction, adx, volatility_level, rsi_level, rsi,
            bb_position, macd_signal, volume_state, atr, atr_percent
        """
        indicators = self.calculator.compute(snapshot.candles)
        adx_value: float = getattr(indicators, "adx", 0.0) or 0.0
        rsi_value: float = float(getattr(indicators, "rsi_14", 50.0) or 50.0)
        atr_value: float = float(getattr(indicators, "atr", 0.0) or 0.0)
        price = snapshot.price

        # Trend direction based on MACD histogram + ADX
        macd_hist: float = float(getattr(indicators, "macd_hist", 0.0) or 0.0)
        if adx_value >= 25:
            trend_direction = "BULLISH" if macd_hist > 0 else "BEARISH"
        else:
            trend_direction = "NEUTRAL"

        # Volatility level based on ATR %
        atr_pct = (atr_value / price * 100) if price > 0 else 0.0
        if atr_pct > 3.0:
            volatility_level = "HIGH"
        elif atr_pct > 1.5:
            volatility_level = "MEDIUM"
        else:
            volatility_level = "LOW"

        # RSI level
        if rsi_value >= 70:
            rsi_level = "OVERBOUGHT"
        elif rsi_value <= 30:
            rsi_level = "OVERSOLD"
        else:
            rsi_level = "NEUTRAL"

        # Bollinger Band position
        bb_upper: float = float(getattr(indicators, "bb_upper", price) or price)
        bb_lower: float = float(getattr(indicators, "bb_lower", price) or price)
        bb_mid: float = float(getattr(indicators, "bb_mid", price) or price)
        if price >= bb_upper:
            bb_position = "UPPER"
        elif price <= bb_lower:
            bb_position = "LOWER"
        elif price > bb_mid:
            bb_position = "UPPER_MIDDLE"
        else:
            bb_position = "LOWER_MIDDLE"

        # MACD signal
        macd_signal_val: float = float(getattr(indicators, "macd_signal", 0.0) or 0.0)
        macd_line: float = float(getattr(indicators, "macd_line", 0.0) or 0.0)
        if macd_line > macd_signal_val and macd_hist > 0:
            macd_signal = "BULLISH"
        elif macd_line < macd_signal_val and macd_hist < 0:
            macd_signal = "BEARISH"
        else:
            macd_signal = "NEUTRAL"

        # Volume state (if OBV slope available)
        obv_slope: float = float(getattr(indicators, "obv_slope", 0.0) or 0.0)
        if obv_slope > 0.5:
            volume_state = "ACCUMULATION"
        elif obv_slope < -0.5:
            volume_state = "DISTRIBUTION"
        else:
            volume_state = "NORMAL"

        return {
            "trend_direction": trend_direction,
            "adx": round(float(adx_value), 2),
            "volatility_level": volatility_level,
            "rsi_level": rsi_level,
            "rsi": round(rsi_value, 2),
            "bb_position": bb_position,
            "macd_signal": macd_signal,
            "volume_state": volume_state,
            "atr": round(atr_value, 6),
            "atr_percent": round(atr_pct, 4),
        }

    @staticmethod
    def _band_relation(price: float, indicators) -> str:
        if price < indicators.bb_lower:
            return "below lower Bollinger band"
        if price > indicators.bb_upper:
            return "above upper Bollinger band"
        if price < indicators.bb_mid:
            return "below middle Bollinger band"
        if price > indicators.bb_mid:
            return "above middle Bollinger band"
        return "at middle Bollinger band"

    def _select_signal(self, price: float, indicators) -> Signal:
        if (
            indicators.rsi_14 < 30
            and price < indicators.bb_lower
            and indicators.macd_hist > 0
        ):
            return Signal.STRONG_BUY
        if indicators.rsi_14 < 40 and price < indicators.bb_mid:
            return Signal.BUY
        if (
            indicators.rsi_14 > 70
            and price > indicators.bb_upper
            and indicators.macd_hist < 0
        ):
            return Signal.STRONG_SELL
        if indicators.rsi_14 > 60 and price > indicators.bb_mid:
            return Signal.SELL
        return Signal.NEUTRAL
