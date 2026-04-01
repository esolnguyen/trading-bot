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

    def analyze(
        self,
        symbol: str,
        snapshot: MarketSnapshot,
        choppiness_threshold: float = 61.8,
        rsi_strong_buy: float = 30.0,
        rsi_buy: float = 40.0,
        rsi_sell: float = 60.0,
        rsi_strong_sell: float = 70.0,
    ) -> TechnicalAnalysis:
        indicators = self.calculator.compute(snapshot.candles)
        signal = self._select_signal(
            snapshot.price, indicators, snapshot, choppiness_threshold,
            rsi_strong_buy, rsi_buy, rsi_sell, rsi_strong_sell,
        )
        band_relation = self._band_relation(snapshot.price, indicators)
        choppy_note = f"; CHOP={indicators.choppiness:.1f}({'choppy' if indicators.choppiness > choppiness_threshold else 'trending'})" if indicators.choppiness else ""
        reasoning = (
            f"RSI={indicators.rsi_14:.2f}; "
            f"MACD_hist={indicators.macd_hist:.4f}; "
            f"ADX={indicators.adx:.1f}; "
            f"ATR={indicators.atr:.2f}; "
            f"price is {band_relation}{choppy_note}."
        )
        return TechnicalAnalysis(
            symbol=symbol,
            signal=signal,
            indicators=indicators,
            reasoning=reasoning,
        )

    def get_market_conditions(self, symbol: str, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """Return a summarised market-conditions dict for downstream services.

        Keys returned:
            trend_direction, adx, volatility_level, rsi_level, rsi,
            bb_position, macd_signal, volume_state, atr, atr_percent,
            choppiness, is_choppy, funding_rate, open_interest
        """
        indicators = self.calculator.compute(snapshot.candles)
        adx_value: float = float(indicators.adx)
        rsi_value: float = float(indicators.rsi_14)
        atr_value: float = float(indicators.atr)
        choppiness: float = float(indicators.choppiness)
        price = snapshot.price

        # Trend direction based on MACD histogram + ADX
        macd_hist: float = float(indicators.macd_hist)
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
        bb_upper: float = float(indicators.bb_upper)
        bb_lower: float = float(indicators.bb_lower)
        bb_mid: float = float(indicators.bb_mid)
        if price >= bb_upper:
            bb_position = "UPPER"
        elif price <= bb_lower:
            bb_position = "LOWER"
        elif price > bb_mid:
            bb_position = "UPPER_MIDDLE"
        else:
            bb_position = "LOWER_MIDDLE"

        # MACD signal
        macd_signal_val: float = float(indicators.macd_signal)
        macd_line: float = float(indicators.macd_line)
        if macd_line > macd_signal_val and macd_hist > 0:
            macd_signal = "BULLISH"
        elif macd_line < macd_signal_val and macd_hist < 0:
            macd_signal = "BEARISH"
        else:
            macd_signal = "NEUTRAL"

        # Volume state via OBV slope
        obv_slope: float = float(indicators.obv_slope)
        if obv_slope > 0.5:
            volume_state = "ACCUMULATION"
        elif obv_slope < -0.5:
            volume_state = "DISTRIBUTION"
        else:
            volume_state = "NORMAL"

        conditions: Dict[str, Any] = {
            "trend_direction": trend_direction,
            "adx": round(adx_value, 2),
            "volatility_level": volatility_level,
            "rsi_level": rsi_level,
            "rsi": round(rsi_value, 2),
            "bb_position": bb_position,
            "macd_signal": macd_signal,
            "volume_state": volume_state,
            "atr": round(atr_value, 6),
            "atr_percent": round(atr_pct, 4),
            "choppiness": round(choppiness, 2),
            "is_choppy": choppiness > 61.8,
        }

        # Futures enrichment from snapshot
        if snapshot.funding_rate is not None:
            fr = snapshot.funding_rate
            conditions["funding_rate"] = round(fr, 6)
            # High positive FR = longs crowded → bearish pressure; high negative = shorts crowded → bullish
            if fr > 0.001:
                conditions["funding_bias"] = "BEARISH"   # longs paying heavily
            elif fr < -0.001:
                conditions["funding_bias"] = "BULLISH"   # shorts paying heavily
            else:
                conditions["funding_bias"] = "NEUTRAL"

        if snapshot.open_interest is not None:
            conditions["open_interest"] = snapshot.open_interest

        return conditions

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

    def _select_signal(
        self,
        price: float,
        indicators,
        snapshot: Optional[MarketSnapshot] = None,
        choppiness_threshold: float = 61.8,
        rsi_strong_buy: float = 30.0,
        rsi_buy: float = 40.0,
        rsi_sell: float = 60.0,
        rsi_strong_sell: float = 70.0,
    ) -> Signal:
        # Regime gate: in a choppy market, suppress directional entry signals
        if indicators.choppiness > choppiness_threshold:
            return Signal.NEUTRAL

        # Funding rate sentiment bias (futures only)
        funding_rate: float = snapshot.funding_rate if (snapshot and snapshot.funding_rate is not None) else 0.0
        fr_bearish_bias = funding_rate > 0.003
        fr_bullish_bias = funding_rate < -0.001

        if (
            indicators.rsi_14 < rsi_strong_buy
            and price < indicators.bb_lower
            and indicators.macd_hist > 0
            and not fr_bearish_bias
        ):
            return Signal.STRONG_BUY
        if (
            indicators.rsi_14 < rsi_buy
            and price < indicators.bb_mid
            and (not fr_bearish_bias or fr_bullish_bias)
        ):
            return Signal.BUY
        if (
            indicators.rsi_14 > rsi_strong_sell
            and price > indicators.bb_upper
            and indicators.macd_hist < 0
            and not fr_bullish_bias
        ):
            return Signal.STRONG_SELL
        if (
            indicators.rsi_14 > rsi_sell
            and price > indicators.bb_mid
            and (not fr_bullish_bias or fr_bearish_bias)
        ):
            return Signal.SELL
        return Signal.NEUTRAL
