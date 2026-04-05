"""Deterministic signal scoring engine — replaces LLM for fast decisions.

Combines indicator signals, ML model outputs, regime context, key level
proximity, and funding rate into a single composite score that maps
directly to a TradeDecision.  Zero latency, zero cost, fully backtestable.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.domain.analysis import Signal, TechnicalAnalysis
from src.domain.trading import Action, TradeDecision

logger = logging.getLogger(__name__)

_SIGNAL_SCORES = {
    Signal.STRONG_BUY: 1.0,
    Signal.BUY: 0.5,
    Signal.NEUTRAL: 0.0,
    Signal.SELL: -0.5,
    Signal.STRONG_SELL: -1.0,
}

_REGIME_MULTIPLIERS = {
    "BULL_TRENDING": {"long": 1.2, "short": 0.6},
    "BULL_CORRECTION": {"long": 1.0, "short": 0.8},
    "BEAR_TRENDING": {"long": 0.6, "short": 1.2},
    "ACCUMULATION": {"long": 0.7, "short": 0.7},
}


class SignalScorer:
    """Compute a composite score from all available signals and return a TradeDecision."""

    def __init__(
        self,
        settings: Any,
        *,
        direction_classifier: Any | None = None,
        key_level_detector: Any | None = None,
        outcome_predictor: Any | None = None,
    ) -> None:
        self.settings = settings
        self._direction_classifier = direction_classifier
        self._key_level_detector = key_level_detector
        self._outcome_predictor = outcome_predictor

    def decide(
        self,
        symbol: str,
        snapshot: Any,
        analysis: TechnicalAnalysis,
        *,
        regime: str | None = None,
        htf_signal: Signal | None = None,
        position_direction: str | None = None,
    ) -> TradeDecision:
        """Score market conditions and return a deterministic trade decision.

        Args:
            symbol: Trading symbol e.g. "BTCUSDT"
            snapshot: MarketSnapshot with price, funding_rate, etc.
            analysis: TechnicalAnalysis with signal + IndicatorSet
            regime: CycleClassifier label (BULL_TRENDING, etc.) or None
            htf_signal: Higher-timeframe Signal for confirmation
            position_direction: "LONG" or "SHORT" if position is open, else None
        """
        indicators = analysis.indicators
        price = snapshot.price
        s = self.settings  # shorthand

        # --- Component scores (each in [-1, +1]) ---
        components: dict[str, float] = {}

        # 1. Technical signal
        components["signal"] = _SIGNAL_SCORES.get(analysis.signal, 0.0)

        # 2. XGBoost direction probability (None = no model available)
        direction_score = self._score_direction(indicators, price, symbol)

        # 3. Trend strength (ADX + EMA alignment)
        components["trend"] = self._score_trend(indicators, price)

        # 4. Momentum (RSI + MACD)
        components["momentum"] = self._score_momentum(indicators)

        # 5. Volume confirmation
        components["volume"] = self._score_volume(indicators)

        # 6. Key level proximity
        components["key_levels"] = self._score_key_levels(price, symbol)

        # --- Weighted composite ---
        # When the direction classifier is unavailable, redistribute its weight
        # proportionally across the remaining components so no score capacity
        # is wasted on a dead 0.0 input.
        weights = {
            "signal": s.scoring_w_signal,
            "trend": s.scoring_w_trend,
            "momentum": s.scoring_w_momentum,
            "volume": s.scoring_w_volume,
            "key_levels": s.scoring_w_key_levels,
        }
        if direction_score is not None:
            components["direction"] = direction_score
            weights["direction"] = s.scoring_w_direction
        else:
            # Redistribute direction weight proportionally to active components
            active_total = sum(weights.values())
            if active_total > 0:
                scale = (active_total + s.scoring_w_direction) / active_total
                weights = {k: v * scale for k, v in weights.items()}
            logger.debug(
                "[scorer] %s direction classifier unavailable — redistributed %.0f%% weight",
                symbol, s.scoring_w_direction * 100,
            )

        raw_score = sum(components[k] * weights[k] for k in components)

        comp_str = " ".join(f"{k}={v:+.3f}(w={weights[k]:.2f})" for k, v in components.items())
        logger.info("[scorer] %s raw=%.4f %s components: %s",
                    symbol, raw_score,
                    "(no direction model)" if direction_score is None else "",
                    comp_str)

        # --- Multipliers ---
        tags: list[str] = []

        # Choppiness penalty
        if indicators.choppiness > s.choppiness_threshold:
            raw_score *= s.scoring_choppiness_penalty
            tags.append(f"chop={indicators.choppiness:.0f}>thr")

        # Regime alignment
        if regime:
            raw_score = self._apply_regime(raw_score, regime, tags)

        # HTF confirmation
        if htf_signal is not None:
            raw_score = self._apply_htf(raw_score, htf_signal, tags)

        # Funding rate bias (futures)
        fr = getattr(snapshot, "funding_rate", None)
        if fr is not None:
            raw_score = self._apply_funding(raw_score, fr, tags)

        logger.info(
            "[scorer] %s final=%.4f threshold=%.2f tags=%s → %s",
            symbol, raw_score, s.scoring_entry_threshold,
            tags or "none",
            "BUY" if raw_score >= s.scoring_entry_threshold else
            ("SELL" if raw_score <= -s.scoring_entry_threshold else "HOLD"),
        )

        # --- Outcome predictor gate ---
        if self._outcome_predictor is not None:
            market_cond = self._build_market_conditions(indicators, price, analysis.signal)
            win_prob = self._outcome_predictor.predict(market_cond)
            if win_prob is not None and win_prob < self._outcome_predictor.threshold:
                tags.append(f"outcome={win_prob:.0%}<thr")
                return self._make_hold(symbol, raw_score, components, tags, f"low_win_prob={win_prob:.0%}")

        # --- Map score → decision ---
        return self._score_to_decision(
            raw_score, symbol, price, components, tags, position_direction,
        )

    # ------------------------------------------------------------------
    # Component scorers — each returns a value in [-1, +1]
    # ------------------------------------------------------------------

    def _score_direction(self, indicators: Any, price: float, symbol: str) -> float | None:
        """Return direction score in [-1, +1], or None if classifier unavailable."""
        if self._direction_classifier is None:
            return None
        prob = self._direction_classifier.predict_proba(indicators, price, symbol)
        if prob is None:
            return None
        return (prob - 0.5) * 2  # [0,1] → [-1,+1]

    def _score_trend(self, indicators: Any, price: float) -> float:
        score = 0.0
        adx = indicators.adx
        # ADX strength: >25 = trending
        if adx >= 25:
            # Determine trend direction from EMA alignment
            ema_bull = indicators.ema_20 > indicators.ema_50
            direction = 1.0 if ema_bull else -1.0
            strength = min(adx / 50.0, 1.0)  # normalise: 50+ ADX → 1.0
            score = direction * strength
        elif adx < 20:
            score = 0.0  # weak/no trend
        else:
            # Moderate trend (20-25 ADX)
            ema_bull = indicators.ema_20 > indicators.ema_50
            score = (0.3 if ema_bull else -0.3)

        # Price vs EMA position adds conviction
        if price > 0 and indicators.ema_50 > 0:
            dist_pct = (price - indicators.ema_50) / indicators.ema_50
            score += max(min(dist_pct * 5, 0.3), -0.3)  # clamp contribution

        return max(min(score, 1.0), -1.0)

    def _score_momentum(self, indicators: Any) -> float:
        score = 0.0

        # RSI component: oversold = bullish, overbought = bearish
        rsi = indicators.rsi_14
        if rsi <= 30:
            score += 0.6   # oversold → bullish
        elif rsi <= 40:
            score += 0.3
        elif rsi >= 70:
            score -= 0.6   # overbought → bearish
        elif rsi >= 60:
            score -= 0.3

        # MACD histogram direction
        hist = indicators.macd_hist
        if hist > 0:
            score += min(0.4, hist * 10)  # positive histogram → bullish
        elif hist < 0:
            score -= min(0.4, abs(hist) * 10)

        return max(min(score, 1.0), -1.0)

    def _score_volume(self, indicators: Any) -> float:
        score = 0.0

        # Volume ratio: >1.5 = high interest (confirms move)
        vol_ratio = indicators.vol_ratio
        if vol_ratio > 2.0:
            score += 0.3
        elif vol_ratio > 1.5:
            score += 0.15
        elif vol_ratio < 0.5:
            score -= 0.2  # very low volume — unreliable signal

        # OBV slope: positive = accumulation, negative = distribution
        obv = indicators.obv_slope
        if obv > 0.5:
            score += 0.4
        elif obv > 0:
            score += 0.1
        elif obv < -0.5:
            score -= 0.4
        elif obv < 0:
            score -= 0.1

        return max(min(score, 1.0), -1.0)

    def _score_key_levels(self, price: float, symbol: str) -> float:
        if self._key_level_detector is None:
            return 0.0

        cache = self._key_level_detector._get_cache(symbol)
        if not cache:
            return 0.0

        levels = cache.get("levels", [])
        if not levels:
            return 0.0

        score = 0.0
        for lvl in levels:
            center = lvl["center"]
            dist_pct = (price - center) / center * 100 if center > 0 else 0
            lvl_type = lvl.get("type", "")
            touches = lvl.get("touches", 1)
            strength = min(touches / 10.0, 1.0)  # normalise touches

            # Near support (price slightly above) → bullish
            if lvl_type == "support" and 0 < dist_pct < 2.0:
                score += 0.5 * strength
            # Near resistance (price slightly below) → bearish
            elif lvl_type == "resistance" and -2.0 < dist_pct < 0:
                score -= 0.5 * strength
            # Breaking above resistance → bullish breakout
            elif lvl_type == "resistance" and 0 < dist_pct < 1.0:
                score += 0.3 * strength
            # Breaking below support → bearish breakdown
            elif lvl_type == "support" and -1.0 < dist_pct < 0:
                score -= 0.3 * strength

        return max(min(score, 1.0), -1.0)

    # ------------------------------------------------------------------
    # Multiplier helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_regime(score: float, regime: str, tags: list[str]) -> float:
        mult = _REGIME_MULTIPLIERS.get(regime)
        if mult is None:
            return score
        side = "long" if score > 0 else "short"
        factor = mult[side]
        tags.append(f"regime={regime}×{factor}")
        return score * factor

    @staticmethod
    def _apply_htf(score: float, htf_signal: Signal, tags: list[str]) -> float:
        htf_dir = _SIGNAL_SCORES.get(htf_signal, 0.0)
        agrees = (score > 0 and htf_dir > 0) or (score < 0 and htf_dir < 0)
        if agrees:
            tags.append(f"htf={htf_signal.value}✓")
            return score  # no penalty
        if abs(htf_dir) > 0:
            tags.append(f"htf={htf_signal.value}✗×0.5")
            return score * 0.5  # disagrees — cut conviction
        return score  # HTF neutral — no effect

    @staticmethod
    def _apply_funding(score: float, funding_rate: float, tags: list[str]) -> float:
        if abs(funding_rate) < 0.0005:
            return score  # neutral
        # High positive FR = longs crowded → bearish bias
        # High negative FR = shorts crowded → bullish bias
        bias = -funding_rate * 100  # scale: 0.001 → -0.1 (slight bearish)
        bias = max(min(bias, 0.15), -0.15)  # clamp
        if abs(bias) >= 0.05:
            tags.append(f"fr={funding_rate:+.4f}")
        return score + bias

    # ------------------------------------------------------------------
    # Decision mapping
    # ------------------------------------------------------------------

    def _score_to_decision(
        self,
        score: float,
        symbol: str,
        price: float,
        components: dict[str, float],
        tags: list[str],
        position_direction: str | None,
    ) -> TradeDecision:
        s = self.settings
        entry_thr = s.scoring_entry_threshold
        exit_thr = s.scoring_exit_threshold
        confidence = min(abs(score) * 100 / entry_thr, 100) if entry_thr > 0 else 0.0
        reasoning = self._format_reasoning(score, components, tags)

        # Close existing position on reversal
        if position_direction == "LONG" and score < -exit_thr:
            return TradeDecision(
                symbol=symbol, action=Action.CLOSE_LONG,
                confidence=confidence, reasoning=reasoning,
                source="signal_scorer",
            )
        if position_direction == "SHORT" and score > exit_thr:
            return TradeDecision(
                symbol=symbol, action=Action.CLOSE_SHORT,
                confidence=confidence, reasoning=reasoning,
                source="signal_scorer",
            )

        # Entry signals (only when no position)
        if position_direction is None:
            if score >= entry_thr:
                return TradeDecision(
                    symbol=symbol, action=Action.BUY,
                    confidence=confidence, reasoning=reasoning,
                    source="signal_scorer",
                )
            if score <= -entry_thr:
                return TradeDecision(
                    symbol=symbol, action=Action.SELL,
                    confidence=confidence, reasoning=reasoning,
                    source="signal_scorer",
                )

        return self._make_hold(symbol, score, components, tags)

    def _make_hold(
        self,
        symbol: str,
        score: float,
        components: dict[str, float],
        tags: list[str],
        extra: str = "",
    ) -> TradeDecision:
        reasoning = self._format_reasoning(score, components, tags)
        if extra:
            reasoning = f"{reasoning} | {extra}"
        return TradeDecision(
            symbol=symbol, action=Action.HOLD,
            confidence=0.0, reasoning=reasoning,
            source="signal_scorer",
        )

    @staticmethod
    def _format_reasoning(score: float, components: dict[str, float], tags: list[str]) -> str:
        parts = " ".join(f"{k}={v:+.2f}" for k, v in components.items())
        tag_str = " ".join(tags) if tags else ""
        return f"score={score:+.3f} [{parts}]{' ' + tag_str if tag_str else ''}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_market_conditions(indicators: Any, price: float, signal: Signal) -> dict[str, Any]:
        """Build the conditions dict that OutcomePredictor expects."""
        rsi = indicators.rsi_14
        if rsi >= 70:
            rsi_level = "OVERBOUGHT"
        elif rsi <= 30:
            rsi_level = "OVERSOLD"
        else:
            rsi_level = "NEUTRAL"

        atr_pct = (indicators.atr / price * 100) if price > 0 else 0.0
        ema_bull = indicators.ema_20 > indicators.ema_50
        macd_bull = indicators.macd_hist > 0

        if indicators.adx >= 25:
            trend = "BULLISH" if ema_bull else "BEARISH"
        else:
            trend = "NEUTRAL"

        obv = indicators.obv_slope
        if obv > 0.5:
            vol_state = "HIGH"
        else:
            vol_state = "NORMAL"

        bb_width = indicators.bb_upper - indicators.bb_lower
        if bb_width > 0:
            bb_pos_val = (price - indicators.bb_lower) / bb_width
            if bb_pos_val > 0.8:
                bb_position = "UPPER"
            elif bb_pos_val < 0.2:
                bb_position = "LOWER"
            else:
                bb_position = "MIDDLE"
        else:
            bb_position = "MIDDLE"

        return {
            "rsi": rsi,
            "adx": indicators.adx,
            "atr_percentage": atr_pct,
            "trend_direction": trend,
            "volume_state": vol_state,
            "bb_position": bb_position,
        }
