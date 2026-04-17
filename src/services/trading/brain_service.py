"""Trading brain service for learning and adaptive parameters.

Handles brain state management, learning from closed trades, and providing AI context.
Helpers live in :mod:`brain_context` and :mod:`brain_reflection`.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.domain.trading.models import Position, TradeRecord
from src.infrastructure.storage.vector_memory import VectorMemoryService
from src.services.trading.brain_context import (
    build_brain_prompt_context,
    build_rich_context_string,
    get_vector_context,
)
from src.services.trading.brain_reflection import (
    run_loss_reflection,
    run_win_reflection,
)

if TYPE_CHECKING:
    from src.infrastructure.storage.persistence import Persistence


class TradingBrainService:
    """Service for managing trading brain and learning from trades.

    Responsibilities:
    - Update brain from closed trades
    - Provide brain context for AI prompts
    - Suggest parameters based on learned data
    - Get dynamic thresholds
    """

    def __init__(
        self,
        logger: logging.Logger,
        persistence: "Persistence",
        vector_memory: VectorMemoryService,
    ):
        self.logger = logger
        self.persistence = persistence
        self.vector_memory = vector_memory

        # Cache for computed stats (invalidated when new trades arrive)
        self._stats_cache: Dict[str, Any] = {}
        self._cache_trade_count: int = 0
        self._reflection_interval: int = 10

        # Initialize trade count from persistent storage
        # so reflection triggers stay consistent across restarts.
        self._trade_count: int = self.vector_memory.trade_count

    def update_from_closed_trade(
        self,
        position: Position,
        close_price: float,
        close_reason: str,
        entry_decision: Optional[TradeRecord] = None,
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract insights from a closed trade and update brain."""
        pnl_pct = position.calculate_pnl(close_price)
        is_win = pnl_pct > 0
        conditions = market_conditions or {}

        condition_str = build_rich_context_string(
            trend_direction=conditions.get("trend_direction", "NEUTRAL"),
            adx=float(conditions.get("adx", 0.0)),
            volatility_level=conditions.get("volatility", "MEDIUM"),
            rsi_level=conditions.get("rsi_level", "NEUTRAL"),
            macd_signal=conditions.get("macd_signal", "NEUTRAL"),
            volume_state=conditions.get("volume_state", "NORMAL"),
            bb_position=conditions.get("bb_position", "MIDDLE"),
            is_weekend=conditions.get("is_weekend", False),
            market_sentiment=conditions.get("market_sentiment", "NEUTRAL"),
            order_book_bias=conditions.get("order_book_bias", "BALANCED"),
        )

        self._stats_cache = {}  # invalidate (new trade added)

        trade_id = f"trade_{position.entry_time.isoformat()}"
        reasoning = entry_decision.reasoning if entry_decision else "N/A"
        self.vector_memory.store_experience(
            trade_id=trade_id,
            market_context=condition_str,
            outcome="WIN" if is_win else "LOSS",
            pnl_pct=pnl_pct,
            direction=position.direction,
            confidence=position.confidence,
            reasoning=reasoning,
            metadata={
                "close_reason": close_reason,
                "adx_at_entry": position.adx_at_entry,
                "rsi_at_entry": position.rsi_at_entry,
                "atr_at_entry": position.atr_at_entry,
                "volatility_level": position.volatility_level,
                "sl_distance_pct": position.sl_distance_pct,
                "tp_distance_pct": position.tp_distance_pct,
                "rr_ratio": position.rr_ratio_at_entry,
                "max_drawdown_pct": position.max_drawdown_pct,
                "max_profit_pct": position.max_profit_pct,
                "fear_greed_index": conditions.get("fear_greed_index", 50),
                "market_regime": conditions.get("trend_direction", "NEUTRAL"),
                "is_weekend": conditions.get("is_weekend", False),
                "position_size_pct": position.size_pct,
                "confluence_count": self._count_strong_confluences(
                    position.confluence_factors
                ),
                "timeframe_alignment": conditions.get("timeframe_alignment"),
                **self._extract_factor_scores(position.confluence_factors),
            },
        )

        self.logger.info(
            "Updated brain from %s trade (%s, P&L: %s%%)",
            position.direction, close_reason, f"{pnl_pct:+.2f}",
        )

        self._trade_count += 1
        if self._trade_count % self._reflection_interval == 0:
            run_win_reflection(self.vector_memory, self.logger)
            run_loss_reflection(self.vector_memory, self.logger)

    def get_context(
        self,
        trend_direction: str = "NEUTRAL",
        adx: float = 0,
        volatility_level: str = "MEDIUM",
        rsi_level: str = "NEUTRAL",
        macd_signal: str = "NEUTRAL",
        volume_state: str = "NORMAL",
        bb_position: str = "MIDDLE",
        is_weekend: bool = False,
        market_sentiment: str = "NEUTRAL",
        order_book_bias: str = "BALANCED",
    ) -> str:
        """Generate formatted brain context for prompt injection."""
        return build_brain_prompt_context(
            self.vector_memory,
            self._get_cached_stats,
            trend_direction=trend_direction,
            adx=adx,
            volatility_level=volatility_level,
            rsi_level=rsi_level,
            macd_signal=macd_signal,
            volume_state=volume_state,
            bb_position=bb_position,
            is_weekend=is_weekend,
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias,
        )

    def get_vector_context(
        self,
        trend_direction: str = "NEUTRAL",
        adx: float = 0,
        volatility_level: str = "MEDIUM",
        rsi_level: str = "NEUTRAL",
        macd_signal: str = "NEUTRAL",
        volume_state: str = "NORMAL",
        bb_position: str = "MIDDLE",
        is_weekend: bool = False,
        market_sentiment: str = "NEUTRAL",
        order_book_bias: str = "BALANCED",
        k: int = 5,
    ) -> str:
        """Semantic-search past trades for the given market context."""
        return get_vector_context(
            self.vector_memory,
            trend_direction=trend_direction,
            adx=adx,
            volatility_level=volatility_level,
            rsi_level=rsi_level,
            macd_signal=macd_signal,
            volume_state=volume_state,
            bb_position=bb_position,
            is_weekend=is_weekend,
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias,
            k=k,
        )

    def get_parameter_suggestions(
        self,
        volatility_level: str = "MEDIUM",
        confidence: str = "MEDIUM",
        current_atr_pct: float = 2.0,
    ) -> Dict[str, float]:
        """Get SL/TP/size suggestions from trading brain."""
        # High volatility = wider stops to avoid premature exits.
        # Low volatility = tighter stops for better risk management.
        volatility_multipliers = {
            "HIGH": {"sl": 2.5, "tp": 4.5},
            "MEDIUM": {"sl": 2.0, "tp": 4.0},
            "LOW": {"sl": 1.5, "tp": 3.0},
        }
        multipliers = volatility_multipliers.get(
            volatility_level.upper(), volatility_multipliers["MEDIUM"]
        )
        recommendations = {
            "sl_pct": current_atr_pct * multipliers["sl"] / 100,
            "tp_pct": current_atr_pct * multipliers["tp"] / 100,
            "size_pct": 0.02,
            "min_rr": 2.0,
            "source": f"atr_fallback_vol_{volatility_level.lower()}",
        }

        # High volatility = reduce size even further for risk management.
        base_size = {"HIGH": 0.03, "MEDIUM": 0.02, "LOW": 0.01}
        volatility_size_adj = {"HIGH": 0.8, "MEDIUM": 1.0, "LOW": 1.0}
        base = base_size.get(confidence.upper(), 0.02)
        vol_adj = volatility_size_adj.get(volatility_level.upper(), 1.0)
        recommendations["size_pct"] = base * vol_adj
        return recommendations

    def get_dynamic_thresholds(self) -> Dict[str, Any]:
        """Get Brain-learned thresholds from vector store."""
        thresholds = self._get_cached_stats(
            "thresholds", self.vector_memory.compute_optimal_thresholds
        )
        return {
            "adx_strong_threshold": thresholds.get("adx_strong_threshold", 25),
            "avg_sl_pct": thresholds.get("avg_sl_pct", 2.5),
            "min_rr_recommended": thresholds.get("min_rr_recommended", 2.0),
            "confidence_threshold": thresholds.get("confidence_threshold", 70),
            "safe_mae_pct": thresholds.get("safe_mae_pct", 0),
            "adx_weak_threshold": thresholds.get("adx_weak_threshold", 20),
            "min_confluences_weak": thresholds.get("min_confluences_weak", 4),
            "min_confluences_standard": thresholds.get("min_confluences_standard", 3),
            "position_reduce_mixed": thresholds.get("position_reduce_mixed", 0.20),
            "position_reduce_divergent": thresholds.get("position_reduce_divergent", 0.35),
            "min_position_size": thresholds.get("min_position_size", 0.10),
            "rr_borderline_min": thresholds.get("rr_borderline_min", 1.5),
            "rr_strong_setup": thresholds.get("rr_strong_setup", 2.5),
        }

    def track_position_update(
        self,
        position: Position,
        old_sl: float,
        old_tp: float,
        new_sl: float,
        new_tp: float,
        current_price: float,
        current_pnl_pct: float,
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Track position update decisions for learning."""
        from datetime import datetime
        conditions = market_conditions or {}

        sl_moved = new_sl != old_sl
        tp_moved = new_tp != old_tp
        if sl_moved and not tp_moved:
            action_type = "SL_TRAIL"
        elif tp_moved and not sl_moved:
            action_type = "TP_EXTEND"
        elif sl_moved and tp_moved:
            action_type = "BOTH"
        else:
            return

        market_context = build_rich_context_string(
            trend_direction=conditions.get("trend_direction", "NEUTRAL"),
            adx=float(conditions.get("adx", 0.0)),
            volatility_level=conditions.get("volatility", "MEDIUM"),
            rsi_level=conditions.get("rsi_level", "NEUTRAL"),
            macd_signal=conditions.get("macd_signal", "NEUTRAL"),
            volume_state=conditions.get("volume_state", "NORMAL"),
            bb_position=conditions.get("bb_position", "MIDDLE"),
            is_weekend=conditions.get("is_weekend", False),
            market_sentiment=conditions.get("market_sentiment", "NEUTRAL"),
            order_book_bias=conditions.get("order_book_bias", "BALANCED"),
        )

        update_id = f"update_{int(datetime.now().timestamp())}"
        reasoning_str = (
            f"Moved {action_type}: SL {old_sl:.2f}→{new_sl:.2f}, "
            f"TP {old_tp:.2f}→{new_tp:.2f}"
        )

        self.vector_memory.store_experience(
            trade_id=update_id,
            market_context=market_context,
            outcome="UPDATE",
            pnl_pct=current_pnl_pct,
            direction=position.direction,
            confidence=position.confidence,
            reasoning=reasoning_str,
            metadata={
                "action_type": action_type,
                "current_price": current_price,
                "sl_change": new_sl - old_sl,
                "tp_change": new_tp - old_tp,
                "pnl_at_update": current_pnl_pct,
                "adx_at_update": conditions.get("adx", 0),
                "volatility": conditions.get("volatility", "MEDIUM"),
            },
        )

        self.logger.debug(
            "Tracked position update: %s at %s%% PnL",
            action_type, f"{current_pnl_pct:+.1f}",
        )

    def _get_cached_stats(self, key: str, compute_fn) -> Dict[str, Any]:
        """Get stats from cache or compute and cache them."""
        current_count = self.vector_memory.experience_count
        if current_count != self._cache_trade_count:
            self._stats_cache = {}
            self._cache_trade_count = current_count
        if key not in self._stats_cache:
            self._stats_cache[key] = compute_fn()
        return self._stats_cache[key]

    @staticmethod
    def _extract_factor_scores(confluence_factors: tuple) -> Dict[str, float]:
        """Extract factor scores into flat dict for vector metadata."""
        scores: Dict[str, float] = {}
        if not confluence_factors:
            return scores
        for factor_name, score in confluence_factors:
            clean_name = factor_name.replace(" ", "_").lower()
            scores[f"{clean_name}_score"] = float(score)
        return scores

    @staticmethod
    def _count_strong_confluences(confluence_factors: tuple) -> int:
        """Count factors with score > 50 (supporting the trade)."""
        if not confluence_factors:
            return 0
        return sum(1 for _, score in confluence_factors if score > 50)
