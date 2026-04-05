"""Hard safety layer between LLM decisions and execution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.config import Settings
from src.domain.trading import Action, RiskValidationResult, TradeDecision

if TYPE_CHECKING:
    from src.domain.trading.models import RiskAssessment


class RiskManager:
    """Validate, clamp, or block trade decisions using hard-coded rules."""

    # Pairs considered correlated — bot won't hold same-direction positions in both
    CORRELATED_PAIRS: tuple[frozenset[str], ...] = (frozenset({"BTCUSDT", "ETHUSDT"}),)

    def __init__(
        self,
        settings: Settings,
        *,
        logger: logging.Logger | None = None,
        outcome_predictor: Any | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)
        self.ALLOWED_SYMBOLS: frozenset[str] = frozenset(
            getattr(settings, "trading_symbols", None) or ["BTCUSDT", "ETHUSDT"]
        )
        self._outcome_predictor = outcome_predictor

    def check_kill_switches(
        self,
        consecutive_losses: int,
        daily_loss_pct: float,
    ) -> tuple[bool, str]:
        """Return (should_halt, reason) based on daily loss and streak limits.

        Called at the start of each cycle before any analysis or execution.
        """
        max_losses = getattr(self.settings, "max_consecutive_losses", 3)
        max_daily = getattr(self.settings, "max_daily_loss_pct", 0.05)

        if consecutive_losses >= max_losses:
            return (
                True,
                f"max_consecutive_losses reached ({consecutive_losses}/{max_losses})",
            )

        if daily_loss_pct >= max_daily:
            return (
                True,
                f"max_daily_loss_pct reached ({daily_loss_pct:.2%} >= {max_daily:.2%})",
            )

        return False, ""

    def validate(
        self,
        decision: TradeDecision,
        balance: dict,
        open_positions: Optional[Dict[str, Any]] = None,
    ) -> RiskValidationResult:
        dry_run = self.settings.bot_dry_run

        if not self.settings.bot_enabled:
            self.logger.warning(
                "Risk rule triggered: BOT_ENABLED is false; overriding to HOLD"
            )
            return RiskValidationResult(
                decision=self._hold(decision, "bot_disabled"),
                dry_run=dry_run,
                status="blocked",
            )

        if dry_run:
            self.logger.info("BOT_DRY_RUN is true; execution stays in dry-run mode")

        if decision.action == Action.HOLD:
            self.logger.debug("HOLD passthrough — no risk checks needed")
            return RiskValidationResult(
                decision=decision, dry_run=dry_run, status="passed"
            )

        # CLOSE/UPDATE actions don't need size clamping or balance checks
        if decision.action in (Action.CLOSE, Action.CLOSE_LONG, Action.CLOSE_SHORT, Action.UPDATE):
            self.logger.debug("%s passthrough — no sizing/balance checks needed", decision.action.value)
            return RiskValidationResult(
                decision=decision, dry_run=dry_run, status="passed"
            )

        # ML outcome gate (B3): block low-probability trades
        if self._outcome_predictor is not None and decision.action in (
            Action.BUY,
            Action.SELL,
        ):
            market_conditions = getattr(decision, "market_conditions", {}) or {}
            should_block, win_prob = self._outcome_predictor.should_block(
                market_conditions
            )
            if should_block and win_prob is not None:
                self.logger.warning(
                    "Risk rule triggered: outcome model win probability %.0f%% below threshold — HOLD",
                    win_prob * 100,
                )
                return RiskValidationResult(
                    decision=self._hold(
                        decision, f"low_win_probability:{win_prob:.0%}"
                    ),
                    dry_run=dry_run,
                    status="blocked",
                )

        # Confidence threshold gate (only active when threshold > 0)
        min_conf = getattr(self.settings, "min_confidence_threshold", 0.0)
        if (
            min_conf > 0
            and decision.confidence < min_conf
            and decision.action
            not in (Action.CLOSE, Action.CLOSE_LONG, Action.CLOSE_SHORT)
        ):
            self.logger.warning(
                "Risk rule triggered: confidence %.1f < threshold %.1f; overriding to HOLD",
                decision.confidence,
                min_conf,
            )
            return RiskValidationResult(
                decision=self._hold(
                    decision, f"low_confidence:{decision.confidence:.1f}<{min_conf:.1f}"
                ),
                dry_run=dry_run,
                status="blocked",
            )

        # Correlation guard — don't open a correlated position in the same direction
        if open_positions and decision.action in (Action.BUY, Action.SELL):
            corr_block = self._check_correlation(decision, open_positions)
            if corr_block:
                self.logger.warning(
                    "Risk rule triggered: correlated position already open — %s",
                    corr_block,
                )
                return RiskValidationResult(
                    decision=self._hold(decision, f"correlated_position:{corr_block}"),
                    dry_run=dry_run,
                    status="blocked",
                )

        if decision.symbol not in self.ALLOWED_SYMBOLS:
            self.logger.warning(
                "Risk rule triggered: symbol %s is not allowed", decision.symbol
            )
            return RiskValidationResult(
                decision=self._hold(decision, "symbol_not_allowed"),
                dry_run=dry_run,
                status="blocked",
            )

        if decision.order_type == "LIMIT" and decision.price is None:
            self.logger.warning("Risk rule triggered: LIMIT order missing price")
            return RiskValidationResult(
                decision=self._hold(decision, "limit_price_required"),
                dry_run=dry_run,
                status="blocked",
            )

        effective_price = self._effective_price(decision, balance)
        if effective_price <= 0:
            self.logger.warning("Risk rule triggered: cannot determine price for %s", decision.symbol)
            return RiskValidationResult(
                decision=self._hold(decision, "unknown_price"),
                dry_run=dry_run,
                status="blocked",
            )

        # Signal scorer (and any other source) may leave quantity=0 — compute it
        # from balance using confidence-based sizing so the order has a real size.
        # For futures, size_pct is the fraction of balance committed as *margin*;
        # the notional is margin × leverage.
        leverage = self._effective_leverage()
        if decision.quantity <= 0 and decision.action in (Action.BUY, Action.SELL):
            usdt_avail = self._usdt_balance(balance)
            confidence_map = {"HIGH": 0.03, "MEDIUM": 0.02, "LOW": 0.01}
            conf_key = "HIGH" if decision.confidence >= 70 else ("MEDIUM" if decision.confidence >= 40 else "LOW")
            size_pct = confidence_map[conf_key]
            margin_alloc = usdt_avail * size_pct
            computed_qty = (margin_alloc * leverage) / effective_price
            self.logger.info(
                "Auto-sizing %s: balance=%.2f size=%.1f%% leverage=%dx margin=%.2f price=%.4f → qty=%.8f",
                decision.symbol, usdt_avail, size_pct * 100, leverage, margin_alloc, effective_price, computed_qty,
            )
            decision = TradeDecision(
                symbol=decision.symbol,
                action=decision.action,
                quantity=computed_qty,
                order_type=decision.order_type,
                price=decision.price,
                reasoning=decision.reasoning,
                confidence=decision.confidence,
                timestamp=decision.timestamp,
                source=decision.source,
            )

        notional = decision.quantity * effective_price
        margin = notional / leverage
        if margin > self.settings.max_order_usdt:
            clamped_quantity = (self.settings.max_order_usdt * leverage) / effective_price
            self.logger.warning(
                "Risk rule triggered: order margin %.4f exceeds max %.4f; clamping quantity to %.8f",
                margin,
                self.settings.max_order_usdt,
                clamped_quantity,
            )
            decision = TradeDecision(
                symbol=decision.symbol,
                action=decision.action,
                quantity=clamped_quantity,
                order_type=decision.order_type,
                price=decision.price,
                reasoning=decision.reasoning,
                confidence=decision.confidence,
                timestamp=decision.timestamp,
                source=decision.source,
            )
            effective_price = self._effective_price(decision, balance)
            notional = decision.quantity * effective_price
            status = "clamped"
        else:
            self.logger.debug("Order size within configured max")
            status = "passed"

        if not dry_run:
            usdt_balance = self._usdt_balance(balance)
            required_margin = (notional / leverage) * 1.01
            if usdt_balance < required_margin:
                self.logger.warning(
                    "Risk rule triggered: insufficient USDT balance %.4f < required margin %.4f",
                    usdt_balance,
                    required_margin,
                )
                return RiskValidationResult(
                    decision=self._hold(
                        decision,
                        f"insufficient_balance: requires {required_margin:.4f} USDT margin, has {usdt_balance:.4f}",
                    ),
                    dry_run=dry_run,
                    status="blocked",
                )

        self.logger.info(
            "Risk validation passed for %s %s", decision.symbol, decision.action.value
        )
        return RiskValidationResult(decision=decision, dry_run=dry_run, status=status)

    def calculate_entry_parameters(
        self,
        signal: str,
        current_price: float,
        capital: float,
        confidence: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_size: Optional[float] = None,
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> "RiskAssessment":
        """
        Calculate all risk parameters for a new position entry.
        """
        from src.domain.trading.models import RiskAssessment

        market_conditions = market_conditions or {}
        direction = "LONG" if signal == "BUY" else "SHORT"

        # 1. Extract or Default ATR/Volatility
        atr = market_conditions.get("atr", current_price * 0.02)
        atr_pct = market_conditions.get("atr_percentage", (atr / current_price) * 100)

        # Determine volatility level
        if atr_pct > 3:
            volatility_level = "HIGH"
        elif atr_pct < 1.5:
            volatility_level = "LOW"
        else:
            volatility_level = "MEDIUM"

        # 2. Dynamic SL/TP Calculation
        # Use 2x ATR for SL, 4x ATR for TP → true 2:1 R/R default
        dynamic_sl_distance = atr * 2
        dynamic_tp_distance = atr * 4

        if direction == "LONG":
            dynamic_sl = current_price - dynamic_sl_distance
            dynamic_tp = current_price + dynamic_tp_distance
        else:  # SHORT
            dynamic_sl = current_price + dynamic_sl_distance
            dynamic_tp = current_price - dynamic_tp_distance

        # 3. Resolve Final SL/TP (AI vs Dynamic)
        if stop_loss and stop_loss > 0:
            final_sl = stop_loss
            self.logger.debug("Using AI-provided SL: $%s", f"{final_sl:,.2f}")
        else:
            final_sl = dynamic_sl
            self.logger.info("Using dynamic SL (2x ATR): $%s", f"{final_sl:,.2f}")

        if take_profit and take_profit > 0:
            final_tp = take_profit
            self.logger.debug("Using AI-provided TP: $%s", f"{final_tp:,.2f}")
        else:
            final_tp = dynamic_tp
            self.logger.info("Using dynamic TP (4x ATR): $%s", f"{final_tp:,.2f}")

        # 4. Circuit Breakers (Clamp Extreme Values)
        sl_distance_raw = abs(current_price - final_sl) / current_price

        # Clamp SL: min 0.5%, max 10%
        if sl_distance_raw > 0.10:
            self.logger.warning(
                "SL distance %s exceeds 10%% max, clamping", f"{sl_distance_raw:.1%}"
            )
            if direction == "LONG":
                final_sl = current_price * 0.90
            else:
                final_sl = current_price * 1.10
        elif sl_distance_raw < 0.005:
            self.logger.warning(
                "SL distance %s below 0.5%% min, expanding", f"{sl_distance_raw:.1%}"
            )
            if direction == "LONG":
                final_sl = current_price * 0.995
            else:
                final_sl = current_price * 1.005

        # Validate Logical Consistency
        if direction == "LONG":
            if final_sl >= current_price:
                self.logger.warning(
                    "Invalid SL for LONG (%s >= %s), using dynamic",
                    final_sl,
                    current_price,
                )
                final_sl = dynamic_sl
            if final_tp <= current_price:
                self.logger.warning(
                    "Invalid TP for LONG (%s <= %s), using dynamic",
                    final_tp,
                    current_price,
                )
                final_tp = dynamic_tp
        else:  # SHORT
            if final_sl <= current_price:
                self.logger.warning(
                    "Invalid SL for SHORT (%s <= %s), using dynamic",
                    final_sl,
                    current_price,
                )
                final_sl = dynamic_sl
            if final_tp >= current_price:
                self.logger.warning(
                    "Invalid TP for SHORT (%s >= %s), using dynamic",
                    final_tp,
                    current_price,
                )
                final_tp = dynamic_tp

        # 5. Position Sizing
        if position_size and position_size > 0:
            final_size_pct = position_size
        else:
            # Dynamic sizing based on confidence
            confidence_map = {"HIGH": 0.03, "MEDIUM": 0.02, "LOW": 0.01}
            final_size_pct = confidence_map.get(confidence.upper(), 0.02)
            self.logger.info(
                "Using confidence-based size: %.1f%%", final_size_pct * 100
            )

        # 6. Calculate Financials
        # For futures, size_pct is the fraction of capital used as margin;
        # the notional position = margin × leverage.
        leverage = self._effective_leverage()
        allocation = capital * final_size_pct * leverage
        quantity = allocation / current_price
        entry_fee = allocation * self.settings.transaction_fee_percent

        # 7. Metrics
        sl_distance_pct = abs(current_price - final_sl) / current_price
        tp_distance_pct = abs(final_tp - current_price) / current_price
        rr_ratio = tp_distance_pct / sl_distance_pct if sl_distance_pct > 0 else 0

        return RiskAssessment(
            direction=direction,
            entry_price=current_price,
            stop_loss=final_sl,
            take_profit=final_tp,
            quantity=quantity,
            size_pct=final_size_pct,
            quote_amount=allocation,
            entry_fee=entry_fee,
            sl_distance_pct=sl_distance_pct,
            tp_distance_pct=tp_distance_pct,
            rr_ratio=rr_ratio,
            volatility_level=volatility_level,
        )

    @staticmethod
    def _hold(decision: TradeDecision, reason: str) -> TradeDecision:
        return TradeDecision(
            symbol=decision.symbol,
            action=Action.HOLD,
            quantity=0.0,
            order_type="MARKET",
            price=None,
            reasoning=reason,
            confidence=0.0,
            timestamp=decision.timestamp,
            source="fallback_hold",
        )

    @staticmethod
    def _usdt_balance(balance: dict) -> float:
        if "USDT" in balance and isinstance(balance["USDT"], dict):
            entry = balance["USDT"]
            return float(entry.get("free", entry.get("available", 0.0)))
        if "USDT" in balance:
            return float(balance["USDT"])
        if (
            "free" in balance
            and isinstance(balance["free"], dict)
            and "USDT" in balance["free"]
        ):
            return float(balance["free"]["USDT"])
        return 0.0

    def _effective_leverage(self) -> int:
        """Return futures leverage if trading futures, else 1 (spot)."""
        if getattr(self.settings, "binance_product", "spot") == "usdt_futures":
            return getattr(self.settings, "futures_leverage", 1)
        return 1

    @staticmethod
    def _effective_price(decision: TradeDecision, balance: dict) -> float:
        if decision.price is not None:
            return float(decision.price)
        ticker_price = balance.get("prices", {}).get(decision.symbol)
        if ticker_price is not None:
            return float(ticker_price)
        return 0.0

    def _check_correlation(
        self, decision: TradeDecision, open_positions: Dict[str, Any]
    ) -> str:
        """Return a description string if decision would create a correlated exposure, else empty string."""
        new_side = decision.action.value  # "BUY" or "SELL"
        for group in self.CORRELATED_PAIRS:
            if decision.symbol not in group:
                continue
            for other_sym in group:
                if other_sym == decision.symbol:
                    continue
                pos = open_positions.get(other_sym)
                if pos is None:
                    continue
                existing_side = pos.get("direction", "")
                # Block if the other position is in the same direction
                if (existing_side == "BUY" and new_side == "BUY") or (
                    existing_side == "SELL" and new_side == "SELL"
                ):
                    return f"{other_sym} already {existing_side}"
        return ""
