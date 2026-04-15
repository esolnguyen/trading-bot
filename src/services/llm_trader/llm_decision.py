"""Send raw OHLCV chart data (+ open position) to Azure OpenAI and parse a trading decision."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from openai import AzureOpenAI

from src.core.config.llm_trader_settings import LLMTraderConfig
from src.domain.trading.models import Action, OpenPosition
from src.infrastructure.binance.market_data import Candle, candles_to_csv
from src.services.llm_trader.skills_loader import load_skills

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMDecision:
    action: Action
    sl: float | None      # stop-loss price  (None → do not place)
    tp: float | None      # take-profit price (None → do not place)
    confidence: int       # 1–10
    reason: str
    raw_response: str


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert futures trader with deep experience in technical price-action analysis.
You will receive:
  1. The last {n} closed 15-minute OHLCV candles for {symbol} (perpetual futures).
  2. A snapshot of the currently open position (if any), including its existing SL/TP orders.

Your job:
  - Decide whether to BUY (open/hold long), SELL (open/hold short), or HOLD (keep current state).
  - If there is an open position and you say HOLD, you are choosing to keep it — provide updated SL/TP.
  - If there is an open position and you flip direction (BUY when SHORT, SELL when LONG), it will be closed and reversed.
  - Always provide a stop_loss and take_profit as exact price levels.

Rules:
  - Base your decision ONLY on the raw price data and position information provided.
  - Think step-by-step: trend, momentum, support/resistance, candle structure, position risk.
  - sl must be on the loss side of the entry; tp must be on the profit side.
  - For a new LONG: sl < current price < tp.
  - For a new SHORT: tp < current price < sl.
  - Reply with a single JSON object and nothing else:
  {{
    "action": "BUY" | "SELL" | "HOLD",
    "stop_loss": <price as number>,
    "take_profit": <price as number>,
    "confidence": <integer 1-10>,
    "reason": "<one or two sentences>"
  }}
"""

_POSITION_SECTION_OPEN = """\
── Current Open Position ────────────────────────────────
Side            : {side}
Entry price     : {entry_price}
Quantity        : {quantity}
Unrealized PnL  : {unrealized_pnl}
Liquidation     : {liquidation_price}
Active SL order : {current_sl}
Active TP order : {current_tp}
─────────────────────────────────────────────────────────
"""

_POSITION_SECTION_NONE = """\
── Current Open Position ────────────────────────────────
No open position.
─────────────────────────────────────────────────────────
"""

_USER_PROMPT = """\
Symbol  : {symbol}
Interval: 15m

{position_section}
── Price Chart (oldest → newest) ────────────────────────
{csv_data}
─────────────────────────────────────────────────────────

What is your trading decision?
"""


# ── Decision engine ───────────────────────────────────────────────────────────

class LLMDecisionEngine:
    def __init__(self, cfg: LLMTraderConfig) -> None:
        self._cfg = cfg
        self._client = AzureOpenAI(
            azure_endpoint=cfg.azure_endpoint,
            api_key=cfg.azure_api_key,
            api_version=cfg.azure_api_version,
        )

    def decide(
        self,
        symbol: str,
        candles: list[Candle],
        position: OpenPosition | None = None,
    ) -> LLMDecision:
        if not candles:
            raise ValueError("No candles provided to LLM decision engine.")

        csv_data = candles_to_csv(candles)
        position_section = _format_position(position)
        system_msg = _SYSTEM_PROMPT.format(n=len(candles), symbol=symbol)
        skills_block = load_skills(self._cfg.skills)
        if skills_block:
            system_msg = f"{system_msg}\n\n{skills_block}"
        user_msg = _USER_PROMPT.format(
            symbol=symbol,
            position_section=position_section,
            csv_data=csv_data,
        )

        logger.info(
            "Sending %d candles + position=%s for %s to Azure OpenAI (%s)…",
            len(candles),
            position.side if position else "NONE",
            symbol,
            self._cfg.azure_deployment,
        )

        response = self._client.chat.completions.create(
            model=self._cfg.azure_deployment,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=300,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or ""
        logger.debug("LLM raw response: %s", raw)
        return _parse_decision(raw)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_position(position: OpenPosition | None) -> str:
    if position is None or not position.is_open:
        return _POSITION_SECTION_NONE
    return _POSITION_SECTION_OPEN.format(
        side=position.side,
        entry_price=position.entry_price,
        quantity=position.quantity,
        unrealized_pnl=f"{position.unrealized_pnl:+.4f} USDT",
        liquidation_price=position.liquidation_price,
        current_sl=position.current_sl if position.current_sl else "none",
        current_tp=position.current_tp if position.current_tp else "none",
    )


def _parse_decision(raw: str) -> LLMDecision:
    try:
        data: dict[str, Any] = json.loads(raw)
        action = Action(data["action"].upper())
        confidence = int(data.get("confidence", 5))
        confidence = max(1, min(10, confidence))
        reason = str(data.get("reason", ""))

        sl_raw = data.get("stop_loss") or data.get("sl")
        tp_raw = data.get("take_profit") or data.get("tp")
        sl = float(sl_raw) if sl_raw is not None else None
        tp = float(tp_raw) if tp_raw is not None else None

        return LLMDecision(
            action=action, sl=sl, tp=tp,
            confidence=confidence, reason=reason, raw_response=raw,
        )
    except Exception as exc:
        logger.warning("Failed to parse LLM response — defaulting to HOLD. Error: %s", exc)
        return LLMDecision(
            action=Action.HOLD, sl=None, tp=None,
            confidence=1, reason="Parse error", raw_response=raw,
        )
