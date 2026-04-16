"""Durable bot persistence for trades and cycle logs."""

from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from src.domain.analysis import PatternResult, TechnicalAnalysis
from src.domain.trading import TradeOutcome
from src.shared.json_io import read_json as _shared_read_json, write_json as _shared_write_json
from src.shared.symbol_utils import slugify_symbol


class Persistence:
    """Append trade and cycle logs to durable files under logs/.

    Also provides JSON-based state persistence for positions, trade history,
    statistics, AI response caching, and analysis timing.
    """

    CSV_HEADER = [
        "timestamp_iso",
        "timeframe",
        "symbol",
        "action",
        "order_type",
        "quantity",
        "price",
        "order_id",
        "executed_price",
        "pnl_usdt",
        "dry_run",
        "source",
        "reasoning",
    ]

    def __init__(
        self,
        log_dir: str | Path = "logs",
        data_dir: str | Path = "data",
        timeframe: str = "",
        trading_engine: str = "",
    ) -> None:
        base_log_dir = Path(log_dir)
        self.timeframe = timeframe
        # Organise logs by trading engine, with an extra timeframe
        # subfolder for the scorer engine (it runs across multiple TFs).
        if trading_engine:
            self.log_dir = base_log_dir / trading_engine
            if trading_engine == "scorer" and timeframe:
                self.log_dir = self.log_dir / timeframe
        elif timeframe:
            self.log_dir = base_log_dir / timeframe
        else:
            self.log_dir = base_log_dir
        self.data_dir = Path(data_dir)
        self.daily_log_dir = self.log_dir / datetime.now().strftime("%Y-%m-%d")
        self.trades_csv_path = self.log_dir / "trades.csv"
        self.bot_log_path = self.daily_log_dir / "bot.log"
        self.daily_log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Existing append methods (unchanged)
    # ------------------------------------------------------------------

    def append_trade(self, outcome: TradeOutcome, timestamp_iso: str) -> None:
        write_header = not self.trades_csv_path.exists() or self.trades_csv_path.stat().st_size == 0
        with self.trades_csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(self.CSV_HEADER)
            writer.writerow(
                [
                    timestamp_iso,
                    self.timeframe,
                    outcome.decision.symbol,
                    outcome.decision.action.value,
                    outcome.decision.order_type,
                    outcome.decision.quantity,
                    outcome.decision.price if outcome.decision.price is not None else "",
                    outcome.order_id if outcome.order_id is not None else "",
                    outcome.executed_price if outcome.executed_price is not None else "",
                    outcome.pnl_usdt if outcome.pnl_usdt is not None else "",
                    outcome.dry_run,
                    outcome.decision.source,
                    outcome.decision.reasoning,
                ]
            )

    def append_cycle_log(
        self,
        *,
        timestamp_iso: str,
        cycle: int,
        symbol: str,
        analysis: TechnicalAnalysis,
        patterns: PatternResult,
        rag_docs_retrieved: int,
        llm_decision: str,
        llm_usage: dict[str, int] | None,
        llm_prompt: dict[str, Any] | None,
        llm_raw_response: str | None,
        decision_source: str,
        decision_reasoning: str,
        llm_error: str | None,
        risk_outcome: str,
        order_id: str | None,
        dry_run: bool,
    ) -> None:
        payload: dict[str, Any] = {
            "ts": timestamp_iso,
            "timeframe": self.timeframe,
            "cycle": cycle,
            "symbol": symbol,
            "signal": analysis.signal.value,
            "patterns": patterns.patterns,
            "rag_docs_retrieved": rag_docs_retrieved,
            "llm_decision": llm_decision,
            "llm_usage": llm_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "llm_prompt": llm_prompt or {},
            "llm_raw_response": llm_raw_response,
            "decision_source": decision_source,
            "decision_reasoning": decision_reasoning,
            "llm_error": llm_error,
            "risk_outcome": risk_outcome,
            "order_id": order_id,
            "dry_run": dry_run,
        }
        with self.bot_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def _read_json(self, path: Path) -> Any:
        return _shared_read_json(path)

    def _write_json(self, path: Path, data: Any) -> None:
        _shared_write_json(path, data)

    # ------------------------------------------------------------------
    # Position state
    # ------------------------------------------------------------------

    def _slug_path(self, kind: str, symbol: str) -> Path:
        """Build a per-symbol JSON path for the given file kind."""
        return self.data_dir / f"{kind}_{slugify_symbol(symbol)}.json"

    def save_position(self, symbol: str, position_data: Optional[dict[str, Any]]) -> None:
        """Persist an open position dict (or None to clear it)."""
        path = self._slug_path("position", symbol)
        if position_data is None:
            path.unlink(missing_ok=True)
        else:
            self._write_json(path, position_data)

    def load_position(self, symbol: str) -> Optional[dict[str, Any]]:
        """Load an open position dict for the given symbol."""
        return self._read_json(self._slug_path("position", symbol))

    async def async_save_position(self, symbol: str, position_data: Optional[dict[str, Any]]) -> None:
        await asyncio.to_thread(self.save_position, symbol, position_data)

    # ------------------------------------------------------------------
    # Trade history
    # ------------------------------------------------------------------

    def _trade_history_path(self, symbol: str) -> Path:
        return self._slug_path("trade_history", symbol)

    def save_trade_record(self, symbol: str, record: dict[str, Any]) -> None:
        """Append a single trade record dict to the trade history file."""
        path = self._trade_history_path(symbol)
        existing: List[dict[str, Any]] = self._read_json(path) or []
        existing.append(record)
        self._write_json(path, existing)

    def load_trade_history(self, symbol: str) -> List[dict[str, Any]]:
        """Load all trade records for the given symbol."""
        return self._read_json(self._trade_history_path(symbol)) or []

    def load_last_n_decisions(self, symbol: str, n: int = 10) -> List[dict[str, Any]]:
        """Return the n most recent trade records."""
        history = self.load_trade_history(symbol)
        return history[-n:] if history else []

    def get_entry_decision_for_position(self, symbol: str) -> Optional[dict[str, Any]]:
        """Return the last BUY/SELL record with no subsequent CLOSE for this symbol."""
        history = self.load_trade_history(symbol)
        open_entry = None
        for record in history:
            action = record.get("action", "")
            if action in ("BUY", "SELL"):
                open_entry = record
            elif action in ("CLOSE", "CLOSE_LONG", "CLOSE_SHORT"):
                open_entry = None
        return open_entry

    async def async_save_trade_record(self, symbol: str, record: dict[str, Any]) -> None:
        await asyncio.to_thread(self.save_trade_record, symbol, record)

    async def async_save_trade_decision(self, decision: Any) -> None:
        """Persist a TradeRecord (strategy path) to trade history.

        Accepts any object with .symbol and .to_dict() (TradeRecord / SerializableMixin).
        """
        symbol = getattr(decision, "symbol", "unknown")
        data = decision.to_dict() if callable(getattr(decision, "to_dict", None)) else dict(decision)
        await self.async_save_trade_record(symbol, data)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _statistics_path(self, symbol: str) -> Path:
        return self._slug_path("statistics", symbol)

    def save_statistics(self, symbol: str, stats: dict[str, Any]) -> None:
        self._write_json(self._statistics_path(symbol), stats)

    def load_statistics(self, symbol: str) -> Optional[dict[str, Any]]:
        return self._read_json(self._statistics_path(symbol))

    async def async_save_statistics(self, symbol: str, stats: dict[str, Any]) -> None:
        await asyncio.to_thread(self.save_statistics, symbol, stats)

    # ------------------------------------------------------------------
    # Previous AI response caching
    # ------------------------------------------------------------------

    def _response_cache_path(self, symbol: str) -> Path:
        return self._slug_path("last_response", symbol)

    def save_previous_response(self, symbol: str, response_data: dict[str, Any]) -> None:
        self._write_json(self._response_cache_path(symbol), response_data)

    def load_previous_response(self, symbol: str) -> Optional[dict[str, Any]]:
        return self._read_json(self._response_cache_path(symbol))

    async def async_save_previous_response(self, symbol: str, response_data: dict[str, Any]) -> None:
        await asyncio.to_thread(self.save_previous_response, symbol, response_data)

    # ------------------------------------------------------------------
    # Last analysis timestamp
    # ------------------------------------------------------------------

    def _analysis_time_path(self, symbol: str) -> Path:
        return self._slug_path("last_analysis", symbol)

    def save_last_analysis_time(self, symbol: str, ts: Optional[datetime] = None) -> None:
        """Persist the last analysis timestamp (defaults to UTC now)."""
        if ts is None:
            ts = datetime.now(tz=timezone.utc)
        self._write_json(self._analysis_time_path(symbol), {"timestamp": ts.isoformat()})

    def get_last_analysis_time(self, symbol: str) -> Optional[datetime]:
        """Return the last analysis timestamp for the symbol, or None."""
        data = self._read_json(self._analysis_time_path(symbol))
        if data and "timestamp" in data:
            try:
                return datetime.fromisoformat(data["timestamp"])
            except ValueError:
                return None
        return None

    async def async_save_last_analysis_time(self, symbol: str, ts: Optional[datetime] = None) -> None:
        await asyncio.to_thread(self.save_last_analysis_time, symbol, ts)
