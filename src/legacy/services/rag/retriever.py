"""Online RAG retrieval for decision-time context assembly."""

from __future__ import annotations

import re
from typing import Any

from src.mcp_servers.shared.domain.analysis import TechnicalAnalysis
from src.mcp_servers.shared.domain.market import MarketSnapshot
from src.mcp_servers.rag_mcp.storage import ChromaStore


class RAGRetriever:
    """Fetch and format local Chroma context for the trading decision."""

    MAX_OUTPUT_CHARS = 4000
    # Per-field character limits to prevent oversized docs from dominating the prompt
    MAX_TITLE_CHARS = 120
    MAX_BODY_CHARS = 300
    MAX_NARRATIVE_CHARS = 400
    MAX_REASONING_CHARS = 200
    # Regex to strip control characters (keep printable ASCII and common unicode)
    _CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    SYMBOL_KEYWORDS = {
        "BTCUSDT": ("btc", "bitcoin"),
        "ETHUSDT": ("eth", "ethereum"),
        "SOLUSDT": ("sol", "solana"),
        "BNBUSDT": ("bnb", "binance"),
    }

    def __init__(self, store: ChromaStore, settings: Any | None = None) -> None:
        self.store = store
        self._news_results: int = (
            getattr(settings, "rag_retrieval_news", 3) if settings else 3
        )
        self._macro_results: int = (
            getattr(settings, "rag_retrieval_macro", 1) if settings else 1
        )
        self._memory_results: int = (
            getattr(settings, "rag_retrieval_memory", 2) if settings else 2
        )

    def retrieve(
        self,
        snapshot: MarketSnapshot,
        analysis: TechnicalAnalysis,
        *,
        include_macro: bool = True,
    ) -> str:
        """Retrieve formatted RAG context for one symbol.

        Args:
            include_macro: When False, the global macro section is omitted.
                           Use ``retrieve_macro`` once and share it across symbols
                           to avoid sending the same macro text N times.
        """
        if all(
            self.store.count(name) == 0 for name in ("news", "macro", "trade_memory")
        ):
            return "=== NO CONTEXT AVAILABLE ==="

        query = f"{snapshot.symbol} price {snapshot.price} signal {analysis.signal.value} {analysis.reasoning[:200]}"

        news_docs = self.store.query("news", query, n_results=self._news_results)
        macro_docs = (
            self.store.query("macro", query, n_results=self._macro_results)
            if include_macro
            else {}
        )
        memory_docs = self.store.query(
            "trade_memory", query, n_results=self._memory_results
        )

        sections = [
            self._format_news_section(news_docs, snapshot.symbol),
            self._format_macro_section(macro_docs) if include_macro else "",
            self._format_trade_memory_section(memory_docs, snapshot.symbol),
        ]
        content = "\n\n".join(section for section in sections if section)
        if not content.strip():
            return "=== NO CONTEXT AVAILABLE ==="
        return content[: self.MAX_OUTPUT_CHARS]

    def retrieve_macro(self, query: str) -> str:
        """Return a formatted macro section using a freeform query string.

        Call this once per cycle when ``single_symbol_decision`` is True, then
        append the result to the combined RAG block instead of repeating it per symbol.
        """
        if self.store.count("macro") == 0:
            return ""
        macro_docs = self.store.query("macro", query, n_results=self._macro_results)
        return self._format_macro_section(macro_docs)

    @classmethod
    def _sanitize(cls, text: str, max_chars: int) -> str:
        """Strip control characters and truncate to prevent prompt injection."""
        cleaned = cls._CONTROL_RE.sub(" ", str(text))
        # Collapse runs of whitespace to single spaces
        cleaned = " ".join(cleaned.split())
        return cleaned[:max_chars]

    def _format_news_section(self, payload: dict, symbol: str) -> str:
        rows = self._pairs_from_query(payload)
        rows = self._filter_news_rows(rows, symbol)
        if not rows:
            return ""

        display_symbol = symbol.replace("USDT", "").replace("BUSD", "")
        lines = [f"=== RECENT NEWS ({display_symbol}) ==="]
        for index, (document, metadata) in enumerate(rows, start=1):
            title = self._sanitize(
                metadata.get("title", "Untitled"), self.MAX_TITLE_CHARS
            )
            source = self._sanitize(metadata.get("source", "unknown"), 60)
            published_at = self._sanitize(metadata.get("published_at", ""), 30)
            body = self._sanitize(
                metadata.get("body", document or ""), self.MAX_BODY_CHARS
            )
            # Fix #7: surface FinBERT sentiment score when present
            sentiment_score = metadata.get("sentiment_score")
            sentiment_tag = ""
            if sentiment_score is not None:
                try:
                    score_f = float(sentiment_score)
                    if score_f >= 0.3:
                        sentiment_tag = " 📈[bullish]"
                    elif score_f <= -0.3:
                        sentiment_tag = " 📉[bearish]"
                    else:
                        sentiment_tag = " ➡[neutral]"
                except (TypeError, ValueError):
                    pass
            lines.append(
                f"[{index}] {title} — {source} ({published_at}){sentiment_tag}"
            )
            lines.append(body)
        return "\n".join(lines)

    def _format_macro_section(self, payload: dict) -> str:
        rows = self._pairs_from_query(payload)
        if not rows:
            return ""

        lines = ["=== MACRO CONTEXT ==="]
        for index, (_document, metadata) in enumerate(rows, start=1):
            source = self._sanitize(metadata.get("source", "unknown"), 60)
            narrative = self._sanitize(
                metadata.get("narrative", ""), self.MAX_NARRATIVE_CHARS
            )
            lines.append(f"[{index}] {source}: {narrative}")
        return "\n".join(lines)

    def _format_trade_memory_section(self, payload: dict, symbol: str) -> str:
        rows = self._pairs_from_query(payload)
        rows = [
            (document, metadata)
            for document, metadata in rows
            if metadata.get("symbol") in ("", None, symbol)
        ]
        if not rows:
            return ""

        lines = ["=== SIMILAR PAST TRADES ==="]
        for index, (_document, metadata) in enumerate(rows, start=1):
            trade_symbol = self._sanitize(metadata.get("symbol", "UNKNOWN"), 20)
            action = self._sanitize(metadata.get("action", "UNKNOWN"), 20)
            timestamp = self._sanitize(metadata.get("timestamp", ""), 30)
            reasoning = self._sanitize(
                str(metadata.get("reasoning", "")), self.MAX_REASONING_CHARS
            )
            pnl = metadata.get("outcome_pnl")
            lines.append(f"[{index}] {trade_symbol} {action} @ {timestamp}")
            lines.append(f"Reasoning: {reasoning}")
            lines.append(f"Outcome PnL: {pnl} USDT")
        return "\n".join(lines)

    @staticmethod
    def _pairs_from_query(payload: dict) -> list[tuple[str, dict]]:
        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        return list(zip(documents, metadatas))

    @classmethod
    def _filter_news_rows(
        cls, rows: list[tuple[str, dict]], symbol: str
    ) -> list[tuple[str, dict]]:
        keywords = cls.SYMBOL_KEYWORDS.get(symbol, ())
        if not keywords:
            return rows

        filtered: list[tuple[str, dict]] = []
        for document, metadata in rows:
            haystack = " ".join(
                str(part)
                for part in (
                    metadata.get("symbol", ""),
                    metadata.get("title", ""),
                    metadata.get("body", ""),
                    document or "",
                )
            ).lower()
            if any(keyword in haystack for keyword in keywords):
                filtered.append((document, metadata))

        return filtered or rows[:2]
