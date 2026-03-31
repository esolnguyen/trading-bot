"""Online RAG retrieval for decision-time context assembly."""

from __future__ import annotations

import re

from src.domain.analysis import TechnicalAnalysis
from src.domain.market import MarketSnapshot
from src.infrastructure.storage import ChromaStore


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
    }

    def __init__(self, store: ChromaStore) -> None:
        self.store = store

    def retrieve(self, snapshot: MarketSnapshot, analysis: TechnicalAnalysis) -> str:
        if all(self.store.count(name) == 0 for name in ("news", "macro", "trade_memory")):
            return "=== NO CONTEXT AVAILABLE ==="

        query = f"{snapshot.symbol} price {snapshot.price} signal {analysis.signal.value} {analysis.reasoning[:200]}"

        news_docs = self.store.query("news", query, n_results=5)
        macro_docs = self.store.query("macro", query, n_results=3)
        memory_docs = self.store.query("trade_memory", query, n_results=3)

        sections = [
            self._format_news_section(news_docs, snapshot.symbol),
            self._format_macro_section(macro_docs),
            self._format_trade_memory_section(memory_docs, snapshot.symbol),
        ]
        content = "\n\n".join(section for section in sections if section)
        if not content.strip():
            return "=== NO CONTEXT AVAILABLE ==="
        return content[: self.MAX_OUTPUT_CHARS]

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
            title = self._sanitize(metadata.get("title", "Untitled"), self.MAX_TITLE_CHARS)
            source = self._sanitize(metadata.get("source", "unknown"), 60)
            published_at = self._sanitize(metadata.get("published_at", ""), 30)
            body = self._sanitize(metadata.get("body", document or ""), self.MAX_BODY_CHARS)
            lines.append(f"[{index}] {title} — {source} ({published_at})")
            lines.append(body)
        return "\n".join(lines)

    def _format_macro_section(self, payload: dict) -> str:
        rows = self._pairs_from_query(payload)
        if not rows:
            return ""

        lines = ["=== MACRO CONTEXT ==="]
        for index, (_document, metadata) in enumerate(rows, start=1):
            source = self._sanitize(metadata.get("source", "unknown"), 60)
            narrative = self._sanitize(metadata.get("narrative", ""), self.MAX_NARRATIVE_CHARS)
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
            reasoning = self._sanitize(str(metadata.get("reasoning", "")), self.MAX_REASONING_CHARS)
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
    def _filter_news_rows(cls, rows: list[tuple[str, dict]], symbol: str) -> list[tuple[str, dict]]:
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
