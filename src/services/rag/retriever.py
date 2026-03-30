"""Online RAG retrieval for decision-time context assembly."""

from __future__ import annotations

from src.domain.analysis import TechnicalAnalysis
from src.domain.market import MarketSnapshot
from src.infrastructure.storage import ChromaStore


class RAGRetriever:
    """Fetch and format local Chroma context for the trading decision."""

    MAX_OUTPUT_CHARS = 4000
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

    def _format_news_section(self, payload: dict, symbol: str) -> str:
        rows = self._pairs_from_query(payload)
        rows = self._filter_news_rows(rows, symbol)
        if not rows:
            return ""

        display_symbol = symbol.replace("USDT", "").replace("BUSD", "")
        lines = [f"=== RECENT NEWS ({display_symbol}) ==="]
        for index, (document, metadata) in enumerate(rows, start=1):
            title = metadata.get("title", "Untitled")
            source = metadata.get("source", "unknown")
            published_at = metadata.get("published_at", "")
            body = metadata.get("body", document or "")[:300]
            lines.append(f"[{index}] {title} — {source} ({published_at})")
            lines.append(body)
        return "\n".join(lines)

    def _format_macro_section(self, payload: dict) -> str:
        rows = self._pairs_from_query(payload)
        if not rows:
            return ""

        lines = ["=== MACRO CONTEXT ==="]
        for index, (_document, metadata) in enumerate(rows, start=1):
            source = metadata.get("source", "unknown")
            narrative = metadata.get("narrative", "")
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
            symbol = metadata.get("symbol", "UNKNOWN")
            action = metadata.get("action", "UNKNOWN")
            timestamp = metadata.get("timestamp", "")
            reasoning = str(metadata.get("reasoning", ""))[:200]
            pnl = metadata.get("outcome_pnl")
            lines.append(f"[{index}] {symbol} {action} @ {timestamp}")
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
