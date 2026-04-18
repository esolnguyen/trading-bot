"""Acceptance tests for S7 RAG retriever."""

from __future__ import annotations

import time

from src.mcp_servers.shared.domain.analysis import (
    IndicatorSet,
    Signal,
    TechnicalAnalysis,
)
from src.mcp_servers.shared.domain.market import MarketSnapshot, OHLCVCandle
from src.legacy.services.rag import RAGRetriever


class FakeStore:
    def __init__(
        self, responses: dict[str, dict], counts: dict[str, int] | None = None
    ) -> None:
        self.responses = responses
        self.counts = counts or {name: 1 for name in ("news", "macro", "trade_memory")}
        self.queries: list[tuple[str, str, int]] = []

    def count(self, collection_name: str) -> int:
        return self.counts.get(collection_name, 0)

    def query(self, collection_name: str, query_text: str, n_results: int) -> dict:
        self.queries.append((collection_name, query_text, n_results))
        return self.responses.get(
            collection_name, {"documents": [[]], "metadatas": [[]]}
        )


def snapshot_fixture() -> MarketSnapshot:
    candle = OHLCVCandle(
        timestamp=1, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0
    )
    return MarketSnapshot(
        symbol="BTCUSDT",
        price=64000.0,
        change_24h_pct=2.5,
        volume_24h=123456.0,
        bid=63990.0,
        ask=64010.0,
        candles=[candle] * 60,
        timestamp=1,
    )


def analysis_fixture() -> TechnicalAnalysis:
    return TechnicalAnalysis(
        symbol="BTCUSDT",
        signal=Signal.BUY,
        indicators=IndicatorSet(
            rsi_14=35.0,
            macd_line=1.0,
            macd_signal=0.8,
            macd_hist=0.2,
            bb_upper=65000.0,
            bb_mid=63000.0,
            bb_lower=61000.0,
            ema_20=63500.0,
            ema_50=62000.0,
            volume_sma_20=1000.0,
        ),
        reasoning="RSI is recovering and price sits above the middle band.",
    )


def test_returns_no_context_when_all_collections_empty() -> None:
    retriever = RAGRetriever(
        FakeStore({}, counts={"news": 0, "macro": 0, "trade_memory": 0})
    )

    result = retriever.retrieve(snapshot_fixture(), analysis_fixture())

    assert result == "=== NO CONTEXT AVAILABLE ==="


def test_returns_formatted_context_under_character_budget() -> None:
    retriever = RAGRetriever(
        FakeStore(
            {
                "news": {
                    "documents": [["Bitcoin body"]],
                    "metadatas": [
                        [
                            {
                                "title": "Bitcoin rally",
                                "source": "example",
                                "published_at": "2026-03-25",
                                "body": "Bitcoin body",
                            }
                        ]
                    ],
                },
                "macro": {
                    "documents": [["Macro text"]],
                    "metadatas": [
                        [
                            {
                                "source": "coingecko",
                                "narrative": "Market sentiment is improving.",
                            }
                        ]
                    ],
                },
                "trade_memory": {
                    "documents": [["Trade memory"]],
                    "metadatas": [
                        [
                            {
                                "symbol": "BTCUSDT",
                                "action": "BUY",
                                "timestamp": "2026-03-24T00:00:00Z",
                                "reasoning": "Momentum aligned",
                                "outcome_pnl": 12.5,
                            }
                        ]
                    ],
                },
            }
        )
    )

    result = retriever.retrieve(snapshot_fixture(), analysis_fixture())

    assert "=== RECENT NEWS (BTC) ===" in result
    assert "=== MACRO CONTEXT ===" in result
    assert "=== SIMILAR PAST TRADES ===" in result
    assert len(result) <= 4000


def test_filters_news_to_symbol_relevant_documents_when_possible() -> None:
    retriever = RAGRetriever(
        FakeStore(
            {
                "news": {
                    "documents": [["Bitcoin body", "TRX body", "Ethereum body"]],
                    "metadatas": [
                        [
                            {
                                "title": "Bitcoin rally",
                                "source": "example",
                                "published_at": "2026-03-25",
                                "body": "Bitcoin body",
                            },
                            {
                                "title": "TRX breakout",
                                "source": "example",
                                "published_at": "2026-03-25",
                                "body": "TRX body",
                            },
                            {
                                "title": "Ethereum strength",
                                "source": "example",
                                "published_at": "2026-03-25",
                                "body": "Ethereum body",
                            },
                        ]
                    ],
                },
                "macro": {"documents": [[]], "metadatas": [[]]},
                "trade_memory": {"documents": [[]], "metadatas": [[]]},
            }
        )
    )

    result = retriever.retrieve(snapshot_fixture(), analysis_fixture())

    assert "Bitcoin rally" in result
    assert "TRX breakout" not in result
    assert "Ethereum strength" not in result


def test_filters_trade_memory_to_same_symbol() -> None:
    retriever = RAGRetriever(
        FakeStore(
            {
                "news": {"documents": [[]], "metadatas": [[]]},
                "macro": {"documents": [[]], "metadatas": [[]]},
                "trade_memory": {
                    "documents": [["btc trade", "eth trade"]],
                    "metadatas": [
                        [
                            {
                                "symbol": "BTCUSDT",
                                "action": "BUY",
                                "timestamp": "t1",
                                "reasoning": "Momentum aligned",
                                "outcome_pnl": 10.0,
                            },
                            {
                                "symbol": "ETHUSDT",
                                "action": "SELL",
                                "timestamp": "t2",
                                "reasoning": "Reversal",
                                "outcome_pnl": 5.0,
                            },
                        ]
                    ],
                },
            }
        )
    )

    result = retriever.retrieve(snapshot_fixture(), analysis_fixture())

    assert "BTCUSDT BUY @ t1" in result
    assert "ETHUSDT SELL @ t2" not in result


def test_query_is_constructed_from_snapshot_and_analysis() -> None:
    store = FakeStore(
        {
            "news": {"documents": [[]], "metadatas": [[]]},
            "macro": {"documents": [[]], "metadatas": [[]]},
            "trade_memory": {"documents": [[]], "metadatas": [[]]},
        }
    )
    retriever = RAGRetriever(store)

    retriever.retrieve(snapshot_fixture(), analysis_fixture())

    assert len(store.queries) == 3
    query_text = store.queries[0][1]
    assert "BTCUSDT" in query_text
    assert "64000.0" in query_text
    assert "BUY" in query_text


def test_retrieve_is_fast_with_local_store() -> None:
    long_body = "crypto " * 200
    store = FakeStore(
        {
            "news": {
                "documents": [[long_body] * 5],
                "metadatas": [
                    [
                        {
                            "title": f"News {i}",
                            "source": "example",
                            "published_at": "2026-03-25",
                            "body": long_body,
                        }
                        for i in range(5)
                    ]
                ],
            },
            "macro": {
                "documents": [["macro"] * 3],
                "metadatas": [
                    [{"source": f"macro-{i}", "narrative": "macro"} for i in range(3)]
                ],
            },
            "trade_memory": {
                "documents": [["memory"] * 3],
                "metadatas": [
                    [
                        {
                            "symbol": "BTCUSDT",
                            "action": "BUY",
                            "timestamp": f"t{i}",
                            "reasoning": "reasoning",
                            "outcome_pnl": i,
                        }
                        for i in range(3)
                    ]
                ],
            },
        },
        counts={"news": 10, "macro": 10, "trade_memory": 10},
    )
    retriever = RAGRetriever(store)

    started = time.perf_counter()
    result = retriever.retrieve(snapshot_fixture(), analysis_fixture())
    duration_ms = (time.perf_counter() - started) * 1000

    assert duration_ms < 100
    assert len(result) <= 4000
