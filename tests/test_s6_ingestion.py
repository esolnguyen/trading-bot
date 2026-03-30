"""Acceptance tests for S6 offline RAG ingestion."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import requests

from src.core.config import Settings
from src.infrastructure.storage import ChromaStore
from src.services.rag.filter import is_relevant_article, truncate_body
from src.services.rag.ingestion_loop import IngestionLoop


def build_settings(tmp_path: Path) -> Settings:
    return Settings(
        azure_endpoint="https://example-resource.openai.azure.com",
        azure_api_key="azure-key-1234",
        azure_deployment="grok-prod",
        binance_api_key="binance-key-1234",
        binance_api_secret="binance-secret-1234",
        cryptocompare_api_key="cc-key-1234",
        chroma_path=str(tmp_path / "chroma"),
    )


class FakeNewsSource:
    def __init__(self, documents):
        self.documents = documents

    async def fetch(self):
        return self.documents


class FakeMacroSource:
    def __init__(self, documents):
        self.documents = documents

    async def fetch(self):
        return self.documents


class Http500Source:
    async def fetch(self):
        response = requests.Response()
        response.status_code = 500
        raise requests.HTTPError("boom", response=response)


def test_news_collection_contains_document_after_one_cycle(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    store = ChromaStore(settings.chroma_path)
    loop = IngestionLoop(
        store,
        settings,
        sources={
            "news": FakeNewsSource(
                [
                    {
                        "title": "Bitcoin rallies after crypto breakout",
                        "url": "https://example.com/1",
                        "body": "Bitcoin and crypto markets moved higher today.",
                        "source": "example",
                        "published_at": "2026-03-25T00:00:00Z",
                        "symbol_tags": "BTC,ETH",
                    }
                ]
            ),
            "coingecko": FakeMacroSource([]),
            "alternative_me": FakeMacroSource([]),
            "defillama": FakeMacroSource([]),
            "ohlcv_history": FakeMacroSource([]),
        },
    )

    asyncio.run(loop.ingest_news_once())

    assert store.count("news") >= 1


def test_duplicate_url_is_stored_once(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    store = ChromaStore(settings.chroma_path)
    source = FakeNewsSource(
        [
            {
                "title": "Ethereum crypto update",
                "url": "https://example.com/dup",
                "body": "Ethereum remains central to crypto markets.",
                "source": "example",
                "published_at": "2026-03-25T00:00:00Z",
                "symbol_tags": "ETH",
            }
        ]
    )
    loop = IngestionLoop(
        store,
        settings,
        sources={
            "news": source,
            "coingecko": FakeMacroSource([]),
            "alternative_me": FakeMacroSource([]),
            "defillama": FakeMacroSource([]),
            "ohlcv_history": FakeMacroSource([]),
        },
    )

    asyncio.run(loop.ingest_news_once())
    asyncio.run(loop.ingest_news_once())

    assert store.count("news") == 1


def test_http_500_is_logged_and_loop_continues(tmp_path: Path, caplog) -> None:
    settings = build_settings(tmp_path)
    store = ChromaStore(settings.chroma_path)
    caplog.set_level(logging.ERROR)
    loop = IngestionLoop(
        store,
        settings,
        sources={
            "news": Http500Source(),
            "coingecko": FakeMacroSource(
                [{"source": "coingecko", "metric": "fear", "value": 50, "narrative": "Crypto is calm.", "timestamp": "2026-03-25T00"}]
            ),
            "alternative_me": FakeMacroSource([]),
            "defillama": FakeMacroSource([]),
            "ohlcv_history": FakeMacroSource([]),
        },
    )

    asyncio.run(loop.ingest_news_once())
    asyncio.run(loop.ingest_macro_once("coingecko"))

    assert "HTTP error in news source news" in caplog.text
    assert store.count("macro") == 1


def test_store_raises_runtime_error_for_unwritable_path(tmp_path: Path) -> None:
    bad_path = tmp_path / "not_a_dir"
    bad_path.write_text("blocked", encoding="utf-8")

    try:
        ChromaStore(str(bad_path))
    except RuntimeError as exc:
        assert "not writable" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for unwritable path")


def test_off_topic_article_filtered_before_embedding(tmp_path: Path) -> None:
    with TemporaryDirectory() as temp_dir:
        settings = build_settings(Path(temp_dir))
        store = ChromaStore(settings.chroma_path)
        loop = IngestionLoop(
            store,
            settings,
            sources={
                "news": FakeNewsSource(
                    [
                        {
                            "title": "Gold prices rise on macro jitters",
                            "url": "https://example.com/gold",
                            "body": "Gold demand is rising while bond yields shift.",
                            "source": "example",
                            "published_at": "2026-03-25T00:00:00Z",
                            "symbol_tags": "",
                        }
                    ]
                ),
                "coingecko": FakeMacroSource([]),
                "alternative_me": FakeMacroSource([]),
                "defillama": FakeMacroSource([]),
                "ohlcv_history": FakeMacroSource([]),
            },
        )

        asyncio.run(loop.ingest_news_once())

        assert store.count("news") == 0


def test_filter_helpers_apply_relevance_and_truncation() -> None:
    assert is_relevant_article("Bitcoin jumps", "crypto market rally") is True
    assert is_relevant_article("Gold jumps", "metals market rally") is False
    assert len(truncate_body("x" * 1500)) == 1000


def test_store_uses_embedding_profile_specific_collection_names(tmp_path: Path) -> None:
    class Embed16:
        def __call__(self, input):
            return [[0.1] * 16 for _ in input]

    class Embed384:
        def __call__(self, input):
            return [[0.1] * 384 for _ in input]

    path = str(tmp_path / "chroma")
    store16 = ChromaStore(path, embedding_function=Embed16())
    store16.add_document(
        "news",
        document_id="doc-16",
        text="btc rally",
        metadata={"source": "test"},
    )

    store384 = ChromaStore(path, embedding_function=Embed384())
    store384.add_document(
        "news",
        document_id="doc-384",
        text="eth rally",
        metadata={"source": "test"},
    )

    assert store16.count("news") == 1
    assert store384.count("news") == 1
    assert store16.collections["news"].name != store384.collections["news"].name
