"""Offline RAG ingestion loop."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import requests

from src.core.config import Settings
from src.infrastructure.storage.chroma_store import ChromaStore
from src.services.rag.filter import is_relevant_article, prepare_news_document, sha256_text
from src.services.rag.sources.alternative_me import AlternativeMeSource
from src.services.rag.sources.coingecko import CoinGeckoFearGreedSource
from src.services.rag.sources.cryptocompare import CryptoCompareNewsSource
from src.services.rag.sources.defillama import DefiLlamaSource
from src.services.rag.sources.ohlcv_history import OHLCVHistorySource


class IngestionLoop:
    """Continuously ingest offline context into ChromaDB."""

    def __init__(
        self,
        store: ChromaStore,
        settings: Settings,
        *,
        sources: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
        sleep_func: Any = asyncio.sleep,
        sentiment_scorer: Any | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)
        self.sleep_func = sleep_func
        self.sentiment_scorer = sentiment_scorer
        self.sources = sources or {
            "news": CryptoCompareNewsSource(settings.cryptocompare_api_key),
            "coingecko": CoinGeckoFearGreedSource(),
            "alternative_me": AlternativeMeSource(),
            "defillama": DefiLlamaSource(),
            "ohlcv_history": OHLCVHistorySource(),
        }

    async def run(self) -> None:
        await asyncio.gather(
            self._run_source_forever("news", self.settings.news_interval, self.ingest_news_once),
            self._run_source_forever("coingecko", self.settings.macro_interval, self.ingest_macro_once),
            self._run_source_forever("alternative_me", self.settings.ohlcv_interval, self.ingest_macro_once),
            self._run_source_forever("defillama", self.settings.ohlcv_interval, self.ingest_macro_once),
            self._run_source_forever("ohlcv_history", self.settings.ohlcv_interval, self.ingest_macro_once),
        )

    async def run_cycle_once(self) -> None:
        await asyncio.gather(
            self.ingest_news_once("news"),
            self.ingest_macro_once("coingecko"),
            self.ingest_macro_once("alternative_me"),
            self.ingest_macro_once("defillama"),
            self.ingest_macro_once("ohlcv_history"),
        )

    async def ingest_news_once(self, source_name: str = "news") -> None:
        source = self.sources[source_name]
        try:
            documents = await source.fetch()
            for document in documents:
                prepared = prepare_news_document(document)
                if not is_relevant_article(prepared.get("title", ""), prepared.get("body", "")):
                    continue

                url = prepared.get("url", "")
                document_id = sha256_text(url)
                if self.store.exists("news", document_id):
                    continue

                # Score sentiment at ingestion time (FinBERT B1)
                if self.sentiment_scorer is not None:
                    text_for_scoring = f"{prepared.get('title', '')} {prepared.get('body', '')[:400]}"
                    score = self.sentiment_scorer.score(text_for_scoring)
                    if score is not None:
                        prepared["sentiment_score"] = score

                self.store.add_document(
                    "news",
                    document_id=document_id,
                    text=f"{prepared.get('title', '')} {prepared.get('body', '')}",
                    metadata=prepared,
                )
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429:
                self.logger.warning("Skipping news cycle after HTTP 429 from %s", source_name)
                return
            self.logger.error("HTTP error in news source %s: %s", source_name, exc)
        except requests.ConnectionError as exc:
            self.logger.error("Connection error in news source %s: %s", source_name, exc)
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Unexpected error in news source %s: %s", source_name, exc)

    async def ingest_macro_once(self, source_name: str) -> None:
        source = self.sources[source_name]
        try:
            documents = await source.fetch()
            for document in documents:
                timestamp = str(document.get("timestamp", ""))
                document_id = f"{document.get('source', source_name)}_{timestamp[:13] or 'unknown'}"
                if self.store.exists("macro", document_id):
                    continue

                self.store.add_document(
                    "macro",
                    document_id=document_id,
                    text=document.get("narrative", ""),
                    metadata=document,
                )
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429:
                self.logger.warning("Skipping macro cycle after HTTP 429 from %s", source_name)
                return
            self.logger.error("HTTP error in macro source %s: %s", source_name, exc)
        except requests.Timeout as exc:
            self.logger.warning("Timeout fetching macro source %s, will retry next cycle: %s", source_name, exc)
        except requests.ConnectionError as exc:
            self.logger.warning("Connection error in macro source %s, will retry next cycle: %s", source_name, exc)
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Unexpected error in macro source %s: %s", source_name, exc)

    async def _run_source_forever(self, source_name: str, cadence: int, runner: Any) -> None:
        while True:
            await runner(source_name)
            await self.sleep_func(cadence)
