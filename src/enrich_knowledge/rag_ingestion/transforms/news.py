"""Pure transforms for the ``news`` Chroma collection.

Handles three concerns, all CPU-only:

1. Relevance filter — drop articles that mention no crypto keywords.
2. Body truncation — cap body length so single documents can't
   dominate retrieval.
3. Natural-key id — SHA-256 of the URL, so re-fetching the same
   article is a no-op on the writer side.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.enrich_knowledge.rag_ingestion.sources.cryptocompare import NewsArticle

_KEYWORDS = ("bitcoin", "btc", "ethereum", "eth", "crypto")
_BODY_LIMIT = 1000


@dataclass(frozen=True)
class NewsRecord:
    """Write-ready record for the Chroma ``news`` collection."""

    document_id: str
    text: str
    metadata: dict[str, str]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def is_relevant(title: str, body: str) -> bool:
    """True when any crypto keyword appears in title or body."""
    haystack = f"{title} {body}".lower()
    return any(keyword in haystack for keyword in _KEYWORDS)


def news_record(article: NewsArticle) -> NewsRecord | None:
    """Turn one article into a write-ready record.

    Returns ``None`` when the article is irrelevant — callers filter
    those out before the writer runs. The body is truncated to keep
    Chroma embeddings manageable; any downstream scoring (sentiment,
    density) works off the truncated text.
    """
    if not article.url:
        return None
    if not is_relevant(article.title, article.body):
        return None

    body = article.body[:_BODY_LIMIT]
    return NewsRecord(
        document_id=_sha256(article.url),
        text=f"{article.title} {body}",
        metadata={
            "title": article.title,
            "url": article.url,
            "body": body,
            "source": article.source,
            "published_at": article.published_at,
            "symbol_tags": article.symbol_tags,
        },
    )


def news_records(articles: list[NewsArticle]) -> list[NewsRecord]:
    """Batch helper — filters irrelevants and returns only keepers."""
    records: list[NewsRecord] = []
    for article in articles:
        record = news_record(article)
        if record is not None:
            records.append(record)
    return records
