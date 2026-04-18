"""Filtering helpers for RAG ingestion."""

from __future__ import annotations

import hashlib


KEYWORDS = ("bitcoin", "btc", "ethereum", "eth", "crypto")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def truncate_body(body: str, limit: int = 1000) -> str:
    return body[:limit]


def is_relevant_article(title: str, body: str) -> bool:
    haystack = f"{title} {body}".lower()
    return any(keyword in haystack for keyword in KEYWORDS)


def prepare_news_document(document: dict[str, str]) -> dict[str, str]:
    prepared = dict(document)
    prepared["body"] = truncate_body(prepared.get("body", ""))
    return prepared
