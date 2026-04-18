"""ChromaDB wrapper for RAG storage."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
from collections.abc import Iterable
from typing import Any


class DeterministicEmbeddingFunction:
    """Small local fallback embedding function for offline and test environments."""

    def __call__(self, input: Iterable[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in input:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = [int.from_bytes(
                digest[index:index + 2], "big") / 65535.0 for index in range(0, 32, 2)]
            vectors.append(vector)
        return vectors


def build_default_embedding_function() -> Any:
    """Use sentence-transformers when installed, otherwise a deterministic fallback."""
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except Exception:  # noqa: BLE001
        return DeterministicEmbeddingFunction()

    try:
        return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    except Exception:  # noqa: BLE001
        return DeterministicEmbeddingFunction()


class ChromaStore:
    """Persistent ChromaDB storage wrapper with fixed collections."""

    COLLECTIONS = ("news", "macro", "trade_memory")

    def __init__(self, path: str, embedding_function: Any | None = None) -> None:
        self.path = Path(path)
        self._validate_path()
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

        import chromadb

        self._client = chromadb.PersistentClient(path=str(self.path))
        self._embedding_function = embedding_function or build_default_embedding_function()
        self._collection_suffix = self._build_collection_suffix()
        self.collections = {
            name: self._client.get_or_create_collection(
                name=self._physical_collection_name(name),
                embedding_function=self._embedding_function,
            )
            for name in self.COLLECTIONS
        }

    def count(self, collection_name: str) -> int:
        return int(self.collections[collection_name].count())

    def exists(self, collection_name: str, document_id: str) -> bool:
        result = self.collections[collection_name].get(
            ids=[document_id], include=[])
        return bool(result.get("ids"))

    def add_document(
        self,
        collection_name: str,
        *,
        document_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        self.collections[collection_name].add(
            ids=[document_id],
            documents=[text],
            metadatas=[metadata],
        )

    def query(self, collection_name: str, query_text: str, n_results: int) -> dict[str, Any]:
        return self.collections[collection_name].query(query_texts=[query_text], n_results=n_results)

    def get_raw_client(self) -> Any:
        """Return the underlying ChromaDB PersistentClient.

        Used by VectorMemoryService to create its own collections directly.
        """
        return self._client

    def _build_collection_suffix(self) -> str:
        sample_vector = self._embedding_function(["dimension probe"])[0]
        dimension = len(sample_vector)
        model_name = getattr(self._embedding_function, "model_name", None)
        base = model_name or self._embedding_function.__class__.__name__
        slug = re.sub(r"[^a-z0-9]+", "_", str(base).lower()
                      ).strip("_") or "default"
        return f"{slug}_{dimension}"

    def _physical_collection_name(self, collection_name: str) -> str:
        return f"{collection_name}_{self._collection_suffix}"

    def _validate_path(self) -> None:
        if self.path.exists() and not self.path.is_dir():
            raise RuntimeError(f"ChromaDB path is not writable: {self.path}")

        try:
            self.path.mkdir(parents=True, exist_ok=True)
            probe = self.path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"ChromaDB path is not writable: {self.path}") from exc
