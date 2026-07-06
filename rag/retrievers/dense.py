"""DenseRetriever — embed the query, vector-search the store."""
from __future__ import annotations

from rag.interfaces import Embedder, VectorStore
from rag.types import SearchResult


class DenseRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int, where: dict | None = None) -> list[SearchResult]:
        qvec = self.embedder.embed([query])[0]
        return self.store.query(qvec, top_k=top_k, where=where)
