"""BM25Retriever — keyword retrieval that stays in sync with its store.

Lazily rebuilds the underlying BM25 index when the store's document count changes.
Instance-scoped (holds its own index + count), so no global cache and multiple
retrievers over different stores never cross-talk.
"""
from __future__ import annotations

from rag.interfaces import KeywordIndex, VectorStore
from rag.types import SearchResult


class BM25Retriever:
    def __init__(self, store: VectorStore, index: KeywordIndex):
        self.store = store
        self.index = index
        self._built_count = -1

    def _ensure_current(self) -> None:
        count = self.store.count()
        if count != self._built_count:
            self.index.build(self.store.get_all())
            self._built_count = count

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        self._ensure_current()
        return self.index.search(query, top_k)
