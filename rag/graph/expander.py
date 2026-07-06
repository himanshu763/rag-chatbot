"""GraphExpander — expands a result set with graph neighbors before rerank.

Owns the store + GraphIndex, rebuilding the index when the corpus size changes
(same lazy-sync as BM25Retriever). Neighbors enter as SearchResults with a dampened
score so the reranker, not the raw score, decides their final rank.
"""
from __future__ import annotations

from rag.config import RagConfig
from rag.graph.index import GraphIndex
from rag.interfaces import VectorStore
from rag.types import SearchResult


class GraphExpander:
    def __init__(self, store: VectorStore, index: GraphIndex, config: RagConfig | None = None):
        self.store = store
        self.index = index
        self.config = config or RagConfig()
        self._built_count = -1

    def _ensure_current(self) -> None:
        count = self.store.count()
        if count != self._built_count:
            self.index.build(self.store.get_all())
            self._built_count = count

    def expand(self, results: list[SearchResult]) -> list[SearchResult]:
        self._ensure_current()
        seed_ids = {r.chunk_id for r in results}
        neighbor_ids = self.index.expand(seed_ids, hops=self.config.graph_hops)
        out = list(results)
        for nid in neighbor_ids:
            rec = self.index.get_record(nid)
            if rec is None:
                continue
            out.append(SearchResult(
                chunk_id=rec.chunk_id,
                text=rec.raw_text,
                score=self.config.graph_neighbor_score,
                metadata=dict(rec.provenance, section_title=rec.section_path),
                parent_text=rec.parent_text,
                freshness=dict(rec.freshness),
                ordinal=rec.ordinal,
                graph_neighbor=True,
            ))
        return out
