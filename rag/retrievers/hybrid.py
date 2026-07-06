"""HybridRetriever — dense + BM25 → RRF fusion → optional rerank.

Composes two retrievers and a reranker. After reranking, RRF scores are restored
(the cross-encoder scores structured/tabular data poorly, so confidence is judged
on RRF scores, not the reranker's).
"""
from __future__ import annotations

import logging

from rag.config import RagConfig
from rag.interfaces import Reranker, Retriever
from rag.retrievers.fusion import rrf_fuse
from rag.types import SearchResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, dense: Retriever, bm25: Retriever,
                 reranker: Reranker | None = None, config: RagConfig | None = None,
                 graph=None):
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker
        self.config = config or RagConfig()
        self.graph = graph

    def retrieve(self, query: str, top_k: int | None = None,
                 final_k: int | None = None, use_reranker: bool = True) -> list[SearchResult]:
        k = top_k or self.config.top_k
        fk = final_k or self.config.final_k

        vec = self.dense.retrieve(query, k)
        kw = self.bm25.retrieve(query, k)
        fused = rrf_fuse([vec, kw], k=self.config.rrf_k)

        # Graph neighbors are appended (dampened score). They only survive into the final set
        # when a reranker promotes them; without a reranker, fused[:fk] below truncates them
        # if base retrieval already fills final_k.
        if self.graph is not None:
            fused = self.graph.expand(fused)

        if use_reranker and self.reranker and len(fused) > fk:
            rrf_scores = {r.chunk_id: r.score for r in fused}
            reranked = self.reranker.rerank(query, fused, top_k=fk)
            for r in reranked:
                r.score = rrf_scores.get(r.chunk_id, r.score)
            return reranked
        return fused[:fk]
