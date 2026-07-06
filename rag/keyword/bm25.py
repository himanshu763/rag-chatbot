"""BM25KeywordIndex — instance-scoped keyword search over ChunkRecords.

No module-level cache (the old `_bm25_cache` global is gone). Each index owns its
own corpus + BM25 model; callers rebuild via .build(records).
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from rag.types import ChunkRecord, SearchResult

_WORD = re.compile(r"\w+")


class BM25KeywordIndex:
    def __init__(self):
        self._corpus: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def build(self, records: list[ChunkRecord]) -> None:
        corpus, tokenized = [], []
        for r in records:
            corpus.append({
                "chunk_id": r.chunk_id,
                "text": r.raw_text,
                "metadata": dict(r.provenance, section_title=r.section_path),
                "parent_text": r.parent_text,
            })
            tokenized.append(_WORD.findall(r.raw_text.lower()))
        self._corpus = corpus
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self._corpus or self._bm25 is None:
            return []
        qtokens = _WORD.findall(query.lower())
        scores = self._bm25.get_scores(qtokens)
        mx = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            d = self._corpus[idx]
            results.append(SearchResult(
                chunk_id=d["chunk_id"], text=d["text"], score=scores[idx] / mx,
                metadata=d["metadata"], parent_text=d.get("parent_text"),
            ))
        return results
