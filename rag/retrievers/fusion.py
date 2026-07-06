"""Reciprocal Rank Fusion — merge ranked result lists into one, deduped + normalized."""
from __future__ import annotations

from rag.types import SearchResult


def rrf_fuse(lists: list[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    scores: dict[str, float] = {}
    best: dict[str, SearchResult] = {}
    for lst in lists:
        for rank, r in enumerate(lst):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            if r.chunk_id not in best or r.score > best[r.chunk_id].score:
                best[r.chunk_id] = r
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    max_possible = len(lists) / (k + 1)     # max fused score = n_lists / (k+1)
    merged = []
    for cid in ordered:
        r = best[cid]
        r.score = scores[cid] / max_possible if max_possible > 0 else scores[cid]
        merged.append(r)
    return merged
