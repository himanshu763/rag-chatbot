"""In-memory fakes for tests — no API calls, no external services.

Each conforms to its Protocol in rag.interfaces so pipeline logic can be tested
end-to-end without ChromaDB / OpenAI / a reranker model.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterator

from rag.types import ChunkRecord, SearchResult

_WORD = re.compile(r"\w+")


def _vec(text: str, dim: int = 8) -> list[float]:
    """Deterministic pseudo-embedding: hash text into a small fixed vector."""
    h = hashlib.sha256(text.encode()).digest()
    return [h[i % len(h)] / 255.0 for i in range(dim)]


class FakeEmbedder:
    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [_vec(t, self.dim) for t in texts]


class FakeVectorStore:
    """Cosine-ish in-memory store keyed by chunk_id."""

    def __init__(self):
        self._records: dict[str, ChunkRecord] = {}

    def add(self, records: list[ChunkRecord]) -> None:
        for r in records:
            self._records[r.chunk_id] = r

    @staticmethod
    def _cos(a: list[float], b: list[float]) -> float:
        num = sum(x * y for x, y in zip(a, b))
        da = sum(x * x for x in a) ** 0.5
        db = sum(y * y for y in b) ** 0.5
        return num / (da * db) if da and db else 0.0

    def query(self, embedding, top_k, where=None):
        scored = []
        for r in self._records.values():
            if where and not self._match(r, where):
                continue
            score = self._cos(embedding, r.embedding) if r.embedding else 0.0
            scored.append(SearchResult(
                chunk_id=r.chunk_id, text=r.raw_text, score=score,
                metadata=dict(r.provenance, section_title=r.section_path),
                parent_text=r.parent_text, freshness=dict(r.freshness), ordinal=r.ordinal,
            ))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _match(r: ChunkRecord, where: dict) -> bool:
        for key, cond in where.items():
            val = r.provenance.get(key)
            if isinstance(cond, dict) and "$eq" in cond:
                if val != cond["$eq"]:
                    return False
            elif val != cond:
                return False
        return True

    def get_all(self) -> list[ChunkRecord]:
        return list(self._records.values())

    def delete_source(self, source_key: str) -> None:
        self._records = {
            cid: r for cid, r in self._records.items()
            if r.provenance.get("source") != source_key
        }

    def count(self) -> int:
        return len(self._records)

    def list_sources(self) -> list[dict]:
        out: dict[str, dict] = {}
        for r in self._records.values():
            key = r.provenance.get("source", "unknown")
            if key not in out:
                out[key] = {"source": key, "title": r.provenance.get("title", key),
                            "type": r.provenance.get("source_type", "?"), "chunks": 0}
            out[key]["chunks"] += 1
        return list(out.values())

    def get_source_hash(self, source_key: str) -> str | None:
        for r in self._records.values():
            if r.provenance.get("source") == source_key:
                return r.source_hash
        return None

    def get_source_freshness(self, source_key: str) -> dict:
        for r in self._records.values():
            if r.provenance.get("source") == source_key:
                return dict(r.freshness)
        return {}

    def touch_source(self, source_key: str, fetched_at: str) -> None:
        for r in self._records.values():
            if r.provenance.get("source") == source_key:
                r.freshness["source_fetched_at"] = fetched_at


class FakeKeywordIndex:
    def __init__(self):
        self._records: list[ChunkRecord] = []

    def build(self, records: list[ChunkRecord]) -> None:
        self._records = list(records)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        qtokens = set(_WORD.findall(query.lower()))
        scored = []
        for r in self._records:
            toks = _WORD.findall(r.raw_text.lower())
            overlap = sum(1 for t in toks if t in qtokens)
            if overlap:
                scored.append(SearchResult(
                    chunk_id=r.chunk_id, text=r.raw_text, score=float(overlap),
                    metadata=dict(r.provenance, section_title=r.section_path),
                    parent_text=r.parent_text,
                ))
        scored.sort(key=lambda s: s.score, reverse=True)
        mx = scored[0].score if scored else 1.0
        for s in scored:
            s.score /= mx
        return scored[:top_k]


class FakeGenerator:
    """Echoes a canned answer; records the system prompt + messages it saw."""

    def __init__(self, answer: str = "canned answer"):
        self.answer = answer
        self.seen_system: list[str] = []
        self.seen_messages: list[list[dict]] = []

    def generate(self, messages: list[dict], system: str = "", max_tokens: int | None = None) -> str:
        self.seen_system.append(system)
        self.seen_messages.append(messages)
        return self.answer

    def stream(self, messages: list[dict], system: str = "", max_tokens: int | None = None) -> Iterator[str]:
        self.seen_system.append(system)
        self.seen_messages.append(messages)
        for tok in self.answer.split():
            yield tok + " "
