"""Response cache — InMemoryCacheStore backend + SemanticCache (exact + semantic match).

Opt-in via config.cache_enabled. Correctness comes from a corpus-version baked into the
cache key by the pipeline; a TTL is the backstop. No external dependencies.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict

from rag.config import RagConfig
from rag.interfaces import Embedder

logger = logging.getLogger(__name__)


class InMemoryCacheStore:
    """key -> (value, expires_at). Insertion-ordered; oldest evicted past max_entries."""

    def __init__(self, max_entries: int = 1000, clock=time.monotonic):
        self.max_entries = max_entries
        self._clock = clock
        self._data: "OrderedDict[str, tuple[object, float | None]]" = OrderedDict()

    def _expired(self, expires_at: float | None) -> bool:
        return expires_at is not None and self._clock() >= expires_at

    def get(self, key: str) -> object | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if self._expired(expires_at):
            self._data.pop(key, None)
            return None
        return value

    def set(self, key: str, value: object, ttl_seconds: float | None = None) -> None:
        expires_at = (self._clock() + ttl_seconds) if ttl_seconds is not None else None
        self._data[key] = (value, expires_at)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)        # evict oldest-inserted

    def items(self) -> list:
        out = []
        for key, (value, expires_at) in list(self._data.items()):
            if self._expired(expires_at):
                self._data.pop(key, None)
                continue
            out.append((key, value))
        return out

    def clear(self) -> None:
        self._data.clear()


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


class SemanticCache:
    def __init__(self, store, embedder: Embedder, config: RagConfig):
        self.store = store
        self.embedder = embedder
        self.config = config

    @staticmethod
    def _history_fp(history: list[dict]) -> str:
        joined = "\n".join(f"{h.get('role')}:{h.get('content')}" for h in (history or []))
        return hashlib.sha256(joined.encode()).hexdigest()[:16]

    def _exact_key(self, query: str, corpus_version: str, history_fp: str) -> str:
        raw = f"{query}\x00{corpus_version}\x00{history_fp}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def lookup(self, query: str, corpus_version: str, history: list[dict]):
        hfp = self._history_fp(history)
        exact = self.store.get(self._exact_key(query, corpus_version, hfp))
        if exact is not None:
            return exact["result"]
        try:
            qvec = self.embedder.embed([query])[0]
        except Exception:
            logger.warning("Cache semantic lookup embed failed — exact-only", exc_info=True)
            return None
        best, best_sim = None, self.config.cache_semantic_threshold
        for _key, entry in self.store.items():
            if entry["corpus_version"] != corpus_version or entry["history_fp"] != hfp:
                continue
            if not entry["embedding"]:
                continue
            sim = _cosine(qvec, entry["embedding"])
            if sim >= best_sim:
                best, best_sim = entry, sim
        return best["result"] if best is not None else None

    def store_result(self, query: str, corpus_version: str, history: list[dict], result) -> None:
        hfp = self._history_fp(history)
        try:
            emb = self.embedder.embed([query])[0]
        except Exception:
            logger.warning("Cache store embed failed — storing exact-only", exc_info=True)
            emb = []
        entry = {"result": result, "embedding": emb,
                 "corpus_version": corpus_version, "history_fp": hfp}
        self.store.set(self._exact_key(query, corpus_version, hfp), entry,
                       ttl_seconds=self.config.cache_ttl_seconds)
