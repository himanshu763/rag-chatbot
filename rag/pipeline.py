"""RagPipeline — the public facade. Composes every module and exposes a small API.

Build with just a config (defaults constructed from it), or inject any component
(store / embedder / generator / reranker / retriever) to swap implementations or
test without external services. No global state — construct as many as you like.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from rag.cache import InMemoryCacheStore, SemanticCache
from rag.config import RagConfig
from rag.entities import KeywordEntityExtractor
from rag.freshness import is_stale
from rag.ingest import RagIngestor
from rag.interfaces import Embedder, Generator, Reranker, Retriever, VectorStore
from rag.keyword.bm25 import BM25KeywordIndex
from rag.orchestrator import Orchestrator
from rag.retrievers.bm25 import BM25Retriever
from rag.retrievers.dense import DenseRetriever
from rag.retrievers.hybrid import HybridRetriever
from rag.types import OrchestratorResult

logger = logging.getLogger(__name__)


class RagPipeline:
    def __init__(
        self,
        config: RagConfig | None = None,
        *,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
        reranker: Reranker | None = "default",
        retriever: Retriever | None = None,
        keyword_index=None,
        entity_extractor=None,
        cache=None,
    ):
        self.config = config or RagConfig.from_env()
        self._ingest_epoch = 0

        self.store = store or self._default_store()
        self.embedder = embedder or self._default_embedder()
        self.generator = generator or self._default_generator()
        self.keyword_index = keyword_index or BM25KeywordIndex()
        self.entity_extractor = entity_extractor or KeywordEntityExtractor()

        if reranker == "default":
            reranker = self._default_reranker()
        self.reranker = reranker

        if retriever is None:
            dense = DenseRetriever(self.store, self.embedder)
            bm25 = BM25Retriever(self.store, self.keyword_index)
            graph = None
            if self.config.graph_enabled:
                from rag.graph.expander import GraphExpander
                from rag.graph.index import GraphIndex
                graph = GraphExpander(self.store, GraphIndex(self.config), self.config)
            retriever = HybridRetriever(dense, bm25, reranker=self.reranker,
                                        config=self.config, graph=graph)
        self.retriever = retriever

        self.ingestor = RagIngestor(self.store, self.embedder, self.config,
                                    entity_extractor=self.entity_extractor)
        self.orchestrator = Orchestrator(self.retriever, self.generator, self.config)

        if cache is not None:
            self.cache = cache
        elif self.config.cache_enabled:
            self.cache = SemanticCache(
                InMemoryCacheStore(self.config.cache_max_entries), self.embedder, self.config)
        else:
            self.cache = None

    # ── default component builders (lazy — only when not injected) ──
    def _default_store(self):
        from rag.stores.chroma import ChromaVectorStore
        return ChromaVectorStore(self.config)

    def _default_embedder(self):
        from rag.embedders.openai import OpenAIEmbedder
        return OpenAIEmbedder(self.config)

    def _default_generator(self):
        from rag.generators.openai import OpenAIGenerator
        return OpenAIGenerator(self.config)

    def _default_reranker(self):
        if self.config.reranker_type == "crossencoder":
            from rag.rerankers.cross_encoder import CrossEncoderReranker
            return CrossEncoderReranker(self.config)
        from rag.rerankers.llm import LLMReranker
        return LLMReranker(self.generator)

    # ── ingestion ──
    def ingest_url(self, url: str) -> dict:
        return self._bump(self.ingestor.ingest_url(url))

    def ingest_file(self, path: str) -> dict:
        return self._bump(self.ingestor.ingest_file(path))

    def ingest_text(self, text: str, title: str = "Pasted Text") -> dict:
        return self._bump(self.ingestor.ingest_text(text, title))

    def list_sources(self) -> list[dict]:
        return self.ingestor.list_sources()

    def total_chunks(self) -> int:
        return self.ingestor.total_chunks()

    def delete_source(self, source_key: str) -> None:
        self.ingestor.delete_source(source_key)
        self._ingest_epoch += 1

    def _bump(self, result: dict) -> dict:
        if result.get("status") == "ok":     # content actually changed
            self._ingest_epoch += 1
        return result

    def _corpus_version(self) -> str:
        return f"{self.store.count()}:{self._ingest_epoch}"

    # ── query ──
    def query(self, text: str, history: list[dict] | None = None):
        if self.cache is None:
            return self.orchestrator.process(text, history)
        version = self._corpus_version()
        hit = self.cache.lookup(text, version, history or [])
        if hit is not None:
            return hit
        result = self.orchestrator.process(text, history)
        self.cache.store_result(text, version, history or [], result)
        return result

    def stream(self, text: str, history: list[dict] | None = None):
        if self.cache is None:
            yield from self.orchestrator.process_stream(text, history)
            return
        version = self._corpus_version()
        hit = self.cache.lookup(text, version, history or [])
        if hit is not None:
            yield hit.response
            yield hit
            return
        final = None
        for chunk in self.orchestrator.process_stream(text, history):
            if isinstance(chunk, OrchestratorResult):
                final = chunk
            yield chunk
        if final is not None:
            self.cache.store_result(text, version, history or [], final)

    # ── freshness ──
    def refresh_source(self, source_key: str) -> dict:
        # route through _bump so a content-changing refresh invalidates the cache
        # (epoch bumps on "ok"; unchanged "skipped" leaves cached answers valid)
        return self._bump(self.ingestor.refresh_source(source_key))

    def sweep_stale(self, now=None, dry_run: bool = False) -> dict:
        now = now or datetime.now(timezone.utc)
        report = {"checked": 0, "stale": [], "refreshed": [], "skipped": []}
        for src in self.list_sources():
            key = src["source"]
            report["checked"] += 1
            fresh = self.store.get_source_freshness(key)
            if not is_stale(fresh, self.config, now):
                continue
            report["stale"].append(key)
            if dry_run:
                continue
            result = self.refresh_source(key)
            if result.get("status") in ("ok", "skipped"):
                report["refreshed"].append(key)
            else:
                report["skipped"].append(key)
        return report

    # ── warm-up (build BM25 index + load reranker model) ──
    def warmup(self) -> dict:
        status: dict[str, str] = {}
        try:
            if isinstance(self.retriever, HybridRetriever):
                self.retriever.bm25.retrieve("warmup", top_k=1)   # forces index build
            status["bm25"] = "ready"
        except Exception:
            logger.warning("BM25 warm-up failed", exc_info=True)
            status["bm25"] = "failed"
        try:
            if self.reranker is not None and hasattr(self.reranker, "_get_model"):
                self.reranker._get_model()
            status["reranker"] = "ready"
        except Exception:
            logger.warning("Reranker warm-up failed", exc_info=True)
            status["reranker"] = "failed"
        return status
