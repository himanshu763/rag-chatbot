"""End-to-end with the REAL ChromaVectorStore (fake embedder/generator, no API keys).

Proves the store persists + retrieves through the full pipeline, and that two
pipelines in one process are isolated — impossible under the old global collection.
"""
from __future__ import annotations

import pytest

pytest.importorskip("chromadb")

from rag.config import RagConfig
from rag.pipeline import RagPipeline
from rag.stores.chroma import ChromaVectorStore
from rag.types import OrchestratorResult
from tests.fakes import FakeEmbedder, FakeGenerator


def _pipeline(tmp_path, name, answer="grounded"):
    cfg = RagConfig(chroma_dir=str(tmp_path / name), collection_name=name,
                    similarity_threshold=0.0, low_confidence_threshold=0.0,
                    min_chunk_size=1, top_k=5, final_k=3)
    return RagPipeline(
        config=cfg,
        store=ChromaVectorStore(cfg),
        embedder=FakeEmbedder(),
        generator=FakeGenerator(answer=answer),
        reranker=None,
    )


def test_real_chroma_ingest_query_roundtrip(tmp_path):
    p = _pipeline(tmp_path, "kb_main", answer="the answer")
    r = p.ingest_text("Alpha beta gamma. Delta epsilon zeta.", title="Doc")
    assert r["status"] == "ok"
    assert p.total_chunks() == r["chunks"]
    res = p.query("alpha beta")
    assert isinstance(res, OrchestratorResult)
    assert res.response == "the answer"
    assert res.sources


def test_real_chroma_idempotent_skip(tmp_path):
    p = _pipeline(tmp_path, "kb_skip")
    p.ingest_text("Same content. More text here.", title="Doc")
    n = p.total_chunks()
    r2 = p.ingest_text("Same content. More text here.", title="Doc")
    assert r2["status"] == "skipped"
    assert p.total_chunks() == n


def test_two_real_pipelines_isolated(tmp_path):
    a = _pipeline(tmp_path, "tenant_a")
    b = _pipeline(tmp_path, "tenant_b")
    a.ingest_text("Alpha beta gamma. Delta.", title="A")
    assert a.total_chunks() >= 1
    assert b.total_chunks() == 0
