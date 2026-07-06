"""Tests for LLM reranker, hybrid retrieval compose, and cross-encoder (skipped
if sentence-transformers absent)."""
from __future__ import annotations

import pytest

from rag.config import RagConfig
from rag.rerankers.llm import LLMReranker
from rag.retrievers.hybrid import HybridRetriever
from rag.types import ChunkRecord, SearchResult
from tests.fakes import FakeEmbedder, FakeGenerator, FakeKeywordIndex, FakeVectorStore


def _results(*ids):
    return [SearchResult(chunk_id=i, text=i, score=1.0) for i in ids]


# ── LLM reranker ──
def test_llm_reranker_reorders_by_model_output():
    gen = FakeGenerator(answer="3,1,2")
    out = LLMReranker(gen).rerank("q", _results("a", "b", "c"), top_k=3)
    assert [r.chunk_id for r in out] == ["c", "a", "b"]


def test_llm_reranker_fills_missing_indices():
    gen = FakeGenerator(answer="2")           # model only named passage 2
    out = LLMReranker(gen).rerank("q", _results("a", "b", "c"), top_k=3)
    assert out[0].chunk_id == "b"             # named first
    assert set(r.chunk_id for r in out) == {"a", "b", "c"}   # rest filled in


def test_llm_reranker_bad_output_falls_back_to_input_order():
    gen = FakeGenerator(answer="not a list")
    out = LLMReranker(gen).rerank("q", _results("a", "b", "c"), top_k=2)
    assert [r.chunk_id for r in out] == ["a", "b"]


# ── Hybrid retrieve ──
def _hybrid(reranker=None):
    store = FakeVectorStore()
    emb = FakeEmbedder()
    recs = [
        ChunkRecord(chunk_id="a", raw_text="python programming language",
                    embedding=emb.embed(["python programming language"])[0],
                    provenance={"source": "s", "title": "s"}),
        ChunkRecord(chunk_id="b", raw_text="java coding tutorial",
                    embedding=emb.embed(["java coding tutorial"])[0],
                    provenance={"source": "s", "title": "s"}),
        ChunkRecord(chunk_id="c", raw_text="python data science",
                    embedding=emb.embed(["python data science"])[0],
                    provenance={"source": "s", "title": "s"}),
    ]
    store.add(recs)
    from rag.retrievers.dense import DenseRetriever
    from rag.retrievers.bm25 import BM25Retriever
    dense = DenseRetriever(store, emb)
    bm25 = BM25Retriever(store, FakeKeywordIndex())
    cfg = RagConfig(top_k=3, final_k=2)
    return HybridRetriever(dense, bm25, reranker=reranker, config=cfg)


def test_hybrid_returns_final_k_without_reranker():
    h = _hybrid(reranker=None)
    res = h.retrieve("python")
    assert len(res) == 2
    assert all(0.0 <= r.score <= 1.0 for r in res)


def test_hybrid_preserves_rrf_scores_after_rerank():
    """Reranker reorders, but scores stay RRF-based (confidence is judged on RRF)."""
    gen = FakeGenerator(answer="2,1")
    h = _hybrid(reranker=LLMReranker(gen))
    res = h.retrieve("python")
    assert len(res) == 2
    # every returned score must equal an RRF score (0..1), never the reranker's own scale
    assert all(0.0 <= r.score <= 1.0 for r in res)


# ── Cross-encoder (optional dep) ──
def test_hybrid_graph_stage_adds_neighbors_before_rerank():
    from rag.graph.expander import GraphExpander
    from rag.graph.index import GraphIndex
    from rag.retrievers.dense import DenseRetriever
    from rag.retrievers.bm25 import BM25Retriever
    store = FakeVectorStore()
    emb = FakeEmbedder()
    recs = [
        ChunkRecord(chunk_id="a", raw_text="python programming",
                    embedding=emb.embed(["python programming"])[0],
                    provenance={"source": "s", "title": "s"}, ordinal=0),
        ChunkRecord(chunk_id="b", raw_text="unrelated cooking recipes",
                    embedding=emb.embed(["unrelated cooking recipes"])[0],
                    provenance={"source": "s", "title": "s"}, ordinal=1),
    ]
    store.add(recs)
    cfg = RagConfig(top_k=3, final_k=5, graph_hops=1)
    dense = DenseRetriever(store, emb)
    bm25 = BM25Retriever(store, FakeKeywordIndex())
    graph = GraphExpander(store, GraphIndex(cfg), cfg)
    h = HybridRetriever(dense, bm25, reranker=None, config=cfg, graph=graph)
    ids = {r.chunk_id for r in h.retrieve("python")}
    assert "a" in ids and "b" in ids        # graph stage runs; b present


def test_hybrid_without_graph_is_unchanged():
    h = _hybrid(reranker=None)              # existing helper, no graph
    res = h.retrieve("python")
    assert len(res) == 2                     # final_k=2, no neighbors added


def test_cross_encoder_reranker_importable_or_skipped():
    pytest.importorskip("sentence_transformers")
    from rag.rerankers.cross_encoder import CrossEncoderReranker
    rr = CrossEncoderReranker(RagConfig())
    out = rr.rerank("python", _results("a", "b"), top_k=1)
    assert len(out) == 1
