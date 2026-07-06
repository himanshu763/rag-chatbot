"""Tests for BM25 keyword index, RRF fusion, and dense/BM25 retrievers.

Focus: instance-scoped BM25 (no module global) that stays in sync with its store,
and correct fusion/normalization behavior.
"""
from __future__ import annotations

import pytest

pytest.importorskip("rank_bm25")

from rag.keyword.bm25 import BM25KeywordIndex
from rag.retrievers.bm25 import BM25Retriever
from rag.retrievers.dense import DenseRetriever
from rag.retrievers.fusion import rrf_fuse
from rag.types import ChunkRecord, SearchResult
from tests.fakes import FakeEmbedder, FakeKeywordIndex, FakeVectorStore


def _rec(cid, text, source="src1", emb=None):
    return ChunkRecord(chunk_id=cid, raw_text=text, embedding=emb,
                       provenance={"source": source, "title": source})


# ── BM25 index ──
def test_bm25_finds_exact_term():
    idx = BM25KeywordIndex()
    idx.build([_rec("a", "the quick brown fox"),
               _rec("b", "lazy dog sleeps"),
               _rec("c", "error code 502 gateway")])
    res = idx.search("502 gateway", top_k=3)
    assert res[0].chunk_id == "c"


def test_bm25_scores_normalized_0_1():
    # >=3 docs with the query term in a minority → positive IDF (avoids BM25's
    # degenerate zero/negative IDF when a term appears in half-or-more of the corpus)
    idx = BM25KeywordIndex()
    idx.build([_rec("a", "alpha beta"), _rec("b", "gamma delta"), _rec("c", "epsilon zeta")])
    res = idx.search("alpha", top_k=3)
    assert res and res[0].score == pytest.approx(1.0)
    assert all(0.0 <= r.score <= 1.0 for r in res)


def test_bm25_empty_index_returns_empty():
    assert BM25KeywordIndex().search("anything", top_k=5) == []


def test_bm25_no_match_returns_empty():
    idx = BM25KeywordIndex()
    idx.build([_rec("a", "alpha beta")])
    assert idx.search("zzz", top_k=5) == []


# ── RRF fusion ──
def test_rrf_merges_and_dedupes():
    l1 = [SearchResult("a", "a", 0.9), SearchResult("b", "b", 0.8)]
    l2 = [SearchResult("b", "b", 0.7), SearchResult("c", "c", 0.6)]
    fused = rrf_fuse([l1, l2], k=60)
    ids = [r.chunk_id for r in fused]
    assert set(ids) == {"a", "b", "c"}
    assert ids.count("b") == 1          # deduped
    assert ids[0] == "b"                # appears in both lists → highest fused score


def test_rrf_scores_normalized_0_1():
    l1 = [SearchResult("a", "a", 0.9)]
    l2 = [SearchResult("a", "a", 0.8)]
    fused = rrf_fuse([l1, l2], k=60)
    assert 0.0 <= fused[0].score <= 1.0


# ── Dense retriever ──
def test_dense_retriever_embeds_query_and_queries_store():
    store = FakeVectorStore()
    emb = FakeEmbedder()
    docs = ["hello world", "goodbye moon"]
    store.add([_rec("a", docs[0], emb=emb.embed([docs[0]])[0]),
               _rec("b", docs[1], emb=emb.embed([docs[1]])[0])])
    r = DenseRetriever(store, emb)
    res = r.retrieve("hello world", top_k=2)
    assert res[0].chunk_id == "a"


# ── BM25 retriever (lazy sync with store, no global) ──
# These exercise the retriever's rebuild-on-count-change sync, not BM25 scoring
# (BM25 math is covered above), so they use a simple overlap-scoring keyword index.
def test_bm25_retriever_rebuilds_when_store_changes():
    store = FakeVectorStore()
    store.add([_rec("a", "python tutorial")])
    r = BM25Retriever(store, FakeKeywordIndex())
    assert [x.chunk_id for x in r.retrieve("python", top_k=5)] == ["a"]
    # add another doc → retriever must pick it up without a manual rebuild
    store.add([_rec("b", "python scripting")])
    ids = {x.chunk_id for x in r.retrieve("python", top_k=5)}
    assert ids == {"a", "b"}


def test_two_bm25_retrievers_are_independent():
    """No shared module cache: two retrievers over different stores don't cross-talk."""
    s1, s2 = FakeVectorStore(), FakeVectorStore()
    s1.add([_rec("a", "python tutorial")])
    s2.add([_rec("b", "python scripting")])
    r1 = BM25Retriever(s1, FakeKeywordIndex())
    r2 = BM25Retriever(s2, FakeKeywordIndex())
    assert [x.chunk_id for x in r1.retrieve("python", 5)] == ["a"]
    assert [x.chunk_id for x in r2.retrieve("python", 5)] == ["b"]
