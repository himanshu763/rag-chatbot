"""RagPipeline facade — end-to-end wiring (ingest → retrieve → generate) with
injected fakes, plus multi-instance isolation (proves no global state)."""
from __future__ import annotations

from rag.config import RagConfig
from rag.pipeline import RagPipeline
from rag.types import OrchestratorResult
from tests.fakes import FakeEmbedder, FakeGenerator, FakeVectorStore


def _pipeline(answer="grounded answer", **cfg_kw):
    cfg = RagConfig(similarity_threshold=0.0, low_confidence_threshold=0.0,
                    top_k=5, final_k=3, min_chunk_size=1, **cfg_kw)
    return RagPipeline(
        config=cfg,
        store=FakeVectorStore(),
        embedder=FakeEmbedder(),
        generator=FakeGenerator(answer=answer),
        reranker=None,
    )


def test_ingest_then_query_end_to_end():
    p = _pipeline(answer="the grounded answer")
    r = p.ingest_text("Alpha beta gamma. Delta epsilon zeta.", title="Doc")
    assert r["status"] == "ok"
    res = p.query("alpha beta")
    assert isinstance(res, OrchestratorResult)
    assert res.response == "the grounded answer"
    assert res.sources


def test_stream_end_to_end_yields_text_then_result():
    p = _pipeline(answer="hello world")
    p.ingest_text("Alpha beta gamma. Delta epsilon.", title="Doc")
    items = list(p.stream("alpha"))
    assert isinstance(items[-1], OrchestratorResult)
    assert "hello" in "".join(i for i in items[:-1] if isinstance(i, str))


def test_total_chunks_and_list_sources():
    p = _pipeline()
    p.ingest_text("Alpha beta. Gamma delta.", title="Doc A")
    assert p.total_chunks() >= 1
    assert any(s["title"] == "Doc A" for s in p.list_sources())


def test_warmup_reports_ready():
    p = _pipeline()
    p.ingest_text("Alpha beta gamma.", title="Doc")
    status = p.warmup()
    assert status["bm25"] == "ready"
    assert status["reranker"] == "ready"      # None reranker → nothing to load


def test_graph_enabled_pipeline_end_to_end():
    cfg = RagConfig(similarity_threshold=0.0, low_confidence_threshold=0.0,
                    top_k=5, final_k=5, min_chunk_size=1, graph_enabled=True)
    p = RagPipeline(config=cfg, store=FakeVectorStore(), embedder=FakeEmbedder(),
                    generator=FakeGenerator(answer="ok"), reranker=None)
    assert p.retriever.graph is not None          # expander wired when graph_enabled
    r = p.ingest_text("The Zephyr X1 is strong. Zephyr X1 spins fast. Zephyr X1 lasts long.", title="D")
    assert r["status"] == "ok"
    res = p.query("Zephyr")
    assert res.response == "ok"
    assert res.sources


def test_graph_disabled_pipeline_has_no_expander():
    p = _pipeline()                          # existing helper, graph off
    assert getattr(p.retriever, "graph", None) is None


def _cache_pipeline(answer="cached-answer"):
    cfg = RagConfig(similarity_threshold=0.0, low_confidence_threshold=0.0,
                    top_k=5, final_k=5, min_chunk_size=1, cache_enabled=True)
    return RagPipeline(config=cfg, store=FakeVectorStore(), embedder=FakeEmbedder(),
                       generator=FakeGenerator(answer=answer), reranker=None)


def test_cache_hit_skips_second_generation():
    p = _cache_pipeline()
    p.ingest_text("Alpha beta gamma. Delta epsilon.", title="D")
    gen = p.generator
    p.query("what is alpha?")
    calls_after_first = len(gen.seen_messages)
    p.query("what is alpha?")                       # identical → cache hit
    assert len(gen.seen_messages) == calls_after_first   # generator NOT called again


def test_cache_invalidated_after_ingest():
    p = _cache_pipeline()
    p.ingest_text("Alpha beta gamma. Delta epsilon.", title="D")
    gen = p.generator
    p.query("what is alpha?")
    n = len(gen.seen_messages)
    p.ingest_text("New content entirely here. Another sentence.", title="D2")   # bumps version
    p.query("what is alpha?")                       # corpus changed → miss → regenerates
    assert len(gen.seen_messages) > n


def test_cache_invalidated_after_refresh_same_chunk_count(tmp_path):
    # content changes but chunk count stays 1 → version must still flip (epoch, not count)
    cfg = RagConfig(cache_enabled=True, min_chunk_size=1,
                    similarity_threshold=0.0, low_confidence_threshold=0.0)
    p = RagPipeline(config=cfg, store=FakeVectorStore(), embedder=FakeEmbedder(),
                    generator=FakeGenerator(answer="x"), reranker=None)
    f = tmp_path / "n.txt"
    f.write_text("Original one sentence.", encoding="utf-8")
    p.ingest_file(str(f))
    p.query("q")
    n = len(p.generator.seen_messages)
    f.write_text("Totally different one sentence.", encoding="utf-8")   # still 1 chunk
    p.refresh_source(str(f))
    p.query("q")
    assert len(p.generator.seen_messages) > n          # must regenerate, not serve stale


def test_cache_disabled_pipeline_has_no_cache():
    p = _pipeline()
    assert getattr(p, "cache", None) is None


def test_cache_stream_hit_replays_text_and_result():
    p = _cache_pipeline(answer="streamy answer")
    p.ingest_text("Alpha beta gamma. Delta epsilon.", title="D")
    list(p.stream("what is alpha?"))                # populate cache
    items = list(p.stream("what is alpha?"))        # hit
    assert isinstance(items[-1], OrchestratorResult)
    text = "".join(i for i in items[:-1] if isinstance(i, str))
    assert "streamy answer" in text


def test_corpus_version_changes_on_ingest():
    p = _pipeline()                      # cache off is fine here
    v1 = p._corpus_version()
    p.ingest_text("Alpha beta gamma. Delta.", title="D")
    v2 = p._corpus_version()
    assert v1 != v2                      # ingest bumped version


def test_corpus_version_stable_without_ingest():
    p = _pipeline()
    assert p._corpus_version() == p._corpus_version()


def test_two_pipelines_are_isolated():
    """Independent stores → independent corpora in one process (no global collection)."""
    a = _pipeline()
    b = _pipeline()
    a.ingest_text("Alpha beta gamma.", title="A")
    assert a.total_chunks() >= 1
    assert b.total_chunks() == 0
