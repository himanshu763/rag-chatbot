"""Orchestrator tests — confidence ladder parity (every branch), context assembly,
NONE early-return (generation skipped), and the streaming contract app.py depends on."""
from __future__ import annotations

from rag.config import RagConfig
from rag.orchestrator import Orchestrator
from rag.types import Confidence, OrchestratorResult, SearchResult
from tests.fakes import FakeGenerator


class FakeRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query, top_k=None, **kw):
        return list(self._results)


def _sr(cid, score, text="body", parent=None, **meta):
    m = {"source": meta.get("source", cid), "title": meta.get("title", cid),
         "section_title": meta.get("section", "")}
    return SearchResult(chunk_id=cid, text=text, score=score, metadata=m, parent_text=parent)


CFG = RagConfig()  # similarity_threshold=0.25, low_confidence_threshold=0.40


# ── confidence ladder (all branches, thresholds from config) ──
def test_confidence_none_below_similarity_threshold():
    assert Orchestrator._assess_confidence([_sr("a", 0.20)], CFG) is Confidence.NONE


def test_confidence_none_when_empty():
    assert Orchestrator._assess_confidence([], CFG) is Confidence.NONE


def test_confidence_low_between_thresholds():
    assert Orchestrator._assess_confidence([_sr("a", 0.30)], CFG) is Confidence.LOW


def test_confidence_high_when_top_and_avg_above():
    res = [_sr("a", 0.9), _sr("b", 0.8)]      # avg 0.85 > 0.40
    assert Orchestrator._assess_confidence(res, CFG) is Confidence.HIGH


def test_confidence_medium_when_top_high_but_avg_low():
    res = [_sr("a", 0.9), _sr("b", 0.1), _sr("c", 0.1)]   # top≥0.40 but avg≈0.37 < 0.40
    assert Orchestrator._assess_confidence(res, CFG) is Confidence.MEDIUM


def _neighbor(cid, score):
    r = _sr(cid, score)
    r.graph_neighbor = True
    return r


def test_confidence_ignores_graph_neighbor_at_top():
    # a dampened neighbor reranked to position 0 must not suppress a strong seed's confidence
    res = [_neighbor("n", 0.15), _sr("a", 0.9), _sr("b", 0.8)]
    assert Orchestrator._assess_confidence(res, CFG) is Confidence.HIGH


def test_confidence_avg_excludes_neighbors():
    # neighbors at flat 0.15 must not drag the seed average below the HIGH threshold
    res = [_sr("a", 0.9), _sr("b", 0.8), _neighbor("n1", 0.15), _neighbor("n2", 0.15)]
    assert Orchestrator._assess_confidence(res, CFG) is Confidence.HIGH


# ── context assembly ──
def test_context_uses_parent_text_when_present():
    ctx = Orchestrator._assemble_context([_sr("a", 0.9, text="child", parent="PARENT BODY")], CFG)
    assert "PARENT BODY" in ctx
    assert "Source 1" in ctx


# ── NONE → generation skipped ──
def test_process_none_returns_fallback_without_generating():
    gen = FakeGenerator(answer="SHOULD NOT APPEAR")
    orch = Orchestrator(FakeRetriever([_sr("a", 0.1)]), gen, CFG)
    res = orch.process("what is x?")
    assert res.fallback is True
    assert res.confidence is Confidence.NONE
    assert "don't have enough information" in res.response
    assert res.response != gen.answer            # generation was skipped


# ── HIGH → grounded answer generated ──
def test_process_high_confidence_generates_answer():
    gen = FakeGenerator(answer="grounded answer")
    orch = Orchestrator(FakeRetriever([_sr("a", 0.9), _sr("b", 0.8)]), gen, CFG)
    res = orch.process("what is x?")
    assert res.response == "grounded answer"
    assert res.confidence is Confidence.HIGH
    assert res.sources and res.sources[0]["source"] == "a"


# ── greeting short-circuit (no retrieval, no generation) ──
def test_process_greeting_short_circuits():
    gen = FakeGenerator(answer="X")
    orch = Orchestrator(FakeRetriever([]), gen, CFG)
    res = orch.process("hello")
    assert res.intent == "greeting"
    assert res.confidence is Confidence.HIGH


# ── streaming contract: str chunks then a final OrchestratorResult ──
def test_process_stream_yields_text_then_result():
    gen = FakeGenerator(answer="hello world")
    orch = Orchestrator(FakeRetriever([_sr("a", 0.9), _sr("b", 0.8)]), gen, CFG)
    items = list(orch.process_stream("what is x?"))
    assert isinstance(items[-1], OrchestratorResult)
    text = "".join(i for i in items[:-1] if isinstance(i, str))
    assert "hello" in text
    assert items[-1].confidence is Confidence.HIGH


def test_process_stream_none_yields_fallback_then_result():
    gen = FakeGenerator(answer="NOPE")
    orch = Orchestrator(FakeRetriever([_sr("a", 0.1)]), gen, CFG)
    items = list(orch.process_stream("what is x?"))
    assert isinstance(items[-1], OrchestratorResult)
    assert items[-1].fallback is True
    assert "don't have enough information" in "".join(i for i in items[:-1] if isinstance(i, str))


# ── source dedupe ──
def test_extract_sources_dedupes_by_source():
    res = [_sr("a", 0.9, source="doc1"), _sr("b", 0.8, source="doc1"), _sr("c", 0.7, source="doc2")]
    srcs = Orchestrator._extract_sources(res)
    assert [s["source"] for s in srcs] == ["doc1", "doc2"]


# ── staleness downgrade ──
from datetime import datetime, timedelta, timezone


def _stale_sr(cid, score, days_old, ttl=30):
    r = _sr(cid, score)
    ts = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    r.freshness = {"source_fetched_at": ts, "ttl_days": ttl}
    return r


def test_stale_top_result_downgrades_high_to_medium():
    conf, note = Orchestrator._apply_staleness(
        Confidence.HIGH, [_stale_sr("a", 0.9, days_old=60)], CFG, datetime.now(timezone.utc))
    assert conf is Confidence.MEDIUM
    assert "days ago" in note


def test_fresh_top_result_no_downgrade():
    conf, note = Orchestrator._apply_staleness(
        Confidence.HIGH, [_stale_sr("a", 0.9, days_old=1)], CFG, datetime.now(timezone.utc))
    assert conf is Confidence.HIGH
    assert note == ""


def test_downgrade_disabled_by_config():
    cfg = RagConfig(freshness_enabled=False)
    conf, note = Orchestrator._apply_staleness(
        Confidence.HIGH, [_stale_sr("a", 0.9, days_old=60)], cfg, datetime.now(timezone.utc))
    assert conf is Confidence.HIGH


def test_low_floors_at_low():
    conf, _ = Orchestrator._apply_staleness(
        Confidence.LOW, [_stale_sr("a", 0.3, days_old=60)], CFG, datetime.now(timezone.utc))
    assert conf is Confidence.LOW


def test_no_freshness_data_no_downgrade():
    conf, note = Orchestrator._apply_staleness(
        Confidence.HIGH, [_sr("a", 0.9)], CFG, datetime.now(timezone.utc))
    assert conf is Confidence.HIGH
    assert note == ""


def test_process_sets_staleness_note_for_stale_answer():
    gen = FakeGenerator(answer="answer")
    orch = Orchestrator(FakeRetriever([_stale_sr("a", 0.9, days_old=60), _stale_sr("b", 0.8, days_old=60)]), gen, CFG)
    res = orch.process("q")
    assert res.confidence is Confidence.MEDIUM
    assert "days ago" in res.staleness_note


# ── reading-order context (graph) ──
def _osr(cid, score, source, ordinal, text="body"):
    r = SearchResult(chunk_id=cid, text=text, score=score,
                     metadata={"source": source, "title": source, "section_title": ""})
    r.ordinal = ordinal
    return r


def test_context_reading_order_groups_by_source_and_ordinal():
    # ordinal-0 chunk has the LOWER score → reading-order (ordinal) diverges from score-order
    cfg = RagConfig(graph_enabled=True, graph_context_ordering=True, context_tokens=10000)
    results = [_osr("b", 0.9, "doc1", 1, "second"), _osr("a", 0.7, "doc1", 0, "first")]
    ctx = Orchestrator._assemble_context(results, cfg)
    assert ctx.index("first") < ctx.index("second")   # ordinal 0 before 1, despite lower score


def test_context_score_order_when_graph_disabled():
    # same data, graph off → score-order: higher-scored "second" (0.9) precedes "first" (0.7)
    cfg = RagConfig(graph_enabled=False, context_tokens=10000)
    results = [_osr("b", 0.9, "doc1", 1, "second"), _osr("a", 0.7, "doc1", 0, "first")]
    ctx = Orchestrator._assemble_context(results, cfg)
    assert ctx.index("second") < ctx.index("first")   # unchanged score-order behavior
