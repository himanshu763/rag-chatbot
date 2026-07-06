"""Tests for rag.types — ChunkRecord schema, SearchResult, Confidence, results."""
from __future__ import annotations

from rag.types import (
    ChunkRecord,
    Confidence,
    CONF_META,
    OrchestratorResult,
    SearchResult,
)


def test_chunk_record_scaffold_fields_default_empty():
    """entities/edges/freshness are reserved for later features — default empty, not None."""
    rec = ChunkRecord(chunk_id="c1", raw_text="hello world")
    assert rec.entities == []
    assert rec.edges == []
    assert rec.freshness == {}
    # mutating one record's scaffold must not leak to another (no shared mutable default)
    rec.entities.append("x")
    rec2 = ChunkRecord(chunk_id="c2", raw_text="bye")
    assert rec2.entities == []


def test_chunk_record_defaults():
    rec = ChunkRecord(chunk_id="c1", raw_text="body")
    assert rec.enriched_text == ""
    assert rec.embedding is None
    assert rec.parent_text is None
    assert rec.section_path == ""
    assert rec.provenance == {}
    assert rec.content_hash == ""
    assert rec.source_hash == ""


def test_chunk_record_carries_all_fields():
    rec = ChunkRecord(
        chunk_id="c1",
        raw_text="body",
        enriched_text="[Context: x] body",
        embedding=[0.1, 0.2],
        parent_text="heading\nbody",
        section_path="Doc > Section",
        provenance={"source": "s", "source_type": "web", "title": "T"},
        content_hash="abc",
        source_hash="def",
    )
    assert rec.embedding == [0.1, 0.2]
    assert rec.provenance["title"] == "T"
    assert rec.section_path == "Doc > Section"


def test_search_result_defaults():
    r = SearchResult(chunk_id="c1", text="t", score=0.5)
    assert r.metadata == {}
    assert r.parent_text is None


def test_confidence_values():
    assert Confidence.HIGH.value == "high"
    assert Confidence("none") is Confidence.NONE
    # every confidence level has render metadata
    for level in Confidence:
        assert level in CONF_META
        assert "icon" in CONF_META[level] and "color" in CONF_META[level]


def test_orchestrator_result_defaults():
    res = OrchestratorResult(response="hi")
    assert res.confidence is Confidence.NONE
    assert res.sources == []
    assert res.fallback is False
    assert res.latency_ms == 0.0
