"""Tests for ChromaVectorStore — persistence, query, scaffold JSON encode/decode,
source management, and per-instance isolation (no global collection)."""
from __future__ import annotations

import pytest

pytest.importorskip("chromadb")

from rag.config import RagConfig
from rag.stores.chroma import ChromaVectorStore
from rag.types import ChunkRecord


def _rec(cid, text, source, emb, **kw):
    return ChunkRecord(
        chunk_id=cid, raw_text=text, embedding=emb,
        provenance={"source": source, "source_type": "text", "title": source},
        source_hash=kw.get("source_hash", "h1"),
        section_path=kw.get("section_path", ""),
        entities=kw.get("entities", []),
        edges=kw.get("edges", []),
    )


def _store(tmp_path, name="coll_main"):
    return ChromaVectorStore(RagConfig(chroma_dir=str(tmp_path / name), collection_name=name))


def test_add_and_count(tmp_path):
    s = _store(tmp_path)
    s.add([_rec("a", "hello world", "src1", [1.0, 0.0]),
           _rec("b", "goodbye moon", "src1", [0.0, 1.0])])
    assert s.count() == 2


def test_query_returns_nearest_first(tmp_path):
    s = _store(tmp_path)
    s.add([_rec("a", "hello world", "src1", [1.0, 0.0]),
           _rec("b", "goodbye moon", "src1", [0.0, 1.0])])
    res = s.query([1.0, 0.0], top_k=2)
    assert res[0].chunk_id == "a"
    assert res[0].score >= res[1].score


def test_scaffold_fields_roundtrip(tmp_path):
    """entities/edges are lists — must survive Chroma's scalar-only metadata via JSON."""
    s = _store(tmp_path)
    s.add([_rec("a", "x", "src1", [1.0, 0.0], entities=["Acme", "Widget"], edges=["b"])])
    got = {r.chunk_id: r for r in s.get_all()}["a"]
    assert got.entities == ["Acme", "Widget"]
    assert got.edges == ["b"]


def test_parent_text_not_leaked_into_query_metadata(tmp_path):
    s = _store(tmp_path)
    s.add([_rec("a", "x", "src1", [1.0, 0.0])])
    r = s.query([1.0, 0.0], top_k=1)[0]
    assert "parent_text" not in r.metadata


def test_delete_source(tmp_path):
    s = _store(tmp_path)
    s.add([_rec("a", "x", "src1", [1.0, 0.0]), _rec("b", "y", "src2", [0.0, 1.0])])
    s.delete_source("src1")
    assert s.count() == 1
    assert s.list_sources()[0]["source"] == "src2"


def test_list_sources_groups_and_counts(tmp_path):
    s = _store(tmp_path)
    s.add([_rec("a", "x", "src1", [1.0, 0.0]), _rec("b", "y", "src1", [0.1, 0.9]),
           _rec("c", "z", "src2", [0.5, 0.5])])
    by = {d["source"]: d for d in s.list_sources()}
    assert by["src1"]["chunks"] == 2
    assert by["src2"]["chunks"] == 1


def test_get_source_hash(tmp_path):
    s = _store(tmp_path)
    s.add([_rec("a", "x", "src1", [1.0, 0.0], source_hash="deadbeef")])
    assert s.get_source_hash("src1") == "deadbeef"
    assert s.get_source_hash("nope") is None


def test_query_surfaces_freshness(tmp_path):
    s = _store(tmp_path)
    rec = _rec("a", "x", "src1", [1.0, 0.0])
    rec.freshness = {"ingested_at": "2026-01-01T00:00:00+00:00", "ttl_days": 30}
    s.add([rec])
    r = s.query([1.0, 0.0], top_k=1)[0]
    assert r.freshness["ttl_days"] == 30
    assert "__freshness_json" not in r.metadata     # still stripped from public metadata


def test_ordinal_roundtrips(tmp_path):
    s = _store(tmp_path)
    rec = _rec("a", "x", "src1", [1.0, 0.0])
    rec.ordinal = 4
    s.add([rec])
    assert s.get_all()[0].ordinal == 4
    assert s.query([1.0, 0.0], top_k=1)[0].ordinal == 4


def test_two_instances_are_isolated(tmp_path):
    """Proves no global collection: separate stores don't share data."""
    a = _store(tmp_path, "tenant_a")
    b = _store(tmp_path, "tenant_b")
    a.add([_rec("a", "x", "src1", [1.0, 0.0])])
    assert a.count() == 1
    assert b.count() == 0
