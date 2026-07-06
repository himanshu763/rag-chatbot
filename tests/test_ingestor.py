"""RagIngestor tests — enriched-embedded / raw-stored invariant, idempotent skip,
provenance + hashes, and status dicts app/ingest depend on."""
from __future__ import annotations

from rag.config import RagConfig
from rag.ingest import RagIngestor
from tests.fakes import FakeEmbedder, FakeVectorStore


def _ingestor():
    store = FakeVectorStore()
    emb = FakeEmbedder()
    return RagIngestor(store, emb, RagConfig(chunk_size=400, min_chunk_size=1)), store, emb


def test_embeds_enriched_but_stores_raw():
    ing, store, emb = _ingestor()
    ing.ingest_text("Alpha beta gamma. Delta epsilon zeta.", title="Doc")
    recs = store.get_all()
    assert recs, "expected stored chunks"
    # raw_text is the plain chunk text (BM25 searches this) — no context prefix
    assert all("[Context:" not in r.raw_text for r in recs)
    # the text handed to the embedder WAS the enriched (context-prefixed) text
    embedded = [t for call in emb.calls for t in call]
    assert any("[Context:" in t for t in embedded)


def test_status_ok_and_chunk_count():
    ing, store, _ = _ingestor()
    r = ing.ingest_text("Alpha beta gamma. Delta epsilon zeta.", title="Doc")
    assert r["status"] == "ok"
    assert r["chunks"] >= 1
    assert r["title"] == "Doc"
    assert store.count() == r["chunks"]


def test_idempotent_skip_on_unchanged_content():
    ing, store, emb = _ingestor()
    ing.ingest_text("Same content here. More text.", title="Doc")
    count_after_first = store.count()
    emb.calls.clear()
    r2 = ing.ingest_text("Same content here. More text.", title="Doc")
    assert r2["status"] == "skipped"
    assert store.count() == count_after_first
    assert emb.calls == []                 # no re-embedding on unchanged content


def test_reingest_changed_content_replaces():
    ing, store, _ = _ingestor()
    ing.ingest_text("Original text. Second sentence.", title="Doc")
    ing.ingest_text("Rewritten text entirely. New sentence.", title="Doc")
    recs = store.get_all()
    assert all("Rewritten" in r.raw_text or "New sentence" in r.raw_text for r in recs)


def test_empty_document_status():
    ing, store, _ = _ingestor()
    r = ing.ingest_text("   ", title="Empty")
    assert r["status"] == "empty"
    assert store.count() == 0


def test_provenance_and_hashes_populated():
    ing, store, _ = _ingestor()
    ing.ingest_text("Alpha beta gamma. Delta.", title="Doc")
    rec = store.get_all()[0]
    assert rec.provenance["source_type"] == "text"
    assert rec.provenance["title"] == "Doc"
    assert rec.content_hash and rec.source_hash


def test_list_sources_and_total_chunks():
    ing, store, _ = _ingestor()
    ing.ingest_text("Alpha beta. Gamma delta.", title="Doc A")
    assert ing.total_chunks() == store.count()
    assert any(s["title"] == "Doc A" for s in ing.list_sources())


def test_freshness_populated_with_ttl_by_type():
    ing, store, _ = _ingestor()
    ing.ingest_text("Alpha beta gamma. Delta.", title="Doc")
    rec = store.get_all()[0]
    assert rec.freshness["ttl_days"] == 365          # text default from ttl_days_by_type
    assert rec.freshness.get("ingested_at")
    assert rec.freshness.get("source_fetched_at")


def test_refresh_unrefreshable_for_pasted_text():
    ing, store, _ = _ingestor()
    ing.ingest_text("Some content. More.", title="Doc")
    r = ing.refresh_source("text:Doc")
    assert r["status"] == "unrefreshable"


def test_refresh_file_reingests(tmp_path):
    store = FakeVectorStore()
    emb = FakeEmbedder()
    ing = RagIngestor(store, emb, RagConfig(min_chunk_size=1))
    p = tmp_path / "n.txt"
    p.write_text("First sentence here. Second one too.", encoding="utf-8")
    ing.ingest_file(str(p))
    r = ing.refresh_source(str(p))
    assert r["status"] in ("ok", "skipped")


def test_ingest_populates_entities_and_ordinals():
    ing, store, _ = _ingestor()
    ing.ingest_text("The Zephyr X1 turbine is strong. Fusion ERP Analytics tracks it.", title="Doc")
    recs = sorted(store.get_all(), key=lambda r: r.ordinal)
    assert [r.ordinal for r in recs] == list(range(len(recs)))    # 0..n-1
    all_ents = {e for r in recs for e in r.entities}
    assert "zephyr x1" in all_ents or "fusion erp analytics" in all_ents


def test_refresh_unstales_unchanged_source(tmp_path):
    """Re-checking stable content must bump fetched_at — else the sweep never un-stales it."""
    from datetime import datetime, timedelta, timezone
    from rag.freshness import is_stale
    store = FakeVectorStore()
    emb = FakeEmbedder()
    ing = RagIngestor(store, emb, RagConfig(min_chunk_size=1))
    p = tmp_path / "n.txt"
    p.write_text("First sentence. Second one.", encoding="utf-8")
    ing.ingest_file(str(p))
    for r in store.get_all():                          # force stale
        r.freshness["source_fetched_at"] = (datetime.now(timezone.utc) - timedelta(days=999)).isoformat()
        r.freshness["ttl_days"] = 30
    res = ing.refresh_source(str(p))                   # content unchanged → "skipped"
    assert res["status"] == "skipped"
    assert not is_stale(store.get_source_freshness(str(p)), RagConfig(), datetime.now(timezone.utc))
