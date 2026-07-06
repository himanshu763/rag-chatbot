from datetime import datetime, timedelta, timezone

from rag.config import RagConfig
from rag.freshness import age_days, ttl_days, is_stale, freshness_note

NOW = datetime(2026, 7, 4, tzinfo=timezone.utc)


def _fresh(days_ago, source_type="web", ttl=None):
    ts = (NOW - timedelta(days=days_ago)).isoformat()
    d = {"source_fetched_at": ts, "ingested_at": ts}
    if ttl is not None:
        d["ttl_days"] = ttl
    return d


def test_age_days_uses_fetched_at():
    assert age_days(_fresh(10), NOW) == 10.0


def test_age_days_falls_back_to_ingested_at():
    ts = (NOW - timedelta(days=5)).isoformat()
    assert age_days({"ingested_at": ts}, NOW) == 5.0


def test_age_days_none_when_missing():
    assert age_days({}, NOW) is None


def test_ttl_days_prefers_chunk_value():
    assert ttl_days({"ttl_days": 12}, RagConfig()) == 12


def test_ttl_days_falls_back_to_default():
    assert ttl_days({}, RagConfig()) == 180


def test_is_stale_boundary():
    cfg = RagConfig()
    assert is_stale(_fresh(31, ttl=30), cfg, NOW) is True
    assert is_stale(_fresh(30, ttl=30), cfg, NOW) is False   # exactly ttl = not stale
    assert is_stale(_fresh(29, ttl=30), cfg, NOW) is False


def test_is_stale_unknown_age_is_not_stale():
    assert is_stale({}, RagConfig(), NOW) is False


def test_freshness_note_wording():
    assert "42 days ago" in freshness_note(_fresh(42), NOW)


def test_freshness_note_empty_when_unknown():
    assert freshness_note({}, NOW) == ""


# ── sweep ──
from rag.config import RagConfig as _Cfg
from rag.pipeline import RagPipeline
from rag.types import ChunkRecord
from tests.fakes import FakeEmbedder, FakeGenerator, FakeVectorStore


def _pipe():
    cfg = _Cfg(similarity_threshold=0.0, low_confidence_threshold=0.0, min_chunk_size=1)
    return RagPipeline(config=cfg, store=FakeVectorStore(), embedder=FakeEmbedder(),
                       generator=FakeGenerator(), reranker=None)


def test_sweep_reports_stale_sources():
    p = _pipe()
    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    p.store.add([ChunkRecord(chunk_id="c1", raw_text="x",
                             provenance={"source": "http://example.com/a", "source_type": "web", "title": "A"},
                             freshness={"source_fetched_at": old, "ttl_days": 30})])
    report = p.sweep_stale(dry_run=True)
    assert "http://example.com/a" in report["stale"]
    assert report["refreshed"] == []          # dry run refreshes nothing


def test_sweep_skips_fresh_sources():
    p = _pipe()
    fresh = datetime.now(timezone.utc).isoformat()
    p.store.add([ChunkRecord(chunk_id="c1", raw_text="x",
                             provenance={"source": "http://example.com/a", "source_type": "web", "title": "A"},
                             freshness={"source_fetched_at": fresh, "ttl_days": 30})])
    report = p.sweep_stale(dry_run=True)
    assert report["stale"] == []


def test_cli_main_dry_run(capsys, monkeypatch):
    from rag import freshness as fmod
    p = _pipe()
    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    p.store.add([ChunkRecord(chunk_id="c1", raw_text="x",
                             provenance={"source": "http://example.com/a", "source_type": "web", "title": "A"},
                             freshness={"source_fetched_at": old, "ttl_days": 30})])
    monkeypatch.setattr(fmod, "_build_pipeline", lambda: p)
    fmod.main(["sweep", "--dry-run"])
    out = capsys.readouterr().out
    assert "http://example.com/a" in out
    assert "stale" in out.lower()
