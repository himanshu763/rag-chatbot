# Freshness / Staleness Checker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make freshness a first-class signal — capture timestamps at ingest, score staleness against per-source-type TTL, surface age on citations, downgrade confidence one level for stale answers, and provide on-demand re-check + a sweep command.

**Architecture:** A pure scoring module (`rag/freshness.py`) is the testable core. Ingest populates the already-scaffolded `ChunkRecord.freshness` dict; the store surfaces it into `SearchResult`; the orchestrator downgrades confidence after assessment (strict no-op when disabled or when data lacks freshness — protects parity). Active recheck + a `sweep_stale()` command (CLI, no daemon) refresh stale sources.

**Tech Stack:** Python 3.12, pytest, ChromaDB, existing `rag/` library. Test runner: `.venv-wsl/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-07-04-freshness-staleness-design.md`

---

## File Structure

- `rag/freshness.py` — **new**. Pure scoring functions + sweep CLI `main()`.
- `rag/config.py` — **modify**. Add freshness config fields + env reads.
- `rag/types.py` — **modify**. `SearchResult.freshness`, `OrchestratorResult.staleness_note`.
- `rag/loaders.py` — **modify**. Capture `source_fetched_at`/`source_last_modified`/`etag`; WebLoader conditional GET.
- `rag/ingest.py` — **modify**. Populate `ChunkRecord.freshness`; `refresh_source()`.
- `rag/stores/chroma.py` — **modify**. Decode `__freshness_json` into `SearchResult.freshness`.
- `rag/orchestrator.py` — **modify**. `_apply_staleness()` after `_assess_confidence` in both paths.
- `rag/pipeline.py` — **modify**. `refresh_source()` + `sweep_stale()` passthrough/compose.
- `app.py` — **modify**. Show age + staleness note on citations.
- Tests: `tests/test_freshness.py` (new), additions to `test_orchestrator.py`, `test_ingestor.py`, `test_chroma_store.py`.

Test/commit convention: `PY=.venv-wsl/bin/python`. Commit after each task.

---

## Task 1: Config fields for freshness

**Files:**
- Modify: `rag/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_config.py`:

```python
def test_freshness_config_defaults():
    cfg = RagConfig()
    assert cfg.freshness_enabled is True
    assert cfg.staleness_downgrade is True
    assert cfg.default_ttl_days == 180
    assert cfg.ttl_days_by_type["web"] == 30
    assert cfg.ttl_days_by_type["pdf"] == 365


def test_freshness_config_from_env(monkeypatch):
    monkeypatch.setenv("FRESHNESS_ENABLED", "false")
    monkeypatch.setenv("DEFAULT_TTL_DAYS", "90")
    cfg = RagConfig.from_env()
    assert cfg.freshness_enabled is False
    assert cfg.default_ttl_days == 90


def test_ttl_maps_are_independent_per_instance():
    a = RagConfig()
    a.ttl_days_by_type["web"] = 7
    b = RagConfig()
    assert b.ttl_days_by_type["web"] == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-wsl/bin/python -m pytest tests/test_config.py -q`
Expected: FAIL — `AttributeError: 'RagConfig' object has no attribute 'freshness_enabled'`.

- [ ] **Step 3: Implement** — in `rag/config.py`, add fields to the `RagConfig` dataclass (after `history_tokens`), keeping `field` import (already imported):

```python
    # Freshness / staleness
    freshness_enabled: bool = True
    staleness_downgrade: bool = True
    default_ttl_days: int = 180
    ttl_days_by_type: dict = field(default_factory=lambda: {
        "web": 30, "pdf": 365, "docx": 365, "csv": 180, "excel": 180, "text": 365,
    })
```

Then in `from_env`, add these keys to the `env = {...}` dict:

```python
            "freshness_enabled": os.getenv("FRESHNESS_ENABLED", "true").lower() != "false",
            "default_ttl_days": int(os.getenv("DEFAULT_TTL_DAYS", "180")),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-wsl/bin/python -m pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rag/config.py tests/test_config.py
git commit -m "feat(freshness): add freshness config fields"
```

---

## Task 2: Pure scoring module `rag/freshness.py`

**Files:**
- Create: `rag/freshness.py`
- Test: `tests/test_freshness.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_freshness.py`:

```python
from datetime import datetime, timedelta, timezone

from rag.config import RagConfig
from rag.freshness import age_days, ttl_days, is_stale, staleness_ratio, freshness_note

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


def test_staleness_ratio():
    assert staleness_ratio(_fresh(60, ttl=30), RagConfig(), NOW) == 2.0
    assert staleness_ratio({}, RagConfig(), NOW) == 0.0


def test_freshness_note_wording():
    assert "42 days ago" in freshness_note(_fresh(42), NOW)


def test_freshness_note_empty_when_unknown():
    assert freshness_note({}, NOW) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-wsl/bin/python -m pytest tests/test_freshness.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.freshness'`.

- [ ] **Step 3: Implement** — create `rag/freshness.py`:

```python
"""Freshness / staleness scoring — pure functions over ChunkRecord.freshness dicts.

Age uses source_fetched_at (falls back to ingested_at). Unknown/unparseable timestamps
yield age None → treated as NOT stale (never downgrade on missing data).
"""
from __future__ import annotations

from datetime import datetime, timezone


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def age_days(freshness: dict, now: datetime) -> float | None:
    ts = freshness.get("source_fetched_at") or freshness.get("ingested_at")
    dt = _parse(ts)
    if dt is None:
        return None
    return (now - dt).total_seconds() / 86400.0


def ttl_days(freshness: dict, config) -> int:
    val = freshness.get("ttl_days")
    return int(val) if val is not None else config.default_ttl_days


def is_stale(freshness: dict, config, now: datetime) -> bool:
    age = age_days(freshness, now)
    if age is None:
        return False
    return age > ttl_days(freshness, config)


def staleness_ratio(freshness: dict, config, now: datetime) -> float:
    age = age_days(freshness, now)
    if age is None:
        return 0.0
    ttl = ttl_days(freshness, config)
    return age / ttl if ttl else 0.0


def freshness_note(freshness: dict, now: datetime) -> str:
    age = age_days(freshness, now)
    if age is None:
        return ""
    return f"last verified {int(age)} days ago"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-wsl/bin/python -m pytest tests/test_freshness.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add rag/freshness.py tests/test_freshness.py
git commit -m "feat(freshness): pure staleness scoring module"
```

---

## Task 3: Ingest captures freshness into ChunkRecord

**Files:**
- Modify: `rag/loaders.py` (WebLoader, file loaders — add fetch metadata)
- Modify: `rag/ingest.py` (`_to_record` populates `freshness`)
- Test: `tests/test_ingestor.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_ingestor.py`:

```python
def test_freshness_populated_with_ttl_by_type():
    ing, store, _ = _ingestor()
    ing.ingest_text("Alpha beta gamma. Delta.", title="Doc")
    rec = store.get_all()[0]
    assert rec.freshness["ttl_days"] == 365          # text default from ttl_days_by_type
    assert rec.freshness.get("ingested_at")
    assert rec.freshness.get("source_fetched_at")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-wsl/bin/python -m pytest tests/test_ingestor.py::test_freshness_populated_with_ttl_by_type -q`
Expected: FAIL — `KeyError: 'ttl_days'` (freshness is `{}`).

- [ ] **Step 3: Implement**

In `rag/ingest.py`, change `_to_record` from a `@staticmethod` to an instance method so it can read `self.config`, and populate `freshness`. Replace the whole method:

```python
    def _to_record(self, chunk: Chunk, enriched: str, embedding: list[float], source_hash: str) -> ChunkRecord:
        m = chunk.metadata
        source_type = m.get("source_type", "")
        ttl = self.config.ttl_days_by_type.get(source_type, self.config.default_ttl_days)
        freshness = {
            "ingested_at": _now_iso(),
            "source_fetched_at": m.get("source_fetched_at", ""),
            "source_last_modified": m.get("source_last_modified", ""),
            "etag": m.get("etag", ""),
            "ttl_days": ttl,
        }
        return ChunkRecord(
            chunk_id=chunk.chunk_id,
            raw_text=chunk.text,
            enriched_text=enriched,
            embedding=embedding,
            parent_text=chunk.parent_text,
            section_path=m.get("section_title", ""),
            provenance={
                "source": m.get("source", ""),
                "source_type": source_type,
                "title": m.get("title", ""),
                "anchor": m.get("anchor", ""),
            },
            content_hash=chunk.content_hash,
            source_hash=source_hash,
            freshness=freshness,
        )
```

Update the call site in `_process` (it currently calls `self._to_record(...)` via the staticmethod name — confirm it reads `self._to_record`): the existing line is
`records = [self._to_record(c, enriched, emb, doc.content_hash) for c, enriched, emb in zip(...)]` — already `self.`, no change needed.

Add a helper near the top of `rag/ingest.py` (after imports):

```python
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
```

In `rag/loaders.py`, ensure fetch metadata is present. `TextLoader.load` already sets
`ingested_at`; add `source_fetched_at`. In `TextLoader.load`, change the metadata dict to include:

```python
            metadata={"source_type": "text", "source": "manual", "title": title,
                      "source_fetched_at": datetime.now(timezone.utc).isoformat(),
                      "ingested_at": datetime.now(timezone.utc).isoformat()},
```

(`datetime`/`timezone` already imported in loaders.py.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-wsl/bin/python -m pytest tests/test_ingestor.py -q`
Expected: PASS (all ingestor tests).

- [ ] **Step 5: Commit**

```bash
git add rag/ingest.py rag/loaders.py tests/test_ingestor.py
git commit -m "feat(freshness): populate ChunkRecord.freshness at ingest"
```

---

## Task 4: Web/file loaders capture last_modified + etag

**Files:**
- Modify: `rag/loaders.py` (WebLoader.load, PDFLoader/DocxLoader/CSVLoader/ExcelLoader/TxtLoader)
- Test: `tests/test_chunking_loaders.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_chunking_loaders.py`:

```python
def test_txt_loader_captures_file_mtime(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("First para.\n\nSecond para.", encoding="utf-8")
    doc = TxtLoader().load(str(p))
    assert doc.metadata.get("source_fetched_at")
    assert doc.metadata.get("source_last_modified")   # from file mtime
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-wsl/bin/python -m pytest tests/test_chunking_loaders.py::test_txt_loader_captures_file_mtime -q`
Expected: FAIL — `assert None` (`source_last_modified` missing).

- [ ] **Step 3: Implement** — in `rag/loaders.py`:

Add a module helper after `_build_session`:

```python
def _file_times(path: str) -> dict:
    """fetched_at (now) + source_last_modified (file mtime), as ISO strings."""
    import os
    now = datetime.now(timezone.utc).isoformat()
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).isoformat()
    except OSError:
        mtime = ""
    return {"source_fetched_at": now, "source_last_modified": mtime}
```

In each file loader (`PDFLoader`, `DocxLoader`, `CSVLoader`, `ExcelLoader`, `TxtLoader`), merge `_file_times(path)` into the returned `RawDocument.metadata`. Example for `TxtLoader.load` — change its metadata to:

```python
            metadata={"source_type": "text", "source": path, "title": Path(path).stem,
                      "ingested_at": datetime.now(timezone.utc).isoformat(),
                      **_file_times(path)},
```

Apply the same `**_file_times(path)` merge to the `metadata={...}` of `PDFLoader.load`,
`DocxLoader.load`, `CSVLoader.load` (the non-empty return), and `ExcelLoader.load`.

For `WebLoader.load`, after `resp = session.get(url, timeout=30)`, capture headers and merge
into metadata:

```python
        fetched = {
            "source_fetched_at": datetime.now(timezone.utc).isoformat(),
            "source_last_modified": resp.headers.get("Last-Modified", ""),
            "etag": resp.headers.get("ETag", ""),
        }
```

and add `**fetched` to the `metadata={...}` dict in the returned `RawDocument`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-wsl/bin/python -m pytest tests/test_chunking_loaders.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rag/loaders.py tests/test_chunking_loaders.py
git commit -m "feat(freshness): loaders capture last_modified + etag"
```

---

## Task 5: SearchResult carries freshness; Chroma surfaces it

**Files:**
- Modify: `rag/types.py` (`SearchResult.freshness`)
- Modify: `rag/stores/chroma.py` (decode `__freshness_json` into query results)
- Test: `tests/test_chroma_store.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_chroma_store.py`:

```python
def test_query_surfaces_freshness(tmp_path):
    s = _store(tmp_path)
    rec = _rec("a", "x", "src1", [1.0, 0.0])
    rec.freshness = {"ingested_at": "2026-01-01T00:00:00+00:00", "ttl_days": 30}
    s.add([rec])
    r = s.query([1.0, 0.0], top_k=1)[0]
    assert r.freshness["ttl_days"] == 30
    assert "__freshness_json" not in r.metadata     # still stripped from public metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-wsl/bin/python -m pytest tests/test_chroma_store.py::test_query_surfaces_freshness -q`
Expected: FAIL — `AttributeError: 'SearchResult' object has no attribute 'freshness'`.

- [ ] **Step 3: Implement**

In `rag/types.py`, add a field to `SearchResult`:

```python
@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    parent_text: str | None = None
    freshness: dict = field(default_factory=dict)
```

In `rag/stores/chroma.py`, `query()` — where each `SearchResult` is built, decode freshness.
Add a helper import at top: `import json` is already imported. Add a static decoder:

```python
    @staticmethod
    def _decode_freshness(raw: dict) -> dict:
        try:
            return json.loads(raw.get(_FRESHNESS_KEY, "") or "{}")
        except Exception:
            return {}
```

Then in the `query` loop, pass `freshness=self._decode_freshness(raw_meta)` to `SearchResult(...)`:

```python
                out.append(SearchResult(
                    chunk_id=res["ids"][0][i],
                    text=res["documents"][0][i],
                    score=1.0 - res["distances"][0][i],
                    metadata=self._public_meta(raw_meta),
                    parent_text=parent,
                    freshness=self._decode_freshness(raw_meta),
                ))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-wsl/bin/python -m pytest tests/test_chroma_store.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rag/types.py rag/stores/chroma.py tests/test_chroma_store.py
git commit -m "feat(freshness): surface freshness in SearchResult from Chroma"
```

---

## Task 6: Orchestrator staleness downgrade + note

**Files:**
- Modify: `rag/types.py` (`OrchestratorResult.staleness_note`)
- Modify: `rag/orchestrator.py` (`_apply_staleness`, wire into both paths)
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_orchestrator.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-wsl/bin/python -m pytest tests/test_orchestrator.py -q`
Expected: FAIL — `AttributeError: type object 'Orchestrator' has no attribute '_apply_staleness'`.

- [ ] **Step 3: Implement**

In `rag/types.py`, add to `OrchestratorResult`:

```python
    staleness_note: str = ""
```

In `rag/orchestrator.py`, add imports at top:

```python
from datetime import datetime, timezone

from rag.freshness import freshness_note, is_stale
```

Add the static method to `Orchestrator`:

```python
    _DOWNGRADE = {Confidence.HIGH: Confidence.MEDIUM,
                  Confidence.MEDIUM: Confidence.LOW,
                  Confidence.LOW: Confidence.LOW}

    @staticmethod
    def _apply_staleness(conf, results, config, now=None):
        """Return (possibly downgraded confidence, note). No-op when disabled or fresh."""
        if not (config.freshness_enabled and config.staleness_downgrade) or not results:
            return conf, ""
        now = now or datetime.now(timezone.utc)
        top = max(results, key=lambda r: r.score)
        fresh = getattr(top, "freshness", {}) or {}
        if is_stale(fresh, config, now):
            return Orchestrator._DOWNGRADE.get(conf, conf), freshness_note(fresh, now)
        return conf, ""
```

Wire into `process` — after `conf = self._assess_confidence(results, self.config)` and the
`if conf == Confidence.NONE:` block, insert:

```python
        conf, staleness_note = self._apply_staleness(conf, results, self.config)
```

and pass `staleness_note=staleness_note` into the final `OrchestratorResult(...)` return.

Wire into `process_stream` — same: after the NONE block, add the `_apply_staleness` call and
pass `staleness_note=staleness_note` into the terminal `OrchestratorResult(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-wsl/bin/python -m pytest tests/test_orchestrator.py -q`
Expected: PASS (existing 12 + 6 new).

- [ ] **Step 5: Commit**

```bash
git add rag/types.py rag/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(freshness): downgrade confidence one level for stale answers"
```

---

## Task 7: Active recheck — RagIngestor.refresh_source

**Files:**
- Modify: `rag/ingest.py` (`refresh_source`)
- Test: `tests/test_ingestor.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_ingestor.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-wsl/bin/python -m pytest tests/test_ingestor.py::test_refresh_unrefreshable_for_pasted_text -q`
Expected: FAIL — `AttributeError: 'RagIngestor' object has no attribute 'refresh_source'`.

- [ ] **Step 3: Implement** — add to `RagIngestor` in `rag/ingest.py`:

```python
    def refresh_source(self, source_key: str) -> dict:
        """Re-fetch a web/file source and re-ingest if changed. Pasted text is unrefreshable."""
        if source_key.startswith(("http://", "https://")):
            return self.ingest_url(source_key)
        if source_key.startswith("text:"):
            return {"source": source_key, "status": "unrefreshable", "chunks": 0}
        import os
        if os.path.exists(source_key):
            return self.ingest_file(source_key)
        return {"source": source_key, "status": "missing", "chunks": 0}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-wsl/bin/python -m pytest tests/test_ingestor.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rag/ingest.py tests/test_ingestor.py
git commit -m "feat(freshness): RagIngestor.refresh_source for active recheck"
```

---

## Task 8: Sweep — pipeline.sweep_stale() + freshness-per-source read

**Files:**
- Modify: `rag/stores/chroma.py` (add `get_source_freshness`) and `tests/fakes.py` (FakeVectorStore mirror)
- Modify: `rag/pipeline.py` (`refresh_source`, `sweep_stale`)
- Test: `tests/test_freshness.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_freshness.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-wsl/bin/python -m pytest tests/test_freshness.py -q`
Expected: FAIL — `AttributeError: 'RagPipeline' object has no attribute 'sweep_stale'`.

- [ ] **Step 3: Implement**

Add `get_source_freshness` to `ChromaVectorStore` (`rag/stores/chroma.py`):

```python
    def get_source_freshness(self, source_key: str) -> dict:
        try:
            existing = self._col.get(where={"source": {"$eq": source_key}}, include=["metadatas"])
        except Exception:
            return {}
        if existing and existing["ids"]:
            return self._decode_freshness(existing["metadatas"][0] or {})
        return {}
```

Mirror it in `FakeVectorStore` (`tests/fakes.py`):

```python
    def get_source_freshness(self, source_key: str) -> dict:
        for r in self._records.values():
            if r.provenance.get("source") == source_key:
                return dict(r.freshness)
        return {}
```

Add to `rag/pipeline.py` (imports at top: `from datetime import datetime, timezone` and
`from rag.freshness import is_stale`):

```python
    def refresh_source(self, source_key: str) -> dict:
        return self.ingestor.refresh_source(source_key)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-wsl/bin/python -m pytest tests/test_freshness.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rag/stores/chroma.py rag/pipeline.py tests/fakes.py tests/test_freshness.py
git commit -m "feat(freshness): pipeline.sweep_stale + get_source_freshness"
```

---

## Task 9: Sweep CLI — `python -m rag.freshness sweep`

**Files:**
- Modify: `rag/freshness.py` (add `main()` + `__main__` guard)
- Test: `tests/test_freshness.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_freshness.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-wsl/bin/python -m pytest tests/test_freshness.py::test_cli_main_dry_run -q`
Expected: FAIL — `AttributeError: module 'rag.freshness' has no attribute 'main'`.

- [ ] **Step 3: Implement** — append to `rag/freshness.py`:

```python
def _build_pipeline():
    from rag import RagConfig, RagPipeline
    return RagPipeline(RagConfig.from_env())


def main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="rag.freshness", description="Freshness sweep")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sweep = sub.add_parser("sweep", help="find + re-ingest stale sources")
    sweep.add_argument("--dry-run", action="store_true", help="report only, don't re-ingest")
    args = parser.parse_args(argv)

    if args.cmd == "sweep":
        pipeline = _build_pipeline()
        report = pipeline.sweep_stale(dry_run=args.dry_run)
        print(f"Checked {report['checked']} source(s) — {len(report['stale'])} stale.")
        for key in report["stale"]:
            mark = "would refresh" if args.dry_run else (
                "refreshed" if key in report["refreshed"] else "skipped")
            print(f"  • {key}  [{mark}]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-wsl/bin/python -m pytest tests/test_freshness.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rag/freshness.py tests/test_freshness.py
git commit -m "feat(freshness): sweep CLI (python -m rag.freshness sweep)"
```

---

## Task 10: UI — show age + staleness note on citations

**Files:**
- Modify: `app.py` (citation rendering + staleness note under badge)
- Manual verification (Streamlit UI, not unit-tested)

- [ ] **Step 1: Implement** — in `app.py`, both places that render the confidence badge
(history render and live render), after the badge markdown, add the staleness note. Find the
two `st.markdown(... conf-badge ...)` blocks. After each, insert:

```python
            if getattr(meta, "staleness_note", ""):    # history block uses `meta`
                st.caption(f"⏳ {meta.staleness_note}")
```

For the live block the variable is `meta_result` — use:

```python
            if getattr(meta_result, "staleness_note", ""):
                st.caption(f"⏳ {meta_result.staleness_note}")
```

- [ ] **Step 2: Verify import + compile**

Run: `.venv-wsl/bin/python -m py_compile app.py && echo OK`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(freshness): show staleness note on chat citations"
```

---

## Task 11: Full verification

- [ ] **Step 1: Full test suite**

Run: `.venv-wsl/bin/python -m pytest -q`
Expected: all pass (prior 84 + freshness additions), 1 skipped (cross-encoder).

- [ ] **Step 2: Parity check — freshness disabled is a no-op**

Run:
```bash
.venv-wsl/bin/python -c "
from datetime import datetime, timezone, timedelta
from rag.config import RagConfig
from rag.orchestrator import Orchestrator
from rag.types import Confidence, SearchResult
r = SearchResult('a','x',0.9); 
r.freshness={'source_fetched_at':(datetime.now(timezone.utc)-timedelta(days=999)).isoformat(),'ttl_days':30}
print('enabled:', Orchestrator._apply_staleness(Confidence.HIGH,[r],RagConfig())[0].value)
print('disabled:', Orchestrator._apply_staleness(Confidence.HIGH,[r],RagConfig(freshness_enabled=False))[0].value)
"
```
Expected: `enabled: medium` / `disabled: high`.

- [ ] **Step 3: CLI smoke (dry-run against real store, no writes)**

Run: `.venv-wsl/bin/python -m rag.freshness sweep --dry-run`
Expected: prints "Checked N source(s) — M stale." Note: existing chunks have no freshness data → age None → not stale → likely 0 stale (safe).

- [ ] **Step 4: Commit any final fixups**

```bash
git add -A && git commit -m "test(freshness): full-suite verification" || echo "nothing to commit"
```

---

## Self-Review Notes (author)

- Spec §1 data → Tasks 3, 4. §2 config → Task 1. §3 scoring → Task 2. §4 surfacing → Task 5.
  §5 downgrade → Task 6. §6 recheck → Task 7. §7 sweep → Tasks 8, 9. §8 UI → Task 10.
- Parity: `_apply_staleness` gated (Task 6 tests `freshness_enabled=False` + no-data no-op);
  existing chunks lack freshness → never stale → verified Task 11 Step 2/3.
- Type consistency: `freshness` dict everywhere; `SearchResult.freshness` (Task 5) consumed by
  `_apply_staleness` (Task 6) and `get_source_freshness` (Task 8). `refresh_source` status
  strings (`ok`/`skipped`/`unrefreshable`/`missing`) consistent between Tasks 7 and 8.
