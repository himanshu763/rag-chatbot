# Freshness / Staleness Checker — Design Spec

**Date:** 2026-07-04
**Status:** Approved, ready for implementation plan
**Depends on:** foundation slice (rag/ library, `ChunkRecord.freshness` scaffold field)

## Context

The `rag/` library stores chunks but has no concept of *when* content was fetched or
whether it's outdated. Retrieval treats a 2-year-old pricing page the same as one fetched
today. This feature makes freshness a first-class signal: capture timestamps at ingest,
score staleness against a per-source-type TTL, surface age on citations, downgrade
confidence for stale answers, and provide on-demand re-check + a sweep command to refresh
stale sources.

`ChunkRecord.freshness` already exists (scaffolded empty in the foundation slice) and is
persisted to Chroma via the `__freshness_json` metadata key — so **no schema migration**.

Scope (confirmed with user): **full** — passive scoring + active recheck + sweep command.
Staleness effect: **downgrade one confidence level + note**. Sweep: **CLI command /
`pipeline.sweep_stale()`, no daemon dependency**.

## Non-Goals

- No background scheduler/daemon (user crons the CLI themselves).
- No hard filtering of stale chunks out of retrieval (downgrade only, not drop).
- No Tier-2 / distributed concerns.

## Components

### 1. `rag/freshness.py` — pure scoring (new)

Pure functions, no I/O — the load-bearing testable core:

```python
def age_days(freshness: dict, now: datetime) -> float | None      # None if no timestamp
def ttl_days(freshness: dict, config) -> int                       # chunk ttl_days or config fallback
def is_stale(freshness: dict, config, now: datetime) -> bool       # age > ttl
def staleness_ratio(freshness: dict, config, now: datetime) -> float  # age/ttl (0 if unknown)
def freshness_note(freshness: dict, now: datetime) -> str          # "last verified 42 days ago"
```

`age_days` uses `source_fetched_at` if present, else `ingested_at`. Missing/unparseable
timestamp → `None` age → treated as **not stale** (safe default: never downgrade on unknown).

Also hosts the sweep CLI `main()` (argparse: `sweep [--dry-run]`), invoked via
`python -m rag.freshness sweep`.

### 2. `RagConfig` additions (`rag/config.py`)

```python
freshness_enabled: bool = True
staleness_downgrade: bool = True
default_ttl_days: int = 180
ttl_days_by_type: dict = field(default_factory=lambda: {
    "web": 30, "pdf": 365, "docx": 365, "csv": 180, "excel": 180, "text": 365})
```

`from_env` reads `FRESHNESS_ENABLED` / `DEFAULT_TTL_DAYS` overrides (ttl map stays code-default).

### 3. Ingest capture (`rag/loaders.py`, `rag/ingest.py`, `rag/types.py`)

- **Loaders** add to `RawDocument.metadata`: `source_fetched_at` (now, at fetch),
  `source_last_modified` (WebLoader: `resp.headers.get("Last-Modified")`; file loaders:
  `datetime.utcfromtimestamp(os.path.getmtime(path))`), `etag` (WebLoader:
  `resp.headers.get("ETag")`). Existing loaders already set `ingested_at` — keep.
- **`RagIngestor._to_record`** builds `ChunkRecord.freshness`:
  `{ingested_at, source_fetched_at, source_last_modified, etag, ttl_days}` where `ttl_days`
  is resolved from `config.ttl_days_by_type.get(source_type, config.default_ttl_days)`.

### 4. Surface in retrieval (`rag/stores/chroma.py`, `rag/types.py`)

- `SearchResult` gains `freshness: dict = field(default_factory=dict)`.
- `ChromaVectorStore.query` decodes `__freshness_json` (currently stripped as internal) into
  `SearchResult.freshness`. `_public_meta` still strips it from `.metadata`.

### 5. Confidence downgrade + note (`rag/orchestrator.py`, `rag/types.py`)

- `OrchestratorResult` gains `staleness_note: str = ""`.
- New pure static: `_apply_staleness(conf, results, config, now) -> (Confidence, str)`.
  When `config.freshness_enabled and config.staleness_downgrade` and the top-ranked result
  `is_stale`: drop one level (HIGH→MEDIUM→LOW→LOW floor; NONE unreached — already returned),
  and return a note from `freshness_note`. Otherwise unchanged.
- Called in both `process` and `process_stream` **after** `_assess_confidence`, **before**
  the NONE early-return check is irrelevant (NONE already returned). Applied only on the
  answered path. The `CONFIDENCE_NOTES` fed to the system prompt use the downgraded level.
- **Parity guarantee:** with `freshness_enabled=False`, or chunks lacking freshness data
  (age `None`), `_apply_staleness` is a no-op → existing confidence tests unaffected.

### 6. Active recheck (`rag/ingest.py`, `rag/loaders.py`)

- `RagIngestor.refresh_source(source_key) -> dict`: only for re-fetchable sources
  (`http(s)://` or existing file path). Pasted text (`text:…`) → `{status: "unrefreshable"}`.
- WebLoader conditional GET: if a stored `etag` exists, send `If-None-Match`; `304 Not
  Modified` → `{status: "unchanged"}` without re-embedding. Otherwise normal fetch →
  `_process` (existing `content_hash` skip still prevents needless re-embed).

### 7. Sweep (`rag/pipeline.py`, `rag/freshness.py`)

- `RagPipeline.sweep_stale(now=None, dry_run=False) -> dict`: for each source from
  `list_sources()`, read one chunk's freshness (via store), test `is_stale`; collect stale
  sources; if not `dry_run`, call `refresh_source` on each re-fetchable one. Returns
  `{checked, stale, refreshed:[...], skipped:[...]}`.
- `RagPipeline.refresh_source(key)` passthrough to ingestor.
- CLI `python -m rag.freshness sweep [--dry-run]` builds `RagPipeline(RagConfig.from_env())`,
  runs `sweep_stale`, prints the report.

### 8. UI (`app.py`)

Citations show age ("· 42d old"); if `result.staleness_note`, render it under the
confidence badge. Small, additive.

## Data Flow

```
ingest:  loader (fetched_at/last_modified/etag) → RagIngestor._to_record
         → ChunkRecord.freshness{...,ttl_days} → Chroma __freshness_json

query:   store.query → SearchResult.freshness → Orchestrator._assess_confidence
         → _apply_staleness (downgrade + note) → generate → OrchestratorResult.staleness_note

sweep:   CLI → pipeline.sweep_stale → per source: is_stale? → refresh_source
         → WebLoader conditional GET (304 skip) / re-ingest if changed
```

## Testing

- **Pure scoring** (`test_freshness.py`): age from fetched_at vs ingested_at; missing
  timestamp → None → not stale; ttl resolution by type + fallback; is_stale boundary
  (age == ttl, age > ttl); freshness_note wording.
- **Confidence downgrade** (`test_orchestrator.py` additions): stale top result → HIGH→MEDIUM
  + note set; `freshness_enabled=False` → no downgrade (parity); no freshness data → no
  downgrade; each ladder step downgrades correctly; note absent when fresh.
- **Ingest capture** (`test_ingestor.py` additions): `ttl_days` resolved by source_type;
  `ingested_at`/`fetched_at` populated; fake loader metadata → freshness dict.
- **Store surfacing** (`test_chroma_store.py` additions): freshness roundtrips into
  `SearchResult.freshness`; absent on old chunks → `{}`.
- **Sweep** (`test_freshness.py`): with fakes, stale source → listed + refresh called;
  fresh → skipped; `text:` source → unrefreshable; `dry_run` → no refresh calls.
- **Active recheck**: 304 path → status unchanged (fake session); changed content → re-ingest.
- Full suite stays green; parity tests unchanged.

## Risks / Notes

- `_apply_staleness` must be gated so it is a strict no-op under the foundation's existing
  data (no freshness dict) — protects "behavior unchanged" for pre-existing chunks.
- Conditional-GET etag path needs the WebLoader to accept a stored etag — small signature
  addition; keep normal `load(url)` working.
- Sweep reads one chunk per source for freshness; assumes chunks from one source share a
  fetch time (true — set once per ingest).
