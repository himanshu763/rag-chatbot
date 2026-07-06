# CAG — Cache-Augmented Generation — Design Spec

**Date:** 2026-07-05
**Status:** Approved, ready for implementation plan
**Depends on:** foundation slice (rag/ library, RagPipeline facade). Independent of freshness/graph.

## Context

Every query runs the full pipeline — embed, hybrid retrieve, rerank, LLM generate — even when
the same or a near-identical question was just asked. This feature adds an **opt-in response
cache** that returns a stored answer on repeat/paraphrased queries, skipping retrieval and
generation entirely.

Two match layers: **exact** (hash) and **semantic** (query-embedding cosine). Correctness comes
from a **corpus version** baked into the cache key — any ingest/delete flips it, so a cached
answer from a superseded knowledge base is never served. A TTL is the backstop.

**Opt-in:** `cache_enabled=False` by default — parity with today is the default.

## Non-Goals

- No Redis/SQLite backend now (pluggable `CacheStore` protocol leaves the door open).
- No provider prompt/prefix caching (OpenAI auto-caches long prefixes; marginal build value).
- No per-source invalidation (corpus-version is coarse but correct); no distributed cache.

## Components

### 1. `CacheStore` protocol + `InMemoryCacheStore` (`rag/interfaces.py`, `rag/cache.py`)

```python
@runtime_checkable
class CacheStore(Protocol):
    def get(self, key: str) -> object | None: ...
    def set(self, key: str, value: object, ttl_seconds: float | None = None) -> None: ...
    def items(self) -> list[tuple[str, object]]: ...      # non-expired entries, for semantic scan
    def clear(self) -> None: ...
```

`InMemoryCacheStore(max_entries)` — dict of `key -> (value, expires_at)`. `get`/`items` drop
expired entries lazily (compare `time.monotonic()`). On `set` past `max_entries`, evict the
oldest-inserted key (insertion-ordered dict). `ttl_seconds=None` → no expiry.

### 2. `SemanticCache` (`rag/cache.py`)

Composes `(store: CacheStore, embedder: Embedder, config: RagConfig)`.

Entry stored as a dict: `{"result": OrchestratorResult, "embedding": list[float],
"corpus_version": str, "history_fp": str}`.

- `_exact_key(query, corpus_version, history_fp) -> str` = `sha256` of the three joined.
- `_history_fp(history) -> str` = `sha256` of the concatenated role+content of history (empty
  history → a fixed constant hash).
- `lookup(query, corpus_version, history) -> OrchestratorResult | None`:
  1. exact: `store.get(_exact_key(...))` → return its `result` if present.
  2. semantic: embed query once; over `store.items()`, consider entries whose `corpus_version`
     and `history_fp` match the current ones; return the `result` of the highest cosine ≥
     `config.cache_semantic_threshold`. None if no candidate qualifies.
- `store_result(query, corpus_version, history, result) -> None`: embed query, build the entry,
  `store.set(_exact_key(...), entry, ttl_seconds=config.cache_ttl_seconds)`.

Cosine is a local helper (same formula as elsewhere). Embedding failures degrade to exact-only
(caught, logged) — never raise into the query path.

### 3. Invalidation — corpus version (`rag/pipeline.py`)

`RagPipeline` holds `self._ingest_epoch = 0`, incremented whenever a content change occurs:
`ingest_url`/`ingest_file`/`ingest_text` returning `status == "ok"`, or `delete_source`.
`_corpus_version() -> str` returns `f"{self.store.count()}:{self._ingest_epoch}"`. Baked into
every cache key → any KB change → all prior entries miss. (`status == "skipped"` does not bump —
content unchanged, cached answers still valid.)

### 4. Pipeline wiring (`rag/pipeline.py`)

- `__init__` gains `cache=None` (keyword). When `config.cache_enabled` and no cache injected,
  build `SemanticCache(InMemoryCacheStore(config.cache_max_entries), self.embedder, config)`.
  When disabled → `self.cache = None`.
- `query(text, history)`:
  - cache off → `orchestrator.process(text, history)` (unchanged).
  - cache on → `v = _corpus_version()`; `hit = cache.lookup(text, v, history or [])`; return hit
    if not None; else `result = orchestrator.process(...)`; `cache.store_result(text, v, history
    or [], result)`; return result.
- `stream(text, history)`:
  - cache off → `yield from orchestrator.process_stream(...)` (unchanged).
  - cache on → `v = _corpus_version()`; `hit = cache.lookup(...)`; if hit: `yield hit.response;
    yield hit; return`. Else iterate `process_stream`, passing chunks through, capture the final
    `OrchestratorResult`, `store_result(...)` it, then `yield` it.
- Ingest wrappers bump `_ingest_epoch` per §3.

### 5. Config (`rag/config.py`)

```python
cache_enabled: bool = False
cache_ttl_seconds: float = 3600.0
cache_semantic_threshold: float = 0.92
cache_max_entries: int = 1000
```
`from_env` reads `CACHE_ENABLED` / `CACHE_TTL_SECONDS`.

## Data Flow

```
query (cache on):
  v = f"{store.count()}:{ingest_epoch}"
  exact_key = hash(query + v + history_fp)
  store.get(exact_key)  → HIT → cached OrchestratorResult
        │ miss
  embed(query) → scan entries (same v + history_fp) → cosine ≥ threshold → HIT
        │ miss
  orchestrator.process → store_result(exact_key → entry, ttl) → result

ingest ok / delete → ingest_epoch += 1 → next query's v differs → all prior entries miss
```

## Testing

- **InMemoryCacheStore** (`test_cache.py`): set/get; TTL expiry (monotonic-clock, injected or
  small sleep-free via ttl=0 semantics); `max_entries` evicts oldest; `items()` omits expired;
  `clear()`.
- **SemanticCache** (`test_cache.py`, FakeEmbedder): exact hit; miss→store→exact hit; paraphrase
  → semantic hit above threshold; dissimilar query → miss; corpus_version mismatch → miss;
  different history_fp → miss; embedder raising → falls back to exact-only, no crash.
- **Pipeline** (`test_pipeline.py`, fakes): `cache_enabled=True` → identical repeat query returns
  cached result and calls the generator only once; after `ingest_text` (ok) → epoch bumps →
  repeat query is a miss (generator called again); `stream` hit replays text + final result;
  `cache_enabled=False` → `pipeline.cache is None`, generator called every time (parity).
- **Config** (`test_config.py`): defaults; env overrides.
- Full suite stays green; 142 existing tests unaffected (opt-in default).

## Risks / Notes

- **Parity is load-bearing:** default off → no cache built → `query`/`stream` unchanged. Verify.
- **History correctness:** key + semantic filter include a history fingerprint, so a follow-up
  with different conversation context never matches a first-turn entry. Most cache value is
  first-turn/FAQ queries; deep-history turns rarely repeat and simply miss — acceptable.
- **Double-embed on miss:** semantic lookup embeds the query, then retrieval embeds again on the
  miss path. Minor; exact hits are free. Not optimized now.
- **Fallback/NONE results are cached** — corpus-version keeps them correct; a repeat out-of-scope
  query returns instantly.
- **Single-instance only:** `_ingest_epoch` is per-pipeline; external ingests by another process
  aren't seen. Fine for Tier-1; Redis backend + shared version is the Tier-2 path.
