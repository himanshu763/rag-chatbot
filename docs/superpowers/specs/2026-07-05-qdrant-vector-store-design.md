# Tier-2: QdrantVectorStore — Design Spec

**Date:** 2026-07-05
**Status:** Approved, ready for implementation plan
**Depends on:** foundation slice (`VectorStore` protocol, `ChunkRecord`/`SearchResult`). First slice of the Tier-2 (distributed scale) roadmap item.

## Context

The library's only `VectorStore` is `ChromaVectorStore` — single-node, single-process, and its
BM25/scan paths load the whole corpus into RAM. To reach beyond low millions the vector store
must become a distributed backend. This slice adds a **Qdrant adapter** conforming to the
existing `VectorStore` protocol, drop-in via config.

Qdrant was chosen because `qdrant-client` runs **in-memory with no server** (`:memory:`), so the
adapter is fully testable in this environment, while the same code points at a real Qdrant
cluster (URL) for production scale (sharding, quantization, millions+). Verified working:
in-memory create/upsert/query/scroll/filter/set_payload/delete, native list & dict payloads,
cosine `score` = similarity (same scale as Chroma's `1 - distance`, so confidence thresholds
transfer unchanged).

**Opt-in:** `vector_store="chroma"` by default — parity is the default.

## Non-Goals

- No OpenSearch/ES keyword index (needs a live cluster; separate later slice).
- No Redis cache backend, no Milvus (separate slices).
- No migration tooling from Chroma→Qdrant (re-ingest is the path).

## Components

### 1. `QdrantVectorStore` (`rag/stores/qdrant.py`)

`QdrantVectorStore(config, client=None)`. Client built from config: `QdrantClient(location=
config.qdrant_location)` (default `":memory:"`) unless `config.qdrant_url` is set, then
`QdrantClient(url=config.qdrant_url)`. Collection name = `config.collection_name`.

**Lazy collection creation:** the collection is created on the first `add`, sized to
`len(records[0].embedding)`, cosine distance. (Mirrors Chroma's dim-inference; avoids coupling
to `config.embedding_dim` and lets tests use small vectors.) Read methods on a missing
collection return empty/zero.

Point identity: Qdrant requires int/UUID ids, but `chunk_id` is a 12-char hex string. Point id =
`str(uuid5(NAMESPACE_URL, chunk_id))` (deterministic); `chunk_id` is stored in the payload as the
source of truth and read back from there.

Payload (native — no JSON encoding, unlike Chroma):
`{chunk_id, raw_text, source, source_type, title, anchor, section_path, content_hash,
source_hash, schema_version, ordinal, parent_text, entities(list), edges(list), freshness(dict)}`.

Methods (all `VectorStore` protocol):
- `add(records)` — ensure collection; `upsert(points=[PointStruct(id, vector, payload)])`.
- `query(embedding, top_k, where=None)` — `query_points(query=embedding, limit=top_k,
  query_filter=_filter(where)).points` → `SearchResult(chunk_id=payload["chunk_id"],
  text=payload["raw_text"], score=hit.score, metadata={source, source_type, title, anchor,
  section_title=section_path}, parent_text, freshness, ordinal)`.
- `get_all()` — `scroll(with_payload=True, with_vectors=True)` paginated (follow `next_offset`)
  → `ChunkRecord`s (rebuild from payload + vector).
- `delete_source(source_key)` — `delete(points_selector=FilterSelector(filter=_source(source_key)))`.
- `count()` — `count(collection).count` (0 if collection missing).
- `list_sources()` — scroll all payloads, group by `source` → `{source, title, type, chunks}`.
- `get_source_hash(source_key)` — scroll with `_source` filter, first payload's `source_hash`.
- `get_source_freshness(source_key)` — same, first payload's `freshness`.
- `touch_source(source_key, fetched_at)` — scroll ids with `_source` filter; for each, merge
  `source_fetched_at=fetched_at` into its `freshness` and `set_payload` (metadata-only, no re-embed).

Helpers: `_filter(where)` translates the Chroma-style `{"field": {"$eq": v}}` / `{"field": v}`
dict into `models.Filter(must=[FieldCondition(key, match=MatchValue(value))])` (None → no filter);
`_source(key)` = filter on `source == key`.

### 2. Config (`rag/config.py`)

```python
vector_store: str = "chroma"          # "chroma" | "qdrant"
qdrant_location: str = ":memory:"     # ":memory:" | local path
qdrant_url: str = ""                  # non-empty → connect to a real cluster (wins over location)
```
`from_env` reads `VECTOR_STORE`, `QDRANT_URL`, `QDRANT_LOCATION`.

### 3. Pipeline (`rag/pipeline.py`)

`_default_store()`:
```python
if self.config.vector_store == "qdrant":
    from rag.stores.qdrant import QdrantVectorStore
    return QdrantVectorStore(self.config)
from rag.stores.chroma import ChromaVectorStore
return ChromaVectorStore(self.config)
```
Default `"chroma"` → unchanged.

### 4. Shared store-contract tests (`tests/test_store_contract.py`)

A parametrized suite (pytest `params`) over three store factories — `ChromaVectorStore`
(tmp dir), `QdrantVectorStore` (`:memory:`), `FakeVectorStore` — asserting identical behavior:
- add + `count`; `query` returns nearest first (orthogonal unit vectors → deterministic order).
- `ordinal`, `freshness`, `entities` round-trip through `get_all`.
- `delete_source` removes only that source; `list_sources` groups + counts.
- `get_source_hash` / `get_source_freshness` return stored values; missing → `None`/`{}`.
- `touch_source` bumps `source_fetched_at`.
- two instances / collections are isolated.

`pyproject.toml`: add `qdrant = ["qdrant-client>=1.7.0"]` extra. Qdrant tests
`pytest.importorskip("qdrant_client")`.

## Data Flow

```
ingest → store.add(records) → Qdrant upsert (uuid5 id, native payload)
query  → embed → store.query → query_points(cosine) → SearchResult(score=similarity)
config vector_store="qdrant" → pipeline builds QdrantVectorStore; else ChromaVectorStore
```

## Testing

- **QdrantVectorStore** covered by the shared contract suite (in-memory) + a couple of
  Qdrant-specific tests: uuid5 point-id mapping keeps `chunk_id` retrievable; `query` respects a
  `where` source filter; read methods on an empty/uncreated collection return `[]`/`0`.
- **Contract suite** runs green for all three backends → proves the drop-in claim.
- **Config** — defaults + env.
- **Pipeline** — `vector_store="chroma"` builds Chroma (default parity); with a construction
  hook, `"qdrant"` builds `QdrantVectorStore` (assert type; no live server needed via `:memory:`).
- Full suite stays green; 165 existing tests unaffected (opt-in default).

## Risks / Notes

- **`:memory:` is ephemeral** — dev/test only. Production sets `QDRANT_URL` (or a local path).
  Documented; the default keeps tests hermetic.
- **Score scale:** Qdrant cosine = similarity, matching Chroma's `1 - distance`. Confidence
  thresholds (0.25/0.40) transfer without change. Verified by the contract suite's ordering test;
  exact threshold parity depends on identical embeddings (same embedder) — true in practice.
- **Lazy dim:** first `add` fixes the collection's vector size. Mixed-dim adds (different embedder
  mid-corpus) error — same practical constraint as Chroma. Acceptable.
- **get_all() scroll cost:** still pulls the whole corpus for BM25/graph index builds — that's a
  KeywordIndex/graph concern (in-memory Tier-1 helpers), not this store. A future OpenSearch
  keyword index removes that path. Out of scope here.
- **Single-collection multi-tenancy:** one collection per `collection_name`, same as Chroma.
