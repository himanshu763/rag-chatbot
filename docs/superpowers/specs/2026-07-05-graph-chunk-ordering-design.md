# Graph-Based Chunk Ordering — Design Spec

**Date:** 2026-07-05
**Status:** Approved, ready for implementation plan
**Depends on:** foundation slice (rag/ library, `ChunkRecord.entities`/`edges` scaffold), freshness feature (independent).

## Context

Retrieval today ranks chunks purely by hybrid score (dense + BM25 + RRF + rerank). It has no
notion of how chunks relate — a chunk that answers the query well but whose *neighbor* holds
the crucial follow-up detail won't surface the neighbor unless it independently scores high.
And context is assembled in score order, which can read as disconnected fragments.

This feature adds a **chunk graph** with two mechanisms (both confirmed in scope):

1. **Expand** — from the hybrid seed set, pull 1–2 hop graph neighbors into the candidate set
   before rerank (recall: related-but-not-lexically-similar chunks, including cross-document).
2. **Order** — assemble the final context in reading order (grouped by source, ordered within
   source) instead of pure score-sort (coherence).

Edges come from **structure + keyword-overlap entities** (no LLM, no new heavy deps).
Entities are extracted once at ingest and persisted in the scaffolded `ChunkRecord.entities`;
the graph itself is built in-memory at query time (the existing `BM25Retriever` pattern),
avoiding cross-document edge-update complexity.

**Opt-in:** `graph_enabled=False` by default — parity with the foundation is the default.

## Non-Goals

- No LLM / spaCy NER entity extraction (pluggable `EntityExtractor` leaves the door open).
- No persisted edges / graph DB (Neo4j). In-memory only — Tier-1.
- No community detection / topic clustering / graph visualization.

## Components

### 1. `EntityExtractor` protocol + `KeywordEntityExtractor` (`rag/interfaces.py`, `rag/entities.py`)

```python
@runtime_checkable
class EntityExtractor(Protocol):
    def extract(self, text: str) -> list[str]: ...
```

`KeywordEntityExtractor.extract(text)` — regex-based: capitalized multi-word proper-noun
phrases (e.g. `Zephyr X1`, `Fusion ERP`) + standalone capitalized tokens length ≥ 3, minus a
small stopword set (sentence-initial common words). Returns **normalized lowercase, deduped**
strings. No dependencies. Deterministic (testable).

### 2. Schema additions (`rag/types.py`)

- `ChunkRecord.ordinal: int = 0` — position of the chunk within its source (0-based, ingest
  order). Drives adjacency edges **and** reading-order assembly.
- `SearchResult.ordinal: int = 0` — surfaced from the store so the orchestrator can order.
- `ChunkRecord.entities` already exists (scaffold) — now populated.

### 3. Store persistence (`rag/stores/chroma.py`)

- `_to_meta` writes `ordinal` (int scalar). `_from_meta` reads it back (default 0).
- `query()` sets `SearchResult.ordinal` from metadata.
- `entities` already round-trips via `__entities_json` (foundation).

### 4. Ingest (`rag/ingest.py`)

- `RagIngestor(__init__)` gains `entity_extractor` (default `KeywordEntityExtractor()`).
- `_process`: assign each chunk a 0-based `ordinal` in chunk order before building records.
- `_to_record`: `entities = self.entity_extractor.extract(chunk.text)`; set `ordinal`.

### 5. `GraphIndex` (`rag/graph/index.py`) — in-memory, query-time

`build(records)` builds an adjacency map `chunk_id -> set[chunk_id]`:
- **Adjacency:** within each source, records sorted by `ordinal`; consecutive pairs linked.
- **Section:** chunks sharing `(source, section_path)` (non-empty section) linked.
- **Sibling:** chunks sharing identical non-empty `parent_text` linked.
- **Entity:** invert `entities` → `entity -> [chunk_ids]`; chunks sharing ≥1 entity linked
  (cross-document included). Guard against huge fan-out: skip entities appearing in more than
  `max_entity_fanout` (default 50) chunks (stopword-like terms).

Methods: `neighbors(chunk_id, hops) -> set[str]` (BFS, excludes the seed), `expand(seed_ids,
hops) -> set[str]` (union of neighbors minus seeds), `get_record(chunk_id) -> ChunkRecord|None`.

### 6. `GraphExpander` (`rag/graph/expander.py`)

`GraphExpander(store, index, config)` — owns store + `GraphIndex`; `_ensure_current()` rebuilds
the index when `store.count()` changes (same lazy-sync as `BM25Retriever`).
`expand(results) -> list[SearchResult]`: BFS-expand the result `chunk_id`s by `config.graph_hops`,
append each new neighbor as a `SearchResult` (text/metadata/ordinal/freshness from its
`ChunkRecord`) with `score = config.graph_neighbor_score` (dampened so rerank, not raw score,
decides their fate). Returns original results + neighbor results (deduped by id).

### 7. `HybridRetriever` graph stage (`rag/retrievers/hybrid.py`)

Add optional `graph: GraphExpander | None = None`. New order:
`vec + bm25 → RRF fuse → (graph.expand if graph) → rerank → final_k`. RRF-score restore after
rerank unchanged (neighbors keep their dampened score, which is fine — confidence still reads
the top result's restored/own score). When `graph is None`, behavior is identical to today.

### 8. Reading-order context (`rag/orchestrator.py`)

`_assemble_context(results, config)`: when `config.graph_enabled and config.graph_context_ordering`,
group results by `metadata["source"]`, order each group by `ordinal` ascending, order groups by
the group's best score descending, then flatten — and apply the existing token-budget logic to
that ordering. Otherwise the current score-sort path is used verbatim. `Source N` labels and
parent/child text selection are unchanged.

### 9. Config (`rag/config.py`)

```python
graph_enabled: bool = False           # opt-in; False = foundation parity
graph_hops: int = 1
graph_neighbor_score: float = 0.15
graph_context_ordering: bool = True   # only consulted when graph_enabled
max_entity_fanout: int = 50
```
`from_env` reads `GRAPH_ENABLED` / `GRAPH_HOPS`.

### 10. Pipeline wiring (`rag/pipeline.py`)

- `RagPipeline(__init__)` gains `entity_extractor=None` (default `KeywordEntityExtractor()`),
  passed to `RagIngestor`.
- When `config.graph_enabled`, build `GraphExpander(store, GraphIndex(config), config)` and pass
  it into `HybridRetriever(..., graph=expander)`. When disabled, `graph=None` (no expander) and
  the orchestrator's context ordering is off → **identical to today**.

## Data Flow

```
ingest:  chunk → KeywordEntityExtractor.extract → ChunkRecord.entities
         + ordinal → Chroma (__entities_json, ordinal)

query (graph on):
  hybrid: vec+bm25 → RRF fuse → GraphExpander.expand (1-2 hop neighbors, dampened)
          → rerank → final_k
  orchestrate: _assess_confidence → _apply_staleness → _assemble_context
               (group by source, order by ordinal) → generate
```

## Testing

- **Entity extraction** (`test_entities.py`): proper-noun phrases extracted + normalized;
  stopwords/lowercase words excluded; deterministic; empty text → `[]`.
- **GraphIndex** (`test_graph_index.py`): adjacency from ordinal; section/sibling edges; entity
  edges incl. cross-document; `max_entity_fanout` skips over-common entities; `neighbors` hop
  counts (1 vs 2); seed excluded from its own neighbors.
- **GraphExpander** (`test_graph_expander.py`, fakes): expands seeds → neighbor SearchResults
  with dampened score; rebuilds on store count-change; two expanders independent (no global).
- **Hybrid graph stage** (`test_rerank_hybrid.py` additions): with graph, neighbor chunks appear
  in the candidate set before rerank; `graph=None` → identical output to today.
- **Reading-order context** (`test_orchestrator.py` additions): graph on → chunks grouped by
  source and ordered by ordinal; graph off → score-order (parity, existing tests unchanged).
- **Ingest** (`test_ingestor.py` additions): entities populated; ordinals 0..n-1 in order.
- **Store** (`test_chroma_store.py` additions): ordinal round-trips into SearchResult.
- **Pipeline e2e** (`test_pipeline.py` additions): `graph_enabled=True` pipeline ingests +
  queries end-to-end (fakes); `graph_enabled=False` unchanged.
- Full suite stays green; 111 existing tests unaffected (opt-in default).

## Risks / Notes

- **Parity is load-bearing:** default `graph_enabled=False` → no expander, score-order context.
  Every existing test must stay green with zero changes. Verify explicitly.
- **Fan-out:** common keyword "entities" (e.g. "Overview") could link everything; `max_entity_fanout`
  caps it. Tune default if recall/precision suffers.
- **get_all() cost:** GraphIndex build pulls the whole corpus (like BM25) — Tier-1 only.
- **Ordinal on legacy chunks:** existing v2/v3 chunks lack `ordinal` → default 0. With graph off
  (default) this is irrelevant; if a user enables graph on a pre-existing store, adjacency/ordering
  degrade gracefully (all ordinal 0 → no adjacency, group order stable) until re-ingest.
