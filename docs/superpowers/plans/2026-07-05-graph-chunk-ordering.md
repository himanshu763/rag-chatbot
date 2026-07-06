# Graph-Based Chunk Ordering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a chunk graph (structural + keyword-entity edges, built in-memory at query) that expands hybrid seeds via graph neighbors and assembles context in reading order — both opt-in via `graph_enabled`.

**Architecture:** Entities extracted once at ingest into `ChunkRecord.entities` + an `ordinal` position. At query, `GraphExpander` (BM25Retriever-style lazy sync) builds a `GraphIndex` from the corpus and expands the fused candidate set before rerank inside `HybridRetriever`. The orchestrator assembles context grouped-by-source, ordered-by-ordinal. `graph_enabled=False` (default) makes every path identical to today.

**Tech Stack:** Python 3.12, pytest, ChromaDB, existing `rag/` library. Runner: `.venv-wsl/bin/python -m pytest`. Commits skipped (user runs them).

**Spec:** `docs/superpowers/specs/2026-07-05-graph-chunk-ordering-design.md`

---

## File Structure

- `rag/entities.py` — **new**. `KeywordEntityExtractor`.
- `rag/graph/__init__.py`, `rag/graph/index.py`, `rag/graph/expander.py` — **new**. `GraphIndex`, `GraphExpander`.
- `rag/interfaces.py` — **modify**. `EntityExtractor` protocol.
- `rag/types.py` — **modify**. `ChunkRecord.ordinal`, `SearchResult.ordinal`.
- `rag/config.py` — **modify**. graph config fields + env.
- `rag/stores/chroma.py` — **modify**. persist/read `ordinal`; surface in query.
- `tests/fakes.py` — **modify**. `FakeVectorStore.query` surfaces `ordinal`.
- `rag/ingest.py` — **modify**. entity extraction + ordinal at ingest.
- `rag/retrievers/hybrid.py` — **modify**. optional graph stage.
- `rag/orchestrator.py` — **modify**. reading-order context assembly.
- `rag/pipeline.py` — **modify**. wire extractor + expander when enabled.
- Tests: `test_entities.py`, `test_graph_index.py`, `test_graph_expander.py` (new); additions to `test_config.py`, `test_chroma_store.py`, `test_ingestor.py`, `test_rerank_hybrid.py`, `test_orchestrator.py`, `test_pipeline.py`.

Convention: `PY=.venv-wsl/bin/python`. No git commits (user's choice).

---

## Task 1: Graph config fields

**Files:** Modify `rag/config.py`; Test `tests/test_config.py`

- [ ] **Step 1: Failing test** — append to `tests/test_config.py`:

```python
def test_graph_config_defaults():
    cfg = RagConfig()
    assert cfg.graph_enabled is False           # opt-in
    assert cfg.graph_hops == 1
    assert cfg.graph_neighbor_score == 0.15
    assert cfg.graph_context_ordering is True
    assert cfg.max_entity_fanout == 50


def test_graph_config_from_env(monkeypatch):
    monkeypatch.setenv("GRAPH_ENABLED", "true")
    monkeypatch.setenv("GRAPH_HOPS", "2")
    cfg = RagConfig.from_env()
    assert cfg.graph_enabled is True
    assert cfg.graph_hops == 2
```

- [ ] **Step 2: Run — expect FAIL** (`AttributeError: graph_enabled`)

Run: `.venv-wsl/bin/python -m pytest tests/test_config.py -q`

- [ ] **Step 3: Implement** — in `rag/config.py`, after the freshness fields:

```python
    # Graph-based retrieval
    graph_enabled: bool = False
    graph_hops: int = 1
    graph_neighbor_score: float = 0.15
    graph_context_ordering: bool = True
    max_entity_fanout: int = 50
```

And in `from_env`'s `env = {...}`:

```python
            "graph_enabled": os.getenv("GRAPH_ENABLED", "false").lower() == "true",
            "graph_hops": int(os.getenv("GRAPH_HOPS", "1")),
```

- [ ] **Step 4: Run — expect PASS**

Run: `.venv-wsl/bin/python -m pytest tests/test_config.py -q`

---

## Task 2: KeywordEntityExtractor + EntityExtractor protocol

**Files:** Create `rag/entities.py`; Modify `rag/interfaces.py`; Test `tests/test_entities.py`

- [ ] **Step 1: Failing test** — create `tests/test_entities.py`:

```python
from rag.entities import KeywordEntityExtractor
from rag.interfaces import EntityExtractor


def test_extractor_conforms_to_protocol():
    assert isinstance(KeywordEntityExtractor(), EntityExtractor)


def test_extracts_proper_noun_phrases_normalized():
    ents = KeywordEntityExtractor().extract("The Zephyr X1 turbine uses Fusion ERP Analytics.")
    assert "zephyr x1" in ents
    assert "fusion erp analytics" in ents


def test_excludes_common_lowercase_words():
    ents = KeywordEntityExtractor().extract("the turbine produces power at rated speed")
    assert ents == []


def test_deduplicates_and_is_deterministic():
    e = KeywordEntityExtractor()
    a = e.extract("Zephyr X1 and Zephyr X1 again")
    assert a == e.extract("Zephyr X1 and Zephyr X1 again")
    assert a.count("zephyr x1") == 1


def test_empty_text():
    assert KeywordEntityExtractor().extract("") == []


def test_drops_sentence_initial_common_word():
    # "The" starts a sentence but is a stopword — not an entity
    ents = KeywordEntityExtractor().extract("The system is ready. Fusion ERP works.")
    assert "the" not in ents
    assert "fusion erp" in ents
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: rag.entities`)

Run: `.venv-wsl/bin/python -m pytest tests/test_entities.py -q`

- [ ] **Step 3: Implement**

In `rag/interfaces.py`, add after the `Embedder` protocol:

```python
@runtime_checkable
class EntityExtractor(Protocol):
    def extract(self, text: str) -> list[str]: ...
```

Create `rag/entities.py`:

```python
"""KeywordEntityExtractor — dependency-free proper-noun / capitalized-phrase extraction.

Returns normalized (lowercase, deduped, order-preserving) entity strings used to build
shared-entity graph edges. Swap for an NER/LLM extractor via the EntityExtractor protocol.
"""
from __future__ import annotations

import re

# Runs of Capitalized tokens (proper-noun phrases), allowing digits/hyphens: "Zephyr X1".
_PHRASE = re.compile(r"\b([A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*)*)\b")

# Common capitalized-but-not-entity words (sentence starters, etc.).
_STOP = {
    "the", "a", "an", "this", "that", "these", "those", "it", "he", "she", "they", "we",
    "i", "you", "and", "or", "but", "if", "then", "for", "to", "of", "in", "on", "at",
    "as", "by", "is", "are", "was", "were", "be", "each", "some", "any", "all", "no",
    "not", "when", "where", "how", "why", "what", "which", "who",
}


class KeywordEntityExtractor:
    def extract(self, text: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for m in _PHRASE.finditer(text or ""):
            phrase = m.group(1).strip()
            norm = phrase.lower()
            # single common word (e.g. sentence-initial "The") → skip
            if " " not in norm and norm in _STOP:
                continue
            # multi-word: strip leading stopword tokens ("The Fusion ERP" → "fusion erp")
            tokens = norm.split()
            while len(tokens) > 1 and tokens[0] in _STOP:
                tokens = tokens[1:]
            norm = " ".join(tokens)
            if not norm or (len(tokens) == 1 and tokens[0] in _STOP):
                continue
            if len(norm) < 3:
                continue
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
        return out
```

- [ ] **Step 4: Run — expect PASS**

Run: `.venv-wsl/bin/python -m pytest tests/test_entities.py -q`

---

## Task 3: Schema — ChunkRecord.ordinal + SearchResult.ordinal

**Files:** Modify `rag/types.py`; Test covered by Tasks 4 & 6 (no standalone test — trivial field additions verified downstream)

- [ ] **Step 1: Implement** — in `rag/types.py`:

`ChunkRecord` — add after `schema_version`:

```python
    ordinal: int = 0                       # position within source (adjacency + reading order)
```

`SearchResult` — add after `freshness`:

```python
    ordinal: int = 0
```

- [ ] **Step 2: Verify import**

Run: `.venv-wsl/bin/python -c "from rag.types import ChunkRecord, SearchResult; print(ChunkRecord('a','x').ordinal, SearchResult('a','x',0.0).ordinal)"`
Expected: `0 0`

---

## Task 4: Chroma persists + surfaces ordinal

**Files:** Modify `rag/stores/chroma.py`, `tests/fakes.py`; Test `tests/test_chroma_store.py`

- [ ] **Step 1: Failing test** — append to `tests/test_chroma_store.py`:

```python
def test_ordinal_roundtrips(tmp_path):
    s = _store(tmp_path)
    rec = _rec("a", "x", "src1", [1.0, 0.0])
    rec.ordinal = 4
    s.add([rec])
    assert s.get_all()[0].ordinal == 4
    assert s.query([1.0, 0.0], top_k=1)[0].ordinal == 4
```

- [ ] **Step 2: Run — expect FAIL** (ordinal 0, not 4)

Run: `.venv-wsl/bin/python -m pytest tests/test_chroma_store.py::test_ordinal_roundtrips -q`

- [ ] **Step 3: Implement**

In `rag/stores/chroma.py` `_to_meta`, add to the returned dict (before the JSON scaffold keys):

```python
            "ordinal": r.ordinal,
```

In `_from_meta`, add to the `ChunkRecord(...)` constructor:

```python
            ordinal=int(meta.get("ordinal", 0)),
```

In `query()`, add to the `SearchResult(...)` constructor:

```python
                    ordinal=int(raw_meta.get("ordinal", 0)),
```

In `tests/fakes.py`, `FakeVectorStore.query` — add `ordinal=r.ordinal` to the `SearchResult(...)`
constructor (both are built from a `ChunkRecord r`). The query builds results from
`self._records.values()`; set `ordinal=r.ordinal`.

- [ ] **Step 4: Run — expect PASS**

Run: `.venv-wsl/bin/python -m pytest tests/test_chroma_store.py -q`

---

## Task 5: Ingest extracts entities + assigns ordinal

**Files:** Modify `rag/ingest.py`; Test `tests/test_ingestor.py`

- [ ] **Step 1: Failing test** — append to `tests/test_ingestor.py`:

```python
def test_ingest_populates_entities_and_ordinals():
    ing, store, _ = _ingestor()
    ing.ingest_text("The Zephyr X1 turbine is strong. Fusion ERP Analytics tracks it.", title="Doc")
    recs = sorted(store.get_all(), key=lambda r: r.ordinal)
    assert [r.ordinal for r in recs] == list(range(len(recs)))    # 0..n-1
    all_ents = {e for r in recs for e in r.entities}
    assert "zephyr x1" in all_ents or "fusion erp analytics" in all_ents
```

- [ ] **Step 2: Run — expect FAIL** (entities empty, ordinals all 0)

Run: `.venv-wsl/bin/python -m pytest tests/test_ingestor.py::test_ingest_populates_entities_and_ordinals -q`

- [ ] **Step 3: Implement** — in `rag/ingest.py`:

Add import at top:

```python
from rag.entities import KeywordEntityExtractor
from rag.interfaces import EntityExtractor
```

Change `__init__` signature + body to accept an extractor:

```python
    def __init__(self, store: VectorStore, embedder: Embedder,
                 config: RagConfig | None = None, chunker: Chunker | None = None,
                 entity_extractor: EntityExtractor | None = None):
        self.store = store
        self.embedder = embedder
        self.config = config or RagConfig()
        self.chunker = chunker or Chunker(self.config)
        self.entity_extractor = entity_extractor or KeywordEntityExtractor()
```

In `_process`, after `chunks = self.chunker.chunk(doc)` and the `if not chunks:` guard, pass the
ordinal through to `_to_record`. Change the record-building line to enumerate:

```python
        records = [self._to_record(c, enriched, emb, doc.content_hash, ordinal)
                   for ordinal, (c, enriched, emb) in enumerate(zip(chunks, embed_texts, embeddings))]
```

Change `_to_record` signature + set entities/ordinal:

```python
    def _to_record(self, chunk: Chunk, enriched: str, embedding: list[float],
                   source_hash: str, ordinal: int = 0) -> ChunkRecord:
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
            entities=self.entity_extractor.extract(chunk.text),
            ordinal=ordinal,
        )
```

- [ ] **Step 4: Run — expect PASS** (whole ingestor file)

Run: `.venv-wsl/bin/python -m pytest tests/test_ingestor.py -q`

---

## Task 6: GraphIndex

**Files:** Create `rag/graph/__init__.py`, `rag/graph/index.py`; Test `tests/test_graph_index.py`

- [ ] **Step 1: Failing test** — create `tests/test_graph_index.py`:

```python
from rag.config import RagConfig
from rag.graph.index import GraphIndex
from rag.types import ChunkRecord


def _rec(cid, source, ordinal, entities=(), section="", parent=""):
    return ChunkRecord(chunk_id=cid, raw_text=cid,
                       provenance={"source": source, "title": source},
                       section_path=section, parent_text=parent or None,
                       entities=list(entities), ordinal=ordinal)


def _index(records):
    idx = GraphIndex(RagConfig())
    idx.build(records)
    return idx


def test_adjacency_links_consecutive_ordinals_same_source():
    idx = _index([_rec("a", "s1", 0), _rec("b", "s1", 1), _rec("c", "s1", 2)])
    assert idx.neighbors("b", hops=1) == {"a", "c"}


def test_no_adjacency_across_sources():
    idx = _index([_rec("a", "s1", 0), _rec("b", "s2", 0)])
    assert idx.neighbors("a", hops=1) == set()


def test_section_edges_link_same_section():
    idx = _index([_rec("a", "s1", 0, section="Intro"), _rec("b", "s1", 5, section="Intro")])
    assert "b" in idx.neighbors("a", hops=1)


def test_entity_edges_link_cross_document():
    idx = _index([_rec("a", "s1", 0, entities=["zephyr x1"]),
                  _rec("b", "s2", 0, entities=["zephyr x1"])])
    assert idx.neighbors("a", hops=1) == {"b"}


def test_max_entity_fanout_skips_overcommon_entity():
    cfg = RagConfig(max_entity_fanout=2)
    recs = [_rec(f"c{i}", "s1", i, entities=["common"]) for i in range(5)]
    idx = GraphIndex(cfg); idx.build(recs)
    # "common" appears in 5 chunks > fanout 2 → no entity edges from it
    # (adjacency still links neighbors, so check a non-adjacent pair)
    assert "c4" not in idx.neighbors("c0", hops=1)


def test_two_hops_reach_further():
    idx = _index([_rec("a", "s1", 0), _rec("b", "s1", 1), _rec("c", "s1", 2)])
    assert idx.neighbors("a", hops=1) == {"b"}
    assert idx.neighbors("a", hops=2) == {"b", "c"}


def test_seed_excluded_from_its_neighbors():
    idx = _index([_rec("a", "s1", 0), _rec("b", "s1", 1)])
    assert "a" not in idx.neighbors("a", hops=2)


def test_expand_unions_neighbors_minus_seeds():
    idx = _index([_rec("a", "s1", 0), _rec("b", "s1", 1), _rec("c", "s1", 2)])
    assert idx.expand({"a", "b"}, hops=1) == {"c"}


def test_get_record():
    idx = _index([_rec("a", "s1", 0)])
    assert idx.get_record("a").chunk_id == "a"
    assert idx.get_record("missing") is None
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: rag.graph.index`)

Run: `.venv-wsl/bin/python -m pytest tests/test_graph_index.py -q`

- [ ] **Step 3: Implement**

Create `rag/graph/__init__.py`:

```python
from rag.graph.expander import GraphExpander
from rag.graph.index import GraphIndex

__all__ = ["GraphIndex", "GraphExpander"]
```

Create `rag/graph/index.py`:

```python
"""GraphIndex — in-memory chunk graph built from a corpus at query time.

Edges: adjacency (consecutive ordinal in a source), section (same source+section),
sibling (same parent_text), entity (shared entity, cross-document). Common entities
appearing in more than config.max_entity_fanout chunks are skipped.
"""
from __future__ import annotations

from collections import defaultdict

from rag.config import RagConfig
from rag.types import ChunkRecord


class GraphIndex:
    def __init__(self, config: RagConfig | None = None):
        self.config = config or RagConfig()
        self._adj: dict[str, set[str]] = defaultdict(set)
        self._records: dict[str, ChunkRecord] = {}

    def _link(self, a: str, b: str) -> None:
        if a != b:
            self._adj[a].add(b)
            self._adj[b].add(a)

    def build(self, records: list[ChunkRecord]) -> None:
        self._adj = defaultdict(set)
        self._records = {r.chunk_id: r for r in records}

        by_source: dict[str, list[ChunkRecord]] = defaultdict(list)
        by_section: dict[tuple, list[str]] = defaultdict(list)
        by_parent: dict[str, list[str]] = defaultdict(list)
        by_entity: dict[str, list[str]] = defaultdict(list)

        for r in records:
            src = r.provenance.get("source", "")
            by_source[src].append(r)
            if r.section_path:
                by_section[(src, r.section_path)].append(r.chunk_id)
            if r.parent_text:
                by_parent[r.parent_text].append(r.chunk_id)
            for e in r.entities:
                by_entity[e].append(r.chunk_id)

        # adjacency: consecutive ordinals within a source
        for recs in by_source.values():
            ordered = sorted(recs, key=lambda r: r.ordinal)
            for prev, cur in zip(ordered, ordered[1:]):
                self._link(prev.chunk_id, cur.chunk_id)

        # section + sibling cliques
        for ids in list(by_section.values()) + list(by_parent.values()):
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    self._link(ids[i], ids[j])

        # entity edges (skip over-common entities)
        for ids in by_entity.values():
            if len(ids) > self.config.max_entity_fanout:
                continue
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    self._link(ids[i], ids[j])

    def neighbors(self, chunk_id: str, hops: int = 1) -> set[str]:
        seen = {chunk_id}
        frontier = {chunk_id}
        for _ in range(max(0, hops)):
            nxt: set[str] = set()
            for node in frontier:
                nxt |= self._adj.get(node, set())
            nxt -= seen
            if not nxt:
                break
            seen |= nxt
            frontier = nxt
        seen.discard(chunk_id)
        return seen

    def expand(self, seed_ids: set[str], hops: int = 1) -> set[str]:
        out: set[str] = set()
        for sid in seed_ids:
            out |= self.neighbors(sid, hops)
        return out - set(seed_ids)

    def get_record(self, chunk_id: str) -> ChunkRecord | None:
        return self._records.get(chunk_id)
```

- [ ] **Step 4: Run — expect PASS**

Run: `.venv-wsl/bin/python -m pytest tests/test_graph_index.py -q`

---

## Task 7: GraphExpander

**Files:** Create `rag/graph/expander.py`; Test `tests/test_graph_expander.py`

- [ ] **Step 1: Failing test** — create `tests/test_graph_expander.py`:

```python
from rag.config import RagConfig
from rag.graph.expander import GraphExpander
from rag.graph.index import GraphIndex
from rag.types import ChunkRecord, SearchResult
from tests.fakes import FakeVectorStore


def _rec(cid, source, ordinal, entities=()):
    return ChunkRecord(chunk_id=cid, raw_text=cid,
                       provenance={"source": source, "title": source},
                       entities=list(entities), ordinal=ordinal)


def _store(recs):
    s = FakeVectorStore(); s.add(recs); return s


def _expander(store, hops=1):
    return GraphExpander(store, GraphIndex(RagConfig()), RagConfig(graph_hops=hops))


def test_expands_seed_with_neighbor_results():
    store = _store([_rec("a", "s1", 0), _rec("b", "s1", 1), _rec("c", "s1", 2)])
    exp = _expander(store)
    seeds = [SearchResult("a", "a", 0.9)]
    out = exp.expand(seeds)
    ids = [r.chunk_id for r in out]
    assert "a" in ids and "b" in ids            # b is adjacency neighbor of a
    b = next(r for r in out if r.chunk_id == "b")
    assert b.score == RagConfig().graph_neighbor_score   # dampened


def test_neighbors_not_duplicated_when_already_seed():
    store = _store([_rec("a", "s1", 0), _rec("b", "s1", 1)])
    exp = _expander(store)
    out = exp.expand([SearchResult("a", "a", 0.9), SearchResult("b", "b", 0.8)])
    assert [r.chunk_id for r in out].count("b") == 1


def test_rebuilds_on_store_count_change():
    store = _store([_rec("a", "s1", 0), _rec("b", "s1", 1)])
    exp = _expander(store)
    exp.expand([SearchResult("a", "a", 0.9)])          # builds at count=2
    store.add([_rec("c", "s1", 2)])                     # now b-c adjacency exists
    out = exp.expand([SearchResult("b", "b", 0.9)])
    assert "c" in [r.chunk_id for r in out]


def test_two_expanders_independent():
    s1 = _store([_rec("a", "s1", 0), _rec("b", "s1", 1)])
    s2 = _store([_rec("x", "s2", 0)])
    e1, e2 = _expander(s1), _expander(s2)
    assert "b" in [r.chunk_id for r in e1.expand([SearchResult("a", "a", 0.9)])]
    assert [r.chunk_id for r in e2.expand([SearchResult("x", "x", 0.9)])] == ["x"]
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: rag.graph.expander`)

Run: `.venv-wsl/bin/python -m pytest tests/test_graph_expander.py -q`

- [ ] **Step 3: Implement** — create `rag/graph/expander.py`:

```python
"""GraphExpander — expands a result set with graph neighbors before rerank.

Owns the store + GraphIndex, rebuilding the index when the corpus size changes
(same lazy-sync as BM25Retriever). Neighbors enter as SearchResults with a dampened
score so the reranker, not the raw score, decides their final rank.
"""
from __future__ import annotations

from rag.config import RagConfig
from rag.graph.index import GraphIndex
from rag.interfaces import VectorStore
from rag.types import SearchResult


class GraphExpander:
    def __init__(self, store: VectorStore, index: GraphIndex, config: RagConfig | None = None):
        self.store = store
        self.index = index
        self.config = config or RagConfig()
        self._built_count = -1

    def _ensure_current(self) -> None:
        count = self.store.count()
        if count != self._built_count:
            self.index.build(self.store.get_all())
            self._built_count = count

    def expand(self, results: list[SearchResult]) -> list[SearchResult]:
        self._ensure_current()
        seed_ids = {r.chunk_id for r in results}
        neighbor_ids = self.index.expand(seed_ids, hops=self.config.graph_hops)
        out = list(results)
        for nid in neighbor_ids:
            rec = self.index.get_record(nid)
            if rec is None:
                continue
            out.append(SearchResult(
                chunk_id=rec.chunk_id,
                text=rec.raw_text,
                score=self.config.graph_neighbor_score,
                metadata=dict(rec.provenance, section_title=rec.section_path),
                parent_text=rec.parent_text,
                freshness=dict(rec.freshness),
                ordinal=rec.ordinal,
            ))
        return out
```

- [ ] **Step 4: Run — expect PASS**

Run: `.venv-wsl/bin/python -m pytest tests/test_graph_expander.py -q`

---

## Task 8: HybridRetriever optional graph stage

**Files:** Modify `rag/retrievers/hybrid.py`; Test `tests/test_rerank_hybrid.py`

- [ ] **Step 1: Failing test** — append to `tests/test_rerank_hybrid.py`:

```python
def test_hybrid_graph_stage_adds_neighbors_before_rerank():
    from rag.graph.expander import GraphExpander
    from rag.graph.index import GraphIndex
    from rag.retrievers.dense import DenseRetriever
    from rag.retrievers.bm25 import BM25Retriever
    from rag.types import ChunkRecord
    store = FakeVectorStore()
    emb = FakeEmbedder()
    recs = [
        ChunkRecord(chunk_id="a", raw_text="python programming",
                    embedding=emb.embed(["python programming"])[0],
                    provenance={"source": "s", "title": "s"}, ordinal=0),
        ChunkRecord(chunk_id="b", raw_text="unrelated cooking recipes",
                    embedding=emb.embed(["unrelated cooking recipes"])[0],
                    provenance={"source": "s", "title": "s"}, ordinal=1),
    ]
    store.add(recs)
    cfg = RagConfig(top_k=3, final_k=5, graph_hops=1)
    dense = DenseRetriever(store, emb)
    bm25 = BM25Retriever(store, FakeKeywordIndex())
    graph = GraphExpander(store, GraphIndex(cfg), cfg)
    h = HybridRetriever(dense, bm25, reranker=None, config=cfg, graph=graph)
    ids = {r.chunk_id for r in h.retrieve("python")}
    assert "a" in ids and "b" in ids        # b pulled in as adjacency neighbor of a


def test_hybrid_without_graph_is_unchanged():
    h = _hybrid(reranker=None)              # existing helper, no graph
    res = h.retrieve("python")
    assert len(res) == 2                     # final_k=2, no neighbors added
```

- [ ] **Step 2: Run — expect FAIL** (`TypeError: unexpected keyword argument 'graph'`)

Run: `.venv-wsl/bin/python -m pytest tests/test_rerank_hybrid.py::test_hybrid_graph_stage_adds_neighbors_before_rerank -q`

- [ ] **Step 3: Implement** — in `rag/retrievers/hybrid.py`:

Update `__init__` to accept `graph`:

```python
    def __init__(self, dense: Retriever, bm25: Retriever,
                 reranker: Reranker | None = None, config: RagConfig | None = None,
                 graph=None):
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker
        self.config = config or RagConfig()
        self.graph = graph
```

In `retrieve`, insert the graph stage between fusion and rerank:

```python
        vec = self.dense.retrieve(query, k)
        kw = self.bm25.retrieve(query, k)
        fused = rrf_fuse([vec, kw], k=self.config.rrf_k)

        if self.graph is not None:
            fused = self.graph.expand(fused)

        if use_reranker and self.reranker and len(fused) > fk:
```

(The rest of `retrieve` — the RRF-score restore + return — is unchanged.)

- [ ] **Step 4: Run — expect PASS**

Run: `.venv-wsl/bin/python -m pytest tests/test_rerank_hybrid.py -q`

---

## Task 9: Reading-order context assembly

**Files:** Modify `rag/orchestrator.py`; Test `tests/test_orchestrator.py`

- [ ] **Step 1: Failing test** — append to `tests/test_orchestrator.py`:

```python
def _osr(cid, score, source, ordinal, text="body"):
    r = SearchResult(chunk_id=cid, text=text, score=score,
                     metadata={"source": source, "title": source, "section_title": ""})
    r.ordinal = ordinal
    return r


def test_context_reading_order_groups_by_source_and_ordinal():
    cfg = RagConfig(graph_enabled=True, graph_context_ordering=True, context_tokens=10000)
    results = [_osr("b", 0.7, "doc1", 1, "second"), _osr("a", 0.9, "doc1", 0, "first"),
               _osr("c", 0.8, "doc2", 0, "other")]
    ctx = Orchestrator._assemble_context(results, cfg)
    # within doc1, ordinal 0 ("first") precedes ordinal 1 ("second")
    assert ctx.index("first") < ctx.index("second")


def test_context_score_order_when_graph_disabled():
    cfg = RagConfig(graph_enabled=False, context_tokens=10000)
    results = [_osr("b", 0.7, "doc1", 1, "second"), _osr("a", 0.9, "doc1", 0, "first")]
    ctx = Orchestrator._assemble_context(results, cfg)
    # score-order: higher-scored "first" (0.9) precedes "second" (0.7) — unchanged behavior
    assert ctx.index("first") < ctx.index("second")
```

- [ ] **Step 2: Run — expect FAIL** (`test_context_reading_order...` — ordering wrong under graph mode)

Run: `.venv-wsl/bin/python -m pytest tests/test_orchestrator.py::test_context_reading_order_groups_by_source_and_ordinal -q`

- [ ] **Step 3: Implement** — in `rag/orchestrator.py`, replace the first line of `_assemble_context`
(the `results_sorted = sorted(...)`) with a mode switch, keeping the rest of the method identical:

```python
    @staticmethod
    def _assemble_context(results: list[SearchResult], config: RagConfig) -> str:
        results_sorted = Orchestrator._order_for_context(results, config)
        blocks, tokens = [], 0
        for i, r in enumerate(results_sorted, 1):
            # ... unchanged body ...
```

Add the ordering helper as a new static method on `Orchestrator`:

```python
    @staticmethod
    def _order_for_context(results: list[SearchResult], config: RagConfig) -> list[SearchResult]:
        if not (config.graph_enabled and config.graph_context_ordering):
            return sorted(results, key=lambda r: r.score, reverse=True)
        # group by source; order within group by ordinal; order groups by best score
        groups: dict[str, list[SearchResult]] = {}
        for r in results:
            groups.setdefault(r.metadata.get("source", ""), []).append(r)
        ordered_groups = sorted(
            groups.values(), key=lambda g: max(x.score for x in g), reverse=True)
        out: list[SearchResult] = []
        for g in ordered_groups:
            out.extend(sorted(g, key=lambda r: getattr(r, "ordinal", 0)))
        return out
```

(Leave the rest of `_assemble_context`'s body — label building, token budget, parent/child
selection — exactly as it is.)

- [ ] **Step 4: Run — expect PASS** (whole orchestrator file, incl. existing parity tests)

Run: `.venv-wsl/bin/python -m pytest tests/test_orchestrator.py -q`

---

## Task 10: Pipeline wiring

**Files:** Modify `rag/pipeline.py`; Test `tests/test_pipeline.py`

- [ ] **Step 1: Failing test** — append to `tests/test_pipeline.py`:

```python
def test_graph_enabled_pipeline_end_to_end():
    cfg = RagConfig(similarity_threshold=0.0, low_confidence_threshold=0.0,
                    top_k=5, final_k=5, min_chunk_size=1, graph_enabled=True)
    p = RagPipeline(config=cfg, store=FakeVectorStore(), embedder=FakeEmbedder(),
                    generator=FakeGenerator(answer="ok"), reranker=None)
    r = p.ingest_text("The Zephyr X1 is strong. Zephyr X1 spins fast. Zephyr X1 lasts long.", title="D")
    assert r["status"] == "ok"
    res = p.query("Zephyr")
    assert res.response == "ok"
    assert res.sources


def test_graph_disabled_pipeline_has_no_expander():
    p = _pipeline()                          # existing helper, graph off
    assert getattr(p.retriever, "graph", None) is None
```

- [ ] **Step 2: Run — expect FAIL** (`AttributeError`/graph not wired, or expander present when off)

Run: `.venv-wsl/bin/python -m pytest tests/test_pipeline.py::test_graph_disabled_pipeline_has_no_expander -q`

- [ ] **Step 3: Implement** — in `rag/pipeline.py`:

Add imports at top:

```python
from rag.entities import KeywordEntityExtractor
```

In `__init__`, accept an extractor and build the expander when enabled. Change the signature to
add `entity_extractor=None`, and set it before building the ingestor:

```python
        self.entity_extractor = entity_extractor or KeywordEntityExtractor()
```

Replace the retriever-construction block with a graph-aware version:

```python
        if retriever is None:
            dense = DenseRetriever(self.store, self.embedder)
            bm25 = BM25Retriever(self.store, self.keyword_index)
            graph = None
            if self.config.graph_enabled:
                from rag.graph.expander import GraphExpander
                from rag.graph.index import GraphIndex
                graph = GraphExpander(self.store, GraphIndex(self.config), self.config)
            retriever = HybridRetriever(dense, bm25, reranker=self.reranker,
                                        config=self.config, graph=graph)
        self.retriever = retriever
```

Pass the extractor into the ingestor:

```python
        self.ingestor = RagIngestor(self.store, self.embedder, self.config,
                                    entity_extractor=self.entity_extractor)
```

And add `entity_extractor=None` to the `__init__` parameter list (keyword-only section, alongside
`store`, `embedder`, etc.).

- [ ] **Step 4: Run — expect PASS**

Run: `.venv-wsl/bin/python -m pytest tests/test_pipeline.py -q`

---

## Task 11: Full verification

- [ ] **Step 1: Full suite**

Run: `.venv-wsl/bin/python -m pytest -q`
Expected: all pass (111 prior + graph additions), 1 skipped (cross-encoder).

- [ ] **Step 2: Parity — graph off is unchanged**

Run:
```bash
.venv-wsl/bin/python -c "
from rag.config import RagConfig
from rag.orchestrator import Orchestrator
from rag.types import SearchResult
def sr(cid, sc, ordv):
    r = SearchResult(cid, cid, sc, {'source':'d','title':'d','section_title':''}); r.ordinal=ordv; return r
res=[sr('b',0.7,0), sr('a',0.9,1)]
print('graph off order:', [x.chunk_id for x in Orchestrator._order_for_context(res, RagConfig())])
print('graph on  order:', [x.chunk_id for x in Orchestrator._order_for_context(res, RagConfig(graph_enabled=True))])
"
```
Expected: `graph off order: ['a', 'b']` (score) / `graph on order: ['b', 'a']` (ordinal).

- [ ] **Step 3: App still compiles**

Run: `.venv-wsl/bin/python -m py_compile app.py ingest.py && echo OK`
Expected: `OK`.

---

## Self-Review Notes (author)

- Spec §1 extractor → Task 2. §2 schema → Task 3. §3 store → Task 4. §4 ingest → Task 5.
  §5 GraphIndex → Task 6. §6 GraphExpander → Task 7. §7 hybrid stage → Task 8. §8 reading-order
  → Task 9. §9 config → Task 1. §10 pipeline → Task 10. Verification → Task 11.
- Type consistency: `GraphIndex.build/neighbors/expand/get_record` (Task 6) consumed verbatim by
  `GraphExpander` (Task 7). `SearchResult.ordinal` (Task 3) set by store (Task 4) + expander
  (Task 7), read by `_order_for_context` (Task 9). `graph=` kwarg on `HybridRetriever` (Task 8)
  passed by pipeline (Task 10). `entity_extractor` on `RagIngestor` (Task 5) + `RagPipeline`
  (Task 10).
- Parity: `graph_enabled=False` default → pipeline builds `graph=None` (Task 10) + `_order_for_context`
  returns score-sort (Task 9) → identical to today. Verified Task 11 Step 2. Existing 111 tests
  must stay green.
