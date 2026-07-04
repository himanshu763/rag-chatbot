# Enhancement Options — RAG Library Evolution

Brainstorm doc. Options only — not a committed design. Lists what *could* be done in
each area, with complexity/dependency/scale notes, so we can pick what to spec first.

**Verified against code** (not just README). Corrections from first draft noted inline.

---

## 0. Framing Corrections (read first)

Three things the design must be honest about:

1. **This is NOT built on langchain.** `requirements.txt` uses raw libs only
   (`openai`, `sentence-transformers`, `chromadb`, `rank-bm25`, `playwright`, `bs4`,
   `PyMuPDF`, `python-docx`, `openpyxl`, `pymongo`). No langchain / llama-index.
   So the goal is a langchain **alternative**, not a wrapper.
   - **Achievable edge over langchain/llama-index:** minimal deps, transparent +
     hackable core, opinionated production defaults baked in (confidence gating,
     hybrid retrieval, fallback ladder), RAG + CAG + graph as first-class citizens.
   - **NOT the edge:** feature count. Those frameworks are mature; won't out-feature
     them. Sell "focused + transparent," not "bigger."

2. **"App-first then library" is fine — the foundation is the problem.** Good libraries
   get *extracted* from real usage. Keep the app as the driver. But this repo runs on
   module-level global state (`settings` singleton `config.py:71`; `_collection`,
   `_bm25_cache`, `_embed_client` in `pipeline.py`; `_reranker`, `_llm_rerank_client`
   in `engine.py`). Every new feature added now wires *into those globals*, turning the
   eventual library conversion into a rewrite. **Kill the globals + define interfaces
   before layering features** — ~10× cheaper now than after 4 more features deepen it.

3. **"Billions of documents" is a different product tier.** Current stack is single-node
   and does not scale past low millions (see §Scale Tiers). Billions = *replace*
   components, not extend them. Doc tiers this explicitly instead of pretending the
   current code scales.

---

## Scale Tiers

| Tier | Corpus size | Vector store | Keyword | Notes |
|------|-------------|--------------|---------|-------|
| **T1** | thousands → low millions | ChromaDB (single-node) | in-memory `rank-bm25` | current stack, after fixing the rebuild-everything BM25 |
| **T2** | 10M → billions | Qdrant / Milvus / Vespa (distributed, quantized) | OpenSearch / Elasticsearch | *replaces* T1 components via the `VectorStore` / `KeywordIndex` interfaces |

**Concrete T1 blockers found in code (must fix even to reach low-millions):**
- `_get_bm25_index()` (`engine.py:77`) — `col.get(...)` loads the **entire corpus into
  RAM** and rebuilds the BM25 index whenever `col.count()` changes. O(all docs) per
  rebuild. Dies well before billions.
- `list_sources()` (`pipeline.py:191`) and `_migrate_schema_if_needed()`
  (`pipeline.py:65`) full-scan the whole collection.
- `chromadb.PersistentClient` is single-node, single-process.

The plug-and-play interfaces below are what make the T1→T2 swap possible without a rewrite.

---

## The Spine: Retriever-Agnostic Chunk Schema + Module Registry

This is the organizing principle everything else hangs off (user's core insight:
"store chunks so any retriever — graph, dense, bm25, others — can use them,
plug-and-play module selection").

### A. One chunk record, every retrieval method

Store each chunk once, with everything any retriever might need:

```
ChunkRecord:
  chunk_id            # stable id
  raw_text            # what BM25 tokenizes + what LLM reads (exists today)
  enriched_text       # context-prefixed text used for embedding (exists today, not stored separately)
  embedding           # dense vector (exists today)
  parent_text         # parent-child expansion (exists today, in metadata)
  section_path        # hierarchy: doc > section > subsection (chunker has this, underused)
  entities            # extracted entities/keywords → graph nodes (NEW)
  edges               # links to related chunk_ids (shared entity / xref / adjacency) (NEW)
  provenance          # source, source_type, url/path, page/loc (partial today)
  freshness           # fetched_at, ingested_at, source_last_modified, ttl (NEW)
  hashes              # content_hash (per-chunk) + source_hash (per-doc) — BOTH EXIST today
  schema_version      # exists today
```

Correction from first draft: per-chunk `content_hash` and per-source `source_hash`
**already exist** (`pipeline.py:165-166`). The genuinely missing pieces are
**timestamps/TTL, entities, and edges**.

### B. Module registry (plug-and-play)

User picks which implementations to wire, per instance:

```
RagPipeline(
  loaders   = [WebLoader, PDFLoader, ...],   # or auto-dispatch by type
  chunker   = StructureAwareChunker(...),
  embedder  = OpenAIEmbedder / AzureEmbedder / LocalHFEmbedder / ...,
  store     = ChromaStore / QdrantStore / ...,           # VectorStore interface
  keyword   = BM25Index / OpenSearchIndex / None,        # KeywordIndex interface
  retrievers= [DenseRetriever, BM25Retriever, GraphRetriever],  # composable
  fusion    = RRFFusion(...),
  reranker  = CrossEncoderReranker / LLMReranker / None,
  generator = OpenAIGenerator / AnthropicGenerator / ...,
  cache     = SemanticCache / None,          # CAG
  policy    = ConfidencePolicy(...),         # gating/fallback ladder
)
```

Retrievers are the key composable: each takes the query + the shared chunk store and
returns scored `chunk_id`s. Dense, BM25, and Graph all conform to one `Retriever`
interface → user enables any subset → fusion merges them. Adding a new retrieval method
= add one class, no core changes.

---

## 1. Library-ification (the enabling work)

- **Kill global state** — per-instance config + injected store/embedder/clients
  (removes every `_singleton` and the `settings = Settings()` global).
- **Define interfaces:** `Loader`, `Chunker`, `Embedder`, `VectorStore`, `KeywordIndex`,
  `Retriever`, `Fusion`, `Reranker`, `Generator`, `Cache`, `ConfidencePolicy`.
- **Facade:** `RagPipeline(config).ingest(source)` / `.query(text)` / `.stream(text)`.
  `app.py` / `ingest.py` become thin reference clients on top.
- **Config:** env-vars become defaults, not the source of truth; config is an object
  passed in.
- **Packaging:** `pyproject.toml`, versioned; `requirements` split into core + extras
  (`[web]`, `[docx]`, `[graph]`, `[qdrant]`…) so users install only what they enable.
- **Test harness:** fake embedder + fake LLM (no API calls); one contract test per
  interface so any implementation is verifiable.

**Why first:** every feature below is a new interface implementation *if* the seams exist,
or another special-case wired into globals *if* they don't.

---

## 2. Freshness / Staleness Checker

- **Capture at ingest (missing today):** `fetched_at`, `ingested_at`,
  `source_last_modified` (HTTP `Last-Modified`/`ETag` for web, file mtime for local),
  optional per-source `ttl`.
- **Change detection (partly exists):** ingest already skips unchanged content via
  `source_hash` (`pipeline.py:138`). Extend to web recrawl using `ETag`/`Last-Modified`
  so unchanged pages skip re-embedding (saves cost).
- **Staleness scoring:** age-decay with configurable half-life per source type
  (pricing page decays fast, whitepaper slow); TTL overrides.
- **Surface at query time:** staleness on citations ("verified 42 days ago"); fold into
  confidence ladder (stale + high-sim → downgrade HIGH→MEDIUM with a note).
- **Lifecycle:** background sweep flags stale chunks → optional auto re-ingest queue;
  optional chunk version history for "what changed" diffing.

**Coupling:** additive to chunk schema + confidence step. Low. Enables safe CAG
invalidation later.

---

## 3. RAG + Graph: Graph-Based Chunk Ordering

- **Build graph:** entity/keyword extraction per chunk (NER or LLM) → nodes; edges from
  shared entities, explicit xrefs, and existing section/parent-child hierarchy (chunker
  tracks hierarchy already — currently unused for traversal). Cross-document edges link
  the same entity across sources.
- **Storage:** T1 = `networkx` in-memory (or edge-lists in chunk metadata, graph built at
  query time). T2 = Neo4j only if scale demands (own ops burden).
- **Retrieve:** seed from existing dense+BM25+RRF+rerank pipeline, expand 1–2 hops to pull
  related-but-not-lexically-similar chunks, re-rank expanded set.
- **Order context** in document/section reading order, not pure score-sort.
- **Bonus:** community detection → topic clusters; reuse repo's existing `graphify-out`
  graph-viz pattern for debugging retrieval graphs.

**Coupling:** implemented as a `GraphRetriever` conforming to the `Retriever` interface →
needs interfaces (§1) stable first.

---

## 4. CAG: Cache-Augmented Generation

- **Layered cache:** (1) exact-match `hash(query + source-set)` → response;
  (2) semantic cache — embed query, match against recent queries (catches paraphrases);
  (3) miss → full pipeline.
- **Provider prompt/prefix caching:** pin system prompt + hot context blocks as a cached
  prefix (Anthropic / OpenAI prompt caching) → cheaper repeat queries on same doc set.
- **Invalidation:** tie cache entries to source version/hash from §2 → auto-invalidate on
  content change; TTL fallback.
- **Storage:** Redis (shared/multi-instance) or local dict/SQLite (dev).
- **Observability:** hit rate + cost/latency saved (proves the feature earns its complexity).

**Coupling:** most valuable *after* §2 (safe invalidation) and §3 (stable retrieval shape).

---

## 5. Ops-Hardening Backlog (from README, unchanged)

Redis · PostgreSQL+pgvector · Celery (async ingest) · Elasticsearch (BM25 at scale) ·
Auth · Monitoring (Prometheus/Grafana) · Rate limiting. Pick opportunistically as real
deployment needs arise. Several overlap with T2 in the Scale Tiers table.

---

## Recommended First Slice

Regardless of app-first vs library-first, the cheap-now/expensive-later work that
**everything else depends on** is one bounded slice:

> **Kill global state + define the retriever-agnostic `ChunkRecord` schema + the core
> `Retriever` / `VectorStore` / `Embedder` interfaces**, keeping the existing app working
> on top as the driver/reference client.

This unblocks 2, 3, and 4 (each becomes a clean plug-in), fixes the multi-instance blocker,
and is verifiable with the existing app end-to-end. It does **not** yet touch billions-scale
(T2) — that's a later, separate swap enabled by these same interfaces.

**Open question for next step:** spec that first slice, or spec one of the visible features
(freshness / graph / CAG) first as a vertical proof-of-concept?
