# Enhancement Options — RAG Library Evolution

Brainstorm doc. Lists what *could* be done in each area before we pick one to spec in detail.
Not a committed design — options only, with rough complexity/dependency notes.

---

## 1. Library-ification

Goal: turn this from "one app" into a base others can import to build specialized RAG systems.

- **Plugin interfaces (core work)**
  - `Loader` ABC — current loaders (Web/PDF/DOCX/CSV/Excel/Text) become implementations, not hardcoded dispatch
  - `Embedder` protocol — support OpenAI, Azure, local HuggingFace, Cohere, swappable per instance
  - `VectorStore` abstraction — Chroma today, pluggable pgvector/Qdrant/Pinecone later
  - `LLMProvider` abstraction — OpenAI/Azure today; add Anthropic/local model backends
  - `Reranker` abstraction — cross-encoder / LLM-rerank already exist as a `reranker_type` switch; formalize as interface
- **Config overhaul**
  - Current `config.py` is a process-wide singleton (`settings = Settings()`) — blocks multi-instance/multi-tenant use
  - Move to per-instance config object passed into a facade, env-vars become defaults only
- **Public API facade**
  - Single entry class, e.g. `RagPipeline(config).ingest(source)` / `.query(text)` / `.stream(text)`
  - `app.py` / `ingest.py` become thin reference clients built on the library, not the library itself
- **Extension hooks**
  - Custom chunking strategy injection, custom prompt templates, custom confidence policy
- **Multi-tenancy**
  - Namespace/collection-per-tenant support in vector store + BM25 index
- **Packaging**
  - `pyproject.toml`, versioned package, optional publish to private/PyPI index
  - Split `requirements.txt` into core vs. optional extras (e.g. `[web]`, `[docx]`, `[graph]`)
- **Testing harness**
  - Fake/mock LLM + embedder for unit tests without API calls; contract tests per interface

**Dependency note:** everything below is easier to bolt on cleanly if this is done first — new features become new interface implementations instead of more special-casing in existing files.

---

## 2. Freshness / Staleness Checker (chunks)

Goal: know when retrieved info might be outdated, and act on it.

- **Metadata capture at ingest**
  - `source_fetched_at`, `source_last_modified` (HTTP `Last-Modified`/`ETag` for web, file mtime for local files), `ingested_at`
  - Content hash per source (already have `content_hash` per commit history — extend to per-chunk)
- **Staleness scoring**
  - Age-based decay function (configurable half-life per source type — web pages decay faster than static PDFs)
  - Per-source TTL overrides (e.g. pricing page = 7 days, static whitepaper = 1 year)
- **Detection / recheck**
  - Scheduled recrawl for web sources — compare hash/ETag, only re-embed on actual change
  - File-watch or manual `--refresh` flag for local files
  - Idempotent re-ingest already exists (replaces old chunks) — extend to skip if hash unchanged (save embedding cost)
- **Surfacing at query time**
  - Attach staleness badge to citations ("last verified 42 days ago")
  - Fold staleness into confidence ladder: fresh+high-sim → HIGH; stale+high-sim → downgrade to MEDIUM with a note
- **Lifecycle**
  - Background sweep job (APScheduler/Celery) flags stale chunks; optional auto re-ingest queue
  - Keep chunk version history instead of hard overwrite (enables "what changed" diffing)

**Dependency note:** mostly additive to ingestion schema + orchestrator confidence step — low coupling to the other three areas.

---

## 3. RAG + Graph: Graph-Based Chunk Ordering

Goal: use relationships between chunks (not just vector/BM25 score) to select and order context.

- **Graph construction**
  - Entity/keyword extraction per chunk (NER model or LLM-based) → nodes
  - Edges: shared entities, explicit cross-references, and existing section/parent-child hierarchy (chunker already tracks this — currently unused for graph traversal)
  - Cross-document edges: same entity mentioned in different sources link across docs
- **Storage options**
  - `networkx` in-memory, rebuilt or cached per collection (fine up to tens of thousands of chunks)
  - Dedicated graph DB (Neo4j) if scale demands it — bigger lift, own ops burden
  - Lightweight middle ground: store edge lists as Chroma metadata, build graph in-memory at query time
- **Retrieval integration**
  - Seed set from existing vector+BM25+RRF+cross-encoder pipeline (unchanged)
  - Expand seeds via 1–2 hop graph neighbors before final context assembly → pulls in related-but-not-lexically-similar chunks
  - Re-rank expanded set (reuse existing reranker)
- **Context ordering**
  - Assemble context in document/section order (respecting hierarchy) rather than pure score-sort — closer to how a human would read related sections
- **Bonus**
  - Community detection (Louvain etc.) → auto-cluster chunks into topics for a browsable "topics" view
  - `graphify-out/graph.html`-style visualization for debugging retrieval graphs (repo already has a graph visualizer pattern from the graphify experiment — could be repurposed)

**Dependency note:** touches `retrieval/engine.py` directly; benefits from library-ification's `VectorStore`/`Embedder` interfaces being stable first so graph layer isn't built against a moving target.

---

## 4. CAG: Cache-Augmented Generation

Goal: skip retrieval and/or generation entirely on repeat or near-duplicate queries.

- **Layered cache strategy**
  1. Exact-match cache: `hash(query + context/source-set)` → cached response (fastest, free)
  2. Semantic cache: embed incoming query, check similarity against recent query cache — catches paraphrases ("cost?" vs "how much does it cost?")
  3. Miss → full pipeline (current behavior)
- **Context/prompt caching**
  - If provider supports prompt/prefix caching (Anthropic prompt caching, OpenAI prompt caching), pin the system prompt + frequently-used context blocks as a cached prefix — cuts cost/latency on repeat queries against the same doc set
- **Invalidation**
  - Tie cache entries to source version/hash from the freshness checker (#2) — auto-invalidate when underlying content changes
  - TTL fallback even without explicit invalidation signal
- **Storage**
  - Redis for shared/multi-instance cache (already on README's production-upgrade list)
  - SQLite/local dict for single-instance/dev use
- **Observability**
  - Track cache hit rate, estimated cost/latency saved — useful metric to prove the feature earns its complexity

**Dependency note:** most valuable once freshness checker (#2) exists (for safe invalidation) and graph/retrieval (#3) is stable (so what's being cached isn't still changing shape).

---

## 5. Ops-Hardening Backlog (already in README, unchanged)

Not re-scoped here — just the existing list for reference when sequencing:

- Redis (session store / cache backend)
- PostgreSQL + pgvector (>1M chunks)
- Celery (async ingestion queue)
- Elasticsearch (BM25 at scale)
- Auth (Streamlit authenticator / OAuth proxy)
- Monitoring (Prometheus/Grafana)
- Rate limiting (per-user query caps)

---

## Suggested Build Order (recommendation, not decided)

1. **Library-ification** — stable interfaces first, everything else builds on the seams
2. **Freshness/staleness** — schema-level, low coupling, unlocks safe cache invalidation later
3. **Graph-based chunk ordering** — changes retrieval engine
4. **CAG** — changes generation path, most valuable once 2 and 3 are stable
5. **Ops-hardening** — pick items opportunistically as real scale/deployment needs arise

Open question for next step: which of 1–4 do we spec first?
