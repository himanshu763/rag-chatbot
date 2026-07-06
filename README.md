# 🧠 RAG Chatbot — All-Python, One Repo

A production-ready Retrieval-Augmented Generation chatbot with a **Streamlit UI**.
Ingest web pages, PDFs, DOCX files, or raw text — then ask questions grounded in your data.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20%2F%20Azure-0078d4)

> **Now a library.** Core logic lives in the importable `rag/` package — swappable
> loaders, embedders, vector store, retrievers, rerankers, and generator wired through
> one injectable `RagPipeline` (no global state). `app.py` and `ingest.py` are thin
> reference clients. See [Library Usage](#library-usage).

---

## Quick Start

```bash
# 1. Clone & install
git clone <this-repo>
cd rag_chatbot
pip install -r requirements.txt

# 2. Configure Azure OpenAI credentials
cp .env.example .env
# Edit .env and fill in:
#   AZURE_OPENAI_API_KEY
#   AZURE_OPENAI_ENDPOINT
#   AZURE_OPENAI_DEPLOYMENT
#   AZURE_OPENAI_API_VERSION

# 3. Ingest your data
python ingest.py

# 4. Run
streamlit run app.py
```

Open `http://localhost:8501` → add sources in sidebar → start chatting.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  STREAMLIT UI (app.py)                                      │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │  Sidebar          │  │  Chat Interface                  │ │
│  │  • Add URL        │  │  • Message history               │ │
│  │  • Upload file    │  │  • Streaming responses           │ │
│  │  • Paste text     │  │  • Confidence badges             │ │
│  │  • Source list     │  │  • Source citations              │ │
│  └────────┬─────────┘  └──────────┬───────────────────────┘ │
└───────────┼─────────────────────────┼───────────────────────┘
            │                         │
    ┌───────▼───────┐        ┌───────▼────────┐
    │  INGESTION     │        │  ORCHESTRATOR   │
    │  PIPELINE      │        │                 │
    │  • WebLoader   │        │  1. Intent      │
    │  • PDFLoader   │        │  2. Rewrite     │
    │  • DocxLoader  │        │  3. Retrieve    │
    │  • TextLoader  │        │  4. Confidence  │
    │  • Chunker     │        │  5. Context     │
    │  • Enricher    │        │  6. Generate    │
    │  • Embedder    │        │  7. Validate    │
    └───────┬───────┘        └──┬──────┬──────┘
            │                   │      │
    ┌───────▼───────────────────▼┐  ┌──▼──────────┐
    │  RETRIEVAL ENGINE          │  │  LLM CLIENT  │
    │  • Vector search (cosine)  │  │  • OpenAI    │
    │  • BM25 keyword search     │  │  • Streaming │
    │  • RRF fusion              │  └──────────────┘
    │  • Cross-encoder reranker  │
    └───────────┬────────────────┘
                │
    ┌───────────▼────────────────┐
    │  CHROMADB (vector store)   │
    │  embeddings + metadata     │
    └────────────────────────────┘
```

---

## Ingesting Data (CLI)

Drop your files and URLs into the `data/` folder, then run one command to embed and store everything in ChromaDB.

### 1. Add sources

```
data/
├── web_urls.txt       ← paste website URLs here (one per line)
├── report.pdf
├── knowledge_base.docx
├── products.csv
└── financials.xlsx
```

**Supported file types:** `.pdf` `.docx` `.doc` `.csv` `.txt` `.xlsx` `.xls`

Edit `data/web_urls.txt` and add URLs (lines starting with `#` are ignored):

```
https://docs.example.com/overview
https://blog.example.com/article-1
```

### 2. Run ingestion

```bash
# Ingest everything in data/ folder + all URLs in data/web_urls.txt
python ingest.py

# URLs only (skip local files)
python ingest.py --urls-only

# Files only (skip URLs)
python ingest.py --files-only

# Pass extra sources directly (stacks on top of data/ folder)
python ingest.py --url https://example.com/page --file ./other/doc.pdf
```

### 3. Verify

```bash
# Check how many chunks are stored
python -c "from ingestion.pipeline import IngestionPipeline; p = IngestionPipeline(); print(p.total_chunks(), 'chunks in store'); [print(' •', s['title'], '-', s['chunks'], 'chunks') for s in p.list_sources()]"
```

Ingestion is **idempotent** — re-running the same source replaces its old chunks, no duplicates.

---

## Project Structure

```
rag_chatbot/
├── app.py                     ← Streamlit reference client (run this)
├── ingest.py                  ← CLI reference client: ingest data/ + web_urls.txt
├── pyproject.toml             ← Package metadata + optional-dependency extras
│
├── rag/                       ← The library (importable, no global state)
│   ├── pipeline.py            ← RagPipeline facade — composes every module
│   ├── config.py              ← RagConfig (per-instance; env supplies defaults)
│   ├── types.py               ← ChunkRecord schema, SearchResult, Confidence, …
│   ├── interfaces.py          ← Protocols: Embedder/VectorStore/Retriever/…
│   ├── orchestrator.py        ← intent → retrieve → confidence → generate
│   ├── ingest.py              ← RagIngestor: load → chunk → enrich → embed → store
│   ├── chunking.py            ← Structure-aware chunking + parent-child
│   ├── loaders.py             ← Web, PDF, DOCX, CSV, Excel, text loaders
│   ├── embedders/             ← OpenAIEmbedder (Azure/OpenAI)
│   ├── generators/            ← OpenAIGenerator (Azure/OpenAI, streaming)
│   ├── stores/                ← ChromaVectorStore
│   ├── keyword/               ← BM25KeywordIndex
│   ├── retrievers/            ← dense + bm25 + RRF fusion + hybrid compose
│   ├── rerankers/             ← cross-encoder + LLM reranker
│   └── session_store.py       ← MongoDB chat-session persistence
│
├── tests/                     ← pytest suite (fakes — no API calls needed)
└── data/                      ← Drop files here (PDF, DOCX, CSV, XLSX, TXT)
    └── web_urls.txt           ← Paste website URLs here for scraping
```

**Swappable modules, one facade, no framework bloat.**

---

## Library Usage

```python
from rag import RagPipeline, RagConfig

pipe = RagPipeline(RagConfig.from_env())     # reads .env for provider/keys
pipe.ingest_file("report.pdf")
pipe.ingest_url("https://docs.example.com/overview")

result = pipe.query("What were Q3 revenues?")
print(result.confidence, result.response)
for src in result.sources:
    print(" •", src["title"], src["score"])
```

Swap any module (or build multiple isolated pipelines in one process):

```python
from rag import RagPipeline, RagConfig
from rag.stores.chroma import ChromaVectorStore

cfg = RagConfig(collection_name="tenant_a", reranker_type="crossencoder")
pipe = RagPipeline(config=cfg, store=ChromaVectorStore(cfg))  # inject your own Embedder/Generator/Retriever too
```

Every component (`Embedder`, `VectorStore`, `KeywordIndex`, `Retriever`, `Reranker`,
`Generator`) is a `typing.Protocol` in `rag/interfaces.py` — implement the methods and
pass your object in. `pip install -e ".[rerank,sessions,ui]"` pulls optional extras.

---

## How It Works

### Ingestion (when you add a source)

1. **Load** — extract text respecting document structure (headings, tables, pages)
2. **Chunk** — split at sentence boundaries, never mid-sentence; ~400 tokens each
3. **Enrich** — add contextual summary metadata per chunk (improves retrieval)
4. **Embed** — generate vector embeddings with `all-MiniLM-L6-v2`
5. **Store** — save to ChromaDB with full metadata for filtered retrieval

### Chat (when you ask a question)

1. **Intent classify** — FACTUAL / COMPARISON / FOLLOW_UP / OUT_OF_SCOPE / GREETING
2. **Query rewrite** — resolve pronouns using conversation history
3. **Hybrid retrieve** — vector search + BM25 keyword search → RRF fusion → cross-encoder rerank
4. **Confidence score** — HIGH / MEDIUM / LOW / NONE based on retrieval scores
5. **Fallback decision** — NONE → skip LLM, return "I don't know" (saves cost, prevents hallucination)
6. **Context assemble** — dedupe, order, format chunks within token budget
7. **LLM generate** — stream the model's response grounded in the context
8. **Display** — show response with confidence badge + expandable source citations

### Confidence Ladder (no dead ends)

| Level  | What happens |
|--------|-------------|
| 🟢 HIGH   | Confident answer with citations |
| 🟡 MEDIUM | Partial answer, notes what's missing |
| 🟠 LOW    | Cautious answer only if directly relevant info found |
| 🔴 NONE   | "I don't have info on that" — LLM skipped entirely |

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **Hybrid retrieval** (vector + BM25) | Catches both semantic meaning AND exact terms (product names, error codes) |
| **Cross-encoder reranker** | Massive precision boost for ~20ms extra latency |
| **Parent-child chunks** | Small chunks for retrieval accuracy, expanded parent for LLM context |
| **Confidence gating** | Skips LLM when retrieval fails → no hallucination, lower cost |
| **Intent classification** | Routes different query types to the right strategy |
| **Query rewriting** | "How much does it cost?" → "What is the Enterprise plan pricing?" |
| **Streamlit** | All-Python, no JS build step, production-ready with one command |

---

## Production Upgrades

When ready to scale:

- **Redis** → replace in-memory session store for multi-instance deployment
- **PostgreSQL + pgvector** → replace ChromaDB for >1M chunks
- **Celery** → async ingestion queue for large batch imports
- **Elasticsearch** → replace in-memory BM25 for scale
- **Auth** → add Streamlit authenticator or deploy behind OAuth proxy
- **Monitoring** → export metrics to Prometheus/Grafana
- **Rate limiting** → per-user query limits to control LLM cost

---

## License

MIT
