"""Core data types — retriever-agnostic chunk schema and result objects.

One ``ChunkRecord`` represents a stored chunk for *every* retrieval method
(dense, BM25, graph, future). ``entities`` / ``edges`` / ``freshness`` are
reserved (scaffold) for the graph and freshness features — persisted but not
yet populated.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from enum import Enum


# ── Raw document (loader output) ──────────────────────────────────────────────
@dataclass
class RawDocument:
    text: str
    metadata: dict = field(default_factory=dict)
    sections: list[dict] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


# ── Chunk (chunker output, pre-embedding) ─────────────────────────────────────
@dataclass
class Chunk:
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    text: str = ""
    metadata: dict = field(default_factory=dict)
    parent_text: str | None = None

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


# ── ChunkRecord (stored representation, retriever-agnostic) ───────────────────
@dataclass
class ChunkRecord:
    """A stored chunk with everything any retriever might need.

    Existing fields are populated today; ``entities``/``edges``/``freshness``
    are scaffold — reserved for the graph and freshness features.
    """
    chunk_id: str
    raw_text: str                          # what BM25 tokenizes + what the LLM reads
    enriched_text: str = ""                # context-prefixed text used for embedding
    embedding: list[float] | None = None
    parent_text: str | None = None
    section_path: str = ""                 # e.g. "Doc > Section > Subsection"
    provenance: dict = field(default_factory=dict)   # source, source_type, title, anchor, ...
    content_hash: str = ""                 # per-chunk
    source_hash: str = ""                  # whole-document hash (skip-unchanged check)
    schema_version: str = "3"
    ordinal: int = 0                       # position within source (adjacency + reading order)
    # ── scaffold (populated by later features) ──
    entities: list = field(default_factory=list)     # graph nodes
    edges: list = field(default_factory=list)        # links to related chunk_ids
    freshness: dict = field(default_factory=dict)     # fetched_at, ingested_at, last_modified, ttl


# ── Search result (retriever output) ──────────────────────────────────────────
@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    parent_text: str | None = None
    freshness: dict = field(default_factory=dict)
    ordinal: int = 0
    graph_neighbor: bool = False           # added by graph expansion (excluded from confidence)


# ── Intent ────────────────────────────────────────────────────────────────────
class Intent(str, Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    FOLLOW_UP = "follow_up"
    CLARIFY = "clarify"
    OUT_OF_SCOPE = "out_of_scope"
    GREETING = "greeting"


# ── Confidence ────────────────────────────────────────────────────────────────
class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


CONF_META = {
    Confidence.HIGH:   {"color": "#22c55e", "icon": "🟢", "label": "High confidence"},
    Confidence.MEDIUM: {"color": "#f59e0b", "icon": "🟡", "label": "Medium — partial info"},
    Confidence.LOW:    {"color": "#f97316", "icon": "🟠", "label": "Low — limited info"},
    Confidence.NONE:   {"color": "#ef4444", "icon": "🔴", "label": "No relevant info found"},
}


# ── Orchestrator result ───────────────────────────────────────────────────────
@dataclass
class OrchestratorResult:
    response: str
    sources: list[dict] = field(default_factory=list)
    confidence: Confidence = Confidence.NONE
    intent: str = ""
    rewritten_query: str = ""
    latency_ms: float = 0.0
    fallback: bool = False
    staleness_note: str = ""
