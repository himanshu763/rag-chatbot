"""rag — plug-and-play RAG library.

Swappable modules (loaders, embedders, vector store, retrievers, rerankers,
generators) wired through one injectable ``RagPipeline``. No global state.

    from rag import RagPipeline, RagConfig
    pipe = RagPipeline(RagConfig.from_env())
    pipe.ingest_file("report.pdf")
    print(pipe.query("What were Q3 revenues?").response)
"""
from __future__ import annotations

from rag.config import RagConfig
from rag.pipeline import RagPipeline
from rag.types import (
    Chunk,
    ChunkRecord,
    Confidence,
    CONF_META,
    Intent,
    OrchestratorResult,
    RawDocument,
    SearchResult,
)

__all__ = [
    "RagPipeline",
    "RagConfig",
    "Chunk",
    "ChunkRecord",
    "Confidence",
    "CONF_META",
    "Intent",
    "OrchestratorResult",
    "RawDocument",
    "SearchResult",
]
