"""RagIngestor — load → chunk → enrich → embed → store.

Instance-scoped (store + embedder + config injected). Idempotent: re-ingesting an
unchanged source is skipped (source hash match); changed content replaces old chunks.

Enrichment note: the *enriched* (context-prefixed) text is embedded, while the
*raw* chunk text is stored as the document (BM25 searches raw text).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from rag.chunking import Chunk, Chunker
from rag.config import RagConfig
from rag.entities import KeywordEntityExtractor
from rag.interfaces import Embedder, EntityExtractor, VectorStore
from rag.loaders import TextLoader, get_loader
from rag.types import ChunkRecord, RawDocument

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enrich(chunk: Chunk) -> str:
    """Return context-prefixed text for embedding; also stores a summary on the chunk."""
    section = chunk.metadata.get("section_title", "")
    src = chunk.metadata.get("source_type", "document")
    first = chunk.text.split(".")[0][:100]
    parts = []
    if section:
        parts.append(f"Section '{section}'")
    parts.append(f"from {src}")
    if first:
        parts.append(f"about: {first}")
    summary = ". ".join(parts)
    chunk.metadata["summary"] = summary
    return f"[Context: {summary}]\n{chunk.text}"


class RagIngestor:
    def __init__(self, store: VectorStore, embedder: Embedder,
                 config: RagConfig | None = None, chunker: Chunker | None = None,
                 entity_extractor: EntityExtractor | None = None):
        self.store = store
        self.embedder = embedder
        self.config = config or RagConfig()
        self.chunker = chunker or Chunker(self.config)
        self.entity_extractor = entity_extractor or KeywordEntityExtractor()

    # ── entry points ──
    def ingest_url(self, url: str) -> dict:
        loader, _ = get_loader(url, self.config)
        return self._process(loader.load(url), url)

    def ingest_file(self, path: str) -> dict:
        loader, _ = get_loader(path, self.config)
        return self._process(loader.load(path), path)

    def ingest_text(self, text: str, title: str = "Pasted Text") -> dict:
        return self._process(TextLoader().load(text, title), f"text:{title}")

    # ── pipeline ──
    def _process(self, doc: RawDocument, source_key: str) -> dict:
        if not doc.text.strip():
            return {"source": source_key, "status": "empty", "chunks": 0}

        # Pin the stored source to source_key so skip/delete are consistent across
        # all source types (loaders like TextLoader default source to "manual").
        doc.metadata["source"] = source_key

        # skip unchanged content (same source hash already stored)
        try:
            stored_hash = self.store.get_source_hash(source_key)
            if stored_hash and stored_hash == doc.content_hash:
                # Unchanged content: bump fetched_at so a re-check counts as re-verification
                # (otherwise a stable source past its TTL would stay stale forever).
                self.store.touch_source(source_key, _now_iso())
                return {"source": source_key, "status": "skipped",
                        "chunks": self.store.count(),
                        "title": doc.metadata.get("title", source_key)}
        except Exception:
            logger.warning("source-hash check failed — full re-ingest", exc_info=True)

        chunks = self.chunker.chunk(doc)
        if not chunks:
            return {"source": source_key, "status": "no_chunks", "chunks": 0}

        embed_texts = [_enrich(c) for c in chunks]
        embeddings = self.embedder.embed(embed_texts)

        records = [self._to_record(c, enriched, emb, doc.content_hash, ordinal)
                   for ordinal, (c, enriched, emb) in enumerate(zip(chunks, embed_texts, embeddings))]

        self.store.delete_source(source_key)
        self.store.add(records)
        logger.info("Stored %d chunks from %s", len(records), source_key)
        return {"source": source_key, "status": "ok", "chunks": len(records),
                "title": doc.metadata.get("title", source_key)}

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

    def refresh_source(self, source_key: str) -> dict:
        """Re-fetch a web/file source and re-ingest if changed. Pasted text is unrefreshable."""
        if source_key.startswith(("http://", "https://")):
            return self.ingest_url(source_key)
        if source_key.startswith("text:"):
            return {"source": source_key, "status": "unrefreshable", "chunks": 0}
        import os
        if os.path.exists(source_key):
            return self.ingest_file(source_key)
        return {"source": source_key, "status": "missing", "chunks": 0}

    # ── management ──
    def list_sources(self) -> list[dict]:
        return self.store.list_sources()

    def total_chunks(self) -> int:
        return self.store.count()

    def delete_source(self, source_key: str) -> None:
        self.store.delete_source(source_key)
