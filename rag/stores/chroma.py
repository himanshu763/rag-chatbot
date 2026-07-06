"""ChromaVectorStore — ChromaDB-backed VectorStore.

Instance-scoped (no global collection): each store owns its own client +
collection, so multiple RagPipelines / tenants coexist in one process.

Chroma metadata holds only str/int/float/bool, so list/dict scaffold fields
(entities, edges, freshness) are JSON-encoded into string keys and decoded back.
"""
from __future__ import annotations

import json
import logging
import os

import chromadb

from rag.config import RagConfig
from rag.types import ChunkRecord, SearchResult

logger = logging.getLogger(__name__)

# internal metadata keys (not surfaced in SearchResult.metadata)
_ENTITIES_KEY = "__entities_json"
_EDGES_KEY = "__edges_json"
_FRESHNESS_KEY = "__freshness_json"
_PARENT_KEY = "parent_text"
_INTERNAL = {_ENTITIES_KEY, _EDGES_KEY, _FRESHNESS_KEY, _PARENT_KEY}


class ChromaVectorStore:
    def __init__(self, config: RagConfig):
        self.config = config
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        abs_path = os.path.abspath(config.chroma_dir)
        self._client = chromadb.PersistentClient(path=abs_path)
        self._col = self._client.get_or_create_collection(
            name=config.collection_name, metadata={"hnsw:space": "cosine"},
        )

    # ── write ──
    def add(self, records: list[ChunkRecord]) -> None:
        if not records:
            return
        self._col.add(
            ids=[r.chunk_id for r in records],
            embeddings=[r.embedding for r in records],
            documents=[r.raw_text for r in records],
            metadatas=[self._to_meta(r) for r in records],
        )
        logger.debug("Chroma add: %d records", len(records))

    def delete_source(self, source_key: str) -> None:
        try:
            existing = self._col.get(where={"source": {"$eq": source_key}})
            if existing and existing["ids"]:
                self._col.delete(ids=existing["ids"])
        except Exception:
            logger.exception("delete_source failed: %s", source_key)

    # ── read ──
    def query(self, embedding: list[float], top_k: int, where: dict | None = None) -> list[SearchResult]:
        kwargs = {"query_embeddings": [embedding], "n_results": top_k,
                  "include": ["documents", "metadatas", "distances"]}
        if where:
            kwargs["where"] = where
        try:
            res = self._col.query(**kwargs)
        except Exception as e:
            logger.error("Chroma query failed: %s", e)
            return []

        out: list[SearchResult] = []
        if res and res["ids"] and res["ids"][0]:
            for i in range(len(res["ids"][0])):
                raw_meta = dict(res["metadatas"][0][i]) if res["metadatas"] else {}
                parent = raw_meta.get(_PARENT_KEY) or None
                out.append(SearchResult(
                    chunk_id=res["ids"][0][i],
                    text=res["documents"][0][i],
                    score=1.0 - res["distances"][0][i],   # cosine distance → similarity
                    metadata=self._public_meta(raw_meta),
                    parent_text=parent,
                    freshness=self._decode_freshness(raw_meta),
                    ordinal=int(raw_meta.get("ordinal", 0)),
                ))
        return out

    def get_all(self) -> list[ChunkRecord]:
        try:
            data = self._col.get(include=["documents", "metadatas", "embeddings"])
        except Exception:
            logger.exception("get_all failed")
            return []
        records: list[ChunkRecord] = []
        embeds = data.get("embeddings")
        for i, cid in enumerate(data["ids"]):
            meta = dict(data["metadatas"][i]) if data["metadatas"] else {}
            emb = list(embeds[i]) if embeds is not None and len(embeds) > i else None
            records.append(self._from_meta(cid, data["documents"][i], meta, emb))
        return records

    def count(self) -> int:
        return self._col.count()

    def list_sources(self) -> list[dict]:
        try:
            data = self._col.get(include=["metadatas"])
        except Exception:
            logger.exception("list_sources failed")
            return []
        sources: dict[str, dict] = {}
        for m in (data["metadatas"] or []):
            key = m.get("source", "unknown")
            if key not in sources:
                sources[key] = {"source": key, "title": m.get("title", key),
                                "type": m.get("source_type", "?"), "chunks": 0}
            sources[key]["chunks"] += 1
        return list(sources.values())

    def get_source_freshness(self, source_key: str) -> dict:
        try:
            existing = self._col.get(where={"source": {"$eq": source_key}}, include=["metadatas"])
        except Exception:
            return {}
        if existing and existing["ids"]:
            return self._decode_freshness(existing["metadatas"][0] or {})
        return {}

    def touch_source(self, source_key: str, fetched_at: str) -> None:
        """Bump source_fetched_at on all chunks of a source — metadata-only, no re-embed."""
        try:
            existing = self._col.get(where={"source": {"$eq": source_key}}, include=["metadatas"])
        except Exception:
            logger.exception("touch_source failed: %s", source_key)
            return
        if not (existing and existing["ids"]):
            return
        new_metas = []
        for meta in existing["metadatas"]:
            m = dict(meta or {})
            fresh = self._decode_freshness(m)
            fresh["source_fetched_at"] = fetched_at
            m[_FRESHNESS_KEY] = json.dumps(fresh)
            new_metas.append(m)
        self._col.update(ids=existing["ids"], metadatas=new_metas)

    def get_source_hash(self, source_key: str) -> str | None:
        try:
            existing = self._col.get(where={"source": {"$eq": source_key}}, include=["metadatas"])
        except Exception:
            logger.exception("get_source_hash failed: %s", source_key)
            return None
        if existing and existing["ids"]:
            return (existing["metadatas"][0] or {}).get("source_hash", "")
        return None

    # ── metadata (de)serialization ──
    @staticmethod
    def _to_meta(r: ChunkRecord) -> dict:
        meta = {
            "source": r.provenance.get("source", ""),
            "source_type": r.provenance.get("source_type", ""),
            "title": r.provenance.get("title", ""),
            "anchor": r.provenance.get("anchor", ""),
            "section_title": r.section_path,
            "content_hash": r.content_hash,
            "source_hash": r.source_hash,
            "schema_version": r.schema_version,
            "ordinal": r.ordinal,
            _PARENT_KEY: r.parent_text or "",
            _ENTITIES_KEY: json.dumps(r.entities),
            _EDGES_KEY: json.dumps(r.edges),
            _FRESHNESS_KEY: json.dumps(r.freshness),
        }
        # Chroma rejects None; keep only scalar str/int/float/bool
        return {k: ("" if v is None else v) for k, v in meta.items()}

    @staticmethod
    def _public_meta(raw: dict) -> dict:
        """Metadata surfaced to retrievers/orchestrator — internal keys stripped."""
        return {k: v for k, v in raw.items() if k not in _INTERNAL}

    @staticmethod
    def _decode_freshness(raw: dict) -> dict:
        try:
            return json.loads(raw.get(_FRESHNESS_KEY, "") or "{}")
        except Exception:
            return {}

    @staticmethod
    def _from_meta(cid: str, document: str, meta: dict, embedding: list[float] | None) -> ChunkRecord:
        def _load(key):
            try:
                return json.loads(meta.get(key, "") or ("{}" if key == _FRESHNESS_KEY else "[]"))
            except Exception:
                return {} if key == _FRESHNESS_KEY else []

        return ChunkRecord(
            chunk_id=cid,
            raw_text=document,
            embedding=embedding,
            parent_text=(meta.get(_PARENT_KEY) or None),
            section_path=meta.get("section_title", ""),
            provenance={
                "source": meta.get("source", ""),
                "source_type": meta.get("source_type", ""),
                "title": meta.get("title", ""),
                "anchor": meta.get("anchor", ""),
            },
            content_hash=meta.get("content_hash", ""),
            source_hash=meta.get("source_hash", ""),
            schema_version=meta.get("schema_version", "3"),
            ordinal=int(meta.get("ordinal", 0)),
            entities=_load(_ENTITIES_KEY),
            edges=_load(_EDGES_KEY),
            freshness=_load(_FRESHNESS_KEY),
        )
