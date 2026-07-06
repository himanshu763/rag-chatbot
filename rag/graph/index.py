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
