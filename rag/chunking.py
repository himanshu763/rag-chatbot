"""Structure-aware chunker — respects heading boundaries, never splits mid-sentence.

Config-driven (chunk_size / chunk_overlap / min_chunk_size); no global settings.
Stores parent_text for parent-child retrieval.
"""
from __future__ import annotations

import re
import uuid

from rag.config import RagConfig
from rag.types import Chunk, RawDocument

SENT_RE = re.compile(r'(?<=[.!?])\s+')


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


class Chunker:
    def __init__(self, config: RagConfig | None = None):
        self.config = config or RagConfig()

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        sections = doc.sections or [{"heading": "", "text": doc.text}]

        for section in sections:
            heading = section.get("heading", "")
            text = section.get("text", "")
            anchor = section.get("anchor", "")
            if not text.strip():
                continue
            meta = {**doc.metadata, "anchor": anchor} if anchor else dict(doc.metadata)
            section_chunks = self._chunk_section(text, heading, meta)
            parent = (f"{heading}\n{text}" if heading else text).strip()
            for c in section_chunks:
                c.parent_text = parent
            all_chunks.extend(section_chunks)
        return all_chunks

    def _chunk_section(self, text: str, heading: str, base_meta: dict) -> list[Chunk]:
        cfg = self.config
        sentences = [s.strip() for s in SENT_RE.split(text) if s.strip()]
        if not sentences:
            return []

        chunks, current, cur_tok = [], [], 0
        for sent in sentences:
            st = _approx_tokens(sent)
            if cur_tok + st > cfg.chunk_size and current:
                chunks.append(self._make(current, heading, base_meta))
                overlap, ot = [], 0
                for s in reversed(current):
                    t = _approx_tokens(s)
                    if ot + t > cfg.chunk_overlap:
                        break
                    overlap.insert(0, s)
                    ot += t
                current, cur_tok = overlap, ot
            current.append(sent)
            cur_tok += st

        if current and _approx_tokens(" ".join(current)) >= cfg.min_chunk_size:
            chunks.append(self._make(current, heading, base_meta))
        return chunks

    @staticmethod
    def _make(sentences: list[str], heading: str, meta: dict) -> Chunk:
        text = " ".join(sentences)
        return Chunk(
            chunk_id=uuid.uuid4().hex[:12],
            text=text,
            metadata={**meta, "section_title": heading, "char_count": len(text)},
        )
