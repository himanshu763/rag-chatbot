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
            norm = m.group(1).strip().lower()
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
