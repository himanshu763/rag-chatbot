"""LLMReranker — rank passages with the configured Generator.

Robust to malformed model output: invalid/missing indices are filled from the
input order so the result set is never lost.
"""
from __future__ import annotations

import logging

from rag.interfaces import Generator
from rag.types import SearchResult

logger = logging.getLogger(__name__)


class LLMReranker:
    def __init__(self, generator: Generator):
        self.generator = generator

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        passages = "\n\n".join(f"[{i + 1}] {r.text[:400]}" for i, r in enumerate(results))
        prompt = (
            f"Query: {query}\n\n"
            f"Passages:\n{passages}\n\n"
            f"Rank these {len(results)} passages from most to least relevant to the query. "
            f"Return ONLY a comma-separated list of passage numbers. Example: 3,1,4,2"
        )
        try:
            resp = self.generator.generate(
                [{"role": "user", "content": prompt}],
                system="You are a relevance ranking assistant. Return only comma-separated passage numbers.",
                max_tokens=60,
            ).strip()
            indices = [int(x.strip()) - 1 for x in resp.split(",") if x.strip().isdigit()]
            valid = [i for i in indices if 0 <= i < len(results)]
            seen = set(valid)
            for i in range(len(results)):
                if i not in seen:
                    valid.append(i)
            return [results[i] for i in valid[:top_k]]
        except Exception:
            logger.warning("LLM rerank failed — using input order", exc_info=True)
            return results[:top_k]
