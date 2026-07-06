"""CrossEncoderReranker — local HuggingFace cross-encoder (optional dep).

Requires the `sentence-transformers` extra. Model is loaded lazily on first use
and held on the instance (no module global).
"""
from __future__ import annotations

import logging
import math

from rag.config import RagConfig
from rag.types import SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, config: RagConfig):
        self.config = config
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.config.cross_encoder_model)
        return self._model

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        try:
            model = self._get_model()
        except Exception:
            logger.warning("Cross-encoder load failed — using input order", exc_info=True)
            return results[:top_k]
        pairs = [(query, r.text) for r in results]
        scores = model.predict(pairs)
        for r, s in zip(results, scores):
            r.score = 1.0 / (1.0 + math.exp(-float(s)))   # sigmoid → 0..1
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
