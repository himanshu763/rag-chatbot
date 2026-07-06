"""Orchestrator — intent → rewrite → retrieve → confidence → generate.

Dependencies (retriever, generator, config) are injected; no global state. The
confidence ladder and context assembly are pure static methods so they can be
tested branch-by-branch.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Generator as Gen

from rag.config import RagConfig
from rag.freshness import freshness_note, is_stale
from rag.interfaces import Generator, Retriever
from rag.types import Confidence, Intent, OrchestratorResult, SearchResult

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
- Answer based ONLY on the context below. NEVER fabricate information.
- If the context fully answers the question, answer clearly and cite which source you used.
- If it partially answers, answer what you can and state what's missing.
- If the context doesn't answer, say "I don't have enough information in my knowledge base to answer that."
- Be concise. Don't repeat the question back.
{confidence_note}

Context:
{context}"""

CONFIDENCE_NOTES = {
    Confidence.HIGH: "The context is highly relevant. Provide a confident, detailed answer.",
    Confidence.MEDIUM: "The context may only partially cover the question. Answer what you can, note gaps.",
    Confidence.LOW: "The context has low relevance. Only answer if you find directly relevant info, otherwise say you don't have enough information.",
}

INTENT_PROMPT = """Classify the user query intent. Categories:
- FACTUAL: any question about a topic, product, feature, data, or concept
- COMPARISON: comparing two or more things
- FOLLOW_UP: refers to previous conversation turn (uses "it", "that", "they", etc.)
- CLARIFY: asking for clarification of a previous answer
- GREETING: pure greeting with no question (hi, hello, good morning)
- OUT_OF_SCOPE: clearly personal/unrelated requests (jokes, weather, cooking, etc.) — use sparingly, default to FACTUAL if unsure

Conversation context: {context}
User query: {query}
Respond with ONLY the category name."""

REWRITE_PROMPT = """Rewrite as a self-contained search query using conversation context. If already clear, return unchanged.
Context: {context}
User: {query}
Rewritten query:"""

_GREETING_ONLY_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|howdy|greetings|good\s+(?:morning|afternoon|evening|day)"
    r"|what'?s\s+up|sup)[\s!?.,]*$",
    re.IGNORECASE,
)


def _fast_classify(query: str) -> Intent | None:
    if _GREETING_ONLY_RE.match(query):
        return Intent.GREETING
    return None


class Orchestrator:
    def __init__(self, retriever: Retriever, generator: Generator, config: RagConfig | None = None):
        self.retriever = retriever
        self.generator = generator
        self.config = config or RagConfig()

    # ── sync ──
    def process(self, user_msg: str, history: list[dict] | None = None) -> OrchestratorResult:
        start = time.time()
        history = history or []
        conv_ctx = self._history_text(history[-6:])

        intent = self._classify(user_msg, conv_ctx)
        logger.info("intent=%s query=%r", intent, user_msg)

        if intent == Intent.GREETING:
            return OrchestratorResult(
                response="Hello! I can answer questions based on my knowledge base. What would you like to know?",
                confidence=Confidence.HIGH, intent=intent.value,
                latency_ms=(time.time() - start) * 1000)
        if intent == Intent.OUT_OF_SCOPE:
            return OrchestratorResult(
                response="That's outside the scope of my knowledge base. I can only answer questions about the documents and pages that have been added. Is there something else I can help with?",
                confidence=Confidence.NONE, intent=intent.value, fallback=True,
                latency_ms=(time.time() - start) * 1000)

        query = self._rewrite(user_msg, conv_ctx) if intent == Intent.FOLLOW_UP else user_msg
        results = self.retriever.retrieve(query)
        conf = self._assess_confidence(results, self.config)
        logger.info("retrieved=%d top=%.3f confidence=%s", len(results),
                    results[0].score if results else 0.0, conf)

        if conf == Confidence.NONE:
            return OrchestratorResult(
                response="I don't have enough information in my knowledge base to answer that. Could you rephrase, or ask about a topic covered in my sources?",
                confidence=conf, intent=intent.value, rewritten_query=query,
                fallback=True, latency_ms=(time.time() - start) * 1000)

        conf, staleness_note = self._apply_staleness(conf, results, self.config)
        system = SYSTEM_PROMPT.format(
            confidence_note=CONFIDENCE_NOTES.get(conf, ""),
            context=self._assemble_context(results, self.config))
        response = self.generator.generate(self._build_messages(history, user_msg), system=system)
        return OrchestratorResult(
            response=response, sources=self._extract_sources(results), confidence=conf,
            intent=intent.value, rewritten_query=query, staleness_note=staleness_note,
            latency_ms=(time.time() - start) * 1000)

    # ── streaming ──
    def process_stream(self, user_msg: str, history: list[dict] | None = None) -> Gen[str | OrchestratorResult, None, None]:
        start = time.time()
        history = history or []
        conv_ctx = self._history_text(history[-6:])

        intent = self._classify(user_msg, conv_ctx)
        if intent in (Intent.GREETING, Intent.OUT_OF_SCOPE):
            result = self.process(user_msg, history)
            yield result.response
            yield result
            return

        query = self._rewrite(user_msg, conv_ctx) if intent == Intent.FOLLOW_UP else user_msg
        results = self.retriever.retrieve(query)
        conf = self._assess_confidence(results, self.config)

        if conf == Confidence.NONE:
            msg = "I don't have enough information in my knowledge base to answer that. Could you rephrase, or ask about a topic covered in my sources?"
            yield msg
            yield OrchestratorResult(
                response=msg, confidence=conf, intent=intent.value,
                rewritten_query=query, fallback=True,
                latency_ms=(time.time() - start) * 1000)
            return

        conf, staleness_note = self._apply_staleness(conf, results, self.config)
        system = SYSTEM_PROMPT.format(
            confidence_note=CONFIDENCE_NOTES.get(conf, ""),
            context=self._assemble_context(results, self.config))
        full = ""
        for chunk in self.generator.stream(self._build_messages(history, user_msg), system=system):
            full += chunk
            yield chunk
        yield OrchestratorResult(
            response=full, sources=self._extract_sources(results), confidence=conf,
            intent=intent.value, rewritten_query=query, staleness_note=staleness_note,
            latency_ms=(time.time() - start) * 1000)

    # ── helpers ──
    def _classify(self, query: str, ctx: str) -> Intent:
        fast = _fast_classify(query)
        if fast:
            return fast
        try:
            resp = self.generator.generate(
                [{"role": "user", "content": INTENT_PROMPT.format(query=query, context=ctx or "None")}],
                system="Respond with ONLY the category.", max_tokens=15,
            ).strip().upper().replace(" ", "_")
            return Intent(resp.lower()) if resp.lower() in Intent._value2member_map_ else Intent.FACTUAL
        except Exception:
            logger.exception("Intent classify failed — defaulting to FACTUAL")
            return Intent.FACTUAL

    def _rewrite(self, query: str, ctx: str) -> str:
        try:
            r = self.generator.generate(
                [{"role": "user", "content": REWRITE_PROMPT.format(query=query, context=ctx)}],
                system="Rewrite concisely.", max_tokens=80,
            ).strip()
            return r or query
        except Exception:
            logger.exception("Query rewrite failed — using original query")
            return query

    _DOWNGRADE = {Confidence.HIGH: Confidence.MEDIUM,
                  Confidence.MEDIUM: Confidence.LOW,
                  Confidence.LOW: Confidence.LOW}

    @staticmethod
    def _apply_staleness(conf, results, config, now=None):
        """Return (possibly downgraded confidence, note). No-op when disabled or fresh."""
        if not (config.freshness_enabled and config.staleness_downgrade) or not results:
            return conf, ""
        now = now or datetime.now(timezone.utc)
        top = results[0]   # same top-result convention as _assess_confidence + top citation
        fresh = getattr(top, "freshness", {}) or {}
        if is_stale(fresh, config, now):
            return Orchestrator._DOWNGRADE.get(conf, conf), freshness_note(fresh, now)
        return conf, ""

    @staticmethod
    def _assess_confidence(results: list[SearchResult], config: RagConfig) -> Confidence:
        if not results:
            return Confidence.NONE
        # Judge confidence on seed (query-matched) results only — graph-expanded neighbors
        # carry a dampened score and would otherwise suppress/depress the verdict.
        seeds = [r for r in results if not getattr(r, "graph_neighbor", False)] or results
        top = seeds[0].score                       # top-ranked seed (order preserved)
        avg = sum(r.score for r in seeds) / len(seeds)
        if top < config.similarity_threshold:
            return Confidence.NONE
        if top < config.low_confidence_threshold:
            return Confidence.LOW
        if top >= config.low_confidence_threshold and avg > config.low_confidence_threshold:
            return Confidence.HIGH
        return Confidence.MEDIUM

    @staticmethod
    def _order_for_context(results: list[SearchResult], config: RagConfig) -> list[SearchResult]:
        if not (config.graph_enabled and config.graph_context_ordering):
            return sorted(results, key=lambda r: r.score, reverse=True)
        # group by source; order within group by ordinal; order groups by best score
        groups: dict[str, list[SearchResult]] = {}
        for r in results:
            groups.setdefault(r.metadata.get("source", ""), []).append(r)
        ordered_groups = sorted(
            groups.values(), key=lambda g: max(x.score for x in g), reverse=True)
        out: list[SearchResult] = []
        for g in ordered_groups:
            out.extend(sorted(g, key=lambda r: getattr(r, "ordinal", 0)))
        return out

    @staticmethod
    def _assemble_context(results: list[SearchResult], config: RagConfig) -> str:
        results_sorted = Orchestrator._order_for_context(results, config)
        blocks, tokens = [], 0
        for i, r in enumerate(results_sorted, 1):
            text = r.parent_text if r.parent_text else r.text
            src = r.metadata.get("title", r.metadata.get("source", "?"))
            sec = r.metadata.get("section_title", "")
            label = f"[Source {i}: {src}" + (f" > {sec}]" if sec else "]")
            block = f"{label}\n{text}"
            est = len(text.split()) * 1.3
            if tokens + est > config.context_tokens:
                block = f"{label}\n{r.text}"
                est = len(r.text.split()) * 1.3
                if tokens + est > config.context_tokens:
                    break
            blocks.append(block)
            tokens += est
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _extract_sources(results: list[SearchResult]) -> list[dict]:
        seen, sources = set(), []
        for r in results:
            key = r.metadata.get("source", "")
            if key and key not in seen:
                seen.add(key)
                sources.append({
                    "source": key,
                    "title": r.metadata.get("title", key),
                    "section": r.metadata.get("section_title", ""),
                    "score": round(r.score, 3),
                })
        return sources

    @staticmethod
    def _build_messages(history: list[dict], user_msg: str) -> list[dict]:
        msgs = [{"role": h["role"], "content": h["content"]} for h in history[-8:]]
        msgs.append({"role": "user", "content": user_msg})
        return msgs

    @staticmethod
    def _history_text(history: list[dict]) -> str:
        return "\n".join(f"{h['role']}: {h['content']}" for h in history) if history else ""
