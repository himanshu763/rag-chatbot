"""MongoDB-backed session store — persists chat sessions across restarts.

Config-driven (Mongo URI/db from RagConfig); pymongo imported lazily so the core
library installs without the `sessions` extra. Message (de)serialization is pure
and independently testable.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from rag.config import RagConfig
from rag.types import Confidence, OrchestratorResult

logger = logging.getLogger(__name__)


def serialize_message(msg: dict) -> dict:
    """In-memory message (may hold an OrchestratorResult) → MongoDB-safe dict."""
    out = {"role": msg["role"], "content": msg["content"]}
    if msg.get("meta") is not None:
        m: OrchestratorResult = msg["meta"]
        out["meta"] = {
            "confidence": m.confidence.value,
            "latency_ms": m.latency_ms,
            "sources": m.sources,
            "intent": m.intent,
            "rewritten_query": m.rewritten_query,
            "fallback": m.fallback,
        }
    return out


def deserialize_message(doc: dict) -> dict:
    """MongoDB document → in-memory message."""
    msg = {"role": doc["role"], "content": doc["content"]}
    if doc.get("meta"):
        d = doc["meta"]
        msg["meta"] = OrchestratorResult(
            response=doc["content"],
            confidence=Confidence(d.get("confidence", "none")),
            latency_ms=d.get("latency_ms", 0.0),
            sources=d.get("sources", []),
            intent=d.get("intent", ""),
            rewritten_query=d.get("rewritten_query", ""),
            fallback=d.get("fallback", False),
        )
    return msg


class SessionStore:
    def __init__(self, config: RagConfig | None = None):
        from pymongo import ASCENDING, MongoClient
        cfg = config or RagConfig()
        self._client = MongoClient(cfg.mongodb_uri, serverSelectionTimeoutMS=3000)
        db = self._client[cfg.mongodb_db]
        self._col = db["chat_sessions"]
        self._client.admin.command("ping")   # fail fast if unreachable
        self._col.create_index([("updated_at", ASCENDING)])

    def load_all(self) -> dict[str, dict]:
        from pymongo import ASCENDING
        sessions: dict[str, dict] = {}
        try:
            for doc in self._col.find({}, sort=[("created_at", ASCENDING)]):
                sid = doc["_id"]
                sessions[sid] = {
                    "name": doc.get("name", sid),
                    "messages": [deserialize_message(m) for m in doc.get("messages", [])],
                }
        except Exception as e:
            logger.error("MongoDB load failed: %s", e)
        return sessions

    def save(self, session_id: str, session: dict) -> None:
        try:
            self._col.update_one(
                {"_id": session_id},
                {"$set": {
                    "name": session["name"],
                    "messages": [serialize_message(m) for m in session["messages"]],
                    "updated_at": datetime.now(timezone.utc),
                }, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
                upsert=True,
            )
        except Exception as e:
            logger.error("MongoDB save failed: %s", e)

    def delete(self, session_id: str) -> None:
        try:
            self._col.delete_one({"_id": session_id})
        except Exception as e:
            logger.error("MongoDB delete failed: %s", e)
