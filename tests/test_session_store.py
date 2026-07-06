"""SessionStore message (de)serialization — pure, no MongoDB needed."""
from __future__ import annotations

from rag.session_store import deserialize_message, serialize_message
from rag.types import Confidence, OrchestratorResult


def test_serialize_plain_message():
    out = serialize_message({"role": "user", "content": "hi"})
    assert out == {"role": "user", "content": "hi"}
    assert "meta" not in out


def test_serialize_message_with_meta():
    meta = OrchestratorResult(response="a", confidence=Confidence.HIGH, latency_ms=12.0,
                              sources=[{"source": "s"}], intent="factual", fallback=False)
    out = serialize_message({"role": "assistant", "content": "a", "meta": meta})
    assert out["meta"]["confidence"] == "high"
    assert out["meta"]["sources"] == [{"source": "s"}]


def test_roundtrip_preserves_meta():
    meta = OrchestratorResult(response="a", confidence=Confidence.MEDIUM, latency_ms=5.0,
                              intent="factual", rewritten_query="q", fallback=True)
    doc = serialize_message({"role": "assistant", "content": "a", "meta": meta})
    back = deserialize_message(doc)
    assert isinstance(back["meta"], OrchestratorResult)
    assert back["meta"].confidence is Confidence.MEDIUM
    assert back["meta"].fallback is True
    assert back["meta"].rewritten_query == "q"


def test_deserialize_plain_message():
    back = deserialize_message({"role": "user", "content": "hi"})
    assert back == {"role": "user", "content": "hi"}
