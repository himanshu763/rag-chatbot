"""Tests for OpenAIEmbedder + OpenAIGenerator using injected fake SDK clients
(no live API). Verifies provider-correct model selection and request shape."""
from __future__ import annotations

from types import SimpleNamespace

from rag.config import RagConfig
from rag.embedders.openai import OpenAIEmbedder
from rag.generators.openai import OpenAIGenerator


# ── fake OpenAI SDK client ──
class _FakeEmbeddings:
    def __init__(self, sink):
        self.sink = sink

    def create(self, model, input):
        self.sink["model"] = model
        self.sink["input"] = input
        data = [SimpleNamespace(embedding=[float(len(t))]) for t in input]
        return SimpleNamespace(data=data)


class _FakeChat:
    def __init__(self, sink):
        self.completions = self
        self.sink = sink

    def create(self, model, messages, max_completion_tokens, temperature, stream=False):
        self.sink.update(model=model, messages=messages,
                         max_completion_tokens=max_completion_tokens, stream=stream)
        if stream:
            chunks = []
            for word in ["hello", "there"]:
                delta = SimpleNamespace(content=word + " ")
                chunks.append(SimpleNamespace(choices=[SimpleNamespace(delta=delta)]))
            return iter(chunks)
        msg = SimpleNamespace(content="full answer")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _FakeClient:
    def __init__(self):
        self.sink = {}
        self.embeddings = _FakeEmbeddings(self.sink)
        self.chat = _FakeChat(self.sink)


def test_embedder_uses_openai_model_and_returns_vectors():
    c = _FakeClient()
    emb = OpenAIEmbedder(RagConfig(llm_provider="openai", openai_embedding_model="text-embedding-3-large"), client=c)
    out = emb.embed(["ab", "abcd"])
    assert out == [[2.0], [4.0]]
    assert c.sink["model"] == "text-embedding-3-large"


def test_embedder_uses_azure_deployment():
    c = _FakeClient()
    emb = OpenAIEmbedder(RagConfig(llm_provider="azure", azure_openai_embedding_deployment="embed-deploy"), client=c)
    emb.embed(["x"])
    assert c.sink["model"] == "embed-deploy"


def test_generator_generate_returns_content_and_prepends_system():
    c = _FakeClient()
    gen = OpenAIGenerator(RagConfig(openai_model="gpt-4o"), client=c)
    out = gen.generate([{"role": "user", "content": "hi"}], system="be nice")
    assert out == "full answer"
    assert c.sink["model"] == "gpt-4o"
    assert c.sink["messages"][0] == {"role": "system", "content": "be nice"}
    assert c.sink["messages"][1] == {"role": "user", "content": "hi"}


def test_generator_max_tokens_override():
    c = _FakeClient()
    gen = OpenAIGenerator(RagConfig(max_tokens=1024), client=c)
    gen.generate([{"role": "user", "content": "hi"}], max_tokens=15)
    assert c.sink["max_completion_tokens"] == 15


def test_generator_stream_yields_deltas():
    c = _FakeClient()
    gen = OpenAIGenerator(RagConfig(), client=c)
    out = "".join(gen.stream([{"role": "user", "content": "hi"}], system="s"))
    assert out == "hello there "
    assert c.sink["stream"] is True
