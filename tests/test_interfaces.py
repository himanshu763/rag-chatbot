"""Contract tests — fakes conform to their Protocols (runtime_checkable)."""
from __future__ import annotations

from rag.interfaces import (
    Embedder,
    Generator,
    KeywordIndex,
    VectorStore,
)
from tests.fakes import (
    FakeEmbedder,
    FakeGenerator,
    FakeKeywordIndex,
    FakeVectorStore,
)


def test_fake_embedder_conforms():
    assert isinstance(FakeEmbedder(), Embedder)


def test_fake_vector_store_conforms():
    assert isinstance(FakeVectorStore(), VectorStore)


def test_fake_keyword_index_conforms():
    assert isinstance(FakeKeywordIndex(), KeywordIndex)


def test_fake_generator_conforms():
    assert isinstance(FakeGenerator(), Generator)


def test_embedder_is_deterministic():
    e = FakeEmbedder()
    assert e.embed(["hello"]) == e.embed(["hello"])
    assert e.embed(["hello"]) != e.embed(["world"])
