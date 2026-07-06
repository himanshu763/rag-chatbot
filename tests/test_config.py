"""Tests for rag.config.RagConfig — per-instance config, env supplies defaults only."""
from __future__ import annotations

from rag.config import RagConfig


def test_defaults_are_sensible():
    cfg = RagConfig()
    assert cfg.chunk_size == 400
    assert cfg.top_k == 10
    assert cfg.final_k == 5
    assert cfg.collection_name == "rag_chunks"
    assert cfg.llm_provider in ("openai", "azure")


def test_per_instance_override_no_global():
    """Two configs are independent — no shared global settings object."""
    a = RagConfig(collection_name="tenant_a", top_k=3)
    b = RagConfig(collection_name="tenant_b", top_k=7)
    assert a.collection_name == "tenant_a" and a.top_k == 3
    assert b.collection_name == "tenant_b" and b.top_k == 7


def test_from_env_reads_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deploy")
    cfg = RagConfig.from_env()
    assert cfg.llm_provider == "azure"
    assert cfg.openai_api_key == "sk-test"
    assert cfg.azure_openai_deployment == "my-deploy"


def test_from_env_explicit_overrides_win(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    cfg = RagConfig.from_env(llm_provider="openai")
    assert cfg.llm_provider == "openai"


def test_embedding_model_for_provider():
    openai_cfg = RagConfig(llm_provider="openai", openai_embedding_model="text-embedding-3-large")
    azure_cfg = RagConfig(llm_provider="azure", azure_openai_embedding_deployment="embed-deploy")
    assert openai_cfg.embedding_model == "text-embedding-3-large"
    assert azure_cfg.embedding_model == "embed-deploy"


def test_generation_model_for_provider():
    openai_cfg = RagConfig(llm_provider="openai", openai_model="gpt-4o")
    azure_cfg = RagConfig(llm_provider="azure", azure_openai_deployment="gpt-deploy")
    assert openai_cfg.generation_model == "gpt-4o"
    assert azure_cfg.generation_model == "gpt-deploy"


def test_freshness_config_defaults():
    cfg = RagConfig()
    assert cfg.freshness_enabled is True
    assert cfg.staleness_downgrade is True
    assert cfg.default_ttl_days == 180
    assert cfg.ttl_days_by_type["web"] == 30
    assert cfg.ttl_days_by_type["pdf"] == 365


def test_freshness_config_from_env(monkeypatch):
    monkeypatch.setenv("FRESHNESS_ENABLED", "false")
    monkeypatch.setenv("DEFAULT_TTL_DAYS", "90")
    cfg = RagConfig.from_env()
    assert cfg.freshness_enabled is False
    assert cfg.default_ttl_days == 90


def test_ttl_maps_are_independent_per_instance():
    a = RagConfig()
    a.ttl_days_by_type["web"] = 7
    b = RagConfig()
    assert b.ttl_days_by_type["web"] == 30


def test_graph_config_defaults():
    cfg = RagConfig()
    assert cfg.graph_enabled is False           # opt-in
    assert cfg.graph_hops == 1
    assert cfg.graph_neighbor_score == 0.15
    assert cfg.graph_context_ordering is True
    assert cfg.max_entity_fanout == 50


def test_graph_config_from_env(monkeypatch):
    monkeypatch.setenv("GRAPH_ENABLED", "true")
    monkeypatch.setenv("GRAPH_HOPS", "2")
    cfg = RagConfig.from_env()
    assert cfg.graph_enabled is True
    assert cfg.graph_hops == 2


def test_cache_config_defaults():
    cfg = RagConfig()
    assert cfg.cache_enabled is False           # opt-in
    assert cfg.cache_ttl_seconds == 3600.0
    assert cfg.cache_semantic_threshold == 0.97
    assert cfg.cache_max_entries == 1000


def test_cache_config_from_env(monkeypatch):
    monkeypatch.setenv("CACHE_ENABLED", "true")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "60")
    cfg = RagConfig.from_env()
    assert cfg.cache_enabled is True
    assert cfg.cache_ttl_seconds == 60.0
