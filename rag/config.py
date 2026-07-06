"""RagConfig — per-instance configuration. Env supplies defaults; no global singleton.

Build with ``RagConfig()`` (pure defaults), ``RagConfig(top_k=3, ...)`` (explicit),
or ``RagConfig.from_env(...)`` (read environment, with explicit kwargs winning).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


@dataclass
class RagConfig:
    # Chunking
    chunk_size: int = 400
    chunk_overlap: int = 80
    min_chunk_size: int = 50

    # Retrieval
    top_k: int = 10
    final_k: int = 5
    similarity_threshold: float = 0.25
    low_confidence_threshold: float = 0.40
    rrf_k: int = 60

    # Provider: "openai" or "azure"
    llm_provider: str = "openai"
    reranker_type: str = "llm"          # "llm" or "crossencoder"

    # Web loader
    use_playwright_requests: bool = False
    web_ssl_verify: bool = True

    # LLM — OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-large"

    # LLM — Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"

    # Generation
    max_tokens: int = 1024
    temperature: float = 0.3

    # Memory / history
    max_turns: int = 10

    # Storage — ChromaDB
    chroma_dir: str = "./chroma_db"
    collection_name: str = "rag_chunks"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Storage — MongoDB (sessions)
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "rag_chatbot"

    # Token budget
    context_tokens: int = 3000
    history_tokens: int = 1000

    # Freshness / staleness
    freshness_enabled: bool = True
    staleness_downgrade: bool = True
    default_ttl_days: int = 180
    ttl_days_by_type: dict = field(default_factory=lambda: {
        "web": 30, "pdf": 365, "docx": 365, "csv": 180, "excel": 180, "text": 365,
    })

    # Graph-based retrieval
    graph_enabled: bool = False
    graph_hops: int = 1
    graph_neighbor_score: float = 0.15
    graph_context_ordering: bool = True
    max_entity_fanout: int = 50

    # Cache-augmented generation
    cache_enabled: bool = False
    cache_ttl_seconds: float = 3600.0
    cache_semantic_threshold: float = 0.97   # conservative: a false hit serves a wrong answer
    cache_max_entries: int = 1000

    # ── provider-resolved helpers ──
    @property
    def embedding_model(self) -> str:
        return (
            self.azure_openai_embedding_deployment
            if self.llm_provider == "azure"
            else self.openai_embedding_model
        )

    @property
    def generation_model(self) -> str:
        return (
            self.azure_openai_deployment
            if self.llm_provider == "azure"
            else self.openai_model
        )

    # ── env constructor ──
    @classmethod
    def from_env(cls, **overrides) -> "RagConfig":
        """Read defaults from environment; explicit ``overrides`` take precedence.

        Loads a .env file if python-dotenv is available (best-effort).
        """
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        env = {
            "llm_provider": os.getenv("LLM_PROVIDER", "openai").lower(),
            "reranker_type": os.getenv("RERANKER_TYPE", "llm").lower(),
            "use_playwright_requests": os.getenv("USE_PLAYWRIGHT_REQUESTS", "false").lower() == "true",
            "web_ssl_verify": os.getenv("WEB_SSL_VERIFY", "true").lower() != "false",
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o"),
            "openai_embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            "azure_openai_api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
            "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "azure_openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            "azure_openai_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            "azure_openai_embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
            "mongodb_uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
            "mongodb_db": os.getenv("MONGODB_DB", "rag_chatbot"),
            "freshness_enabled": os.getenv("FRESHNESS_ENABLED", "true").lower() != "false",
            "default_ttl_days": int(os.getenv("DEFAULT_TTL_DAYS", "180")),
            "graph_enabled": os.getenv("GRAPH_ENABLED", "false").lower() == "true",
            "graph_hops": int(os.getenv("GRAPH_HOPS", "1")),
            "cache_enabled": os.getenv("CACHE_ENABLED", "false").lower() == "true",
            "cache_ttl_seconds": float(os.getenv("CACHE_TTL_SECONDS", "3600")),
        }
        valid = {f.name for f in fields(cls)}
        merged = {**env, **{k: v for k, v in overrides.items() if k in valid}}
        return cls(**merged)
