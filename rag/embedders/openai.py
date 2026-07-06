"""OpenAIEmbedder — dense embeddings via OpenAI or Azure OpenAI.

The SDK client is built from config, or injected (for tests). No global client.
"""
from __future__ import annotations

from rag.config import RagConfig


def _build_client(config: RagConfig):
    from openai import AzureOpenAI, OpenAI
    if config.llm_provider == "azure":
        return AzureOpenAI(
            api_key=config.azure_openai_api_key,
            azure_endpoint=config.azure_openai_endpoint,
            api_version=config.azure_openai_api_version,
        )
    return OpenAI(api_key=config.openai_api_key)


class OpenAIEmbedder:
    def __init__(self, config: RagConfig, client=None):
        self.config = config
        self.client = client or _build_client(config)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.config.embedding_model, input=texts)
        return [item.embedding for item in resp.data]
