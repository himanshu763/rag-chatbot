"""OpenAIGenerator — chat completion + streaming via OpenAI or Azure OpenAI.

Conforms to the Generator protocol: generate(messages, system, max_tokens) and
stream(...). Client built from config or injected (tests). No global client.
"""
from __future__ import annotations

from typing import Iterator

from rag.config import RagConfig
from rag.embedders.openai import _build_client


class OpenAIGenerator:
    def __init__(self, config: RagConfig, client=None):
        self.config = config
        self.client = client or _build_client(config)
        self.model = config.generation_model

    def _messages(self, messages: list[dict], system: str) -> list[dict]:
        return [
            {"role": "system", "content": system or "You are a helpful assistant."},
            *messages,
        ]

    def generate(self, messages: list[dict], system: str = "", max_tokens: int | None = None) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
            messages=self._messages(messages, system),
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict], system: str = "", max_tokens: int | None = None) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
            messages=self._messages(messages, system),
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
