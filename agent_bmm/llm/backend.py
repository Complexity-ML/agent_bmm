# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
LLM Backend — Unified interface for any LLM provider.

Supports vLLM (local), OpenAI API, Anthropic Claude API, or any
OpenAI-compatible endpoint. The agent doesn't care which LLM runs
behind the API — it just needs hidden states or text completions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import aiohttp
import orjson


def _load_dotenv():
    """Load .env file into os.environ (no external dependency)."""
    for path in [".env", "../.env"]:
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        value = value.replace("\x00", "")
                        if key and value and not os.environ.get(key):
                            os.environ[key] = value
        except FileNotFoundError:
            continue


@dataclass
class LLMConfig:
    """Configuration for an LLM backend."""

    provider: str = "openai"  # "openai", "anthropic", "vllm", "local"
    base_url: str = "http://localhost:8081/v1"
    api_key: str = ""
    model: str = ""
    max_tokens: int = 512
    temperature: float = 0.7

    def __post_init__(self):
        # Load .env file if it exists
        _load_dotenv()
        if not self.api_key:
            if self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY", "")
            elif self.provider == "anthropic":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")


class LLMBackend:
    """
    Async LLM client — works with any OpenAI-compatible API.

    For vLLM: point base_url to your vLLM server.
    For OpenAI: use default base_url.
    For Claude: set provider="anthropic".
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                json_serialize=lambda x: orjson.dumps(x).decode(),
            )
        return self._session

    async def complete(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text completion."""
        session = await self._get_session()
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
            **kwargs,
        }
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers["Content-Type"] = "application/json"

        url = f"{self.config.base_url}/completions"
        async with session.post(url, json=payload, headers=headers) as resp:
            data = await resp.json()
            return data["choices"][0]["text"]

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a chat completion."""
        session = await self._get_session()
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
            **kwargs,
        }
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers["Content-Type"] = "application/json"

        url = f"{self.config.base_url}/chat/completions"
        async with session.post(url, json=payload, headers=headers) as resp:
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_token: Any = None,
        **kwargs: Any,
    ) -> str:
        """Stream a chat completion token by token. Returns full response."""
        import json as json_mod

        session = await self._get_session()
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
            "stream": True,
            **kwargs,
        }
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers["Content-Type"] = "application/json"

        url = f"{self.config.base_url}/chat/completions"
        full_response = []

        async with session.post(url, json=payload, headers=headers) as resp:
            async for line in resp.content:
                line = line.decode(errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json_mod.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content", "")
                    if content:
                        full_response.append(content)
                        if on_token:
                            on_token(content)
                except (json_mod.JSONDecodeError, KeyError, IndexError):
                    continue

        return "".join(full_response)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
