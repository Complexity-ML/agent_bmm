# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Model Fallback Chain — Try models in order, fallback on failure.

Usage:
    llm = FallbackLLM(["gpt-4o", "gpt-4o-mini", "ollama:codellama"])
    response = await llm.chat(messages)  # tries each model in order
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

from agent_bmm.llm.auto_detect import detect_provider
from agent_bmm.llm.backend import LLMBackend, LLMConfig

console = Console()


class FallbackLLM:
    """LLM with fallback chain — tries models in order until one succeeds."""

    def __init__(self, models: list[str]):
        if not models:
            raise ValueError("At least one model required")
        self.models = models
        self._backends: list[LLMBackend] = []
        for model in models:
            provider, base_url, api_key = detect_provider(model)
            # Strip ollama: prefix for the actual model name
            actual_model = model.split(":", 1)[1] if model.startswith("ollama:") else model
            self._backends.append(
                LLMBackend(
                    LLMConfig(
                        provider=provider,
                        base_url=base_url,
                        api_key=api_key,
                        model=actual_model,
                    )
                )
            )

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Try each model in order until one succeeds."""
        last_error = None
        for i, backend in enumerate(self._backends):
            try:
                return await backend.chat(messages, **kwargs)
            except Exception as e:
                last_error = e
                console.print(f"  [yellow]Fallback:[/] {self.models[i]} failed ({e}), trying next...")
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    async def chat_stream(
        self, messages: list[dict[str, str]], on_token: Any = None, **kwargs: Any
    ) -> str:
        """Try each model in order with streaming."""
        last_error = None
        for i, backend in enumerate(self._backends):
            try:
                return await backend.chat_stream(messages, on_token=on_token, **kwargs)
            except Exception as e:
                last_error = e
                console.print(f"  [yellow]Fallback:[/] {self.models[i]} failed ({e}), trying next...")
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    async def close(self):
        for backend in self._backends:
            await backend.close()
