# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Rate Limiter — Respect API rate limits with exponential backoff.

Wraps LLMBackend to add rate limiting, retry on 429, and request queuing.

Configured via agent-bmm.yaml:
    llm:
      rate_limit:
        requests_per_minute: 60
        max_retries: 5
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from agent_bmm.llm.backend import LLMBackend


class RateLimitedLLM:
    """LLMBackend wrapper with rate limiting and exponential backoff."""

    def __init__(
        self,
        backend: LLMBackend,
        requests_per_minute: int = 60,
        max_retries: int = 5,
    ):
        self.backend = backend
        self.rpm = requests_per_minute
        self.max_retries = max_retries
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def _wait_for_slot(self):
        """Wait until we have a rate limit slot available."""
        async with self._lock:
            now = time.time()
            # Remove timestamps older than 60s
            self._timestamps = [t for t in self._timestamps if now - t < 60]
            if len(self._timestamps) >= self.rpm:
                wait = 60 - (now - self._timestamps[0]) + 0.1
                await asyncio.sleep(wait)
            self._timestamps.append(time.time())

    async def _retry_with_backoff(self, fn, *args, **kwargs) -> Any:
        """Call fn with exponential backoff on failure."""
        last_error = None
        for attempt in range(self.max_retries):
            await self._wait_for_slot()
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # Only retry on rate limit or transient errors
                if "429" in err_str or "rate" in err_str or "timeout" in err_str:
                    delay = (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                else:
                    raise
        raise RuntimeError(f"Max retries ({self.max_retries}) exceeded. Last error: {last_error}")

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        return await self._retry_with_backoff(self.backend.chat, messages, **kwargs)

    async def chat_stream(self, messages: list[dict[str, str]], **kwargs) -> str:
        return await self._retry_with_backoff(self.backend.chat_stream, messages, **kwargs)

    async def complete(self, prompt: str, **kwargs) -> str:
        return await self._retry_with_backoff(self.backend.complete, prompt, **kwargs)

    async def close(self):
        await self.backend.close()

    @property
    def config(self):
        return self.backend.config
