# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Streaming — Real-time token streaming for agent responses.

Supports both SSE (Server-Sent Events) for HTTP clients
and WebSocket for interactive applications.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Callable

import aiohttp


class StreamingHandler:
    """
    Handles real-time streaming of agent responses.

    Yields tokens as they arrive from the LLM, interleaved
    with tool execution status updates.
    """

    def __init__(self):
        self._listeners: list[Callable[[str, str], Any]] = []
        self._buffer: list[str] = []

    def on_event(self, callback: Callable[[str, str], Any]):
        """Register an event listener. callback(event_type, data)."""
        self._listeners.append(callback)

    async def _emit(self, event_type: str, data: str):
        for listener in self._listeners:
            result = listener(event_type, data)
            if asyncio.iscoroutine(result):
                await result

    async def stream_llm(
        self,
        base_url: str,
        model: str,
        messages: list[dict],
        api_key: str = "",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream tokens from an LLM endpoint."""
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                async for line in resp.content:
                    line = line.decode().strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        await self._emit("done", "")
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]
                        content = delta.get("content", "")
                        if content:
                            self._buffer.append(content)
                            await self._emit("token", content)
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def emit_tool_start(self, tool_name: str, query: str):
        """Notify listeners that a tool is starting."""
        await self._emit("tool_start", json.dumps({"tool": tool_name, "query": query}))

    async def emit_tool_result(self, tool_name: str, result: str):
        """Notify listeners that a tool finished."""
        await self._emit(
            "tool_result", json.dumps({"tool": tool_name, "result": result[:500]})
        )

    async def emit_thinking(self, thought: str):
        """Notify listeners of agent thinking."""
        await self._emit("thinking", thought)

    @property
    def full_response(self) -> str:
        return "".join(self._buffer)


class SSEFormatter:
    """Format events as Server-Sent Events for HTTP streaming."""

    @staticmethod
    def format(event_type: str, data: str) -> str:
        return f"event: {event_type}\ndata: {data}\n\n"

    @staticmethod
    def done() -> str:
        return "data: [DONE]\n\n"
