# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
WebSocket Client SDK — Python client for Agent BMM server.

Usage:
    from agent_bmm.server.client import AgentClient

    async with AgentClient("ws://localhost:8765") as client:
        # Simple query
        answer = await client.ask("What is 2+2?")

        # Streaming
        async for event in client.ask_stream("Explain AI"):
            if event["type"] == "token":
                print(event["data"], end="")

        # List tools
        tools = await client.list_tools()
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class AgentClient:
    """Async WebSocket client for Agent BMM server."""

    def __init__(self, url: str = "ws://localhost:8765", timeout: float = 60.0):
        self.url = url
        self.timeout = timeout
        self._ws = None

    async def connect(self):
        """Connect to the server."""
        import websockets

        self._ws = await websockets.connect(self.url, max_size=2**20)
        logger.info("Connected to %s", self.url)

    async def close(self):
        """Close the connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _send(self, data: dict):
        """Send a message to the server."""
        if not self._ws:
            raise RuntimeError("Not connected. Call connect() or use async with.")
        await self._ws.send(json.dumps(data))

    async def _recv(self) -> dict:
        """Receive a message from the server."""
        if not self._ws:
            raise RuntimeError("Not connected.")
        raw = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout)
        return json.loads(raw)

    async def ask(self, query: str) -> str:
        """Send a query and wait for the final answer."""
        await self._send({"type": "query", "text": query})

        while True:
            msg = await self._recv()
            msg_type = msg.get("type", "")

            if msg_type == "answer":
                return msg.get("data", "")
            elif msg_type == "error":
                raise RuntimeError(f"Server error: {msg.get('data', '')}")
            elif msg_type == "done":
                return ""
            # Skip intermediate events (thinking, route, tool_start, etc.)

    async def ask_stream(self, query: str) -> AsyncIterator[dict]:
        """Send a query and stream all events."""
        await self._send({"type": "query", "text": query})

        while True:
            msg = await self._recv()
            yield msg

            if msg.get("type") in ("done", "error"):
                break

    async def list_tools(self) -> list[dict]:
        """List available tools on the server."""
        await self._send({"type": "tools"})
        msg = await self._recv()
        return msg.get("data", [])

    async def ping(self) -> bool:
        """Ping the server."""
        await self._send({"type": "ping"})
        msg = await self._recv()
        return msg.get("type") == "pong"
