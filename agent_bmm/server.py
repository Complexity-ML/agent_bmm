# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
WebSocket Server — Real-time agent API.

Exposes the BMM agent via WebSocket for integration into
web apps, mobile apps, or any client that speaks WS.

Protocol:
    Client → {"type": "query", "text": "What is X?"}
    Server → {"type": "thinking", "data": "..."}
    Server → {"type": "route", "experts": [0, 2], "names": ["search", "rag"]}
    Server → {"type": "tool_start", "tool": "search", "query": "..."}
    Server → {"type": "tool_result", "tool": "search", "result": "..."}
    Server → {"type": "token", "data": "The"}
    Server → {"type": "token", "data": " answer"}
    Server → {"type": "answer", "data": "The answer is..."}
    Server → {"type": "done"}
"""

from __future__ import annotations

import asyncio
import json
import time

import websockets
from websockets.server import serve

from agent_bmm.agent import Agent
from agent_bmm.core.logging import AgentLogger, console


class AgentWebSocketServer:
    """
    WebSocket server for BMM agent.

    Usage:
        server = AgentWebSocketServer(agent, port=8765)
        await server.start()
    """

    def __init__(
        self,
        agent: Agent,
        host: str = "0.0.0.0",
        port: int = 8765,
        max_connections: int = 100,
    ):
        self.agent = agent
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self._connections: set = set()
        self._logger = AgentLogger()

    async def _handle_client(self, websocket):
        """Handle a single WebSocket client."""
        self._connections.add(websocket)
        client_id = id(websocket)
        console.print(f"[cyan]Client {client_id} connected[/] ({len(self._connections)} total)")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "data": "Invalid JSON"
                    }))
                    continue

                msg_type = data.get("type", "")

                if msg_type == "query":
                    query = data.get("text", "")
                    if not query:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "data": "Empty query"
                        }))
                        continue

                    # Process query with streaming events
                    await self._process_query(websocket, query)

                elif msg_type == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))

                elif msg_type == "tools":
                    # List available tools
                    chain = self.agent._build_chain()
                    tools = []
                    for i in range(chain.tools.num_tools):
                        t = chain.tools.get(i)
                        tools.append({
                            "index": i,
                            "name": t.name,
                            "description": t.description,
                        })
                    await websocket.send(json.dumps({
                        "type": "tools",
                        "data": tools,
                    }))

                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "data": f"Unknown message type: {msg_type}"
                    }))

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._connections.discard(websocket)
            console.print(f"[dim]Client {client_id} disconnected[/]")

    async def _process_query(self, websocket, query: str):
        """Process a query and stream events to the client."""
        t0 = time.time()

        await websocket.send(json.dumps({
            "type": "start",
            "query": query,
        }))

        try:
            # Build chain and run
            chain = self.agent._build_chain()

            # Override chain to emit WS events
            answer = await self._run_with_events(chain, websocket, query)

            await websocket.send(json.dumps({
                "type": "answer",
                "data": answer,
                "time_ms": (time.time() - t0) * 1000,
            }))

        except Exception as e:
            await websocket.send(json.dumps({
                "type": "error",
                "data": str(e),
            }))

        await websocket.send(json.dumps({"type": "done"}))

    async def _run_with_events(self, chain, websocket, query: str) -> str:
        """Run agent chain with WebSocket event streaming."""
        chain.memory.clear_chain()
        chain.memory.add_turn("user", query)

        for step in range(chain.config.max_steps):
            # Think
            thought = await chain._think()
            await websocket.send(json.dumps({
                "type": "thinking",
                "step": step + 1,
                "data": thought[:500],
            }))

            if chain.config.stop_on_final_answer and "[FINAL]" in thought:
                return thought.split("[FINAL]")[-1].strip()

            # Route
            tool_ids = chain._route(thought)
            await websocket.send(json.dumps({
                "type": "route",
                "step": step + 1,
                "expert_ids": tool_ids,
            }))

            # Act
            unique_tools = set(tool_ids)
            for tid in unique_tools:
                if tid < chain.tools.num_tools:
                    tool = chain.tools.get(tid)
                    await websocket.send(json.dumps({
                        "type": "tool_start",
                        "tool": tool.name,
                        "query": query[:200],
                    }))

            results = await chain._act(tool_ids, query)

            for result in results:
                chain.memory.add_tool_result(result)
                await websocket.send(json.dumps({
                    "type": "tool_result",
                    "tool": result.tool_name,
                    "result": result.result[:500],
                }))

        return await chain._finalize()

    async def start(self):
        """Start the WebSocket server."""
        console.print(
            f"[bold cyan]Agent BMM WebSocket Server[/] "
            f"listening on ws://{self.host}:{self.port}"
        )
        async with serve(
            self._handle_client,
            self.host,
            self.port,
            max_size=2**20,  # 1MB max message
        ):
            await asyncio.Future()  # run forever


async def run_server(
    agent: Agent,
    host: str = "0.0.0.0",
    port: int = 8765,
):
    """Convenience function to start the server."""
    server = AgentWebSocketServer(agent, host, port)
    await server.start()
