# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Remote Client — Connect to a remote Agent BMM server.

Persistent WebSocket connection with real-time event display.

Usage:
    agent-bmm remote ws://gpu-server:8765
    > Create a Flask API
    [agent works on remote GPU...]
    Done!
    > Add authentication
    [agent works...]
    Done!
    > quit
"""

from __future__ import annotations

import asyncio
import json

from rich.console import Console
from rich.panel import Panel

console = Console()


class RemoteClient:
    """Persistent WebSocket client for remote Agent BMM server."""

    def __init__(self, url: str = "ws://localhost:8765"):
        self.url = url
        self._ws = None

    async def connect(self):
        """Connect to the remote server."""
        import websockets

        self._ws = await websockets.connect(self.url, max_size=2**20)
        console.print(f"  [green]Connected to {self.url}[/]")

        # Get available tools
        await self._ws.send(json.dumps({"type": "tools"}))
        resp = json.loads(await self._ws.recv())
        if resp.get("type") == "tools":
            tools = resp.get("data", [])
            names = [t["name"] for t in tools]
            console.print(f"  [dim]Tools: {', '.join(names)}[/]")

    async def query(self, text: str):
        """Send a query and stream results."""
        if not self._ws:
            console.print("[red]Not connected[/]")
            return

        await self._ws.send(json.dumps({"type": "query", "text": text}))

        while True:
            try:
                msg = json.loads(await asyncio.wait_for(self._ws.recv(), timeout=120))
            except asyncio.TimeoutError:
                console.print("  [yellow]Timeout waiting for response[/]")
                break

            event = msg.get("type", "")

            if event == "start":
                pass
            elif event == "thinking":
                step = msg.get("step", "?")
                thought = msg.get("data", "")[:100]
                console.print(f"  [dim]Step {step}[/] [yellow]Think:[/] {thought}")
            elif event == "route":
                ids = msg.get("expert_ids", [])
                console.print(f"  [dim]Step {msg.get('step', '?')}[/] [cyan]Route:[/] experts={ids}")
            elif event == "tool_start":
                tool = msg.get("tool", "?")
                console.print(f"  [magenta]Tool:[/] {tool}")
            elif event == "tool_result":
                tool = msg.get("tool", "?")
                result = msg.get("result", "")[:200]
                console.print(f"  [green]Result:[/] {tool} → {result}")
            elif event == "token":
                # Streaming token
                console.print(msg.get("data", ""), end="")
            elif event == "answer":
                answer = msg.get("data", "")
                time_ms = msg.get("time_ms", 0)
                console.print()
                console.print(
                    Panel(
                        f"[bold white]{answer}[/]",
                        title="[bold green]Answer[/]",
                        subtitle=f"[dim]{time_ms:.0f}ms[/]",
                        border_style="green",
                    )
                )
            elif event == "error":
                console.print(f"  [red]Error: {msg.get('data', '')}[/]")
            elif event == "done":
                break
            elif event == "pong":
                pass

    async def close(self):
        if self._ws:
            await self._ws.close()

    async def run_interactive(self):
        """Interactive chat loop over WebSocket."""
        console.print()
        console.print(
            Panel(
                f"[bold white]Agent BMM Remote[/]\n"
                f"[dim]Server: {self.url}[/]\n"
                f"[dim]Type your request. 'quit' to exit.[/]",
                border_style="cyan",
            )
        )

        try:
            await self.connect()
        except Exception as e:
            console.print(f"[red]Connection failed: {e}[/]")
            return

        while True:
            try:
                console.print()
                query = console.input("[bold cyan]> [/]").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break
            if query.lower() == "ping":
                await self._ws.send(json.dumps({"type": "ping"}))
                resp = json.loads(await self._ws.recv())
                console.print(f"  [green]{resp.get('type', 'ok')}[/]")
                continue
            if query.lower() == "tools":
                await self._ws.send(json.dumps({"type": "tools"}))
                resp = json.loads(await self._ws.recv())
                for t in resp.get("data", []):
                    console.print(f"  [cyan]{t['name']}[/] — {t['description']}")
                continue

            await self.query(query)

        await self.close()
        console.print("[dim]Disconnected.[/]")


def run_remote(url: str):
    """Start interactive remote session."""
    client = RemoteClient(url)
    try:
        asyncio.run(client.run_interactive())
    except KeyboardInterrupt:
        console.print("\n[dim]Disconnected.[/]")
