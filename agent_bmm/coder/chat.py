# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Chat Mode — Interactive conversation with the coding agent.

Usage:
    agent-bmm chat -d ./my_project
    agent-bmm chat -m gpt-4o -d .

Type your requests, the agent executes. Type 'quit' to exit.
"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel

from agent_bmm.coder.cost import CostTracker
from agent_bmm.coder.engine import CoderAgent

console = Console()


class ChatSession:
    """Interactive coding agent chat loop."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        project_dir: str = ".",
        max_steps: int = 20,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.project_dir = project_dir
        self.max_steps = max_steps
        self.cost = CostTracker(model)
        self._turn = 0

    def run(self):
        """Start the interactive chat loop."""
        console.print()
        console.print(
            Panel(
                f"[bold white]Agent BMM Chat[/]\n"
                f"[dim]Model: {self.model} | Project: {self.project_dir}[/]\n"
                f"[dim]Type your request. 'quit' to exit, 'cost' for stats.[/]",
                border_style="cyan",
            )
        )

        while True:
            try:
                console.print()
                query = console.input("[bold cyan]> [/]").strip()
            except (KeyboardInterrupt, EOFError):
                self._goodbye()
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                self._goodbye()
                break
            if query.lower() == "cost":
                self.cost.print_summary()
                continue
            if query.lower() == "help":
                self._print_help()
                continue

            self._turn += 1
            coder = CoderAgent(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                project_dir=self.project_dir,
                max_steps=self.max_steps,
            )

            try:
                asyncio.run(coder.arun(query))
                # Estimate cost from history length
                total_chars = sum(len(m.get("content", "")) for m in coder.history)
                self.cost.add_request(total_chars)
            except KeyboardInterrupt:
                console.print("\n  [yellow]Interrupted.[/]")

    def _goodbye(self):
        console.print()
        self.cost.print_summary()
        console.print("[dim]Goodbye![/]")

    def _print_help(self):
        console.print(
            Panel(
                "[bold]Commands:[/]\n"
                "  [cyan]quit[/]  — Exit chat\n"
                "  [cyan]cost[/]  — Show token usage and cost\n"
                "  [cyan]help[/]  — Show this help\n\n"
                "[bold]Tips:[/]\n"
                '  "Create a Flask app with login"\n'
                '  "Read main.py and add error handling"\n'
                '  "Run the tests and fix any failures"\n'
                '  "Refactor utils.py to use dataclasses"',
                title="[bold cyan]Help[/]",
                border_style="dim",
            )
        )
