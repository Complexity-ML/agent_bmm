# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Logging — Rich terminal output with BMM dispatch tracing.

Traces every step of the agent loop: routing decisions, expert activations,
tool executions, and performance metrics. All tied to BMM dispatch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class BMMTraceEntry:
    """A single BMM dispatch trace entry."""

    step: int
    timestamp: float
    expert_ids: list[int]
    expert_names: list[str]
    query: str
    routing_strategy: str
    dispatch_time_ms: float = 0.0
    tool_results: dict[str, str] = field(default_factory=dict)
    tool_times_ms: dict[str, float] = field(default_factory=dict)


class AgentLogger:
    """
    Rich terminal logger for BMM agent tracing.

    Displays real-time routing decisions, expert activations,
    and performance metrics in a beautiful terminal UI.
    """

    def __init__(self, verbose: bool = True, trace: bool = True):
        self.verbose = verbose
        self.trace = trace
        self.entries: list[BMMTraceEntry] = []
        self._step = 0
        self._start_time = 0.0

    def start(self, query: str):
        """Log agent start."""
        self._start_time = time.time()
        self._step = 0
        if self.verbose:
            console.print()
            console.print(
                Panel(
                    f"[bold white]{query}[/]",
                    title="[bold cyan]Agent BMM[/]",
                    border_style="cyan",
                )
            )

    def log_think(self, thought: str):
        """Log LLM thinking step."""
        self._step += 1
        if self.verbose:
            console.print(
                f"  [dim]Step {self._step}[/] [yellow]Think:[/] "
                f"{thought[:100]}{'...' if len(thought) > 100 else ''}"
            )

    def log_route(
        self,
        expert_ids: list[int],
        expert_names: list[str],
        routing_strategy: str,
        dispatch_time_ms: float,
    ):
        """Log BMM routing decision."""
        if self.verbose:
            # Distribution table
            counts: dict[str, int] = {}
            for name in expert_names:
                counts[name] = counts.get(name, 0) + 1

            parts = []
            for name, count in counts.items():
                pct = count / len(expert_names) * 100
                parts.append(f"[bold green]{name}[/] {pct:.0f}%")

            console.print(
                f"  [dim]Step {self._step}[/] [cyan]Route:[/] "
                f"{' | '.join(parts)} "
                f"[dim]({dispatch_time_ms:.1f}ms, {routing_strategy})[/]"
            )

        if self.trace:
            self.entries.append(
                BMMTraceEntry(
                    step=self._step,
                    timestamp=time.time(),
                    expert_ids=expert_ids,
                    expert_names=expert_names,
                    query="",
                    routing_strategy=routing_strategy,
                    dispatch_time_ms=dispatch_time_ms,
                )
            )

    def log_tool_start(self, tool_name: str, query: str):
        """Log tool execution start."""
        if self.verbose:
            console.print(
                f"  [dim]Step {self._step}[/] [magenta]Tool:[/] "
                f"[bold]{tool_name}[/] ← {query[:80]}"
            )

    def log_tool_result(self, tool_name: str, result: str, time_ms: float):
        """Log tool execution result."""
        if self.verbose:
            truncated = result[:120].replace("\n", " ")
            console.print(
                f"  [dim]Step {self._step}[/] [green]Result:[/] "
                f"[bold]{tool_name}[/] → {truncated} "
                f"[dim]({time_ms:.0f}ms)[/]"
            )

        if self.trace and self.entries:
            self.entries[-1].tool_results[tool_name] = result[:500]
            self.entries[-1].tool_times_ms[tool_name] = time_ms

    def log_answer(self, answer: str):
        """Log final answer."""
        elapsed = (time.time() - self._start_time) * 1000
        if self.verbose:
            console.print()
            console.print(
                Panel(
                    f"[bold white]{answer}[/]",
                    title="[bold green]Answer[/]",
                    subtitle=f"[dim]{self._step} steps · {elapsed:.0f}ms[/]",
                    border_style="green",
                )
            )

    def log_error(self, error: str):
        """Log an error."""
        if self.verbose:
            console.print(f"  [bold red]Error:[/] {error}")

    def print_trace(self):
        """Print the full BMM dispatch trace table."""
        if not self.entries:
            return

        table = Table(title="BMM Dispatch Trace", border_style="cyan")
        table.add_column("Step", style="dim", width=5)
        table.add_column("Routing", style="cyan", width=12)
        table.add_column("Experts", style="green")
        table.add_column("Tools", style="magenta")
        table.add_column("Time", style="yellow", width=8)

        for entry in self.entries:
            # Expert distribution
            counts: dict[str, int] = {}
            for name in entry.expert_names:
                counts[name] = counts.get(name, 0) + 1
            expert_str = " ".join(f"{n}:{c}" for n, c in counts.items())

            # Tool results
            tool_str = " ".join(
                f"{n}({t:.0f}ms)" for n, t in entry.tool_times_ms.items()
            )

            table.add_row(
                str(entry.step),
                entry.routing_strategy,
                expert_str,
                tool_str or "-",
                f"{entry.dispatch_time_ms:.1f}ms",
            )

        console.print(table)

    def print_stats(self):
        """Print performance statistics."""
        if not self.entries:
            return

        total_dispatch = sum(e.dispatch_time_ms for e in self.entries)
        total_tools = sum(
            sum(t for t in e.tool_times_ms.values()) for e in self.entries
        )
        total = (time.time() - self._start_time) * 1000

        table = Table(title="Performance", border_style="yellow")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="yellow")

        table.add_row("Total time", f"{total:.0f}ms")
        table.add_row("BMM dispatch", f"{total_dispatch:.1f}ms")
        table.add_row("Tool execution", f"{total_tools:.0f}ms")
        table.add_row("LLM overhead", f"{total - total_dispatch - total_tools:.0f}ms")
        table.add_row("Steps", str(len(self.entries)))

        # Expert usage distribution
        all_experts: dict[str, int] = {}
        for entry in self.entries:
            for name in entry.expert_names:
                all_experts[name] = all_experts.get(name, 0) + 1
        total_routes = sum(all_experts.values())
        for name, count in sorted(all_experts.items()):
            table.add_row(
                f"Expert: {name}",
                f"{count}/{total_routes} ({count / total_routes * 100:.0f}%)",
            )

        console.print(table)
