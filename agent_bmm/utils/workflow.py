# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Workflow — YAML-based agent automation. Zero code needed.

Usage:
    agent-bmm workflow my_tasks.yaml
    agent-bmm workflow my_tasks.yaml --dry-run
    agent-bmm workflow my_tasks.yaml -o results.json

Workflow file format:
    name: "My Workflow"
    model: gpt-4o-mini
    tools: [browser, code, math, file, sql]
    max_steps: 5

    # Simple: list of tasks
    tasks:
      - "Search for latest AI news"
      - "Summarize the top 3 articles"
      - "Save summary to news.txt"

    # Advanced: tasks with dependencies
    tasks:
      - id: search
        prompt: "Search for latest AI news"
        tools: [browser]
      - id: analyze
        prompt: "Analyze sentiment of these articles: {{search.result}}"
        depends_on: [search]
        tools: [code]
      - id: report
        prompt: "Write a report combining: {{search.result}} and {{analyze.result}}"
        depends_on: [search, analyze]
        tools: [file]
        output: report.txt
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_bmm.agent import Agent
from agent_bmm.tools.builtin import (
    APITool,
    CodeExecTool,
    FileIOTool,
    GitHubTool,
    MathTool,
    SQLTool,
    WebSearchTool,
)

console = Console()

BUILTIN_TOOLS = {
    "browser": WebSearchTool,
    "search": WebSearchTool,
    "code": CodeExecTool,
    "math": MathTool,
    "file": FileIOTool,
    "sql": SQLTool,
    "api": APITool,
    "github": GitHubTool,
}

# Try to import browser tool (requires playwright)
try:
    from agent_bmm.tools.builtin import BrowserTool

    BUILTIN_TOOLS["browser"] = BrowserTool
except ImportError:
    pass


@dataclass
class TaskDef:
    """A single task in a workflow."""

    id: str
    prompt: str
    tools: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    output: str = ""
    result: str = ""


@dataclass
class WorkflowDef:
    """A complete workflow definition."""

    name: str = "Unnamed Workflow"
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    tools: list[str] = field(default_factory=lambda: ["browser", "code", "math"])
    max_steps: int = 5
    tasks: list[TaskDef] = field(default_factory=list)


def parse_workflow(path: str) -> WorkflowDef:
    """Parse a YAML or JSON workflow file."""
    p = Path(path)
    content = p.read_text()

    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(content) or {}
        except ImportError:
            raise ImportError("PyYAML required for .yaml files: pip install pyyaml")
    else:
        data = json.loads(content)

    wf = WorkflowDef(
        name=data.get("name", p.stem),
        model=data.get("model", "gpt-4o-mini"),
        base_url=data.get("base_url", "https://api.openai.com/v1"),
        tools=data.get("tools", ["browser", "code", "math"]),
        max_steps=data.get("max_steps", 5),
    )

    raw_tasks = data.get("tasks", [])
    for i, task in enumerate(raw_tasks):
        if isinstance(task, str):
            wf.tasks.append(TaskDef(id=f"step_{i + 1}", prompt=task))
        elif isinstance(task, dict):
            wf.tasks.append(
                TaskDef(
                    id=task.get("id", f"step_{i + 1}"),
                    prompt=task.get("prompt", ""),
                    tools=task.get("tools", []),
                    depends_on=task.get("depends_on", []),
                    output=task.get("output", ""),
                )
            )

    return wf


def _resolve_template(prompt: str, results: dict[str, str]) -> str:
    """Replace {{task_id.result}} templates with actual results."""

    def replacer(match):
        ref = match.group(1)
        if ref.endswith(".result"):
            task_id = ref[: -len(".result")]
        else:
            task_id = ref
        return results.get(task_id, f"[{ref} not found]")

    return re.sub(r"\{\{(\w+(?:\.\w+)?)\}\}", replacer, prompt)


class WorkflowRunner:
    """
    Executes a workflow — runs tasks in order with dependency resolution.

    Simple tasks run sequentially.
    Tasks with depends_on wait for dependencies to complete.
    Independent tasks can run in parallel (future).
    """

    def __init__(self, workflow: WorkflowDef, dry_run: bool = False):
        self.workflow = workflow
        self.dry_run = dry_run
        self.results: dict[str, str] = {}

    async def run(self) -> dict[str, str]:
        """Execute the workflow and return results."""
        t0 = time.time()

        console.print()
        console.print(
            Panel(
                f"[bold white]{self.workflow.name}[/]\n"
                f"[dim]Model: {self.workflow.model} | Tools: {', '.join(self.workflow.tools)} | "
                f"Tasks: {len(self.workflow.tasks)}[/]",
                title="[bold cyan]Workflow[/]",
                border_style="cyan",
            )
        )

        # Build agent
        agent = Agent(
            model=self.workflow.model,
            base_url=self.workflow.base_url,
            max_steps=self.workflow.max_steps,
        )

        # Register tools
        tool_names = set(self.workflow.tools)
        for task in self.workflow.tasks:
            tool_names.update(task.tools)

        for name in tool_names:
            if name in BUILTIN_TOOLS:
                tool = BUILTIN_TOOLS[name]()
                agent.add_tool(tool.name, tool.description, fn=tool.fn, async_fn=tool.async_fn)

        # Execute tasks
        for i, task in enumerate(self.workflow.tasks):
            # Check dependencies
            for dep in task.depends_on:
                if dep not in self.results:
                    console.print(f"  [red]Task {task.id}: dependency '{dep}' not completed[/]")
                    continue

            # Resolve templates
            prompt = _resolve_template(task.prompt, self.results)

            console.print(f"\n  [bold cyan]Task {i + 1}/{len(self.workflow.tasks)}:[/] [white]{task.id}[/]")
            console.print(f"  [dim]{prompt[:100]}{'...' if len(prompt) > 100 else ''}[/]")

            if self.dry_run:
                console.print("  [yellow]DRY RUN — skipped[/]")
                self.results[task.id] = "[dry run]"
                continue

            # Run
            try:
                result = await agent.ask(prompt)
                self.results[task.id] = result
                console.print(f"  [green]Done:[/] {result[:150]}{'...' if len(result) > 150 else ''}")

                # Save output file if specified
                if task.output:
                    Path(task.output).write_text(result)
                    console.print(f"  [dim]Saved to {task.output}[/]")

            except Exception as e:
                self.results[task.id] = f"Error: {e}"
                console.print(f"  [red]Error: {e}[/]")

        # Summary
        elapsed = time.time() - t0
        console.print()

        table = Table(title="Results", border_style="green")
        table.add_column("Task", style="bold")
        table.add_column("Status", width=8)
        table.add_column("Result", max_width=60)

        for task in self.workflow.tasks:
            result = self.results.get(task.id, "not run")
            status = "[green]OK[/]" if not result.startswith("Error") else "[red]FAIL[/]"
            table.add_row(task.id, status, result[:60])

        console.print(table)
        console.print(f"\n[dim]Completed in {elapsed:.1f}s[/]")

        return self.results

    def run_sync(self) -> dict[str, str]:
        """Synchronous wrapper."""
        return asyncio.run(self.run())


async def run_workflow(path: str, dry_run: bool = False, output: str | None = None) -> dict[str, str]:
    """Load and run a workflow file."""
    wf = parse_workflow(path)
    runner = WorkflowRunner(wf, dry_run=dry_run)
    results = await runner.run()

    if output:
        Path(output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        console.print(f"\n[green]Results saved to {output}[/]")

    return results
