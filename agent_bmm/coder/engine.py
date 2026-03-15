# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Coder Engine — A coding agent like Claude Code.

Reads your codebase, understands context, edits files,
runs commands, tests, and commits. All via LLM + tools.

Usage:
    from agent_bmm.coder import CoderAgent

    coder = CoderAgent(model="gpt-4o-mini", project_dir=".")
    coder.run("Add a login page to this Flask app")
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from agent_bmm.llm.backend import LLMBackend, LLMConfig

console = Console()

# Files to ignore when indexing
IGNORE_PATTERNS = {
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    ".env",
    "dist",
    "build",
    ".eggs",
    "*.pyc",
    "*.so",
    "*.egg-info",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

MAX_FILE_SIZE = 50_000  # chars
MAX_CONTEXT_FILES = 30


class CoderAgent:
    """
    Coding agent — reads, writes, runs, tests, commits.

    Like Claude Code but powered by any LLM via agent-bmm.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        project_dir: str = ".",
        max_steps: int = 10,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.max_steps = max_steps
        self.llm = LLMBackend(
            LLMConfig(
                model=model,
                base_url=base_url,
                api_key=api_key,
                provider="openai",
            )
        )
        self.history: list[dict[str, str]] = []
        self._indexed_files: dict[str, str] = {}

    # === Codebase Tools ===

    def index_project(self) -> dict[str, str]:
        """Index all source files in the project."""
        files = {}
        for path in sorted(self.project_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.project_dir)
            parts = set(rel.parts)
            if parts & IGNORE_PATTERNS:
                continue
            if any(path.name.endswith(p.replace("*", "")) for p in IGNORE_PATTERNS if "*" in p):
                continue
            if path.suffix not in {
                ".py",
                ".js",
                ".ts",
                ".tsx",
                ".jsx",
                ".html",
                ".css",
                ".yaml",
                ".yml",
                ".json",
                ".toml",
                ".md",
                ".txt",
                ".sh",
                ".bash",
                ".sql",
                ".rs",
                ".go",
                ".java",
                ".c",
                ".h",
                ".cpp",
            }:
                continue
            try:
                content = path.read_text(errors="replace")
                if len(content) <= MAX_FILE_SIZE:
                    files[str(rel)] = content
            except Exception:
                continue
            if len(files) >= MAX_CONTEXT_FILES:
                break
        self._indexed_files = files
        return files

    def read_file(self, path: str) -> str:
        """Read a file from the project."""
        p = (self.project_dir / path).resolve()
        if not str(p).startswith(str(self.project_dir)):
            return f"Error: {path} is outside project"
        if not p.exists():
            return f"Error: {path} not found"
        try:
            return p.read_text(errors="replace")[:MAX_FILE_SIZE]
        except Exception as e:
            return f"Error reading {path}: {e}"

    def write_file(self, path: str, content: str) -> str:
        """Write/create a file in the project."""
        p = (self.project_dir / path).resolve()
        if not str(p).startswith(str(self.project_dir)):
            return f"Error: {path} is outside project"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} chars to {path}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        """Replace text in a file (like sed)."""
        p = (self.project_dir / path).resolve()
        if not str(p).startswith(str(self.project_dir)):
            return f"Error: {path} is outside project"
        if not p.exists():
            return f"Error: {path} not found"
        content = p.read_text(errors="replace")
        if old not in content:
            return f"Error: text not found in {path}"
        content = content.replace(old, new, 1)
        p.write_text(content)
        return f"Edited {path}"

    def list_files(self, path: str = ".") -> str:
        """List files in a directory."""
        p = (self.project_dir / path).resolve()
        if not p.exists():
            return f"Error: {path} not found"
        entries = sorted(p.iterdir())
        lines = []
        for e in entries[:50]:
            rel = e.relative_to(self.project_dir)
            if e.name in IGNORE_PATTERNS:
                continue
            suffix = "/" if e.is_dir() else f" ({e.stat().st_size}b)"
            lines.append(f"  {rel}{suffix}")
        return "\n".join(lines) or "(empty)"

    def search_code(self, query: str, path: str = ".") -> str:
        """Search for text across all project files."""
        results = []
        (self.project_dir / path).resolve()
        for fpath, content in (self._indexed_files or self.index_project()).items():
            for i, line in enumerate(content.splitlines(), 1):
                if query.lower() in line.lower():
                    results.append(f"  {fpath}:{i}: {line.strip()}")
                    if len(results) >= 20:
                        return "\n".join(results)
        return "\n".join(results) or f"No matches for '{query}'"

    def run_command(self, cmd: str) -> str:
        """Run a shell command in the project directory."""
        # Block dangerous commands
        dangerous = ["rm -rf /", "mkfs", "dd if=", ":(){", "fork bomb"]
        if any(d in cmd for d in dangerous):
            return "Error: blocked dangerous command"
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.project_dir,
                timeout=30,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nStderr:\n{result.stderr}"
            if len(output) > 5000:
                output = output[:5000] + "\n... (truncated)"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out (30s)"
        except Exception as e:
            return f"Error: {e}"

    def git_status(self) -> str:
        """Get git status."""
        return self.run_command("git status --short")

    def git_diff(self) -> str:
        """Get git diff."""
        return self.run_command("git diff")

    def git_commit(self, message: str) -> str:
        """Stage all and commit."""
        self.run_command("git add -A")
        return self.run_command(f'git commit -m "{message}"')

    # === Agent Loop ===

    def _build_system_prompt(self) -> str:
        """Build system prompt with project context."""
        files = self._indexed_files or self.index_project()
        file_list = "\n".join(f"  {f}" for f in files.keys())

        return f"""You are a coding agent. You read, write, and modify code in a project.

Project: {self.project_dir.name}
Files ({len(files)}):
{file_list}

Available actions (respond with JSON):
  {{"action": "read", "path": "src/main.py"}}
  {{"action": "write", "path": "src/new.py", "content": "print('hello')"}}
  {{"action": "edit", "path": "src/main.py", "old": "old code", "new": "new code"}}
  {{"action": "list", "path": "src/"}}
  {{"action": "search", "query": "def main"}}
  {{"action": "run", "cmd": "python test.py"}}
  {{"action": "git_status"}}
  {{"action": "git_diff"}}
  {{"action": "git_commit", "message": "feat: add login"}}
  {{"action": "done", "summary": "What I did..."}}

Rules:
- Respond with ONE action per message (JSON only, no markdown)
- Read files before editing them
- Test your changes with "run"
- When finished, use "done" with a summary"""

    def _execute_action(self, action: dict) -> str:
        """Execute a parsed action."""
        act = action.get("action", "")

        if act == "read":
            return self.read_file(action.get("path", ""))
        elif act == "write":
            result = self.write_file(action.get("path", ""), action.get("content", ""))
            console.print(f"  [green]Write:[/] {action.get('path')}")
            return result
        elif act == "edit":
            result = self.edit_file(action.get("path", ""), action.get("old", ""), action.get("new", ""))
            console.print(f"  [yellow]Edit:[/] {action.get('path')}")
            return result
        elif act == "list":
            return self.list_files(action.get("path", "."))
        elif act == "search":
            return self.search_code(action.get("query", ""))
        elif act == "run":
            cmd = action.get("cmd", "")
            console.print(f"  [cyan]Run:[/] {cmd}")
            return self.run_command(cmd)
        elif act == "git_status":
            return self.git_status()
        elif act == "git_diff":
            return self.git_diff()
        elif act == "git_commit":
            msg = action.get("message", "update")
            console.print(f"  [magenta]Commit:[/] {msg}")
            return self.git_commit(msg)
        elif act == "done":
            return "__DONE__:" + action.get("summary", "Done")
        else:
            return f"Unknown action: {act}"

    def _parse_action(self, response: str) -> dict | None:
        """Parse LLM response into an action dict."""
        response = response.strip()
        # Try to find JSON in the response
        for start in [response.find("{"), 0]:
            if start == -1:
                continue
            end = response.rfind("}") + 1
            if end <= start:
                continue
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                continue
        return None

    async def _step(self) -> str | None:
        """Run one agent step. Returns summary if done, None otherwise."""
        response = await self.llm.chat(self.history)
        self.history.append({"role": "assistant", "content": response})

        action = self._parse_action(response)
        if action is None:
            self.history.append({"role": "user", "content": "Please respond with a valid JSON action."})
            return None

        act_name = action.get("action", "?")
        console.print(f"  [dim]Step {len(self.history) // 2}[/] [bold]{act_name}[/]")

        result = self._execute_action(action)

        if result.startswith("__DONE__:"):
            return result[9:]

        self.history.append({"role": "user", "content": f"Result:\n{result}"})
        return None

    async def arun(self, task: str) -> str:
        """Run the coding agent on a task (async)."""
        t0 = time.time()

        console.print()
        console.print(Panel(f"[bold white]{task}[/]", title="[bold cyan]Coder Agent[/]", border_style="cyan"))

        # Index project
        files = self.index_project()
        console.print(f"  [dim]Indexed {len(files)} files[/]")

        # Init conversation
        self.history = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": task},
        ]

        # Agent loop
        for step in range(self.max_steps):
            summary = await self._step()
            if summary:
                elapsed = time.time() - t0
                console.print()
                console.print(
                    Panel(
                        f"[bold white]{summary}[/]",
                        title="[bold green]Done[/]",
                        subtitle=f"[dim]{step + 1} steps · {elapsed:.1f}s[/]",
                        border_style="green",
                    )
                )
                return summary

        return "Max steps reached"

    def run(self, task: str) -> str:
        """Run the coding agent on a task (sync)."""
        return asyncio.run(self.arun(task))
