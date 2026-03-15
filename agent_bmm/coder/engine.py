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
import difflib
import json
import subprocess
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from agent_bmm.llm.backend import LLMBackend, LLMConfig

console = Console()

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

MAX_FILE_SIZE = 50_000
MAX_CONTEXT_FILES = 30


class CoderAgent:
    """Coding agent — reads, writes, runs, tests, commits."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        project_dir: str = ".",
        max_steps: int = 20,
        auto_commit: bool = False,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.max_steps = max_steps
        self.auto_commit = auto_commit
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
        self._checkpoints: list[str] = []

    # === Git checkpoint (rollback safety) ===

    def _checkpoint(self):
        """Create a git checkpoint before editing."""
        try:
            r = subprocess.run(
                "git rev-parse --is-inside-work-tree",
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.project_dir,
            )
            if r.returncode != 0:
                return
            subprocess.run("git add -A", shell=True, cwd=self.project_dir, capture_output=True)
            r = subprocess.run(
                "git stash push -m 'agent-bmm-checkpoint'",
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.project_dir,
            )
            if "No local changes" not in r.stdout:
                self._checkpoints.append("stash")
        except Exception:
            pass

    def rollback(self) -> str:
        """Rollback to last checkpoint."""
        if not self._checkpoints:
            return "No checkpoint to rollback to"
        try:
            subprocess.run("git stash pop", shell=True, cwd=self.project_dir, capture_output=True)
            self._checkpoints.pop()
            return "Rolled back to last checkpoint"
        except Exception as e:
            return f"Rollback failed: {e}"

    # === Codebase Tools ===

    def index_project(self) -> dict[str, str]:
        """Index all source files."""
        files = {}
        for path in sorted(self.project_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.project_dir)
            if set(rel.parts) & IGNORE_PATTERNS:
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
        p = (self.project_dir / path).resolve()
        if not str(p).startswith(str(self.project_dir)):
            return f"Error: {path} is outside project"
        if not p.exists():
            return f"Error: {path} not found"
        try:
            return p.read_text(errors="replace")[:MAX_FILE_SIZE]
        except Exception as e:
            return f"Error: {e}"

    def write_file(self, path: str, content: str) -> str:
        p = (self.project_dir / path).resolve()
        if not str(p).startswith(str(self.project_dir)):
            return f"Error: {path} is outside project"
        self._checkpoint()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self._indexed_files[path] = content
        return f"Written {len(content)} chars to {path}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        """Replace text in a file. Supports fuzzy matching."""
        p = (self.project_dir / path).resolve()
        if not str(p).startswith(str(self.project_dir)):
            return f"Error: {path} is outside project"
        if not p.exists():
            return f"Error: {path} not found"

        self._checkpoint()
        content = p.read_text(errors="replace")

        # Exact match
        if old in content:
            new_content = content.replace(old, new, 1)
            p.write_text(new_content)
            self._indexed_files[path] = new_content
            self._show_diff(path, content, new_content)
            return f"Edited {path}"

        # Fuzzy match
        old_lines = old.strip().splitlines()
        content_lines = content.splitlines()
        best_ratio = 0.0
        best_start = -1

        for i in range(len(content_lines) - len(old_lines) + 1):
            block = "\n".join(content_lines[i : i + len(old_lines)])
            ratio = difflib.SequenceMatcher(None, old.strip(), block.strip()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i

        if best_ratio > 0.6 and best_start >= 0:
            new_lines = (
                content_lines[:best_start] + new.strip().splitlines() + content_lines[best_start + len(old_lines) :]
            )
            new_content = "\n".join(new_lines)
            p.write_text(new_content)
            self._indexed_files[path] = new_content
            self._show_diff(path, content, new_content)
            return f"Edited {path} (fuzzy {best_ratio:.0%} at line {best_start + 1})"

        return f"Error: text not found in {path}. Read the file first."

    def _show_diff(self, path: str, old: str, new: str):
        """Show colored diff in terminal."""
        diff = difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
        lines = list(diff)[:30]
        if not lines:
            return
        text = ""
        for line in lines:
            if line.startswith("+") and not line.startswith("+++"):
                text += f"[green]{line.rstrip()}[/]\n"
            elif line.startswith("-") and not line.startswith("---"):
                text += f"[red]{line.rstrip()}[/]\n"
            else:
                text += f"[dim]{line.rstrip()}[/]\n"
        console.print(Panel(text.rstrip(), title=f"[yellow]diff {path}[/]", border_style="yellow"))

    def list_files(self, path: str = ".") -> str:
        p = (self.project_dir / path).resolve()
        if not p.exists():
            return f"Error: {path} not found"
        lines = []
        for e in sorted(p.iterdir())[:50]:
            if e.name in IGNORE_PATTERNS:
                continue
            rel = e.relative_to(self.project_dir)
            suffix = "/" if e.is_dir() else f" ({e.stat().st_size}b)"
            lines.append(f"  {rel}{suffix}")
        return "\n".join(lines) or "(empty)"

    def search_code(self, query: str) -> str:
        results = []
        for fpath, content in (self._indexed_files or self.index_project()).items():
            for i, line in enumerate(content.splitlines(), 1):
                if query.lower() in line.lower():
                    results.append(f"  {fpath}:{i}: {line.strip()}")
                    if len(results) >= 20:
                        return "\n".join(results)
        return "\n".join(results) or f"No matches for '{query}'"

    def run_command(self, cmd: str) -> str:
        dangerous = ["rm -rf /", "mkfs", "dd if=", ":(){", "format c:"]
        if any(d in cmd.lower() for d in dangerous):
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
            if result.returncode != 0:
                output += f"\n(exit code: {result.returncode})"
            if len(output) > 5000:
                output = output[:5000] + "\n... (truncated)"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: timed out (30s)"
        except Exception as e:
            return f"Error: {e}"

    def git_status(self) -> str:
        return self.run_command("git status --short")

    def git_diff(self) -> str:
        return self.run_command("git diff")

    def git_commit(self, message: str) -> str:
        self.run_command("git add -A")
        return self.run_command(f'git commit -m "{message}"')

    # === Agent Loop ===

    def _build_system_prompt(self) -> str:
        files = self._indexed_files or self.index_project()
        file_list = "\n".join(f"  {f}" for f in files.keys()) or "  (empty project)"

        return (
            "You are a coding agent. You MUST respond with ONLY a JSON object. "
            "No text, no markdown, no explanation — JUST the JSON.\n\n"
            f"Project: {self.project_dir.name}\n"
            f"Files:\n{file_list}\n\n"
            "WORKFLOW: First WRITE the code, THEN RUN to test it. "
            "NEVER run a file before writing it.\n\n"
            "RULES:\n"
            "- Your ENTIRE response must be a single JSON object\n"
            "- Create files in the project ROOT\n"
            "- ALWAYS write before run\n"
            "- Read a file before editing it\n\n"
            "ACTIONS:\n"
            '{"action":"write","path":"game.py","content":"import pygame\\n..."}\n'
            '{"action":"read","path":"game.py"}\n'
            '{"action":"edit","path":"game.py","old":"old code","new":"new code"}\n'
            '{"action":"run","cmd":"python game.py"}\n'
            '{"action":"list","path":"."}\n'
            '{"action":"search","query":"def main"}\n'
            '{"action":"done","summary":"Created snake game"}\n\n'
            "EXAMPLE — if asked to create a hello.py:\n"
            'Step 1: {"action":"write","path":"hello.py","content":"print(\'Hello World\')"}\n'
            'Step 2: {"action":"run","cmd":"python hello.py"}\n'
            'Step 3: {"action":"done","summary":"Created hello.py"}'
        )

    def _execute_action(self, action: dict) -> str:
        act = action.get("action", "")
        if act == "read":
            return self.read_file(action.get("path", ""))
        elif act == "write":
            r = self.write_file(action.get("path", ""), action.get("content", ""))
            console.print(f"  [green]Write:[/] {action.get('path')}")
            return r
        elif act == "edit":
            r = self.edit_file(action.get("path", ""), action.get("old", ""), action.get("new", ""))
            console.print(f"  [yellow]Edit:[/] {action.get('path')}")
            return r
        elif act == "list":
            return self.list_files(action.get("path", "."))
        elif act == "search":
            return self.search_code(action.get("query", ""))
        elif act == "run":
            cmd = action.get("cmd", "")
            console.print(f"  [cyan]Run:[/] {cmd}")
            result = self.run_command(cmd)
            if result and result != "(no output)":
                console.print(Panel(result[:500], title="[cyan]Output[/]", border_style="dim"))
            return result
        elif act == "git_status":
            return self.git_status()
        elif act == "git_diff":
            return self.git_diff()
        elif act == "git_commit":
            console.print(f"  [magenta]Commit:[/] {action.get('message', '')}")
            return self.git_commit(action.get("message", "update"))
        elif act == "rollback":
            return self.rollback()
        elif act == "done":
            return "__DONE__:" + action.get("summary", "Done")
        return f"Unknown action: {act}"

    def _parse_action(self, response: str) -> dict | None:
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                response = parts[1].strip()
        start = response.find("{")
        if start == -1:
            return None
        end = response.rfind("}") + 1
        if end <= start:
            return None
        try:
            return json.loads(response[start:end])
        except json.JSONDecodeError:
            return None

    async def _step(self, step_num: int) -> str | None:
        try:
            response = await self.llm.chat(self.history)
        except Exception as e:
            self.history.append({"role": "user", "content": f"LLM Error: {e}. Try again."})
            return None

        self.history.append({"role": "assistant", "content": response})
        action = self._parse_action(response)

        if action is None:
            self.history.append(
                {
                    "role": "user",
                    "content": 'Invalid. Reply with JSON only: {"action": "read", "path": "file.py"}',
                }
            )
            console.print(f"  [dim]Step {step_num}[/] [red]Bad JSON — retry[/]")
            return None

        act = action.get("action", "?")
        console.print(f"  [dim]Step {step_num}[/] [bold]{act}[/]")
        result = self._execute_action(action)

        if result.startswith("__DONE__:"):
            return result[9:]

        remaining = self.max_steps - step_num
        urgency = f"\n\n({remaining} steps left. Wrap up soon.)" if remaining <= 2 else ""
        self.history.append({"role": "user", "content": f"Result:\n{result}{urgency}"})
        return None

    async def arun(self, task: str) -> str:
        t0 = time.time()
        console.print()
        console.print(Panel(f"[bold white]{task}[/]", title="[bold cyan]Coder Agent[/]", border_style="cyan"))

        files = self.index_project()
        console.print(f"  [dim]Indexed {len(files)} files[/]")

        self.history = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": task},
        ]

        try:
            return await self._run_loop(t0)
        except KeyboardInterrupt:
            await self.llm.close()
            console.print("\n  [yellow]Interrupted by user.[/]")
            return "Interrupted"

    async def _run_loop(self, t0: float) -> str:
        for step in range(1, self.max_steps + 1):
            summary = await self._step(step)
            if summary:
                elapsed = time.time() - t0
                console.print()
                console.print(
                    Panel(
                        f"[bold white]{summary}[/]",
                        title="[bold green]Done[/]",
                        subtitle=f"[dim]{step} steps · {elapsed:.1f}s[/]",
                        border_style="green",
                    )
                )
                if self.auto_commit:
                    self.git_commit(f"agent-bmm: {summary[:50]}")
                await self.llm.close()
                return summary

        await self.llm.close()
        console.print(f"\n  [yellow]Max steps ({self.max_steps}) reached.[/]")
        return "Max steps reached"

    def run(self, task: str) -> str:
        try:
            return asyncio.run(self.arun(task))
        except KeyboardInterrupt:
            console.print("\n  [yellow]Interrupted.[/]")
            return "Interrupted"
