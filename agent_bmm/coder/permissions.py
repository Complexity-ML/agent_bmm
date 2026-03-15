# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Permissions — Ask user before dangerous actions.

Like Claude Code: the agent asks permission before writing files,
running commands, or making git commits.
"""

from __future__ import annotations

from enum import Enum

from rich.console import Console

console = Console()


class PermissionLevel(Enum):
    ASK = "ask"  # Ask user every time
    ALLOW_READS = "allow_reads"  # Auto-allow reads, ask for writes/runs
    YOLO = "yolo"  # Allow everything (dangerous!)


class PermissionManager:
    """Manages user permissions for agent actions."""

    def __init__(self, level: PermissionLevel = PermissionLevel.ALLOW_READS):
        self.level = level
        self._allowed_patterns: set[str] = set()
        self._session_allows: dict[str, bool] = {}

    def check(self, action: str, detail: str = "") -> bool:
        """Check if an action is allowed. Prompts user if needed."""
        if self.level == PermissionLevel.YOLO:
            return True

        # Always allow these
        safe_actions = {"list", "search", "read", "git_status", "git_diff", "done"}
        if action in safe_actions:
            return True

        if self.level == PermissionLevel.ALLOW_READS:
            if action in safe_actions:
                return True

        # Check session cache
        cache_key = f"{action}:{detail}"
        if cache_key in self._session_allows:
            return self._session_allows[cache_key]

        # Ask user
        return self._ask(action, detail)

    def _ask(self, action: str, detail: str) -> bool:
        """Prompt user for permission."""
        colors = {
            "write": "green",
            "edit": "yellow",
            "run": "cyan",
            "git_commit": "magenta",
        }
        color = colors.get(action, "white")

        console.print(f"\n  [bold {color}]{action}[/]: {detail[:100]}")
        try:
            response = console.input("  [dim]Allow? (y/n/always) [/]").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return False

        if response in ("y", "yes"):
            return True
        elif response in ("a", "always"):
            self._session_allows[f"{action}:{detail}"] = True
            return True
        return False
