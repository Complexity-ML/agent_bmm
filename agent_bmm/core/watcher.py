# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
File Watcher — Watch project files for external changes and re-index.

Uses polling (no external deps) to detect file modifications.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Callable

IGNORE_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv", ".env", "dist", "build"}


class FileWatcher:
    """Watch a directory for file changes and trigger re-indexing."""

    def __init__(
        self,
        root: str | Path,
        on_change: Callable[[list[str]], None],
        interval: float = 2.0,
        extensions: set[str] | None = None,
    ):
        self.root = Path(root).resolve()
        self.on_change = on_change
        self.interval = interval
        self.extensions = extensions or {".py", ".js", ".ts", ".tsx", ".html", ".css", ".yaml", ".json", ".toml"}
        self._mtimes: dict[str, float] = {}
        self._running = False

    def _scan(self) -> dict[str, float]:
        """Scan all watched files and return path → mtime map."""
        mtimes = {}
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if set(path.relative_to(self.root).parts) & IGNORE_DIRS:
                continue
            if path.suffix not in self.extensions:
                continue
            try:
                mtimes[str(path)] = os.path.getmtime(path)
            except OSError:
                continue
        return mtimes

    def _detect_changes(self) -> list[str]:
        """Detect changed, added, or deleted files since last scan."""
        new_mtimes = self._scan()
        changed = []

        # Modified or added
        for path, mtime in new_mtimes.items():
            if path not in self._mtimes or self._mtimes[path] < mtime:
                changed.append(path)

        # Deleted
        for path in self._mtimes:
            if path not in new_mtimes:
                changed.append(path)

        self._mtimes = new_mtimes
        return changed

    async def start(self):
        """Start watching in an async loop."""
        self._mtimes = self._scan()
        self._running = True
        while self._running:
            await asyncio.sleep(self.interval)
            changed = self._detect_changes()
            if changed:
                self.on_change(changed)

    def stop(self):
        """Stop watching."""
        self._running = False

    def check_once(self) -> list[str]:
        """Check for changes once (non-async). Initialize on first call."""
        if not self._mtimes:
            self._mtimes = self._scan()
            return []
        return self._detect_changes()
