# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
File I/O Tool — Read, write, list, and search files.
Sandboxed to a configurable root directory.
"""

from __future__ import annotations

from pathlib import Path

from agent_bmm.tools.registry import Tool


def create_file_io_tool(
    root_dir: str = ".",
    max_read_size: int = 10000,
    allow_write: bool = False,
) -> Tool:
    """
    Create a file I/O tool.

    Commands:
        "read <path>"           — Read file contents
        "list <path>"           — List directory contents
        "search <path> <query>" — Search for text in files
        "write <path> <content>" — Write to file (if allow_write=True)
    """
    root = Path(root_dir).resolve()

    def _safe_path(path_str: str) -> Path | str:
        """Resolve path and ensure it's within root."""
        p = (root / path_str).resolve()
        if not str(p).startswith(str(root)):
            return f"Error: path {path_str} is outside sandbox {root}"
        return p

    def _execute(query: str) -> str:
        parts = query.strip().split(None, 2)
        if not parts:
            return "Error: no command. Use: read, list, search, write"

        cmd = parts[0].lower()

        if cmd == "read" and len(parts) >= 2:
            p = _safe_path(parts[1])
            if isinstance(p, str):
                return p
            if not p.exists():
                return f"Error: {parts[1]} not found"
            if not p.is_file():
                return f"Error: {parts[1]} is not a file"
            content = p.read_text(errors="replace")
            if len(content) > max_read_size:
                content = content[:max_read_size] + f"\n... (truncated at {max_read_size} chars)"
            return content

        elif cmd == "list" and len(parts) >= 2:
            p = _safe_path(parts[1])
            if isinstance(p, str):
                return p
            if not p.exists():
                return f"Error: {parts[1]} not found"
            if not p.is_dir():
                return f"Error: {parts[1]} is not a directory"
            entries = sorted(p.iterdir())
            lines = []
            for e in entries[:100]:
                rel = e.relative_to(root)
                suffix = "/" if e.is_dir() else f" ({e.stat().st_size} bytes)"
                lines.append(f"  {rel}{suffix}")
            if len(entries) > 100:
                lines.append(f"  ... ({len(entries)} total)")
            return "\n".join(lines) or "(empty directory)"

        elif cmd == "search" and len(parts) >= 3:
            search_path = parts[1]
            search_query = parts[2]
            p = _safe_path(search_path)
            if isinstance(p, str):
                return p
            if not p.exists():
                return f"Error: {search_path} not found"

            results = []
            files = [p] if p.is_file() else list(p.rglob("*"))
            for f in files[:50]:
                if not f.is_file():
                    continue
                try:
                    content = f.read_text(errors="replace")
                    for i, line in enumerate(content.splitlines(), 1):
                        if search_query.lower() in line.lower():
                            rel = f.relative_to(root)
                            results.append(f"  {rel}:{i}: {line.strip()}")
                            if len(results) >= 20:
                                break
                except Exception:
                    continue
                if len(results) >= 20:
                    break
            return "\n".join(results) or f"No matches for '{search_query}'"

        elif cmd == "write" and len(parts) >= 3 and allow_write:
            p = _safe_path(parts[1])
            if isinstance(p, str):
                return p
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(parts[2])
            return f"Written {len(parts[2])} chars to {parts[1]}"

        elif cmd == "write" and not allow_write:
            return "Error: write not allowed (set allow_write=True)"

        else:
            return "Error: unknown command. Use: read, list, search, write"

    return Tool(
        name="file_io",
        description="Read, list, and search files on disk",
        fn=_execute,
    )


FileIOTool = create_file_io_tool
