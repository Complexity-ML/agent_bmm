# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Built-in tools — lazy-loaded to avoid importing heavy deps at startup.

Tools are only imported when actually accessed, not at module load time.
This keeps `import agent_bmm` fast even if optional deps are missing.
"""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy import: only load tool when accessed."""
    _MAP = {
        "APITool": "agent_bmm.tools.builtin.api",
        "BrowserTool": "agent_bmm.tools.builtin.browser",
        "CodeExecTool": "agent_bmm.tools.builtin.code_exec",
        "DockerTool": "agent_bmm.tools.builtin.docker",
        "FileIOTool": "agent_bmm.tools.builtin.file_io",
        "GitHubTool": "agent_bmm.tools.builtin.github",
        "MathTool": "agent_bmm.tools.builtin.math_tool",
        "SlackTool": "agent_bmm.tools.builtin.slack",
        "SQLTool": "agent_bmm.tools.builtin.sql",
        "WebSearchTool": "agent_bmm.tools.builtin.web_search",
        "ImageTool": "agent_bmm.tools.builtin.image",
        "AudioTool": "agent_bmm.tools.builtin.audio",
    }
    if name in _MAP:
        import importlib

        module = importlib.import_module(_MAP[name])
        return getattr(module, name)
    raise AttributeError(f"module 'agent_bmm.tools.builtin' has no attribute {name!r}")


__all__ = [
    "WebSearchTool",
    "SQLTool",
    "APITool",
    "FileIOTool",
    "CodeExecTool",
    "MathTool",
    "GitHubTool",
    "SlackTool",
    "DockerTool",
    "BrowserTool",
    "ImageTool",
    "AudioTool",
]
