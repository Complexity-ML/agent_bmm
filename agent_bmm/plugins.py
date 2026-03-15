# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Plugin System — Installable tools via pip.

Discover and load tools from installed packages that provide
the `agent_bmm.tools` entry point.

Creating a plugin:
    # pyproject.toml of your plugin package
    [project.entry-points."agent_bmm.tools"]
    my_tool = "my_package.tool:create_tool"

    # my_package/tool.py
    from agent_bmm.tools.registry import Tool
    def create_tool(**kwargs) -> Tool:
        return Tool(name="my_tool", description="...", fn=my_fn)

Usage:
    from agent_bmm.plugins import discover_plugins, load_plugin
    plugins = discover_plugins()
    tool = load_plugin("my_tool")
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from agent_bmm.tools.registry import Tool


def discover_plugins() -> dict[str, str]:
    """
    Discover installed plugins that provide agent_bmm tools.

    Returns:
        dict mapping plugin name → module path
    """
    plugins = {}

    if sys.version_info >= (3, 12):
        from importlib.metadata import entry_points

        eps = entry_points(group="agent_bmm.tools")
    else:
        from importlib.metadata import entry_points

        all_eps = entry_points()
        eps = all_eps.get("agent_bmm.tools", [])

    for ep in eps:
        plugins[ep.name] = f"{ep.value}"

    return plugins


def load_plugin(name: str, **kwargs: Any) -> Tool | None:
    """
    Load a plugin by name and return a Tool instance.

    Args:
        name: Plugin name (from entry_points).
        kwargs: Arguments to pass to the plugin's create function.
    """
    plugins = discover_plugins()
    if name not in plugins:
        return None

    module_path, _, attr_name = plugins[name].rpartition(":")
    if not attr_name:
        attr_name = "create_tool"
        module_path = plugins[name]

    try:
        module = importlib.import_module(module_path)
        factory = getattr(module, attr_name)
        return factory(**kwargs)
    except Exception as e:
        print(f"Failed to load plugin {name}: {e}")
        return None


def load_all_plugins(**kwargs: Any) -> list[Tool]:
    """Load all discovered plugins."""
    tools = []
    for name in discover_plugins():
        tool = load_plugin(name, **kwargs)
        if tool is not None:
            tools.append(tool)
    return tools


# ── Plugin marketplace ──

REGISTRY_URL = "https://raw.githubusercontent.com/Complexity-ML/agent_bmm/main/plugins.json"


def search_plugins(query: str = "") -> list[dict]:
    """Search the community plugin registry."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(REGISTRY_URL, timeout=5) as resp:
            registry = json.loads(resp.read())
    except Exception:
        return []

    if not query:
        return registry

    q = query.lower()
    return [p for p in registry if q in p.get("name", "").lower() or q in p.get("description", "").lower()]


def install_plugin(name: str) -> str:
    """Install a plugin from PyPI."""
    import subprocess
    try:
        result = subprocess.run(
            ["pip", "install", name],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return f"Installed {name}"
        return f"Failed: {result.stderr}"
    except Exception as e:
        return f"Error: {e}"
