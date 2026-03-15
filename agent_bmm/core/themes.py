# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Color Themes — Light/dark/minimal themes for Rich terminal output.

Usage:
    from agent_bmm.core.themes import get_theme, set_theme

    set_theme("dark")  # or "light", "minimal"
    theme = get_theme()
    console.print(f"[{theme.success}]Done![/]")
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Theme:
    """Terminal color theme."""

    name: str
    primary: str
    secondary: str
    success: str
    warning: str
    error: str
    muted: str
    border: str
    panel_title: str


THEMES: dict[str, Theme] = {
    "dark": Theme(
        name="dark",
        primary="bold cyan",
        secondary="bold white",
        success="bold green",
        warning="yellow",
        error="bold red",
        muted="dim",
        border="cyan",
        panel_title="bold cyan",
    ),
    "light": Theme(
        name="light",
        primary="bold blue",
        secondary="bold black",
        success="bold green",
        warning="dark_orange",
        error="bold red",
        muted="grey50",
        border="blue",
        panel_title="bold blue",
    ),
    "minimal": Theme(
        name="minimal",
        primary="bold",
        secondary="bold",
        success="bold",
        warning="bold",
        error="bold red",
        muted="dim",
        border="dim",
        panel_title="bold",
    ),
}

_current_theme: str = "dark"


def get_theme() -> Theme:
    """Get the current theme."""
    return THEMES[_current_theme]


def set_theme(name: str):
    """Set the active theme. Options: dark, light, minimal."""
    global _current_theme
    if name not in THEMES:
        raise ValueError(f"Unknown theme '{name}'. Available: {', '.join(THEMES)}")
    _current_theme = name


def list_themes() -> list[str]:
    """List available theme names."""
    return list(THEMES.keys())
