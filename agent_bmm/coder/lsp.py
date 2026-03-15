# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
LSP Integration — Run type checkers after edits.

Runs mypy or pyright to catch type errors before the user sees them.
Configured via agent-bmm.yaml under coder.lsp.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_type_check(
    project_dir: str | Path,
    files: list[str] | None = None,
    checker: str = "auto",
) -> str:
    """
    Run a type checker on the project or specific files.

    Args:
        project_dir: Project root directory.
        files: Specific files to check (None = whole project).
        checker: "mypy", "pyright", or "auto" (detect what's installed).

    Returns:
        Type checker output or empty string if clean.
    """
    project_dir = Path(project_dir)

    if checker == "auto":
        checker = _detect_checker()
        if not checker:
            return ""

    if checker == "mypy":
        cmd = ["mypy", "--no-error-summary", "--no-color-output"]
    elif checker == "pyright":
        cmd = ["pyright", "--outputjson"]
    else:
        return f"Unknown checker: {checker}"

    if files:
        cmd.extend(files)
    else:
        cmd.append(".")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_dir,
            timeout=60,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n{result.stderr}"
        return output.strip()
    except FileNotFoundError:
        return f"{checker} not installed"
    except subprocess.TimeoutExpired:
        return f"{checker} timed out (60s)"


def _detect_checker() -> str | None:
    """Detect which type checker is available."""
    for checker in ("mypy", "pyright"):
        try:
            subprocess.run(
                [checker, "--version"],
                capture_output=True,
                timeout=5,
            )
            return checker
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None
