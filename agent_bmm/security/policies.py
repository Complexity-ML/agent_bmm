# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Security Policies — Command whitelist (#73), file blacklist (#72),
network isolation (#71), input sanitization (#33).

Configured via agent-bmm.yaml:
    security:
      command_whitelist: [python, npm, git, pip, pytest, ruff]
      file_blacklist: [.env, .git/config, credentials.json, id_rsa]
      network_isolation: false
"""

from __future__ import annotations

import re
from pathlib import Path

# ── #73 Command Whitelist ──

DEFAULT_COMMAND_WHITELIST = {
    "python",
    "python3",
    "pip",
    "npm",
    "npx",
    "node",
    "git",
    "pytest",
    "ruff",
    "mypy",
    "pyright",
    "cargo",
    "go",
    "make",
    "echo",
    "cat",
    "ls",
    "dir",
    "cd",
    "mkdir",
    "cp",
    "mv",
    "head",
    "tail",
    "grep",
    "find",
    "wc",
}

BLOCKED_COMMANDS = {
    "rm -rf /",
    "mkfs",
    "dd if=",
    ":(){",
    "format c:",
    "del /f /s /q",
    "shutdown",
    "reboot",
    "curl | sh",
    "wget | sh",
}


def check_command(cmd: str, whitelist: set[str] | None = None) -> tuple[bool, str]:
    """
    Check if a command is allowed.

    Returns (allowed, reason).
    """
    cmd_lower = cmd.strip().lower()

    # Block dangerous patterns
    for blocked in BLOCKED_COMMANDS:
        if blocked in cmd_lower:
            return False, f"Blocked dangerous command pattern: {blocked}"

    # Extract the base command (first word, strip path)
    base = cmd.strip().split()[0] if cmd.strip() else ""
    base = Path(base).name  # strip path like /usr/bin/python → python

    allowed = whitelist or DEFAULT_COMMAND_WHITELIST
    if base in allowed:
        return True, ""

    return False, f"Command '{base}' not in whitelist. Allowed: {', '.join(sorted(allowed))}"


# ── #72 File Whitelist/Blacklist ──

DEFAULT_FILE_BLACKLIST = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "service-account.json",
    "id_rsa",
    "id_ed25519",
    ".ssh/config",
    ".git/config",
    ".netrc",
    ".npmrc",
    "token.json",
    "secrets.yaml",
    "secrets.yml",
}

SENSITIVE_PATTERNS = [
    r".*\.pem$",
    r".*\.key$",
    r".*_secret.*",
    r".*password.*",
    r".*credential.*",
]


def check_file_access(
    path: str,
    blacklist: set[str] | None = None,
    action: str = "read",
) -> tuple[bool, str]:
    """
    Check if file access is allowed.

    Returns (allowed, reason).
    """
    p = Path(path)
    name = p.name
    rel = str(p)

    blocked = blacklist or DEFAULT_FILE_BLACKLIST

    # Exact match
    if name in blocked or rel in blocked:
        return False, f"Access to '{path}' blocked (sensitive file)"

    # Pattern match
    for pattern in SENSITIVE_PATTERNS:
        if re.match(pattern, name, re.IGNORECASE):
            return False, f"Access to '{path}' blocked (matches sensitive pattern)"

    return True, ""


# ── #33 Input Sanitization ──


def sanitize_sql(query: str) -> str:
    """Sanitize SQL input to prevent injection."""
    # Remove common injection patterns
    dangerous = ["--", ";--", "/*", "*/", "xp_", "UNION SELECT", "DROP TABLE", "DELETE FROM"]
    result = query
    for pattern in dangerous:
        result = result.replace(pattern, "")
    return result


def sanitize_path(path: str) -> str:
    """Sanitize file path to prevent traversal."""
    # Remove directory traversal
    cleaned = path.replace("..", "").replace("~", "")
    # Remove null bytes
    cleaned = cleaned.replace("\x00", "")
    return cleaned


def sanitize_html(text: str) -> str:
    """Sanitize HTML to prevent XSS."""
    replacements = {"<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#x27;", "&": "&amp;"}
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text
