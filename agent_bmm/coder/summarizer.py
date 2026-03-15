# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Codebase Summarizer — Auto-generate project overview for LLM context.

Produces a concise summary of architecture, key files, dependencies,
and patterns. Uses less tokens than sending full file contents.
"""

from __future__ import annotations

import re
from pathlib import Path


def summarize_project(project_dir: str | Path, files: dict[str, str]) -> str:
    """
    Generate a concise project summary from indexed files.

    Returns a string suitable for inclusion in the system prompt.
    """
    project = Path(project_dir)
    parts = []

    # Project name and type
    parts.append(f"Project: {project.name}")
    parts.append(f"Files: {len(files)}")

    # Detect project type
    project_type = _detect_project_type(files)
    if project_type:
        parts.append(f"Type: {project_type}")

    # Dependencies
    deps = _extract_dependencies(files)
    if deps:
        parts.append(f"Dependencies: {', '.join(deps[:15])}")

    # Key patterns
    patterns = _extract_patterns(files)
    if patterns:
        parts.append(f"Patterns: {', '.join(patterns)}")

    # File tree (abbreviated)
    dirs = set()
    for path in files:
        d = str(Path(path).parent)
        if d != ".":
            dirs.add(d)
    if dirs:
        parts.append(f"Directories: {', '.join(sorted(dirs)[:10])}")

    # Entry points
    entries = _find_entry_points(files)
    if entries:
        parts.append(f"Entry points: {', '.join(entries[:5])}")

    return "\n".join(parts)


def _detect_project_type(files: dict[str, str]) -> str:
    """Detect project type from file patterns."""
    names = set(files.keys())
    if "package.json" in names:
        return "Node.js/JavaScript"
    if "pyproject.toml" in names or "setup.py" in names:
        return "Python package"
    if "Cargo.toml" in names:
        return "Rust"
    if "go.mod" in names:
        return "Go"
    if "pom.xml" in names:
        return "Java/Maven"
    if any(f.endswith(".py") for f in names):
        return "Python"
    if any(f.endswith(".ts") or f.endswith(".tsx") for f in names):
        return "TypeScript"
    return ""


def _extract_dependencies(files: dict[str, str]) -> list[str]:
    """Extract dependency names from config files."""
    deps = []

    # Python
    for name in ("pyproject.toml", "requirements.txt", "setup.py"):
        if name in files:
            content = files[name]
            # Simple regex for package names
            for match in re.findall(r'"([a-zA-Z][\w-]+)(?:[>=<]|\[)', content):
                if match not in deps:
                    deps.append(match)
            for match in re.findall(r"^([a-zA-Z][\w-]+)", content, re.MULTILINE):
                if match not in deps and match.lower() not in ("name", "version", "description"):
                    deps.append(match)

    # Node.js
    if "package.json" in files:
        import json

        try:
            pkg = json.loads(files["package.json"])
            for section in ("dependencies", "devDependencies"):
                deps.extend(pkg.get(section, {}).keys())
        except json.JSONDecodeError:
            pass

    return deps[:20]


def _extract_patterns(files: dict[str, str]) -> list[str]:
    """Detect coding patterns used in the project."""
    patterns = set()
    all_content = "\n".join(files.values())

    checks = {
        "async/await": r"async\s+def|await\s+",
        "dataclasses": r"@dataclass",
        "type hints": r"def\s+\w+\(.*:\s*\w+",
        "pytest": r"def\s+test_",
        "FastAPI": r"from\s+fastapi",
        "Flask": r"from\s+flask",
        "React": r"from\s+['\"]react",
        "SQLAlchemy": r"from\s+sqlalchemy",
        "pydantic": r"from\s+pydantic",
        "CLI (argparse)": r"argparse\.ArgumentParser",
    }

    for name, pattern in checks.items():
        if re.search(pattern, all_content):
            patterns.add(name)

    return sorted(patterns)


def _find_entry_points(files: dict[str, str]) -> list[str]:
    """Find likely entry point files."""
    candidates = []
    for path, content in files.items():
        if "__name__" in content and "__main__" in content:
            candidates.append(path)
        elif path in ("main.py", "app.py", "cli.py", "server.py", "index.ts", "index.js"):
            candidates.append(path)
    return candidates
