# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Code Execution Tool — Run Python code in a sandboxed environment.
Uses AST validation to block dangerous operations.
"""

from __future__ import annotations

import ast
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout

from agent_bmm.tools.registry import Tool

# Blocked modules and builtins
_BLOCKED_MODULES = {
    "os",
    "subprocess",
    "shutil",
    "sys",
    "importlib",
    "ctypes",
    "socket",
    "http",
    "urllib",
    "requests",
    "pathlib",
    "glob",
    "tempfile",
    "signal",
    "threading",
    "multiprocessing",
}

_BLOCKED_BUILTINS = {
    "exec",
    "eval",
    "compile",
    "__import__",
    "open",
    "breakpoint",
    "exit",
    "quit",
}


def _validate_ast(code: str) -> str | None:
    """Validate code AST. Returns error message or None if safe."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod in _BLOCKED_MODULES:
                    return f"Blocked import: {mod}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split(".")[0]
                if mod in _BLOCKED_MODULES:
                    return f"Blocked import: {mod}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in _BLOCKED_BUILTINS:
                    return f"Blocked builtin: {node.func.id}"

    return None


def create_code_exec_tool(
    timeout: float = 5.0,
    max_output: int = 5000,
) -> Tool:
    """
    Create a sandboxed Python code execution tool.

    The code runs in a restricted namespace with math/json/re available.
    Dangerous operations (file I/O, network, system calls) are blocked.
    """

    def _execute(code: str) -> str:
        # Validate
        error = _validate_ast(code)
        if error:
            return f"Validation Error: {error}"

        # Restricted namespace
        safe_builtins = (
            {k: v for k, v in __builtins__.items() if k not in _BLOCKED_BUILTINS}
            if isinstance(__builtins__, dict)
            else {
                k: getattr(__builtins__, k)
                for k in dir(__builtins__)
                if k not in _BLOCKED_BUILTINS and not k.startswith("_")
            }
        )

        namespace = {
            "__builtins__": safe_builtins,
            "print": print,
            "range": range,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "isinstance": isinstance,
            "type": type,
            "True": True,
            "False": False,
            "None": None,
        }

        # Import safe modules
        import datetime
        import json
        import math
        import re

        namespace.update(
            {
                "math": math,
                "json": json,
                "re": re,
                "datetime": datetime,
            }
        )

        # Execute with captured output
        stdout = io.StringIO()
        stderr = io.StringIO()

        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(code, namespace)
        except Exception:
            tb = traceback.format_exc()
            return f"Runtime Error:\n{tb}"

        output = stdout.getvalue()
        errors = stderr.getvalue()

        result = ""
        if output:
            result += output
        if errors:
            result += f"\nStderr:\n{errors}"
        if not result:
            result = "(no output)"

        if len(result) > max_output:
            result = result[:max_output] + "\n... (truncated)"

        return result

    return Tool(
        name="code_exec",
        description="Execute Python code in a sandboxed environment",
        fn=_execute,
    )


CodeExecTool = create_code_exec_tool
