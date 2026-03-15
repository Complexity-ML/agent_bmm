# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Math Tool — Evaluate mathematical expressions safely.
Supports arithmetic, trig, statistics, and unit conversions.
"""

from __future__ import annotations

import math
import re

from agent_bmm.tools.registry import Tool

# Safe math functions
_MATH_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "log2": math.log2, "exp": math.exp, "pow": pow,
    "abs": abs, "round": round, "ceil": math.ceil, "floor": math.floor,
    "pi": math.pi, "e": math.e, "inf": math.inf,
    "sum": sum, "min": min, "max": max,
    "factorial": math.factorial, "gcd": math.gcd,
}


def create_math_tool() -> Tool:
    """Create a safe math evaluation tool."""

    def _evaluate(expr: str) -> str:
        expr = expr.strip()

        # Block dangerous patterns
        if any(kw in expr for kw in ["import", "exec", "eval", "open", "__"]):
            return "Error: blocked expression"

        try:
            result = eval(expr, {"__builtins__": {}}, _MATH_FUNCS)
            if isinstance(result, float):
                if result == int(result) and not math.isinf(result):
                    return str(int(result))
                return f"{result:.6g}"
            return str(result)
        except ZeroDivisionError:
            return "Error: division by zero"
        except Exception as e:
            return f"Math Error: {e}"

    return Tool(
        name="math",
        description="Evaluate mathematical expressions (arithmetic, trig, log, stats)",
        fn=_evaluate,
    )


MathTool = create_math_tool
