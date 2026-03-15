# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Decorators — Simple API for creating tools and agents.

Usage:
    from agent_bmm import tool, Agent

    @tool("search", "Search the web")
    def search(query: str) -> str:
        return "results..."

    @tool("calc", "Do math")
    async def calc(expr: str) -> str:
        return str(eval(expr))

    agent = Agent(model="gpt-4o-mini", tools="all")
    agent.add_tool(search)
    agent.add_tool(calc)
    agent.run("What is 2+2?")
"""

from __future__ import annotations

import asyncio
from functools import wraps

from agent_bmm.tools.registry import Tool


def tool(name: str, description: str = ""):
    """
    Decorator to create a tool from any function.

    @tool("my_tool", "Does something cool")
    def my_tool(query: str) -> str:
        return "result"
    """

    def decorator(fn):
        desc = description or fn.__doc__ or f"Tool: {name}"
        if asyncio.iscoroutinefunction(fn):
            t = Tool(name=name, description=desc, async_fn=fn)
        else:
            t = Tool(name=name, description=desc, fn=fn)
        # Attach tool metadata to the function
        fn._agent_bmm_tool = t
        fn.name = name
        fn.description = desc

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapper._agent_bmm_tool = t
        wrapper.name = name
        wrapper.description = desc
        return wrapper

    return decorator
