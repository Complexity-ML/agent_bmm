# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Tool Registry — Define and register tools for BMM agent dispatch.

Tools are Python callables with typed schemas. The registry maps
tool indices to their implementations for post-routing execution.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class Tool:
    """
    A tool that can be dispatched by the BMM router.

    Args:
        name: Human-readable tool name.
        description: What the tool does (used for embedding-based routing).
        fn: The callable that executes the tool.
        async_fn: Async version of fn (for parallel I/O tools).
        schema: Input/output schema (pydantic model or dict).
    """

    name: str
    description: str
    fn: Callable[..., Any] | None = None
    async_fn: Callable[..., Awaitable[Any]] | None = None
    schema: dict | None = None
    index: int = -1  # assigned by registry

    def __call__(self, *args, **kwargs) -> Any:
        if self.fn is not None:
            return self.fn(*args, **kwargs)
        raise NotImplementedError(f"Tool {self.name} has no sync implementation")

    async def acall(self, *args, **kwargs) -> Any:
        if self.async_fn is not None:
            return await self.async_fn(*args, **kwargs)
        if self.fn is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.fn(*args, **kwargs))
        raise NotImplementedError(f"Tool {self.name} has no implementation")


class ToolRegistry:
    """
    Registry of tools available to the BMM agent.

    Maps tool indices (used by BMMRouter) to Tool implementations.
    Supports both sync and async execution.
    """

    def __init__(self):
        self._tools: dict[int, Tool] = {}
        self._name_to_idx: dict[str, int] = {}
        self._next_idx = 0

    def register(self, tool: Tool) -> int:
        """Register a tool and return its index."""
        idx = self._next_idx
        tool.index = idx
        self._tools[idx] = tool
        self._name_to_idx[tool.name] = idx
        self._next_idx += 1
        return idx

    def get(self, index: int) -> Tool:
        """Get tool by index."""
        return self._tools[index]

    def get_by_name(self, name: str) -> Tool:
        """Get tool by name."""
        return self._tools[self._name_to_idx[name]]

    @property
    def num_tools(self) -> int:
        return len(self._tools)

    @property
    def descriptions(self) -> list[str]:
        """All tool descriptions in index order."""
        return [self._tools[i].description for i in range(self.num_tools)]

    def execute(self, tool_idx: int, query: str) -> Any:
        """Execute a tool synchronously."""
        return self._tools[tool_idx](query)

    async def aexecute(self, tool_idx: int, query: str) -> Any:
        """Execute a tool asynchronously."""
        return await self._tools[tool_idx].acall(query)

    async def batch_execute(
        self, tool_ids: list[int], queries: list[str]
    ) -> list[Any]:
        """Execute multiple tools in parallel (async)."""
        tasks = [
            self.aexecute(tid, q) for tid, q in zip(tool_ids, queries)
        ]
        return await asyncio.gather(*tasks)
