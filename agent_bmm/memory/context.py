# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Context Memory — Tracks conversation history and tool results.

Maintains a sliding window of interactions for the agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_name: str
    tool_index: int
    query: str
    result: str
    confidence: float = 1.0


@dataclass
class Turn:
    """A single turn in the agent conversation."""

    role: str  # "user", "assistant", "tool"
    content: str
    tool_results: list[ToolResult] = field(default_factory=list)


class ContextMemory:
    """
    Sliding window context for the agent loop.

    Keeps the last `max_turns` turns plus all tool results
    from the current reasoning chain.
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.turns: list[Turn] = []
        self.current_chain: list[ToolResult] = []

    def add_turn(self, role: str, content: str):
        self.turns.append(Turn(role=role, content=content))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def add_tool_result(self, result: ToolResult):
        self.current_chain.append(result)
        self.add_turn(
            "tool",
            f"[{result.tool_name}] {result.result}"
        )

    def clear_chain(self):
        self.current_chain = []

    def to_messages(self) -> list[dict[str, str]]:
        """Convert to OpenAI-style message list."""
        return [
            {"role": t.role if t.role != "tool" else "assistant", "content": t.content}
            for t in self.turns
        ]

    @property
    def last_tool_results(self) -> list[ToolResult]:
        return self.current_chain
