# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Agent BMM — GPU-accelerated agent framework.

Simple usage:
    from agent_bmm import Agent, tool

    @tool("greet", "Say hello")
    def greet(name: str) -> str:
        return f"Hello {name}!"

    agent = Agent(model="gpt-4o-mini", tools="all")
    agent.add_tool(greet)
    agent.run("Say hello to Boris")

Workflow (zero code):
    agent-bmm workflow my_tasks.yaml
"""

__version__ = "0.1.0"

from agent_bmm.agent import Agent
from agent_bmm.tools.registry import Tool, ToolRegistry
from agent_bmm.utils.decorators import tool

__all__ = ["Agent", "tool", "Tool", "ToolRegistry"]
