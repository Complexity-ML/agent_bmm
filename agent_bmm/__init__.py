# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Agent BMM — GPU-accelerated agent framework with Batched Matrix Multiply dispatch.

Replaces sequential tool dispatch (LangChain-style) with parallel
BMM-based routing and execution on GPU.
"""

__version__ = "0.1.0"

from agent_bmm.core.router import BMMRouter
from agent_bmm.core.chain import AgentChain
from agent_bmm.tools.registry import Tool, ToolRegistry
from agent_bmm.agent import Agent

__all__ = ["Agent", "AgentChain", "BMMRouter", "Tool", "ToolRegistry"]
