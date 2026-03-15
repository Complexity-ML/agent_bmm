# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Agent Chain — ReAct loop with BMM-parallel tool dispatch.

The chain runs a Think → Route → Act → Observe loop:
  1. Think:   LLM generates reasoning + tool selection hints
  2. Route:   BMMRouter selects tools in parallel (GPU)
  3. Act:     Tools execute in parallel (async)
  4. Observe: Results fed back to LLM
  5. Repeat until done or max_steps reached
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import torch

from agent_bmm.core.router import BMMRouter
from agent_bmm.llm.backend import LLMBackend
from agent_bmm.memory.context import ContextMemory, ToolResult
from agent_bmm.tools.registry import Tool, ToolRegistry


@dataclass
class ChainConfig:
    """Configuration for an agent chain."""

    max_steps: int = 5
    stop_on_final_answer: bool = True
    parallel_tools: bool = True


class AgentChain:
    """
    ReAct agent loop with BMM-parallel tool dispatch.

    Combines an LLM (for reasoning) with a BMMRouter (for tool selection)
    and a ToolRegistry (for execution). The routing happens on GPU via BMM,
    the tool execution happens in parallel via asyncio.
    """

    def __init__(
        self,
        llm: LLMBackend,
        router: BMMRouter,
        tools: ToolRegistry,
        config: ChainConfig | None = None,
    ):
        self.llm = llm
        self.router = router
        self.tools = tools
        self.config = config or ChainConfig()
        self.memory = ContextMemory()

    async def run(self, query: str) -> str:
        """
        Execute the full agent chain.

        Args:
            query: User's input question/task.

        Returns:
            Final answer from the agent.
        """
        self.memory.clear_chain()
        self.memory.add_turn("user", query)

        for step in range(self.config.max_steps):
            # 1. Think — LLM reasons about what to do
            thought = await self._think()

            # Check if LLM wants to give final answer
            if self.config.stop_on_final_answer and "[FINAL]" in thought:
                answer = thought.split("[FINAL]")[-1].strip()
                self.memory.add_turn("assistant", answer)
                return answer

            # 2. Route — BMM selects tools (GPU, parallel)
            tool_ids = self._route(thought)

            # 3. Act — Execute selected tools (async, parallel)
            results = await self._act(tool_ids, query)

            # 4. Observe — Feed results back
            for result in results:
                self.memory.add_tool_result(result)

        # Max steps reached — ask LLM for final answer
        return await self._finalize()

    async def _think(self) -> str:
        """LLM reasoning step."""
        messages = self.memory.to_messages()
        system = self._build_system_prompt()
        messages.insert(0, {"role": "system", "content": system})
        return await self.llm.chat(messages)

    def _route(self, thought: str) -> list[int]:
        """BMM routing — select tools based on LLM thought."""
        # Encode thought as a simple embedding (hash-based for now)
        # In production, use the LLM's hidden states directly
        h = self._text_to_tensor(thought)
        with torch.no_grad():
            _, expert_ids = self.router(h)
        return expert_ids.tolist()

    async def _act(self, tool_ids: list[int], query: str) -> list[ToolResult]:
        """Execute tools — parallel if configured."""
        results = []
        unique_tools = set(tool_ids)

        if self.config.parallel_tools and len(unique_tools) > 1:
            # Parallel execution via asyncio
            tasks = []
            for tid in unique_tools:
                if tid < self.tools.num_tools:
                    tool = self.tools.get(tid)
                    tasks.append(self._execute_tool(tool, query))
            tool_results = await asyncio.gather(*tasks, return_exceptions=True)
            for tid, res in zip(unique_tools, tool_results):
                if isinstance(res, Exception):
                    res = f"Error: {res}"
                tool = self.tools.get(tid)
                results.append(
                    ToolResult(
                        tool_name=tool.name,
                        tool_index=tid,
                        query=query,
                        result=str(res),
                    )
                )
        else:
            # Sequential execution
            for tid in unique_tools:
                if tid < self.tools.num_tools:
                    tool = self.tools.get(tid)
                    try:
                        res = await self._execute_tool(tool, query)
                    except Exception as e:
                        res = f"Error: {e}"
                    results.append(
                        ToolResult(
                            tool_name=tool.name,
                            tool_index=tid,
                            query=query,
                            result=str(res),
                        )
                    )

        return results

    async def _execute_tool(self, tool: Tool, query: str) -> str:
        """Execute a single tool."""
        return await tool.acall(query)

    async def _finalize(self) -> str:
        """Ask LLM for final answer after max steps."""
        self.memory.add_turn(
            "user",
            "Based on all the information gathered, provide your final answer. Prefix it with [FINAL].",
        )
        response = await self._think()
        if "[FINAL]" in response:
            return response.split("[FINAL]")[-1].strip()
        return response

    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        tool_list = "\n".join(
            f"  Tool {i}: {self.tools.get(i).name} — {self.tools.get(i).description}"
            for i in range(self.tools.num_tools)
        )
        return (
            "You are an AI agent with access to tools. "
            "Think step by step about what tools to use.\n"
            f"Available tools:\n{tool_list}\n\n"
            "When you have enough information, prefix your final answer with [FINAL]."
        )

    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """Convert text to a tensor for BMM routing."""
        # Simple hash-based embedding (placeholder)
        # In production, use the LLM's actual hidden states
        h = torch.zeros(1, self.router.hidden_size)
        for i, c in enumerate(text.encode()[: self.router.hidden_size]):
            h[0, i % self.router.hidden_size] += float(c) / 256.0
        return h
