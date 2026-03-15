# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Agent — High-level API for Agent BMM.

Usage:
    agent = Agent(
        model="your-model",
        base_url="http://localhost:8081/v1",
    )
    agent.add_tool("search", "Search the web", search_fn)
    agent.add_tool("calc", "Do math", calc_fn)

    answer = await agent.ask("What is the GDP of France?")
"""

from __future__ import annotations

import asyncio

from agent_bmm.core.router import BMMRouter
from agent_bmm.core.chain import AgentChain, ChainConfig
from agent_bmm.tools.registry import Tool, ToolRegistry
from agent_bmm.llm.backend import LLMBackend, LLMConfig


class Agent:
    """
    High-level agent with BMM-parallel tool dispatch.

    The simplest way to create an agent:

        agent = Agent(model="my-model", base_url="http://localhost:8081/v1")
        agent.add_tool("search", "Search the web", my_search_fn)
        answer = await agent.ask("What is quantum computing?")
    """

    def __init__(
        self,
        model: str = "",
        base_url: str = "http://localhost:8081/v1",
        api_key: str = "",
        provider: str = "openai",
        hidden_size: int = 256,
        expert_size: int = 128,
        routing: str = "round_robin",
        max_steps: int = 5,
    ):
        self._hidden_size = hidden_size
        self._expert_size = expert_size
        self._routing = routing
        self._max_steps = max_steps

        self._llm_config = LLMConfig(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
        self._tools = ToolRegistry()
        self._chain: AgentChain | None = None

    def add_tool(
        self,
        name: str,
        description: str,
        fn=None,
        async_fn=None,
        schema: dict | None = None,
    ) -> int:
        """Register a tool. Returns the tool index."""
        tool = Tool(
            name=name,
            description=description,
            fn=fn,
            async_fn=async_fn,
            schema=schema,
        )
        return self._tools.register(tool)

    def _build_chain(self) -> AgentChain:
        """Build the agent chain (lazy init)."""
        if self._chain is None:
            router = BMMRouter(
                hidden_size=self._hidden_size,
                num_tools=max(self._tools.num_tools, 1),
                expert_size=self._expert_size,
                routing=self._routing,
            )
            llm = LLMBackend(self._llm_config)
            config = ChainConfig(max_steps=self._max_steps)
            self._chain = AgentChain(llm, router, self._tools, config)
        return self._chain

    async def ask(self, query: str) -> str:
        """Ask the agent a question. Returns the final answer."""
        chain = self._build_chain()
        return await chain.run(query)

    def ask_sync(self, query: str) -> str:
        """Synchronous version of ask()."""
        return asyncio.run(self.ask(query))
