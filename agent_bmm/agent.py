# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Agent — Simple API for Agent BMM.

Usage:
    from agent_bmm import Agent, tool

    @tool("search", "Search the web")
    def search(query: str) -> str:
        return "results..."

    agent = Agent(model="gpt-4o-mini", tools="all")
    agent.add_tool(search)
    agent.run("Find the latest AI news")
"""

from __future__ import annotations

import asyncio

from agent_bmm.core.chain import AgentChain, ChainConfig
from agent_bmm.core.router import BMMRouter
from agent_bmm.llm.backend import LLMBackend, LLMConfig
from agent_bmm.tools.registry import Tool, ToolRegistry


class Agent:
    """
    Agent BMM — GPU-accelerated agent with parallel tool dispatch.

    Simple:
        agent = Agent(model="gpt-4o-mini", tools="all")
        agent.run("What is the weather in Paris?")

    Custom tools:
        @tool("my_tool", "Does stuff")
        def my_tool(q): return "result"

        agent = Agent(model="gpt-4o-mini")
        agent.add_tool(my_tool)
        agent.run("Use my_tool")
    """

    def __init__(
        self,
        model: str = "",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        provider: str = "openai",
        tools: str | list[str] | None = None,
        max_steps: int = 5,
        hidden_size: int = 256,
        expert_size: int = 128,
        routing: str = "round_robin",
    ):
        self._hidden_size = hidden_size
        self._expert_size = expert_size
        self._routing = routing
        self._max_steps = max_steps

        # Auto-detect provider if not specified
        if not base_url or not api_key:
            from agent_bmm.llm.auto_detect import detect_provider

            detected_provider, detected_url, detected_key = detect_provider(model, base_url)
            provider = provider or detected_provider
            base_url = base_url or detected_url
            api_key = api_key or detected_key

        self._llm_config = LLMConfig(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
        self._tools = ToolRegistry()
        self._chain: AgentChain | None = None

        # Auto-load builtin tools
        if tools:
            self._load_builtin_tools(tools)

    def _load_builtin_tools(self, tools: str | list[str]):
        """Load built-in tools by name or 'all'."""
        from agent_bmm.tools.builtin import (
            APITool,
            CodeExecTool,
            FileIOTool,
            GitHubTool,
            MathTool,
            SQLTool,
            WebSearchTool,
        )

        BUILTINS = {
            "search": WebSearchTool,
            "math": MathTool,
            "code": CodeExecTool,
            "file": FileIOTool,
            "sql": SQLTool,
            "api": APITool,
            "github": GitHubTool,
        }

        # Try browser (optional dep)
        try:
            from agent_bmm.tools.builtin import BrowserTool

            BUILTINS["browser"] = BrowserTool
        except ImportError:
            pass

        if tools == "all":
            names = list(BUILTINS.keys())
        elif isinstance(tools, str):
            names = [t.strip() for t in tools.split(",")]
        else:
            names = tools

        for name in names:
            if name in BUILTINS:
                t = BUILTINS[name]()
                self._tools.register(t)

    def add_tool(self, tool_or_name, description: str = "", fn=None, async_fn=None):
        """
        Register a tool. Accepts:
            - A @tool decorated function
            - A Tool object
            - (name, description, fn) arguments
        """
        # @tool decorated function
        if hasattr(tool_or_name, "_agent_bmm_tool"):
            self._chain = None
            return self._tools.register(tool_or_name._agent_bmm_tool)

        # Tool object
        if isinstance(tool_or_name, Tool):
            self._chain = None
            return self._tools.register(tool_or_name)

        # Name + description + fn
        t = Tool(name=tool_or_name, description=description, fn=fn, async_fn=async_fn)
        self._chain = None
        return self._tools.register(t)

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
        """Ask the agent a question (async)."""
        chain = self._build_chain()
        return await chain.run(query)

    def run(self, query: str) -> str:
        """Ask the agent a question (sync). The simple way."""
        return asyncio.run(self.ask(query))
