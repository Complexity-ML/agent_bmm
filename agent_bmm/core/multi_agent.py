# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Multi-Agent — Multiple agents collaborating via BMM dispatch.

Each agent is an expert in the BMM router. The orchestrator routes
queries to the best agent via BMM, and agents can delegate to each
other through the same mechanism.

Architecture:
    Query → BMM Router → Agent₀ (research)
                       → Agent₁ (code)
                       → Agent₂ (analysis)
                       → Agent₃ (summary)
    Results → BMM Aggregator → Final Answer
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import torch

from agent_bmm.core.logging import AgentLogger
from agent_bmm.core.router import BMMRouter
from agent_bmm.llm.backend import LLMBackend, LLMConfig
from agent_bmm.memory.context import ContextMemory


@dataclass
class AgentRole:
    """Definition of a specialized agent role."""

    name: str
    description: str
    system_prompt: str
    llm_config: LLMConfig | None = None  # None = use orchestrator's LLM


class SubAgent:
    """
    A specialized sub-agent with its own role and memory.

    Each sub-agent acts as a BMM expert — the orchestrator
    routes queries to the most appropriate sub-agent.
    """

    def __init__(self, role: AgentRole, llm: LLMBackend):
        self.role = role
        self.llm = llm
        self.memory = ContextMemory(max_turns=10)

    async def process(self, query: str, context: str = "") -> str:
        """Process a query with this agent's specialization."""
        messages = [
            {"role": "system", "content": self.role.system_prompt},
        ]
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": query})

        response = await self.llm.chat(messages)
        self.memory.add_turn("user", query)
        self.memory.add_turn("assistant", response)
        return response


class MultiAgentOrchestrator:
    """
    BMM-based multi-agent orchestrator.

    Routes queries to specialized sub-agents via batched matrix multiply.
    Agents process in parallel, then results are aggregated.

    Usage:
        orch = MultiAgentOrchestrator(llm_config)
        orch.add_agent(AgentRole("researcher", "Research topics", "You research..."))
        orch.add_agent(AgentRole("coder", "Write code", "You write code..."))
        orch.add_agent(AgentRole("analyst", "Analyze data", "You analyze..."))

        answer = await orch.run("Build a web scraper for news")
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        hidden_size: int = 256,
        routing: str = "learned",
        max_rounds: int = 3,
    ):
        self.llm_config = llm_config
        self.hidden_size = hidden_size
        self.routing = routing
        self.max_rounds = max_rounds

        self._agents: list[SubAgent] = []
        self._router: BMMRouter | None = None
        self._logger = AgentLogger()
        self._llm = LLMBackend(llm_config)

    def add_agent(self, role: AgentRole) -> int:
        """Add a sub-agent. Returns its index."""
        llm = LLMBackend(role.llm_config or self.llm_config)
        agent = SubAgent(role, llm)
        idx = len(self._agents)
        self._agents.append(agent)
        self._router = None  # rebuild router
        return idx

    def _build_router(self):
        """Build BMM router with one expert per agent."""
        if self._router is None:
            self._router = BMMRouter(
                hidden_size=self.hidden_size,
                num_tools=len(self._agents),
                expert_size=self.hidden_size // 2,
                routing=self.routing,
            )

    async def run(self, query: str) -> str:
        """
        Run multi-agent collaboration.

        1. Route query to best agent(s) via BMM
        2. Selected agents process in parallel
        3. Aggregate results
        4. Repeat if needed
        """
        self._build_router()
        self._logger.start(query)

        context = ""
        for round_num in range(self.max_rounds):
            # Route via BMM
            t0 = time.time()
            x = self._text_to_tensor(query + " " + context)
            with torch.no_grad():
                _, expert_ids = self._router(x)
            dispatch_ms = (time.time() - t0) * 1000

            selected = list(set(expert_ids.tolist()))
            agent_names = [self._agents[i].role.name for i in selected]

            self._logger.log_route(
                expert_ids=selected,
                expert_names=agent_names,
                routing_strategy=self.routing,
                dispatch_time_ms=dispatch_ms,
            )

            # Dispatch to selected agents in parallel
            tasks = []
            for idx in selected:
                if idx < len(self._agents):
                    agent = self._agents[idx]
                    self._logger.log_tool_start(agent.role.name, query)
                    tasks.append(self._run_agent(agent, query, context))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate results
            parts = []
            for idx, result in zip(selected, results):
                agent = self._agents[idx]
                t_ms = 0.0
                if isinstance(result, tuple):
                    result, t_ms = result
                if isinstance(result, Exception):
                    result = f"Error: {result}"
                    t_ms = 0.0
                self._logger.log_tool_result(agent.role.name, str(result), t_ms)
                parts.append(f"[{agent.role.name}]: {result}")

            context = "\n\n".join(parts)

            # Check if we have enough info
            if round_num < self.max_rounds - 1:
                needs_more = await self._needs_more_rounds(query, context)
                if not needs_more:
                    break

        # Final synthesis
        answer = await self._synthesize(query, context)
        self._logger.log_answer(answer)
        return answer

    async def _run_agent(
        self, agent: SubAgent, query: str, context: str
    ) -> tuple[str, float]:
        """Run a single sub-agent and return (result, time_ms)."""
        t0 = time.time()
        result = await agent.process(query, context)
        return result, (time.time() - t0) * 1000

    async def _needs_more_rounds(self, query: str, context: str) -> bool:
        """Ask LLM if more agent rounds are needed."""
        messages = [
            {"role": "system", "content": "Answer YES or NO only."},
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Agent results so far:\n{context[:1000]}\n\n"
                    "Do we need more agent rounds to fully answer this query?"
                ),
            },
        ]
        response = await self._llm.chat(messages, max_tokens=10)
        return "yes" in response.lower()

    async def _synthesize(self, query: str, context: str) -> str:
        """Synthesize final answer from all agent results."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Synthesize a clear, comprehensive answer from the agent results below. "
                    "Combine the best information from each agent."
                ),
            },
            {
                "role": "user",
                "content": (f"Original query: {query}\n\nAgent results:\n{context}"),
            },
        ]
        return await self._llm.chat(messages)

    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """Simple text → tensor for routing."""
        h = torch.zeros(1, self.hidden_size)
        for i, c in enumerate(text.encode()[: self.hidden_size]):
            h[0, i % self.hidden_size] += float(c) / 256.0
        return h
