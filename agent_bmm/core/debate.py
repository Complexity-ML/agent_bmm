# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Multi-Agent Debate — Agents argue positions, BMM routes rebuttals.

Two or more agents take opposing positions on a question,
argue back and forth, then a judge synthesizes the consensus.
"""

from __future__ import annotations

import asyncio

from agent_bmm.llm.backend import LLMBackend


class DebateAgent:
    """An agent that argues a position."""

    def __init__(self, name: str, position: str, llm: LLMBackend):
        self.name = name
        self.position = position
        self.llm = llm
        self.arguments: list[str] = []

    async def argue(self, query: str, opponent_args: list[str]) -> str:
        """Generate an argument, considering opponent's points."""
        opponent_text = "\n".join(f"- {a}" for a in opponent_args[-3:]) or "None yet."
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are arguing the position: {self.position}. "
                    "Be concise (2-3 sentences). Address opponent's points."
                ),
            },
            {
                "role": "user",
                "content": f"Topic: {query}\nOpponent's arguments:\n{opponent_text}\n\nYour argument:",
            },
        ]
        response = await self.llm.chat(messages, max_tokens=200)
        self.arguments.append(response)
        return response


class Debate:
    """Multi-agent debate with consensus synthesis."""

    def __init__(self, llm: LLMBackend, rounds: int = 3):
        self.llm = llm
        self.rounds = rounds

    async def run(self, query: str, positions: list[str]) -> str:
        """
        Run a debate between agents with different positions.

        Returns synthesized consensus.
        """
        agents = [
            DebateAgent(f"Agent_{i}", pos, self.llm)
            for i, pos in enumerate(positions)
        ]

        for round_num in range(self.rounds):
            tasks = []
            for agent in agents:
                # Collect opponent arguments
                opponent_args = []
                for other in agents:
                    if other is not agent:
                        opponent_args.extend(other.arguments[-2:])
                tasks.append(agent.argue(query, opponent_args))
            await asyncio.gather(*tasks)

        # Judge synthesizes
        all_args = []
        for agent in agents:
            all_args.append(f"**{agent.name}** ({agent.position}):")
            for arg in agent.arguments:
                all_args.append(f"  - {arg}")

        messages = [
            {
                "role": "system",
                "content": "Synthesize a balanced conclusion from this debate. Be fair to all sides.",
            },
            {
                "role": "user",
                "content": f"Topic: {query}\n\n" + "\n".join(all_args),
            },
        ]
        return await self.llm.chat(messages)
