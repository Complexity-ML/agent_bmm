# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Conversation Branching — Explore multiple reasoning paths in parallel.

Fork a conversation into N branches, each exploring a different approach.
BMM routes each branch to different tool sets. Best branch wins.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agent_bmm.llm.backend import LLMBackend
from agent_bmm.memory.context import ContextMemory


@dataclass
class Branch:
    """A single conversation branch."""

    id: int
    memory: ContextMemory
    approach: str
    result: str = ""
    score: float = 0.0


class ConversationBrancher:
    """Fork conversations into parallel exploration paths."""

    def __init__(self, llm: LLMBackend, num_branches: int = 3):
        self.llm = llm
        self.num_branches = num_branches

    async def branch_and_explore(
        self,
        query: str,
        base_memory: ContextMemory,
    ) -> Branch:
        """
        Fork into N branches, explore in parallel, return best.

        1. Ask LLM for N different approaches
        2. Fork memory for each
        3. Explore each in parallel
        4. Score results, return best
        """
        approaches = await self._generate_approaches(query)
        branches = []

        for i, approach in enumerate(approaches[: self.num_branches]):
            mem = ContextMemory(max_turns=base_memory.max_turns)
            # Copy existing turns
            for turn in base_memory.turns:
                mem.add_turn(turn.role, turn.content)
            branches.append(Branch(id=i, memory=mem, approach=approach))

        # Explore all branches in parallel
        tasks = [self._explore_branch(b, query) for b in branches]
        await asyncio.gather(*tasks)

        # Score and return best
        scored = await self._score_branches(query, branches)
        return max(scored, key=lambda b: b.score)

    async def _generate_approaches(self, query: str) -> list[str]:
        """Ask LLM for different approaches to the query."""
        messages = [
            {"role": "system", "content": "List 3 different approaches to solve this problem, one per line."},
            {"role": "user", "content": query},
        ]
        response = await self.llm.chat(messages, max_tokens=200)
        return [line.strip() for line in response.splitlines() if line.strip()]

    async def _explore_branch(self, branch: Branch, query: str):
        """Explore a single branch."""
        messages = branch.memory.to_messages()
        messages.append({
            "role": "user",
            "content": f"Approach: {branch.approach}\n\nNow solve: {query}",
        })
        branch.result = await self.llm.chat(messages)
        branch.memory.add_turn("assistant", branch.result)

    async def _score_branches(self, query: str, branches: list[Branch]) -> list[Branch]:
        """Score each branch's result."""
        summaries = "\n".join(
            f"Branch {b.id} ({b.approach}): {b.result[:200]}"
            for b in branches
        )
        messages = [
            {"role": "system", "content": "Rate each branch 1-10. Reply with just numbers, one per line."},
            {"role": "user", "content": f"Query: {query}\n\n{summaries}"},
        ]
        response = await self.llm.chat(messages, max_tokens=50)
        scores = []
        for line in response.splitlines():
            try:
                scores.append(float(line.strip().split()[0]))
            except (ValueError, IndexError):
                scores.append(5.0)

        for i, branch in enumerate(branches):
            branch.score = scores[i] if i < len(scores) else 5.0
        return branches
