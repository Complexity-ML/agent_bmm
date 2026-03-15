# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Agent Self-Reflection — Evaluate own reasoning and adjust strategy.

After each agent step, a reflection prompt evaluates quality
and suggests adjustments for the next step.
"""

from __future__ import annotations

from agent_bmm.llm.backend import LLMBackend

REFLECTION_PROMPT = """Evaluate the reasoning quality of this agent step.

Query: {query}
Agent thought: {thought}
Action taken: {action}
Result: {result}

Rate 1-10 and suggest improvements. Reply as JSON:
{{"score": 7, "feedback": "...", "should_retry": false, "adjusted_approach": ""}}"""


class SelfReflector:
    """Agent self-reflection — evaluates and adjusts strategy."""

    def __init__(self, llm: LLMBackend, min_score: int = 4):
        self.llm = llm
        self.min_score = min_score
        self._history: list[dict] = []

    async def reflect(
        self,
        query: str,
        thought: str,
        action: str,
        result: str,
    ) -> dict:
        """
        Reflect on a step. Returns reflection dict with score, feedback, etc.
        """
        import json

        prompt = REFLECTION_PROMPT.format(
            query=query[:500],
            thought=thought[:500],
            action=action[:200],
            result=result[:500],
        )
        messages = [
            {"role": "system", "content": "You are a reasoning quality evaluator. Reply with JSON only."},
            {"role": "user", "content": prompt},
        ]
        response = await self.llm.chat(messages, max_tokens=200)

        try:
            reflection = json.loads(response)
        except json.JSONDecodeError:
            reflection = {"score": 5, "feedback": response, "should_retry": False}

        self._history.append(reflection)
        return reflection

    @property
    def avg_score(self) -> float:
        if not self._history:
            return 0.0
        scores = [r.get("score", 5) for r in self._history]
        return sum(scores) / len(scores)

    @property
    def should_adjust(self) -> bool:
        """True if recent scores suggest strategy needs adjustment."""
        if len(self._history) < 2:
            return False
        recent = self._history[-3:]
        avg = sum(r.get("score", 5) for r in recent) / len(recent)
        return avg < self.min_score
