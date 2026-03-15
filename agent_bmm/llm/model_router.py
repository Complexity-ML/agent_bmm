# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Model Router — Auto-select LLM per query based on complexity.

Routes simple queries to cheap/fast models, complex ones to powerful models.
Uses keyword heuristics + token count to estimate complexity.

Configured via agent-bmm.yaml:
    llm:
      model_router:
        simple: gpt-4o-mini      # fast, cheap
        medium: gpt-4o            # balanced
        complex: claude-sonnet-4-20250514  # powerful
"""

from __future__ import annotations

import re

from agent_bmm.llm.auto_detect import detect_provider
from agent_bmm.llm.backend import LLMBackend, LLMConfig

# Complexity signals
COMPLEX_KEYWORDS = {
    "refactor", "architect", "design", "optimize", "debug", "security",
    "performance", "migrate", "analyze", "explain why", "trade-off",
    "compare", "implement from scratch", "full stack",
}
SIMPLE_KEYWORDS = {
    "hello", "print", "create file", "rename", "delete", "list",
    "read", "show", "what is", "how to", "fix typo",
}


def estimate_complexity(query: str) -> str:
    """Estimate query complexity: 'simple', 'medium', or 'complex'."""
    q = query.lower()
    words = set(re.findall(r"\w+", q))

    complex_hits = sum(1 for kw in COMPLEX_KEYWORDS if kw in q)
    simple_hits = sum(1 for kw in SIMPLE_KEYWORDS if kw in q)

    # Long queries with code tend to be complex
    if len(query) > 500 or complex_hits >= 2:
        return "complex"
    if complex_hits > simple_hits:
        return "medium"
    if len(query) < 100 and (simple_hits > 0 or len(words) < 10):
        return "simple"
    return "medium"


class ModelRouter:
    """Routes queries to the best LLM based on complexity."""

    def __init__(self, models: dict[str, str] | None = None):
        """
        Args:
            models: Dict mapping complexity → model name.
                    e.g. {"simple": "gpt-4o-mini", "medium": "gpt-4o", "complex": "claude-sonnet-4-20250514"}
        """
        self.models = models or {
            "simple": "gpt-4o-mini",
            "medium": "gpt-4o-mini",
            "complex": "gpt-4o-mini",
        }
        self._backends: dict[str, LLMBackend] = {}

    def _get_backend(self, complexity: str) -> LLMBackend:
        """Get or create backend for complexity level."""
        if complexity not in self._backends:
            model = self.models.get(complexity, self.models["medium"])
            provider, base_url, api_key = detect_provider(model)
            self._backends[complexity] = LLMBackend(
                LLMConfig(provider=provider, base_url=base_url, api_key=api_key, model=model)
            )
        return self._backends[complexity]

    def route(self, query: str) -> tuple[LLMBackend, str]:
        """Route query to best model. Returns (backend, complexity)."""
        complexity = estimate_complexity(query)
        return self._get_backend(complexity), complexity

    async def close(self):
        for backend in self._backends.values():
            await backend.close()
