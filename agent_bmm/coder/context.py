# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Context Window Management — Keep conversation within token limits.

Truncates old messages when the context gets too large,
keeping the system prompt and recent messages.
"""

from __future__ import annotations

CHARS_PER_TOKEN = 4


class ContextManager:
    """Manages conversation history to fit within token limits."""

    def __init__(self, max_tokens: int = 100_000):
        self.max_tokens = max_tokens

    def estimate_tokens(self, history: list[dict[str, str]]) -> int:
        """Estimate total tokens in conversation."""
        return sum(len(m.get("content", "")) for m in history) // CHARS_PER_TOKEN

    def truncate(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Truncate history to fit within max_tokens.

        Keeps:
        - System prompt (first message)
        - Last N messages that fit
        - Adds a summary of truncated messages
        """
        if not history:
            return history

        total = self.estimate_tokens(history)
        if total <= self.max_tokens:
            return history

        # Always keep system prompt
        system = [history[0]] if history[0].get("role") == "system" else []
        rest = history[1:] if system else history

        # Keep messages from the end until we hit the limit
        budget = self.max_tokens - self.estimate_tokens(system) - 100  # 100 for summary
        kept = []
        for msg in reversed(rest):
            msg_tokens = len(msg.get("content", "")) // CHARS_PER_TOKEN
            if budget - msg_tokens < 0:
                break
            kept.insert(0, msg)
            budget -= msg_tokens

        truncated_count = len(rest) - len(kept)
        if truncated_count > 0:
            summary = {
                "role": "user",
                "content": f"[{truncated_count} earlier messages truncated to fit context window]",
            }
            return system + [summary] + kept

        return system + kept

    def truncate_long_result(self, result: str, max_chars: int = 3000) -> str:
        """Truncate a single tool result if too long."""
        if len(result) <= max_chars:
            return result
        half = max_chars // 2
        return result[:half] + f"\n\n... ({len(result)} chars, truncated) ...\n\n" + result[-half:]
