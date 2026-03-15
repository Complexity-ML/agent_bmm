# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Cost Tracker — Estimate token usage and API costs.

Tracks total characters sent/received, estimates tokens,
and calculates approximate cost based on model pricing.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()

# Approximate pricing per 1M tokens (input/output)
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Anthropic
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-haiku-3-5": (0.25, 1.25),
    # Local/free
    "default": (0.0, 0.0),
}

CHARS_PER_TOKEN = 4  # rough estimate


class CostTracker:
    """Track token usage and estimate costs."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.total_input_chars = 0
        self.total_output_chars = 0
        self.num_requests = 0

    def _get_pricing(self) -> tuple[float, float]:
        """Get (input_price, output_price) per 1M tokens."""
        for key, pricing in MODEL_PRICING.items():
            if key in self.model:
                return pricing
        return MODEL_PRICING["default"]

    def add_request(self, total_chars: int, output_chars: int = 0):
        """Record a request. total_chars = all chars in conversation."""
        self.num_requests += 1
        # Rough split: 70% input, 30% output
        if output_chars:
            self.total_input_chars += total_chars - output_chars
            self.total_output_chars += output_chars
        else:
            self.total_input_chars += int(total_chars * 0.7)
            self.total_output_chars += int(total_chars * 0.3)

    @property
    def input_tokens(self) -> int:
        return self.total_input_chars // CHARS_PER_TOKEN

    @property
    def output_tokens(self) -> int:
        return self.total_output_chars // CHARS_PER_TOKEN

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost(self) -> float:
        """Estimated cost in USD."""
        input_price, output_price = self._get_pricing()
        return (self.input_tokens * input_price + self.output_tokens * output_price) / 1_000_000

    def print_summary(self):
        """Print cost summary to terminal."""
        table = Table(title="Cost Summary", border_style="yellow")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="yellow")

        table.add_row("Model", self.model)
        table.add_row("Requests", str(self.num_requests))
        table.add_row("Input tokens", f"~{self.input_tokens:,}")
        table.add_row("Output tokens", f"~{self.output_tokens:,}")
        table.add_row("Total tokens", f"~{self.total_tokens:,}")
        table.add_row("Estimated cost", f"${self.estimated_cost:.4f}")

        console.print(table)
