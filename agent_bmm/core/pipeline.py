# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Tool Chaining / Pipelines — Output of tool A feeds into tool B.

Define pipelines where each stage processes the previous result.
BMM routes each stage to the appropriate tool.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_bmm.tools.registry import ToolRegistry


@dataclass
class PipelineStage:
    """A single stage in a tool pipeline."""

    tool_name: str
    transform: str = ""  # Optional query template: "summarize: {input}"


class ToolPipeline:
    """Chain tools together: output of A → input of B."""

    def __init__(self, registry: ToolRegistry, stages: list[PipelineStage]):
        self.registry = registry
        self.stages = stages

    async def run(self, initial_input: str) -> str:
        """Execute the pipeline, passing results between stages."""
        current = initial_input

        for stage in self.stages:
            tool = self.registry.get_by_name(stage.tool_name)
            if tool is None:
                return f"Error: tool '{stage.tool_name}' not found"

            # Apply template if provided
            query = stage.transform.replace("{input}", current) if stage.transform else current

            try:
                result = await tool.acall(query)
                current = str(result)
            except Exception as e:
                return f"Pipeline failed at '{stage.tool_name}': {e}"

        return current

    def __repr__(self) -> str:
        names = " → ".join(s.tool_name for s in self.stages)
        return f"Pipeline({names})"
