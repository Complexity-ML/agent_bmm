#!/usr/bin/env python3
# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Quickstart — Agent BMM in 20 lines.

Usage:
    # Start a vLLM server first:
    # vllm serve your-model --port 8081

    python examples/quickstart.py
"""

import asyncio
import logging
import sys

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from agent_bmm import Agent
from agent_bmm.tools.builtin import MathTool, CodeExecTool, WebSearchTool


async def main():
    # Create agent
    agent = Agent(
        model="/workspace/checkpoints/run2-full-final/final",
        base_url="http://localhost:8081/v1",
        routing="round_robin",
        max_steps=3,
    )

    # Add tools
    agent.add_tool(**MathTool().__dict__)
    agent.add_tool(**CodeExecTool().__dict__)
    agent.add_tool(**WebSearchTool().__dict__)

    # Ask
    answer = await agent.ask("What is the square root of 144?")
    logger.info("Answer: %s", answer)


if __name__ == "__main__":
    asyncio.run(main())
