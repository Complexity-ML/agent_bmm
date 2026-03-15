#!/usr/bin/env python3
# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
"""
vLLM Integration Guide — Using agent-bmm with vLLM + token-routed models.

Prerequisites:
    pip install vllm
    vllm serve meta-llama/Llama-3-8B-Instruct --port 8081

Usage:
    # Option 1: via config
    # agent-bmm.yaml:
    #   llm:
    #     provider: auto
    #     model: meta-llama/Llama-3-8B-Instruct
    #     base_url: http://localhost:8081/v1

    # Option 2: via env vars
    # export AGENT_BMM_MODEL=meta-llama/Llama-3-8B-Instruct
    # export AGENT_BMM_BASE_URL=http://localhost:8081/v1

    # Option 3: via Python API (below)

    python examples/vllm_guide.py
"""

import asyncio
import logging
import sys

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from agent_bmm.agent import Agent


async def main():
    # Create agent pointing to local vLLM server
    agent = Agent(
        model="meta-llama/Llama-3-8B-Instruct",
        base_url="http://localhost:8081/v1",  # vLLM OpenAI-compatible endpoint
        provider="openai",  # vLLM speaks OpenAI protocol
        tools="all",
    )

    logger.info("Agent BMM + vLLM Integration")
    logger.info("=" * 40)
    logger.info("Model: %s", agent._llm_config.model)
    logger.info("URL:   %s", agent._llm_config.base_url)

    # Example queries
    queries = [
        "What is 42 * 17?",
        "Search for the latest Python release",
    ]

    for query in queries:
        logger.info("Q: %s", query)
        try:
            answer = await agent.ask(query)
            logger.info("A: %s\n", answer)
        except Exception as e:
            logger.error("Error: %s", e)
            logger.info("Make sure vLLM is running: vllm serve <model> --port 8081")
            break


if __name__ == "__main__":
    asyncio.run(main())
