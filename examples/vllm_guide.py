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
import sys

sys.path.insert(0, ".")

from agent_bmm.agent import Agent


async def main():
    # Create agent pointing to local vLLM server
    agent = Agent(
        model="meta-llama/Llama-3-8B-Instruct",
        base_url="http://localhost:8081/v1",  # vLLM OpenAI-compatible endpoint
        provider="openai",                     # vLLM speaks OpenAI protocol
        tools="all",
    )

    print("Agent BMM + vLLM Integration")
    print("=" * 40)
    print(f"Model: {agent._llm_config.model}")
    print(f"URL:   {agent._llm_config.base_url}")
    print()

    # Example queries
    queries = [
        "What is 42 * 17?",
        "Search for the latest Python release",
    ]

    for query in queries:
        print(f"Q: {query}")
        try:
            answer = await agent.ask(query)
            print(f"A: {answer}\n")
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure vLLM is running: vllm serve <model> --port 8081\n")
            break


if __name__ == "__main__":
    asyncio.run(main())
