# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Auto-detect LLM provider from base_url or model name.

No config needed — just pass the model name and we figure out the rest.
"""

from __future__ import annotations

import os


def detect_provider(model: str = "", base_url: str = "") -> tuple[str, str, str]:
    """
    Auto-detect provider, base_url, and api_key from model name or base_url.

    Returns:
        (provider, base_url, api_key)

    Examples:
        detect_provider(model="gpt-4o") → ("openai", "https://api.openai.com/v1", "sk-...")
        detect_provider(model="claude-sonnet-4-20250514") → ("anthropic", "https://api.anthropic.com/v1", "sk-ant-...")
        detect_provider(base_url="http://localhost:8081/v1") → ("openai", "http://localhost:8081/v1", "")
    """

    # Detect from base_url
    if base_url:
        if "anthropic.com" in base_url:
            return "anthropic", base_url, os.environ.get("ANTHROPIC_API_KEY", "")
        if "openai.com" in base_url:
            return "openai", base_url, os.environ.get("OPENAI_API_KEY", "")
        if "groq.com" in base_url:
            return "openai", base_url, os.environ.get("GROQ_API_KEY", "")
        if "together.xyz" in base_url:
            return "openai", base_url, os.environ.get("TOGETHER_API_KEY", "")
        if "localhost" in base_url or "127.0.0.1" in base_url:
            return "openai", base_url, ""
        # Unknown URL — assume OpenAI-compatible
        return "openai", base_url, ""

    # Detect from model name
    if model.startswith("gpt-") or model.startswith("o1-") or model.startswith("o3-"):
        return "openai", "https://api.openai.com/v1", os.environ.get("OPENAI_API_KEY", "")

    if model.startswith("claude-"):
        return "anthropic", "https://api.anthropic.com/v1", os.environ.get("ANTHROPIC_API_KEY", "")

    # Ollama local models: "ollama:codellama", "ollama:llama3"
    if model.startswith("ollama:"):
        ollama_model = model.split(":", 1)[1]
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return "openai", f"{ollama_url}/v1", f"ollama:{ollama_model}"

    model_lower = model.lower()
    if "llama" in model_lower or "mistral" in model_lower or "qwen" in model_lower:
        # Open models — check for Groq/Together first, fallback to local
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            return "openai", "https://api.groq.com/openai/v1", groq_key
        together_key = os.environ.get("TOGETHER_API_KEY", "")
        if together_key:
            return "openai", "https://api.together.xyz/v1", together_key
        # Assume local vLLM
        return "openai", "http://localhost:8081/v1", ""

    # Default: OpenAI
    return "openai", "https://api.openai.com/v1", os.environ.get("OPENAI_API_KEY", "")
