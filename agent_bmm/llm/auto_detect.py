# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Auto-detect LLM provider from base_url or model name.

Provider registry is data-driven — no hardcoded URLs scattered in code.
All URLs and env var names are in PROVIDERS / MODEL_PREFIXES tables.
"""

from __future__ import annotations

import os

# ── Provider registry (the ONLY place URLs live) ──

PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "provider": "openai",  # Groq is OpenAI-compatible
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "provider": "openai",  # Together is OpenAI-compatible
    },
    "ollama": {
        "base_url_env": "OLLAMA_BASE_URL",
        "base_url_default": "http://localhost:11434",
        "api_key_env": "",
        "provider": "openai",  # Ollama exposes OpenAI-compatible /v1
    },
    "local": {
        "base_url_env": "AGENT_BMM_LOCAL_URL",
        "base_url_default": "http://localhost:8081/v1",
        "api_key_env": "",
        "provider": "openai",
    },
}

# model prefix → provider name
MODEL_PREFIXES: list[tuple[str, str]] = [
    ("gpt-", "openai"),
    ("o1-", "openai"),
    ("o3-", "openai"),
    ("o4-", "openai"),
    ("claude-", "anthropic"),
    ("ollama:", "ollama"),
]

# Keywords in model name → try these providers in order
MODEL_KEYWORDS: dict[str, list[str]] = {
    "llama": ["groq", "together", "local"],
    "mistral": ["groq", "together", "local"],
    "qwen": ["groq", "together", "local"],
    "gemma": ["groq", "together", "local"],
    "phi": ["local"],
}

# URL substring → provider name
URL_HINTS: list[tuple[str, str]] = [
    ("anthropic.com", "anthropic"),
    ("openai.com", "openai"),
    ("groq.com", "groq"),
    ("together.xyz", "together"),
    ("localhost", "local"),
    ("127.0.0.1", "local"),
]


def _resolve_provider(name: str) -> tuple[str, str, str]:
    """Resolve a provider name to (provider_type, base_url, api_key)."""
    info = PROVIDERS.get(name, PROVIDERS["openai"])
    provider_type = info.get("provider", name)

    # Base URL: check env var first, then default
    if "base_url_env" in info and info["base_url_env"]:
        base_url = os.environ.get(info["base_url_env"], info.get("base_url_default", ""))
    else:
        base_url = info.get("base_url", "")

    # API key from env
    api_key_env = info.get("api_key_env", "")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""

    return provider_type, base_url, api_key


def detect_provider(model: str = "", base_url: str = "") -> tuple[str, str, str]:
    """
    Auto-detect provider, base_url, and api_key from model name or base_url.

    Returns:
        (provider, base_url, api_key)
    """
    # 1. Detect from base_url
    if base_url:
        for hint, provider_name in URL_HINTS:
            if hint in base_url:
                provider_type, _, api_key = _resolve_provider(provider_name)
                return provider_type, base_url, api_key
        # Unknown URL — assume OpenAI-compatible
        return "openai", base_url, ""

    # 2. Detect from model prefix
    for prefix, provider_name in MODEL_PREFIXES:
        if model.startswith(prefix):
            if provider_name == "ollama":
                ollama_model = model.split(":", 1)[1]
                _, base_url, _ = _resolve_provider("ollama")
                return "openai", f"{base_url}/v1", f"ollama:{ollama_model}"
            return _resolve_provider(provider_name)

    # 3. Detect from model name keywords
    model_lower = model.lower()
    for keyword, provider_chain in MODEL_KEYWORDS.items():
        if keyword in model_lower:
            for provider_name in provider_chain:
                provider_type, base_url, api_key = _resolve_provider(provider_name)
                if api_key or provider_name == "local":
                    return provider_type, base_url, api_key

    # 4. Default: OpenAI
    return _resolve_provider("openai")
