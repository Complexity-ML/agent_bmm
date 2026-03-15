# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Config — Single source of truth for all agent-bmm settings.

Loads from (in priority order):
  1. CLI arguments (highest)
  2. Environment variables (AGENT_BMM_*)
  3. agent-bmm.yaml / agent-bmm.yml / agent-bmm.json
  4. .env file (for secrets only)
  5. Built-in defaults (lowest)

Usage:
    from agent_bmm.config import load_config, get_config

    cfg = load_config()          # load once at startup
    cfg = get_config()           # get cached config anywhere
    model = cfg["llm"]["model"]  # access any setting
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# ── Built-in defaults (the ONLY place defaults live) ──

DEFAULTS: dict[str, Any] = {
    "llm": {
        "provider": "auto",
        "model": "gpt-4o-mini",
        "base_url": "",
        "api_key": "",
        "fallback_models": [],
    },
    "coder": {
        "max_steps": 20,
        "permission": "allow_reads",
        "stream": True,
        "auto_commit": False,
        "token_budget": 0,
        "max_file_size": 50000,
        "max_context_files": 30,
        "context_window": 100000,
    },
    "router": {
        "hidden_size": 256,
        "expert_size": 128,
        "routing": "round_robin",
    },
    "theme": "dark",
    "tools": ["search", "math", "code"],
    "ollama": {
        "base_url": "http://localhost:11434",
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8765,
    },
    "watcher": {
        "enabled": False,
        "interval": 2.0,
    },
}

# ── .env loader ──

def _load_dotenv():
    """Load .env file into os.environ (no external dependency)."""
    for path in [".env", "../.env"]:
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("'\"").replace("\x00", "")
                        if key and value and not os.environ.get(key):
                            os.environ[key] = value
        except FileNotFoundError:
            continue


# ── Env var mapping ──

ENV_MAP = {
    "AGENT_BMM_MODEL": ("llm", "model"),
    "AGENT_BMM_PROVIDER": ("llm", "provider"),
    "AGENT_BMM_BASE_URL": ("llm", "base_url"),
    "AGENT_BMM_MAX_STEPS": ("coder", "max_steps"),
    "AGENT_BMM_PERMISSION": ("coder", "permission"),
    "AGENT_BMM_STREAM": ("coder", "stream"),
    "AGENT_BMM_TOKEN_BUDGET": ("coder", "token_budget"),
    "AGENT_BMM_THEME": ("theme", None),
    "AGENT_BMM_SERVER_HOST": ("server", "host"),
    "AGENT_BMM_SERVER_PORT": ("server", "port"),
    "AGENT_BMM_OLLAMA_URL": ("ollama", "base_url"),
    "AGENT_BMM_WATCHER": ("watcher", "enabled"),
    "OPENAI_API_KEY": ("llm", "api_key"),
    "ANTHROPIC_API_KEY": ("llm", "api_key"),
}

# Type coercion for env vars
_INT_KEYS = {
    "max_steps", "token_budget", "port", "hidden_size", "expert_size",
    "max_file_size", "max_context_files", "context_window",
}
_FLOAT_KEYS = {"interval"}
_BOOL_KEYS = {"stream", "auto_commit", "enabled"}


def _coerce(key: str, value: str) -> Any:
    """Coerce string env value to the right type."""
    if key in _INT_KEYS:
        return int(value)
    if key in _FLOAT_KEYS:
        return float(value)
    if key in _BOOL_KEYS:
        return value.lower() in ("true", "1", "yes")
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override wins."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ── Config loader ──

_cached_config: dict[str, Any] | None = None


def load_config(path: str | None = None, cli_overrides: dict | None = None) -> dict[str, Any]:
    """
    Load config from all sources, merge in priority order.

    Args:
        path: Explicit path to config file. Auto-detects if None.
        cli_overrides: Dict of CLI arg overrides (highest priority).

    Returns:
        Merged config dict.
    """
    global _cached_config

    # 1. Start with defaults
    config = json.loads(json.dumps(DEFAULTS))  # deep copy

    # 2. Load .env for secrets
    _load_dotenv()

    # 3. Load config file
    file_config = _load_config_file(path)
    if file_config:
        config = _deep_merge(config, file_config)

    # 4. Apply env var overrides
    for env_key, target in ENV_MAP.items():
        value = os.environ.get(env_key)
        if value is None:
            continue
        section, key = target
        if key is None:
            # Top-level key
            config[section] = value
        else:
            if section not in config:
                config[section] = {}
            config[section][key] = _coerce(key, value)

    # 5. Apply CLI overrides (highest priority)
    if cli_overrides:
        config = _deep_merge(config, cli_overrides)

    # 6. Auto-detect provider if set to "auto"
    if config["llm"]["provider"] == "auto" and config["llm"]["model"]:
        from agent_bmm.llm.auto_detect import detect_provider
        provider, base_url, api_key = detect_provider(
            config["llm"]["model"], config["llm"]["base_url"]
        )
        if config["llm"]["provider"] == "auto":
            config["llm"]["provider"] = provider
        if not config["llm"]["base_url"]:
            config["llm"]["base_url"] = base_url
        if not config["llm"]["api_key"]:
            config["llm"]["api_key"] = api_key

    _cached_config = config
    return config


def _load_config_file(path: str | None) -> dict:
    """Load a YAML or JSON config file."""
    if path is None:
        for default in ["agent-bmm.yaml", "agent-bmm.yml", "agent-bmm.json"]:
            if Path(default).exists():
                path = default
                break
    if path is None:
        return {}

    p = Path(path)
    if not p.exists():
        return {}

    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except ImportError:
            pass
    return json.loads(text)


def get_config() -> dict[str, Any]:
    """Get the cached config. Calls load_config() if not loaded yet."""
    global _cached_config
    if _cached_config is None:
        return load_config()
    return _cached_config


def load_profile(name: str) -> dict[str, Any]:
    """Load a named config profile from ~/.agent-bmm/profiles/<name>.yaml."""
    profile_dir = Path.home() / ".agent-bmm" / "profiles"
    for ext in (".yaml", ".yml", ".json"):
        path = profile_dir / f"{name}{ext}"
        if path.exists():
            file_config = _load_config_file(str(path))
            if file_config:
                return load_config(cli_overrides=file_config)
    raise FileNotFoundError(
        f"Profile '{name}' not found in {profile_dir}. "
        f"Create {profile_dir / name}.yaml"
    )


def generate_default_config() -> str:
    """Generate a default agent-bmm.yaml config file content."""
    return """# agent-bmm configuration
# Docs: https://github.com/Complexity-ML/agent_bmm

llm:
  provider: auto          # auto, openai, anthropic
  model: gpt-4o-mini      # or claude-sonnet-4-20250514, ollama:codellama, etc.
  base_url: ""            # leave empty for auto-detect
  # api_key: ""           # prefer .env file: OPENAI_API_KEY or ANTHROPIC_API_KEY
  fallback_models: []     # e.g. [gpt-4o, gpt-4o-mini, ollama:codellama]

coder:
  max_steps: 20
  permission: allow_reads  # ask, allow_reads, yolo
  stream: true
  auto_commit: false
  token_budget: 0          # 0 = unlimited
  max_file_size: 50000
  max_context_files: 30
  context_window: 100000

router:
  hidden_size: 256
  expert_size: 128
  routing: round_robin     # learned, round_robin, embedding

theme: dark                # dark, light, minimal

tools:
  - search
  - math
  - code

ollama:
  base_url: http://localhost:11434

server:
  host: 0.0.0.0
  port: 8765

watcher:
  enabled: false
  interval: 2.0
"""
