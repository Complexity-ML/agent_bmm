# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Config Schema — Pydantic validation for agent-bmm.yaml.

Validates config with helpful error messages. Used by config.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    provider: str = Field("auto", description="LLM provider: auto, openai, anthropic")
    model: str = Field("gpt-4o-mini", description="Model name")
    base_url: str = Field("", description="API base URL (empty = auto-detect)")
    api_key: str = Field("", description="API key (prefer .env)")
    fallback_models: list[str] = Field(default_factory=list)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid = {"auto", "openai", "anthropic", "local"}
        if v not in valid:
            raise ValueError(f"provider must be one of {valid}, got '{v}'")
        return v


class CoderConfig(BaseModel):
    max_steps: int = Field(20, ge=1, le=100, description="Max agent steps")
    permission: str = Field("allow_reads", description="Permission level")
    stream: bool = Field(True)
    auto_commit: bool = Field(False)
    token_budget: int = Field(0, ge=0, description="Max tokens (0=unlimited)")
    max_file_size: int = Field(50000, ge=1000)
    max_context_files: int = Field(30, ge=1)
    context_window: int = Field(100000, ge=1000)

    @field_validator("permission")
    @classmethod
    def validate_permission(cls, v: str) -> str:
        valid = {"ask", "allow_reads", "yolo"}
        if v not in valid:
            raise ValueError(f"permission must be one of {valid}, got '{v}'")
        return v


class RouterConfig(BaseModel):
    hidden_size: int = Field(256, ge=16)
    expert_size: int = Field(128, ge=16)
    routing: str = Field("round_robin")

    @field_validator("routing")
    @classmethod
    def validate_routing(cls, v: str) -> str:
        valid = {"learned", "round_robin", "embedding"}
        if v not in valid:
            raise ValueError(f"routing must be one of {valid}, got '{v}'")
        return v


class ServerConfig(BaseModel):
    host: str = Field("0.0.0.0")
    port: int = Field(8765, ge=1, le=65535)


class WatcherConfig(BaseModel):
    enabled: bool = Field(False)
    interval: float = Field(2.0, ge=0.1)


class AgentBMMConfig(BaseModel):
    """Full agent-bmm configuration schema."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    coder: CoderConfig = Field(default_factory=CoderConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    theme: str = Field("dark")
    tools: list[str] = Field(default_factory=lambda: ["search", "math", "code"])

    @field_validator("theme")
    @classmethod
    def validate_theme(cls, v: str) -> str:
        valid = {"dark", "light", "minimal"}
        if v not in valid:
            raise ValueError(f"theme must be one of {valid}, got '{v}'")
        return v


def validate_config(config: dict) -> AgentBMMConfig:
    """Validate a config dict. Raises ValidationError with helpful messages."""
    return AgentBMMConfig(**config)
