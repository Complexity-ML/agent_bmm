# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Security — Rate limiting, API key management, and auth.

Protects the agent API from abuse and manages multi-user access.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10
    cooldown_seconds: float = 60.0


class RateLimiter:
    """
    Token bucket rate limiter.

    Tracks request counts per client and enforces limits.
    """

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def check(self, client_id: str) -> tuple[bool, str]:
        """
        Check if a request is allowed.

        Returns:
            (allowed, reason) — True if allowed, reason string if denied.
        """
        now = time.time()
        requests = self._buckets[client_id]

        # Clean old entries
        minute_ago = now - 60
        hour_ago = now - 3600
        self._buckets[client_id] = [t for t in requests if t > hour_ago]
        requests = self._buckets[client_id]

        # Check per-minute
        recent = sum(1 for t in requests if t > minute_ago)
        if recent >= self.config.requests_per_minute:
            return False, f"Rate limit: {self.config.requests_per_minute}/min exceeded"

        # Check per-hour
        if len(requests) >= self.config.requests_per_hour:
            return False, f"Rate limit: {self.config.requests_per_hour}/hour exceeded"

        # Check burst
        last_second = sum(1 for t in requests if t > now - 1)
        if last_second >= self.config.burst_size:
            return False, f"Burst limit: {self.config.burst_size}/sec exceeded"

        # Allow
        self._buckets[client_id].append(now)
        return True, ""

    def get_usage(self, client_id: str) -> dict:
        """Get current usage stats for a client."""
        now = time.time()
        requests = self._buckets.get(client_id, [])
        return {
            "requests_last_minute": sum(1 for t in requests if t > now - 60),
            "requests_last_hour": sum(1 for t in requests if t > now - 3600),
            "limit_per_minute": self.config.requests_per_minute,
            "limit_per_hour": self.config.requests_per_hour,
        }


@dataclass
class APIKey:
    """An API key with permissions."""

    key_hash: str
    name: str
    created_at: float
    permissions: set[str] = field(default_factory=lambda: {"query", "tools"})
    rate_limit: RateLimitConfig | None = None
    active: bool = True


class APIKeyManager:
    """
    API key management for multi-user access.

    Generates, validates, and revokes API keys.
    Keys are stored as SHA-256 hashes (never in plaintext).
    """

    def __init__(self):
        self._keys: dict[str, APIKey] = {}  # hash → APIKey
        self._rate_limiters: dict[str, RateLimiter] = {}

    def generate_key(
        self,
        name: str,
        permissions: set[str] | None = None,
        rate_limit: RateLimitConfig | None = None,
    ) -> str:
        """
        Generate a new API key.

        Returns the raw key (only shown once). Store it securely!
        """
        raw_key = f"abmm_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(raw_key)

        self._keys[key_hash] = APIKey(
            key_hash=key_hash,
            name=name,
            created_at=time.time(),
            permissions=permissions or {"query", "tools"},
            rate_limit=rate_limit,
        )

        if rate_limit:
            self._rate_limiters[key_hash] = RateLimiter(rate_limit)

        return raw_key

    def validate(self, raw_key: str) -> tuple[bool, APIKey | None, str]:
        """
        Validate an API key.

        Returns:
            (valid, api_key, reason)
        """
        key_hash = self._hash_key(raw_key)
        api_key = self._keys.get(key_hash)

        if api_key is None:
            return False, None, "Invalid API key"

        if not api_key.active:
            return False, None, "API key revoked"

        # Check rate limit
        limiter = self._rate_limiters.get(key_hash)
        if limiter:
            allowed, reason = limiter.check(key_hash)
            if not allowed:
                return False, api_key, reason

        return True, api_key, ""

    def check_permission(self, raw_key: str, permission: str) -> bool:
        """Check if a key has a specific permission."""
        valid, api_key, _ = self.validate(raw_key)
        if not valid or api_key is None:
            return False
        return permission in api_key.permissions

    def revoke(self, raw_key: str) -> bool:
        """Revoke an API key."""
        key_hash = self._hash_key(raw_key)
        if key_hash in self._keys:
            self._keys[key_hash].active = False
            return True
        return False

    def list_keys(self) -> list[dict]:
        """List all API keys (without hashes)."""
        return [
            {
                "name": k.name,
                "created_at": k.created_at,
                "permissions": list(k.permissions),
                "active": k.active,
            }
            for k in self._keys.values()
        ]

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()
