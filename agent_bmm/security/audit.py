# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Audit Logging (#32) + Encryption at Rest (#34) + Per-key Rate Limiting (#31).

Logs all API key usage, tool executions, and routing decisions.
Optionally encrypts the conversation DB and stored secrets.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from agent_bmm.security.security import RateLimitConfig, RateLimiter

# ── #32 Audit Logging ──

logger = logging.getLogger("agent_bmm.audit")


@dataclass
class AuditEvent:
    timestamp: float
    event_type: str  # "auth", "tool_exec", "route", "file_access", "command"
    client_id: str
    detail: str
    allowed: bool


class AuditLogger:
    """Log all agent actions for compliance and debugging."""

    def __init__(self, log_path: str | Path = "agent-bmm-audit.jsonl"):
        self.log_path = Path(log_path)
        self._events: list[AuditEvent] = []

    def log(self, event_type: str, client_id: str, detail: str, allowed: bool = True):
        event = AuditEvent(
            timestamp=time.time(),
            event_type=event_type,
            client_id=client_id,
            detail=detail[:500],
            allowed=allowed,
        )
        self._events.append(event)
        logger.info(f"[{event_type}] client={client_id} allowed={allowed} {detail[:100]}")

        # Append to JSONL file
        with open(self.log_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": event.timestamp,
                        "type": event.event_type,
                        "client": event.client_id,
                        "detail": event.detail,
                        "allowed": event.allowed,
                    }
                )
                + "\n"
            )

    def get_events(self, client_id: str | None = None, limit: int = 100) -> list[AuditEvent]:
        events = self._events
        if client_id:
            events = [e for e in events if e.client_id == client_id]
        return events[-limit:]


# ── #34 Encryption at Rest ──


class EncryptionManager:
    """Encrypt/decrypt data at rest using Fernet symmetric encryption."""

    def __init__(self, key: str | None = None):
        self._key = key
        self._fernet = None

    def _get_fernet(self):
        if self._fernet is None:
            from cryptography.fernet import Fernet

            if self._key:
                self._fernet = Fernet(self._key.encode() if isinstance(self._key, str) else self._key)
            else:
                # Generate and store key
                self._key = Fernet.generate_key()
                self._fernet = Fernet(self._key)
        return self._fernet

    def encrypt(self, data: str) -> bytes:
        return self._get_fernet().encrypt(data.encode())

    def decrypt(self, token: bytes) -> str:
        return self._get_fernet().decrypt(token).decode()

    @property
    def key(self) -> str:
        self._get_fernet()
        return self._key if isinstance(self._key, str) else self._key.decode()


# ── #31 Per-key Rate Limiting Tiers ──

RATE_TIERS: dict[str, RateLimitConfig] = {
    "free": RateLimitConfig(requests_per_minute=10, requests_per_hour=100, burst_size=3),
    "pro": RateLimitConfig(requests_per_minute=60, requests_per_hour=2000, burst_size=10),
    "enterprise": RateLimitConfig(requests_per_minute=300, requests_per_hour=10000, burst_size=50),
}


class TieredRateLimiter:
    """Rate limiting with different tiers per API key."""

    def __init__(self):
        self._limiters: dict[str, RateLimiter] = {}
        self._key_tiers: dict[str, str] = {}  # key_hash → tier name

    def set_tier(self, key_hash: str, tier: str):
        config = RATE_TIERS.get(tier, RATE_TIERS["free"])
        self._key_tiers[key_hash] = tier
        self._limiters[key_hash] = RateLimiter(config)

    def check(self, key_hash: str) -> tuple[bool, str]:
        if key_hash not in self._limiters:
            self.set_tier(key_hash, "free")
        return self._limiters[key_hash].check(key_hash)

    def get_tier(self, key_hash: str) -> str:
        return self._key_tiers.get(key_hash, "free")
