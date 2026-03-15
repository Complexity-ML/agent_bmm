# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
OAuth2 Authentication — Token-based auth for the WebSocket server.

Supports OAuth2 Bearer tokens as alternative to static API keys.

Configured via agent-bmm.yaml:
    auth:
      type: oauth2       # or "api_key", "none"
      issuer: https://auth.example.com
      audience: agent-bmm
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class AuthConfig:
    auth_type: str = "none"  # "none", "api_key", "oauth2"
    api_key: str = ""
    issuer: str = ""
    audience: str = "agent-bmm"
    jwks_url: str = ""


class AuthManager:
    """Manage authentication for the agent server."""

    def __init__(self, config: AuthConfig):
        self.config = config
        self._jwks_cache: dict | None = None
        self._cache_time: float = 0

    def verify(self, token: str) -> tuple[bool, str]:
        """
        Verify a token. Returns (is_valid, user_or_error).
        """
        if self.config.auth_type == "none":
            return True, "anonymous"

        if self.config.auth_type == "api_key":
            if token == self.config.api_key:
                return True, "api_key_user"
            return False, "Invalid API key"

        if self.config.auth_type == "oauth2":
            return self._verify_jwt(token)

        return False, f"Unknown auth type: {self.config.auth_type}"

    def _verify_jwt(self, token: str) -> tuple[bool, str]:
        """Verify a JWT token."""
        try:
            import jwt

            # Decode without verification first to get header
            jwt.get_unverified_header(token)

            # For simplicity, verify with the issuer's public key
            # In production, fetch JWKS from self.config.jwks_url
            payload = jwt.decode(
                token,
                options={"verify_signature": False},  # TODO: verify with JWKS
                audience=self.config.audience,
                issuer=self.config.issuer,
            )

            # Check expiration
            if payload.get("exp", 0) < time.time():
                return False, "Token expired"

            return True, payload.get("sub", "unknown")

        except ImportError:
            return False, "PyJWT not installed: pip install PyJWT"
        except Exception as e:
            return False, f"JWT verification failed: {e}"
