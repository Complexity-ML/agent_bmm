# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Retry — Error recovery and retry logic for tools and LLM calls.

Exponential backoff with jitter, circuit breaker, and fallback.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: tuple[type, ...] = (Exception,)


@dataclass
class CircuitBreaker:
    """
    Circuit breaker — stops calling a failing service after N failures.

    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    _failures: int = 0
    _last_failure: float = 0.0
    _state: str = "closed"

    @property
    def is_open(self) -> bool:
        if self._state == "open":
            if time.time() - self._last_failure > self.recovery_timeout:
                self._state = "half_open"
                return False
            return True
        return False

    def record_success(self):
        self._failures = 0
        self._state = "closed"

    def record_failure(self):
        self._failures += 1
        self._last_failure = time.time()
        if self._failures >= self.failure_threshold:
            self._state = "open"


async def retry_async(
    fn: Callable[..., Awaitable[Any]],
    *args: Any,
    config: RetryConfig | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    fallback: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Execute an async function with retry logic.

    Args:
        fn: Async function to call.
        config: Retry configuration.
        circuit_breaker: Optional circuit breaker.
        fallback: Fallback function if all retries fail.
    """
    config = config or RetryConfig()

    if circuit_breaker and circuit_breaker.is_open:
        if fallback:
            return fallback(*args, **kwargs)
        raise RuntimeError("Circuit breaker is open")

    last_error = None
    for attempt in range(config.max_retries + 1):
        try:
            result = await fn(*args, **kwargs)
            if circuit_breaker:
                circuit_breaker.record_success()
            return result
        except config.retry_on as e:
            last_error = e
            if circuit_breaker:
                circuit_breaker.record_failure()

            if attempt < config.max_retries:
                delay = min(
                    config.base_delay * (config.exponential_base ** attempt),
                    config.max_delay,
                )
                if config.jitter:
                    delay *= 0.5 + random.random()
                await asyncio.sleep(delay)

    if fallback:
        return fallback(*args, **kwargs)
    raise last_error


def retry_sync(
    fn: Callable[..., Any],
    *args: Any,
    config: RetryConfig | None = None,
    fallback: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Synchronous retry wrapper."""
    config = config or RetryConfig()
    last_error = None

    for attempt in range(config.max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except config.retry_on as e:
            last_error = e
            if attempt < config.max_retries:
                delay = min(
                    config.base_delay * (config.exponential_base ** attempt),
                    config.max_delay,
                )
                if config.jitter:
                    delay *= 0.5 + random.random()
                time.sleep(delay)

    if fallback:
        return fallback(*args, **kwargs)
    raise last_error
