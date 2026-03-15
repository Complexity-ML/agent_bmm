# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
import asyncio
import sys
sys.path.insert(0, ".")

from agent_bmm.core.retry import retry_async, retry_sync, RetryConfig, CircuitBreaker


def test_retry_sync_success():
    calls = [0]
    def fn():
        calls[0] += 1
        if calls[0] < 3:
            raise ValueError("not yet")
        return "ok"

    result = retry_sync(fn, config=RetryConfig(max_retries=5, base_delay=0.01))
    assert result == "ok"
    assert calls[0] == 3
    print("OK: retry sync success after 3 attempts")


def test_retry_sync_fail():
    def fn():
        raise ValueError("always fail")

    try:
        retry_sync(fn, config=RetryConfig(max_retries=2, base_delay=0.01))
        assert False, "Should have raised"
    except ValueError:
        pass
    print("OK: retry sync fails after max_retries")


def test_retry_sync_fallback():
    def fn():
        raise ValueError("fail")

    result = retry_sync(
        fn,
        config=RetryConfig(max_retries=1, base_delay=0.01),
        fallback=lambda: "fallback_value",
    )
    assert result == "fallback_value"
    print("OK: retry sync fallback")


def test_retry_async():
    calls = [0]
    async def fn():
        calls[0] += 1
        if calls[0] < 2:
            raise ConnectionError("retry me")
        return "async ok"

    result = asyncio.run(
        retry_async(fn, config=RetryConfig(max_retries=3, base_delay=0.01))
    )
    assert result == "async ok"
    print("OK: retry async")


def test_circuit_breaker():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
    assert not cb.is_open

    cb.record_failure()
    cb.record_failure()
    assert not cb.is_open

    cb.record_failure()
    assert cb.is_open

    cb.record_success()
    assert not cb.is_open
    print("OK: circuit breaker")


if __name__ == "__main__":
    test_retry_sync_success()
    test_retry_sync_fail()
    test_retry_sync_fallback()
    test_retry_async()
    test_circuit_breaker()
    print("\nAll retry tests passed!")
