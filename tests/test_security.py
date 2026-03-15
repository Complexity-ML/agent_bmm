# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
import sys, time
sys.path.insert(0, ".")

from agent_bmm.security import RateLimiter, RateLimitConfig, APIKeyManager


def test_rate_limiter():
    limiter = RateLimiter(RateLimitConfig(requests_per_minute=5, burst_size=3))

    for i in range(3):
        ok, _ = limiter.check("client1")
        assert ok, f"Request {i} should be allowed"

    # Burst limit
    ok, reason = limiter.check("client1")
    assert not ok
    assert "Burst" in reason
    print("OK: rate limiter burst")


def test_rate_limiter_per_minute():
    limiter = RateLimiter(RateLimitConfig(requests_per_minute=3, burst_size=100))
    for i in range(3):
        ok, _ = limiter.check("c1")
        assert ok
    ok, reason = limiter.check("c1")
    assert not ok
    assert "/min" in reason
    print("OK: rate limiter per minute")


def test_api_key_lifecycle():
    mgr = APIKeyManager()
    key = mgr.generate_key("test-user", permissions={"query"})
    assert key.startswith("abmm_")

    valid, api_key, reason = mgr.validate(key)
    assert valid
    assert api_key.name == "test-user"
    print(f"OK: generated key {key[:15]}...")


def test_api_key_permissions():
    mgr = APIKeyManager()
    key = mgr.generate_key("limited", permissions={"query"})
    assert mgr.check_permission(key, "query")
    assert not mgr.check_permission(key, "admin")
    print("OK: permissions")


def test_api_key_revoke():
    mgr = APIKeyManager()
    key = mgr.generate_key("temp")
    valid, _, _ = mgr.validate(key)
    assert valid

    mgr.revoke(key)
    valid, _, reason = mgr.validate(key)
    assert not valid
    assert "revoked" in reason
    print("OK: revoke")


def test_invalid_key():
    mgr = APIKeyManager()
    valid, _, reason = mgr.validate("abmm_fake_key_12345")
    assert not valid
    assert "Invalid" in reason
    print("OK: invalid key rejected")


if __name__ == "__main__":
    test_rate_limiter()
    test_rate_limiter_per_minute()
    test_api_key_lifecycle()
    test_api_key_permissions()
    test_api_key_revoke()
    test_invalid_key()
    print("\nAll security tests passed!")
