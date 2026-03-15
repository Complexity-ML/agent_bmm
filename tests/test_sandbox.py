# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
import asyncio
import sys

sys.path.insert(0, ".")

from agent_bmm.sandbox import Sandbox, SandboxConfig, SandboxResult, IsolationLevel


def test_no_isolation():
    sb = Sandbox(SandboxConfig(level=IsolationLevel.NONE))
    result = asyncio.run(sb.run("print(2 + 2)"))
    assert "4" in result.output
    assert result.exit_code == 0
    print("OK: no isolation")


def test_no_isolation_error():
    sb = Sandbox(SandboxConfig(level=IsolationLevel.NONE))
    result = asyncio.run(sb.run("raise ValueError('boom')"))
    assert result.exit_code == 1
    assert "boom" in result.error
    print("OK: no isolation error")


def test_process_sandbox():
    sb = Sandbox(SandboxConfig(level=IsolationLevel.PROCESS, timeout_seconds=5))
    result = asyncio.run(sb.run("print('hello from sandbox')"))
    assert "hello from sandbox" in result.output
    assert result.exit_code == 0
    assert result.duration_ms > 0
    print(f"OK: process sandbox ({result.duration_ms:.0f}ms)")


def test_process_sandbox_error():
    sb = Sandbox(SandboxConfig(level=IsolationLevel.PROCESS, timeout_seconds=5))
    result = asyncio.run(sb.run("import sys; sys.exit(42)"))
    assert result.exit_code == 42
    print("OK: process sandbox error code")


def test_process_sandbox_timeout():
    sb = Sandbox(SandboxConfig(level=IsolationLevel.PROCESS, timeout_seconds=2))
    result = asyncio.run(sb.run("import time; time.sleep(10)"))
    assert result.killed
    assert "Timeout" in result.error
    print("OK: process sandbox timeout")


def test_process_sandbox_output_limit():
    sb = Sandbox(SandboxConfig(level=IsolationLevel.PROCESS, timeout_seconds=5, max_output_bytes=100))
    result = asyncio.run(sb.run("print('x' * 10000)"))
    assert len(result.output) <= 200  # some overhead for truncation message
    assert "truncated" in result.output
    print("OK: process sandbox output limit")


def test_process_sandbox_math():
    sb = Sandbox(SandboxConfig(level=IsolationLevel.PROCESS))
    result = asyncio.run(sb.run("import math; print(math.sqrt(144))"))
    assert "12" in result.output
    print("OK: process sandbox math")


if __name__ == "__main__":
    test_no_isolation()
    test_no_isolation_error()
    test_process_sandbox()
    test_process_sandbox_error()
    test_process_sandbox_timeout()
    test_process_sandbox_output_limit()
    test_process_sandbox_math()
    print("\nAll sandbox tests passed!")
