# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
"""
Integration tests — full agent loop with a mock LLM.

Tests the CoderAgent end-to-end without hitting a real LLM API.
Uses predefined responses to verify the agent loop, action parsing,
file I/O, command execution, error recovery, and step limits.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from agent_bmm.coder.engine import CoderAgent


class _MockConfig:
    model = "mock"
    base_url = ""
    api_key = ""
    provider = "openai"


class MockLLM:
    """Mock LLM backend that returns predefined responses in sequence."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
        self.messages_log: list[list[dict]] = []
        self.config = _MockConfig()

    async def chat(self, messages, **kwargs) -> str:
        self.messages_log.append(messages)
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response

    async def chat_stream(self, messages, on_token=None, **kwargs) -> str:
        response = await self.chat(messages)
        if on_token:
            on_token(response)
        return response

    async def close(self):
        pass


def _make_agent(tmp_dir: str, max_steps: int = 5) -> CoderAgent:
    """Create a CoderAgent with YOLO permissions and no streaming."""
    agent = CoderAgent(
        model="mock",
        project_dir=tmp_dir,
        max_steps=max_steps,
        permission="yolo",
        stream=False,
    )
    return agent


def _resp(action: dict) -> str:
    """Encode an action dict as the JSON string the mock LLM returns."""
    return json.dumps(action)


# === Test cases ===


def test_coder_write_and_done():
    """Agent writes a file then completes."""
    with tempfile.TemporaryDirectory() as tmp:
        agent = _make_agent(tmp)
        agent.llm = MockLLM([
            _resp({"action": "write", "path": "hello.py", "content": "print('hello world')"}),
            _resp({"action": "done", "summary": "Created hello.py"}),
        ])

        result = asyncio.run(agent.arun("Create hello.py"))

        assert result == "Created hello.py"
        created = Path(tmp) / "hello.py"
        assert created.exists()
        assert created.read_text() == "print('hello world')"
    print("OK: write and done")


def test_coder_read_edit_done():
    """Agent reads a file, edits it, then completes."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create a file to edit
        target = Path(tmp) / "app.py"
        target.write_text("def greet():\n    return 'hello'\n")

        agent = _make_agent(tmp)
        agent.llm = MockLLM([
            _resp({"action": "read", "path": "app.py"}),
            _resp({
                "action": "edit",
                "path": "app.py",
                "old": "return 'hello'",
                "new": "return 'bonjour'",
            }),
            _resp({"action": "done", "summary": "Updated greeting to French"}),
        ])

        result = asyncio.run(agent.arun("Change greeting to French"))

        assert result == "Updated greeting to French"
        assert "bonjour" in target.read_text()
        assert "hello" not in target.read_text()
    print("OK: read, edit, done")


def test_coder_run_command():
    """Agent runs a shell command then completes."""
    with tempfile.TemporaryDirectory() as tmp:
        agent = _make_agent(tmp)
        agent.llm = MockLLM([
            _resp({"action": "run", "cmd": "echo hello-from-agent"}),
            _resp({"action": "done", "summary": "Ran echo command"}),
        ])

        result = asyncio.run(agent.arun("Run echo"))

        assert result == "Ran echo command"
        # Verify the LLM saw the command output in its context
        messages = agent.llm.messages_log[-1]
        found = any("hello-from-agent" in m.get("content", "") for m in messages)
        assert found, "Command output should be in conversation history"
    print("OK: run command")


def test_coder_bad_json_recovery():
    """Agent sends invalid JSON, then recovers with a valid action."""
    with tempfile.TemporaryDirectory() as tmp:
        agent = _make_agent(tmp)
        agent.llm = MockLLM([
            "This is not JSON at all, I'm confused",
            _resp({"action": "done", "summary": "Recovered from bad JSON"}),
        ])

        result = asyncio.run(agent.arun("Do something"))

        assert result == "Recovered from bad JSON"
        assert agent.llm.call_count == 2, "Should have retried after bad JSON"
    print("OK: bad JSON recovery")


def test_coder_max_steps():
    """Agent hits the max_steps limit without completing."""
    with tempfile.TemporaryDirectory() as tmp:
        agent = _make_agent(tmp, max_steps=3)
        # Always return a read action — never "done"
        agent.llm = MockLLM([
            _resp({"action": "list", "path": "."}),
        ])

        result = asyncio.run(agent.arun("List files forever"))

        assert result == "Max steps reached"
        assert agent.llm.call_count == 3
    print("OK: max steps limit")


def test_coder_search_code():
    """Agent searches for code patterns."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create some files to search
        (Path(tmp) / "main.py").write_text("def main():\n    print('start')\n")
        (Path(tmp) / "utils.py").write_text("def helper():\n    return 42\n")

        agent = _make_agent(tmp)
        agent.llm = MockLLM([
            _resp({"action": "search", "query": "def main"}),
            _resp({"action": "done", "summary": "Found main function"}),
        ])

        result = asyncio.run(agent.arun("Find main function"))

        assert result == "Found main function"
    print("OK: search code")


def test_coder_write_then_run():
    """Agent writes a Python file then runs it — full create-and-test flow."""
    with tempfile.TemporaryDirectory() as tmp:
        agent = _make_agent(tmp)
        agent.llm = MockLLM([
            _resp({"action": "write", "path": "test_math.py", "content": "print(2 + 2)"}),
            _resp({"action": "run", "cmd": "python test_math.py"}),
            _resp({"action": "done", "summary": "Created and tested math script"}),
        ])

        result = asyncio.run(agent.arun("Create a math test"))

        assert result == "Created and tested math script"
        assert (Path(tmp) / "test_math.py").exists()
        # Verify the run result (4) appeared in history
        messages = agent.llm.messages_log[-1]
        found = any("4" in m.get("content", "") for m in messages)
        assert found, "Python output should be in conversation history"
    print("OK: write then run")


def test_coder_multiple_edits():
    """Agent makes multiple sequential edits to the same file."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "counter.py"
        target.write_text("count = 0\nprint(count)\n")

        agent = _make_agent(tmp)
        agent.llm = MockLLM([
            _resp({"action": "read", "path": "counter.py"}),
            _resp({"action": "edit", "path": "counter.py", "old": "count = 0", "new": "count = 10"}),
            _resp({"action": "edit", "path": "counter.py", "old": "print(count)", "new": "print(f'Count: {count}')"}),
            _resp({"action": "done", "summary": "Updated counter"}),
        ])

        result = asyncio.run(agent.arun("Update counter"))

        assert result == "Updated counter"
        content = target.read_text()
        assert "count = 10" in content
        assert "Count:" in content
    print("OK: multiple edits")


# === Run all tests ===

if __name__ == "__main__":
    test_coder_write_and_done()
    test_coder_read_edit_done()
    test_coder_run_command()
    test_coder_bad_json_recovery()
    test_coder_max_steps()
    test_coder_search_code()
    test_coder_write_then_run()
    test_coder_multiple_edits()
    print("\nAll integration tests passed!")
