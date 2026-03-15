# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
import asyncio
import sys
sys.path.insert(0, ".")

from agent_bmm.tools.registry import Tool, ToolRegistry
from agent_bmm.tools.builtin.math_tool import MathTool
from agent_bmm.tools.builtin.code_exec import CodeExecTool
from agent_bmm.tools.builtin.file_io import FileIOTool


def test_registry():
    reg = ToolRegistry()
    idx0 = reg.register(Tool("a", "tool a", fn=lambda q: "a"))
    idx1 = reg.register(Tool("b", "tool b", fn=lambda q: "b"))
    assert idx0 == 0
    assert idx1 == 1
    assert reg.num_tools == 2
    assert reg.get(0).name == "a"
    assert reg.get_by_name("b").index == 1
    assert reg.descriptions == ["tool a", "tool b"]
    print("OK: registry")


def test_tool_sync():
    tool = Tool("test", "test tool", fn=lambda q: f"result: {q}")
    assert tool("hello") == "result: hello"
    print("OK: sync tool execution")


def test_tool_async():
    async def afn(q):
        return f"async: {q}"
    tool = Tool("test", "test", async_fn=afn)
    result = asyncio.run(tool.acall("hello"))
    assert result == "async: hello"
    print("OK: async tool execution")


def test_batch_execute():
    reg = ToolRegistry()
    reg.register(Tool("a", "a", fn=lambda q: f"a:{q}"))
    reg.register(Tool("b", "b", fn=lambda q: f"b:{q}"))
    results = asyncio.run(reg.batch_execute([0, 1], ["q1", "q2"]))
    assert results == ["a:q1", "b:q2"]
    print("OK: batch execute")


def test_math_tool():
    tool = MathTool()
    assert tool("2 + 2") == "4"
    assert tool("sqrt(16)") == "4"
    assert tool("pi") == "3.14159"
    assert tool("sin(0)") == "0"
    assert "Error" in tool("1/0")
    assert "blocked" in tool("import os")
    print("OK: math tool")


def test_code_exec():
    tool = CodeExecTool()
    assert "42" in tool("print(6 * 7)")
    assert "Hello" in tool("print('Hello')")
    assert "Error" in tool("import os")
    assert "Blocked" in tool("open('file.txt')")
    assert "SyntaxError" in tool("def ()")
    print("OK: code exec tool")


def test_file_io():
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    # Create test file
    with open(os.path.join(tmpdir, "test.txt"), "w") as f:
        f.write("hello world\nfoo bar")

    tool = FileIOTool(root_dir=tmpdir)
    assert "hello world" in tool(f"read test.txt")
    assert "test.txt" in tool(f"list .")
    assert "hello" in tool(f"search . hello")
    assert "Error" in tool(f"read ../etc/passwd")
    print("OK: file_io tool")


if __name__ == "__main__":
    test_registry()
    test_tool_sync()
    test_tool_async()
    test_batch_execute()
    test_math_tool()
    test_code_exec()
    test_file_io()
    print("\nAll tool tests passed!")
