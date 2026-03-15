# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Sandbox — Isolated execution environment for agent tools.

Three isolation levels:
    - NONE:      No isolation (tools run in main process)
    - PROCESS:   Subprocess with resource limits (no Docker needed)
    - CONTAINER: Docker container with full isolation

The sandbox wraps tool execution — tools think they run normally,
but the sandbox intercepts and isolates dangerous operations.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum


class IsolationLevel(Enum):
    NONE = "none"
    PROCESS = "process"
    CONTAINER = "container"


@dataclass
class SandboxConfig:
    """Sandbox configuration."""

    level: IsolationLevel = IsolationLevel.PROCESS
    timeout_seconds: float = 30.0
    max_memory_mb: int = 512
    max_output_bytes: int = 100_000
    allowed_network: bool = True
    work_dir: str = ""
    docker_image: str = "python:3.12-slim"
    docker_network: str = "none"  # "none" = no network, "bridge" = allow


@dataclass
class SandboxResult:
    """Result from sandboxed execution."""

    output: str
    exit_code: int
    duration_ms: float
    killed: bool = False
    error: str = ""


class ProcessSandbox:
    """
    Subprocess sandbox with resource limits.

    Runs code in a separate Python process with:
    - Memory limit
    - CPU time limit
    - Timeout
    - No access to parent process memory
    """

    def __init__(self, config: SandboxConfig):
        self.config = config

    async def execute(self, code: str, language: str = "python") -> SandboxResult:
        """Execute code in a sandboxed subprocess."""
        t0 = time.time()

        # Write code to temp file
        suffix = ".py" if language == "python" else ".sh"
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            if language == "python":
                mem_bytes = self.config.max_memory_mb * 1024 * 1024
                timeout_s = int(self.config.timeout_seconds)
                lines = [
                    "# -*- coding: utf-8 -*-",
                    "import sys",
                    "try:",
                    "    import resource, signal",
                    f"    resource.setrlimit(resource.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))",
                    f"    signal.alarm({timeout_s})",
                    "except (ImportError, OSError):",
                    "    pass",
                    code,
                ]
                f.write("\n".join(lines))
            else:
                f.write(code)
            temp_path = f.name

        try:
            cmd = [sys.executable, temp_path] if language == "python" else ["bash", temp_path]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.work_dir or None,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout_seconds)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return SandboxResult(
                    output="",
                    exit_code=-1,
                    duration_ms=(time.time() - t0) * 1000,
                    killed=True,
                    error=f"Timeout after {self.config.timeout_seconds}s",
                )

            output = stdout.decode(errors="replace")
            errors = stderr.decode(errors="replace")

            if len(output) > self.config.max_output_bytes:
                output = output[: self.config.max_output_bytes] + "\n... (truncated)"

            full_output = output
            if errors:
                full_output += f"\nStderr:\n{errors}"

            return SandboxResult(
                output=full_output,
                exit_code=proc.returncode,
                duration_ms=(time.time() - t0) * 1000,
            )
        finally:
            os.unlink(temp_path)


class ContainerSandbox:
    """
    Docker container sandbox with full isolation.

    Each execution runs in a fresh container that is destroyed after.
    No state persists between executions.
    """

    def __init__(self, config: SandboxConfig):
        self.config = config

    async def execute(self, code: str, language: str = "python") -> SandboxResult:
        """Execute code in an isolated Docker container."""
        t0 = time.time()

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()) as f:
            f.write(code)
            temp_path = f.name

        try:
            cmd = [
                "docker",
                "run",
                "--rm",
                f"--memory={self.config.max_memory_mb}m",
                "--cpus=1",
                f"--network={self.config.docker_network}",
                "--read-only",
                "--tmpfs",
                "/tmp:size=50m",
                "-v",
                f"{temp_path}:/code/script.py:ro",
                "-w",
                "/code",
                self.config.docker_image,
                "python",
                "/code/script.py",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout_seconds + 10)
            except asyncio.TimeoutError:
                # Kill the container
                ps = await asyncio.create_subprocess_exec(
                    "docker",
                    "ps",
                    "-q",
                    "--filter",
                    f"ancestor={self.config.docker_image}",
                    stdout=asyncio.subprocess.PIPE,
                )
                ps_out, _ = await ps.communicate()
                for cid in ps_out.decode().strip().split("\n"):
                    if cid:
                        await asyncio.create_subprocess_exec("docker", "kill", cid)

                return SandboxResult(
                    output="",
                    exit_code=-1,
                    duration_ms=(time.time() - t0) * 1000,
                    killed=True,
                    error=f"Container timeout after {self.config.timeout_seconds}s",
                )

            output = stdout.decode(errors="replace")
            errors = stderr.decode(errors="replace")

            if len(output) > self.config.max_output_bytes:
                output = output[: self.config.max_output_bytes] + "\n... (truncated)"

            full_output = output
            if errors and proc.returncode != 0:
                full_output += f"\nStderr:\n{errors}"

            return SandboxResult(
                output=full_output,
                exit_code=proc.returncode,
                duration_ms=(time.time() - t0) * 1000,
            )
        finally:
            os.unlink(temp_path)


class Sandbox:
    """
    Unified sandbox interface.

    Automatically selects the right isolation level:
    - PROCESS: fast, no Docker needed
    - CONTAINER: full isolation, needs Docker

    Usage:
        sandbox = Sandbox(SandboxConfig(level=IsolationLevel.PROCESS))
        result = await sandbox.run("print(2 + 2)")
        print(result.output)  # "4"
    """

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()

        if self.config.level == IsolationLevel.CONTAINER:
            self._backend = ContainerSandbox(self.config)
        elif self.config.level == IsolationLevel.PROCESS:
            self._backend = ProcessSandbox(self.config)
        else:
            self._backend = None

    async def run(self, code: str, language: str = "python") -> SandboxResult:
        """Run code in the sandbox."""
        if self._backend is None:
            # No sandbox — direct execution (dangerous!)
            t0 = time.time()
            try:
                import io
                from contextlib import redirect_stderr, redirect_stdout

                stdout = io.StringIO()
                stderr = io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exec(code, {"__builtins__": __builtins__})
                return SandboxResult(
                    output=stdout.getvalue(),
                    exit_code=0,
                    duration_ms=(time.time() - t0) * 1000,
                )
            except Exception as e:
                return SandboxResult(
                    output="",
                    exit_code=1,
                    duration_ms=(time.time() - t0) * 1000,
                    error=str(e),
                )

        return await self._backend.execute(code, language)

    async def run_tool(self, tool_fn, *args, **kwargs) -> SandboxResult:
        """
        Run a tool function in the sandbox.

        Serializes the function call, executes in sandbox,
        deserializes the result.
        """
        # For PROCESS/CONTAINER, we serialize the call
        code = f"""
import json
import sys
sys.path.insert(0, '.')

# Execute the tool
result = ({tool_fn.__name__})({json.dumps(args)}, **{json.dumps(kwargs)})
print(json.dumps({{"result": str(result)}}))
"""
        return await self.run(code)
