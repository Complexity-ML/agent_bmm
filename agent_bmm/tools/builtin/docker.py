# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Docker Tool — Manage containers via Docker Engine API.
"""

from __future__ import annotations

import aiohttp

from agent_bmm.tools.registry import Tool


def create_docker_tool(
    docker_host: str = "unix:///var/run/docker.sock",
    timeout: float = 30.0,
) -> Tool:
    """
    Create a Docker management tool.

    Commands:
        "ps"                        — List running containers
        "images"                    — List images
        "logs <container_id>"       — Get container logs
        "run <image> [cmd]"         — Run a container
        "stop <container_id>"       — Stop a container
        "exec <container_id> <cmd>" — Execute command in container
    """

    async def _docker(query: str) -> str:
        parts = query.strip().split(None, 2)
        if not parts:
            return "Error: use ps/images/logs/run/stop/exec"

        cmd = parts[0].lower()

        # Use HTTP connector for unix socket or TCP
        if docker_host.startswith("unix://"):
            connector = aiohttp.UnixConnector(path=docker_host[7:])
            base = "http://localhost"
        else:
            connector = None
            base = docker_host.rstrip("/")

        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                if cmd == "ps":
                    async with session.get(
                        f"{base}/containers/json",
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        containers = await resp.json()
                        if not containers:
                            return "No running containers"
                        return "\n".join(
                            f"  {c['Id'][:12]} {c['Image']} {c['State']} {c.get('Status', '')}"
                            for c in containers
                        )

                elif cmd == "images":
                    async with session.get(
                        f"{base}/images/json",
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        images = await resp.json()
                        return (
                            "\n".join(
                                f"  {img.get('RepoTags', ['<none>'])[0]} {img['Size'] // 1024 // 1024}MB"
                                for img in images[:20]
                            )
                            or "No images"
                        )

                elif cmd == "logs" and len(parts) >= 2:
                    cid = parts[1]
                    async with session.get(
                        f"{base}/containers/{cid}/logs?stdout=true&stderr=true&tail=50",
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        logs = await resp.text()
                        return logs[:3000] or "No logs"

                elif cmd == "run" and len(parts) >= 2:
                    image = parts[1]
                    run_cmd = parts[2] if len(parts) > 2 else ""
                    payload = {
                        "Image": image,
                        "Cmd": run_cmd.split() if run_cmd else None,
                    }
                    async with session.post(
                        f"{base}/containers/create",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        data = await resp.json()
                        cid = data.get("Id", "")[:12]
                    # Start it
                    await session.post(f"{base}/containers/{cid}/start")
                    return f"Started container {cid} from {image}"

                elif cmd == "stop" and len(parts) >= 2:
                    cid = parts[1]
                    async with session.post(
                        f"{base}/containers/{cid}/stop",
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        if resp.status in (204, 304):
                            return f"Stopped {cid}"
                        return f"Error stopping: {resp.status}"

                elif cmd == "exec" and len(parts) >= 3:
                    cid = parts[1]
                    exec_cmd = parts[2]
                    # Create exec
                    payload = {
                        "AttachStdout": True,
                        "AttachStderr": True,
                        "Cmd": exec_cmd.split(),
                    }
                    async with session.post(
                        f"{base}/containers/{cid}/exec",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        data = await resp.json()
                        exec_id = data["Id"]

                    # Start exec
                    async with session.post(
                        f"{base}/exec/{exec_id}/start",
                        json={"Detach": False},
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        output = await resp.text()
                        return output[:3000] or "(no output)"

                else:
                    return "Error: use ps/images/logs/run/stop/exec"

        except Exception as e:
            return f"Docker Error: {e}"

    return Tool(
        name="docker",
        description="Manage Docker containers: list, run, stop, exec, logs",
        async_fn=_docker,
    )


DockerTool = create_docker_tool
