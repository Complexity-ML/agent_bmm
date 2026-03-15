# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
GitHub Tool — Interact with GitHub repos, issues, PRs.
"""

from __future__ import annotations

import os

import aiohttp
import orjson

from agent_bmm.tools.registry import Tool


def create_github_tool(
    token: str | None = None,
    timeout: float = 15.0,
) -> Tool:
    """
    Create a GitHub API tool.

    Commands:
        "repos <owner>"                — List repos
        "issues <owner/repo>"          — List open issues
        "pr <owner/repo>"              — List open PRs
        "pr <owner/repo> <number>"     — Get PR details
        "search <query>"               — Search repos
        "readme <owner/repo>"          — Get README
    """
    api_token = token or os.environ.get("GITHUB_TOKEN", "")

    async def _github(query: str) -> str:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AgentBMM/0.1",
        }
        if api_token:
            headers["Authorization"] = f"token {api_token}"

        parts = query.strip().split(None, 2)
        if not parts:
            return "Error: use repos/issues/pr/search/readme"

        cmd = parts[0].lower()
        base = "https://api.github.com"

        try:
            async with aiohttp.ClientSession() as session:
                if cmd == "repos" and len(parts) >= 2:
                    url = f"{base}/users/{parts[1]}/repos?sort=updated&per_page=10"
                elif cmd == "issues" and len(parts) >= 2:
                    url = f"{base}/repos/{parts[1]}/issues?state=open&per_page=10"
                elif cmd == "pr" and len(parts) >= 2:
                    if len(parts) == 3 and parts[2].isdigit():
                        url = f"{base}/repos/{parts[1]}/pulls/{parts[2]}"
                    else:
                        url = f"{base}/repos/{parts[1]}/pulls?state=open&per_page=10"
                elif cmd == "search" and len(parts) >= 2:
                    q = " ".join(parts[1:])
                    url = f"{base}/search/repositories?q={q}&per_page=5"
                elif cmd == "readme" and len(parts) >= 2:
                    url = f"{base}/repos/{parts[1]}/readme"
                    headers["Accept"] = "application/vnd.github.v3.raw"
                else:
                    return "Error: unknown command. Use repos/issues/pr/search/readme"

                async with session.get(
                    url, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 404:
                        return "Not found"
                    if resp.status != 200:
                        return f"GitHub API error: {resp.status}"

                    if "raw" in headers.get("Accept", ""):
                        text = await resp.text()
                        return text[:3000]

                    data = await resp.json()

                    if cmd == "search":
                        items = data.get("items", [])
                        return "\n".join(
                            f"  {r['full_name']} ({r['stargazers_count']} stars): {r.get('description', '')}"
                            for r in items
                        ) or "No results"

                    if isinstance(data, list):
                        lines = []
                        for item in data[:10]:
                            num = item.get("number", "")
                            title = item.get("title", item.get("name", ""))
                            state = item.get("state", "")
                            lines.append(f"  #{num} [{state}] {title}")
                        return "\n".join(lines) or "No results"

                    if isinstance(data, dict):
                        title = data.get("title", data.get("name", ""))
                        body = data.get("body", "")[:500]
                        state = data.get("state", "")
                        return f"{title} [{state}]\n{body}"

                    return str(data)[:2000]
        except Exception as e:
            return f"GitHub Error: {e}"

    return Tool(
        name="github",
        description="Search GitHub repos, list issues and PRs, read READMEs",
        async_fn=_github,
    )


GitHubTool = create_github_tool
