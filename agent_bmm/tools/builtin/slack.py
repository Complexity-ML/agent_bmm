# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Slack Tool — Send messages and read channels via Slack API.
"""

from __future__ import annotations

import os

import aiohttp
import orjson

from agent_bmm.tools.registry import Tool


def create_slack_tool(
    token: str | None = None,
    default_channel: str = "#general",
    timeout: float = 10.0,
) -> Tool:
    """
    Create a Slack messaging tool.

    Commands:
        "send <channel> <message>"  — Send a message
        "read <channel> [count]"    — Read recent messages
        "channels"                  — List channels
    """
    api_token = token or os.environ.get("SLACK_BOT_TOKEN", "")

    async def _slack(query: str) -> str:
        if not api_token:
            return "Error: SLACK_BOT_TOKEN not set"

        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        parts = query.strip().split(None, 2)
        if not parts:
            return "Error: use send/read/channels"

        cmd = parts[0].lower()
        base = "https://slack.com/api"

        try:
            async with aiohttp.ClientSession() as session:
                if cmd == "send" and len(parts) >= 3:
                    channel = parts[1]
                    message = parts[2]
                    payload = {"channel": channel, "text": message}
                    async with session.post(
                        f"{base}/chat.postMessage",
                        json=payload, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        data = await resp.json()
                        if data.get("ok"):
                            return f"Sent to {channel}"
                        return f"Slack Error: {data.get('error', 'unknown')}"

                elif cmd == "read" and len(parts) >= 2:
                    channel = parts[1]
                    count = int(parts[2]) if len(parts) > 2 else 10
                    payload = {"channel": channel, "limit": min(count, 50)}
                    async with session.post(
                        f"{base}/conversations.history",
                        json=payload, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        data = await resp.json()
                        if not data.get("ok"):
                            return f"Slack Error: {data.get('error', 'unknown')}"
                        messages = data.get("messages", [])
                        return "\n".join(
                            f"  [{m.get('user', '?')}]: {m.get('text', '')[:200]}"
                            for m in messages[:count]
                        ) or "No messages"

                elif cmd == "channels":
                    async with session.post(
                        f"{base}/conversations.list",
                        json={"types": "public_channel", "limit": 20},
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        data = await resp.json()
                        if not data.get("ok"):
                            return f"Slack Error: {data.get('error', 'unknown')}"
                        channels = data.get("channels", [])
                        return "\n".join(
                            f"  #{c['name']} ({c.get('num_members', 0)} members)"
                            for c in channels
                        ) or "No channels"

                else:
                    return "Error: use send/read/channels"

        except Exception as e:
            return f"Slack Error: {e}"

    return Tool(
        name="slack",
        description="Send and read Slack messages, list channels",
        async_fn=_slack,
    )


SlackTool = create_slack_tool
