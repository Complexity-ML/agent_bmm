# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
API Tool — Call any REST API endpoint.
Supports GET/POST with headers, auth, and JSON parsing.
"""

from __future__ import annotations

from typing import Any

import aiohttp
import orjson

from agent_bmm.tools.registry import Tool


def create_api_tool(
    name: str = "api",
    description: str = "Call a REST API endpoint",
    base_url: str = "",
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> Tool:
    """
    Create an API calling tool.

    The query should be formatted as: "METHOD URL [JSON_BODY]"
    Examples:
        "GET https://api.example.com/data"
        "POST https://api.example.com/search {\"query\": \"test\"}"
    """
    default_headers = {"Content-Type": "application/json"}
    if headers:
        default_headers.update(headers)

    async def _call_api(query: str) -> str:
        parts = query.strip().split(None, 2)
        if len(parts) < 2:
            return "Error: format should be 'METHOD URL [JSON_BODY]'"

        method = parts[0].upper()
        url = parts[1]
        if base_url and not url.startswith("http"):
            url = f"{base_url.rstrip('/')}/{url.lstrip('/')}"

        body = None
        if len(parts) > 2:
            try:
                body = orjson.loads(parts[2])
            except Exception:
                body = {"query": parts[2]}

        try:
            async with aiohttp.ClientSession() as session:
                kwargs: dict[str, Any] = {
                    "headers": default_headers,
                    "timeout": aiohttp.ClientTimeout(total=timeout),
                }
                if body and method in ("POST", "PUT", "PATCH"):
                    kwargs["json"] = body

                async with session.request(method, url, **kwargs) as resp:
                    status = resp.status
                    try:
                        data = await resp.json()
                        text = orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()
                    except Exception:
                        text = await resp.text()

                    if len(text) > 2000:
                        text = text[:2000] + "\n... (truncated)"
                    return f"Status: {status}\n{text}"
        except Exception as e:
            return f"API Error: {e}"

    return Tool(
        name=name,
        description=description,
        async_fn=_call_api,
    )


APITool = create_api_tool
