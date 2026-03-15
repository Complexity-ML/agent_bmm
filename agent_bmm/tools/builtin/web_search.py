# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Web Search Tool — Search the web via DuckDuckGo (no API key needed)
or any custom search endpoint.
"""

from __future__ import annotations

import re
from html import unescape
from urllib.parse import quote_plus

import aiohttp

from agent_bmm.tools.registry import Tool


def create_web_search(
    max_results: int = 5,
    timeout: float = 10.0,
    endpoint: str | None = None,
    api_key: str | None = None,
) -> Tool:
    """
    Create a web search tool.

    Uses DuckDuckGo HTML search by default (no API key).
    Set endpoint + api_key for custom search (SerpAPI, Brave, etc).
    """

    async def _search_ddg(query: str) -> str:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AgentBMM/0.1)"}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                html = await resp.text()

        # Parse results from HTML
        results = []
        snippets = re.findall(r'class="result__snippet">(.*?)</a>', html, re.DOTALL)
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
        links = re.findall(r'class="result__url"[^>]*href="(.*?)"', html)

        for i in range(min(max_results, len(snippets))):
            title = re.sub(r"<.*?>", "", titles[i] if i < len(titles) else "")
            snippet = re.sub(r"<.*?>", "", snippets[i])
            link = links[i] if i < len(links) else ""
            results.append(
                f"{i + 1}. {unescape(title.strip())}\n"
                f"   {unescape(snippet.strip())}\n"
                f"   {link}"
            )

        if not results:
            return f"No results found for: {query}"
        return "\n\n".join(results)

    async def _search_custom(query: str) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {"query": query, "max_results": max_results}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                data = await resp.json()
                results = data.get("results", [])
                return (
                    "\n\n".join(
                        f"{i + 1}. {r.get('title', '')}\n   {r.get('snippet', '')}"
                        for i, r in enumerate(results[:max_results])
                    )
                    or f"No results for: {query}"
                )

    search_fn = _search_custom if endpoint else _search_ddg

    return Tool(
        name="web_search",
        description="Search the web for current information",
        async_fn=search_fn,
    )


WebSearchTool = create_web_search
