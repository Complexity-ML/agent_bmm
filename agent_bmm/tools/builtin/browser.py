# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Browser Tool — Control a real headless browser via Playwright.

No API keys, no rate limits. Uses the host's browser to:
- Search Google/Bing/any search engine
- Navigate to any URL and read content
- Fill forms, click buttons, interact with pages
- Take screenshots
- Extract structured data from any website
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from agent_bmm.tools.registry import Tool


class BrowserSession:
    """
    Persistent browser session using Playwright.

    Reuses a single browser instance across tool calls
    for performance (no cold start per query).
    """

    def __init__(self, headless: bool = True, timeout: float = 15000):
        self.headless = headless
        self.timeout = timeout
        self._playwright = None
        self._browser = None
        self._context = None

    async def _ensure_browser(self):
        if self._browser is None:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless
            )
            self._context = await self._browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )

    async def new_page(self):
        await self._ensure_browser()
        page = await self._context.new_page()
        page.set_default_timeout(self.timeout)
        return page

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._browser = None
        self._playwright = None


# Global session (reused across calls)
_session = BrowserSession()


async def _search_google(query: str, max_results: int = 5) -> str:
    """Search Google and extract results."""
    page = await _session.new_page()
    try:
        await page.goto(
            f"https://www.google.com/search?q={query}&hl=en",
            wait_until="domcontentloaded",
        )

        # Extract search results
        results = []
        items = await page.query_selector_all("div.g")
        for item in items[:max_results]:
            title_el = await item.query_selector("h3")
            snippet_el = await item.query_selector("div[data-sncf]")
            link_el = await item.query_selector("a")

            title = await title_el.inner_text() if title_el else ""
            snippet = await snippet_el.inner_text() if snippet_el else ""
            link = await link_el.get_attribute("href") if link_el else ""

            if title:
                results.append(f"{len(results)+1}. {title}\n   {snippet}\n   {link}")

        return "\n\n".join(results) if results else f"No results for: {query}"
    finally:
        await page.close()


async def _read_page(url: str, max_chars: int = 5000) -> str:
    """Navigate to URL and extract text content."""
    page = await _session.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded")
        # Get main content text
        content = await page.evaluate("""() => {
            // Try main content selectors
            const selectors = ['main', 'article', '#content', '.content', 'body'];
            for (const sel of selectors) {
                const el = document.querySelector(sel);
                if (el && el.innerText.length > 100) {
                    return el.innerText;
                }
            }
            return document.body.innerText;
        }""")
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... (truncated)"
        return content
    finally:
        await page.close()


async def _screenshot(url: str) -> str:
    """Take a screenshot of a page."""
    page = await _session.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded")
        path = f"/tmp/screenshot_{hash(url) % 10000}.png"
        await page.screenshot(path=path, full_page=False)
        return f"Screenshot saved to {path}"
    finally:
        await page.close()


async def _fill_form(url: str, fields: str) -> str:
    """Navigate to URL and fill form fields. Format: 'selector=value;selector2=value2'"""
    page = await _session.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded")
        pairs = [f.strip().split("=", 1) for f in fields.split(";") if "=" in f]
        for selector, value in pairs:
            await page.fill(selector.strip(), value.strip())
        return f"Filled {len(pairs)} fields on {url}"
    finally:
        await page.close()


async def _extract_links(url: str, max_links: int = 20) -> str:
    """Extract all links from a page."""
    page = await _session.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded")
        links = await page.evaluate(f"""() => {{
            const links = [];
            document.querySelectorAll('a[href]').forEach(a => {{
                const text = a.innerText.trim();
                const href = a.href;
                if (text && href && !href.startsWith('javascript:')) {{
                    links.push(text.substring(0, 80) + ' -> ' + href);
                }}
            }});
            return links.slice(0, {max_links});
        }}""")
        return "\n".join(f"  {l}" for l in links) if links else "No links found"
    finally:
        await page.close()


async def _extract_tables(url: str) -> str:
    """Extract tables from a page as text."""
    page = await _session.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded")
        tables = await page.evaluate("""() => {
            const results = [];
            document.querySelectorAll('table').forEach((table, idx) => {
                const rows = [];
                table.querySelectorAll('tr').forEach(tr => {
                    const cells = [];
                    tr.querySelectorAll('td, th').forEach(td => {
                        cells.push(td.innerText.trim());
                    });
                    if (cells.length > 0) rows.push(cells.join(' | '));
                });
                if (rows.length > 0) {
                    results.push('Table ' + (idx+1) + ':\\n' + rows.join('\\n'));
                }
            });
            return results;
        }""")
        return "\n\n".join(tables) if tables else "No tables found"
    finally:
        await page.close()


def create_browser_tool(
    headless: bool = True,
    timeout: float = 15000,
    max_results: int = 5,
) -> Tool:
    """
    Create a browser tool powered by Playwright.

    Commands:
        "search <query>"           — Google search
        "read <url>"               — Read page content
        "screenshot <url>"         — Take screenshot
        "links <url>"              — Extract all links
        "tables <url>"             — Extract tables
        "fill <url> sel=val;..."   — Fill form fields
    """
    global _session
    _session = BrowserSession(headless=headless, timeout=timeout)

    async def _browser_tool(query: str) -> str:
        parts = query.strip().split(None, 1)
        if len(parts) < 2:
            return "Error: use search/read/screenshot/links/tables/fill <arg>"

        cmd = parts[0].lower()
        arg = parts[1]

        try:
            if cmd == "search":
                return await _search_google(arg, max_results)
            elif cmd == "read":
                return await _read_page(arg)
            elif cmd == "screenshot":
                return await _screenshot(arg)
            elif cmd == "links":
                return await _extract_links(arg)
            elif cmd == "tables":
                return await _extract_tables(arg)
            elif cmd == "fill":
                url_and_fields = arg.split(None, 1)
                if len(url_and_fields) < 2:
                    return "Error: fill <url> <selector=value;...>"
                return await _fill_form(url_and_fields[0], url_and_fields[1])
            else:
                return "Error: unknown command. Use search/read/screenshot/links/tables/fill"
        except Exception as e:
            return f"Browser Error: {e}"

    return Tool(
        name="browser",
        description="Browse the web using a real browser — search Google, read pages, fill forms, extract data",
        async_fn=_browser_tool,
    )


BrowserTool = create_browser_tool
