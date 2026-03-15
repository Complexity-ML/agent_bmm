# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Image Analysis Tool — Analyze images using multimodal models.

Supports OpenAI Vision API (gpt-4o) or local CLIP embeddings.
Requires: pip install agent-bmm[gpu] for CLIP.
"""

from __future__ import annotations

import base64
from pathlib import Path

from agent_bmm.tools.registry import Tool


class ImageTool:
    """Analyze images via multimodal LLM or CLIP."""

    name = "image"
    description = "Analyze images — describe content, extract text, compare"

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    @property
    def fn(self):
        return self._analyze

    @property
    def async_fn(self):
        return self._async_analyze

    def _analyze(self, query: str) -> str:
        """Sync wrapper."""
        import asyncio

        return asyncio.run(self._async_analyze(query))

    async def _async_analyze(self, query: str) -> str:
        """Analyze an image. Query format: '<path> <question>' or just '<path>'."""
        parts = query.strip().split(None, 1)
        if not parts:
            return "Error: provide image path and optional question"

        image_path = parts[0]
        question = parts[1] if len(parts) > 1 else "Describe this image in detail."

        p = Path(image_path)
        if not p.exists():
            return f"Error: image '{image_path}' not found"

        # Encode image to base64
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode()
        ext = p.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "png")

        # Call multimodal API
        import os

        import aiohttp

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return "Error: OPENAI_API_KEY required for image analysis"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                    ],
                }
            ],
            "max_tokens": 500,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            ) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    def to_tool(self) -> Tool:
        return Tool(name=self.name, description=self.description, fn=self.fn, async_fn=self.async_fn)
