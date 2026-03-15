# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Audio Transcription Tool — Transcribe audio with Whisper.

Uses OpenAI Whisper API or local whisper model.
"""

from __future__ import annotations

from pathlib import Path

from agent_bmm.tools.registry import Tool


class AudioTool:
    """Transcribe audio files using Whisper."""

    name = "audio"
    description = "Transcribe audio files — speech to text via Whisper"

    def __init__(self, model: str = "whisper-1", local: bool = False):
        self.model = model
        self.local = local

    @property
    def fn(self):
        return self._transcribe

    @property
    def async_fn(self):
        return self._async_transcribe

    def _transcribe(self, query: str) -> str:
        import asyncio
        return asyncio.run(self._async_transcribe(query))

    async def _async_transcribe(self, query: str) -> str:
        """Transcribe an audio file. Query = file path."""
        audio_path = query.strip()
        p = Path(audio_path)
        if not p.exists():
            return f"Error: audio file '{audio_path}' not found"

        if self.local:
            return self._transcribe_local(p)
        return await self._transcribe_api(p)

    async def _transcribe_api(self, path: Path) -> str:
        """Transcribe via OpenAI Whisper API."""
        import os

        import aiohttp

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return "Error: OPENAI_API_KEY required for audio transcription"

        data = aiohttp.FormData()
        data.add_field("file", path.read_bytes(), filename=path.name)
        data.add_field("model", self.model)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/audio/transcriptions",
                data=data,
                headers={"Authorization": f"Bearer {api_key}"},
            ) as resp:
                result = await resp.json()
                return result.get("text", str(result))

    def _transcribe_local(self, path: Path) -> str:
        """Transcribe locally using whisper package."""
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(str(path))
            return result["text"]
        except ImportError:
            return "Error: `pip install openai-whisper` required for local transcription"

    def to_tool(self) -> Tool:
        return Tool(name=self.name, description=self.description, fn=self.fn, async_fn=self.async_fn)
