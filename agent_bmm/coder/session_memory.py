# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Session Memory — Remember user preferences and project context across sessions.

Stores coding style preferences, common commands, project structure insights,
and previous task outcomes in SQLite via persistence.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_bmm.persistence import ConversationStore


class SessionMemory:
    """Persistent memory across coding sessions."""

    def __init__(self, project_dir: str | Path, db_path: str | None = None):
        self.project_dir = Path(project_dir).resolve()
        db = db_path or str(self.project_dir / ".agent-bmm.db")
        self._store = ConversationStore(db)
        self._prefs = self._load_prefs()

    def _load_prefs(self) -> dict:
        """Load stored preferences."""
        msgs = self._store.search_messages("__PREF__", limit=50)
        prefs = {}
        for msg in msgs:
            try:
                data = json.loads(msg["content"].replace("__PREF__", ""))
                prefs.update(data)
            except (json.JSONDecodeError, KeyError):
                continue
        return prefs

    def remember(self, key: str, value: str):
        """Remember a preference or fact."""
        self._prefs[key] = value
        conv_id = self._get_or_create_conv()
        self._store.add_message(conv_id, "system", f"__PREF__{json.dumps({key: value})}")

    def recall(self, key: str) -> str | None:
        """Recall a stored preference."""
        return self._prefs.get(key)

    def get_context(self) -> str:
        """Get all remembered context as a string for the system prompt."""
        if not self._prefs:
            return ""
        lines = [f"- {k}: {v}" for k, v in self._prefs.items()]
        return "User preferences from previous sessions:\n" + "\n".join(lines)

    def save_task_outcome(self, task: str, outcome: str, steps: int):
        """Record a task outcome for future reference."""
        conv_id = self._get_or_create_conv()
        self._store.add_message(
            conv_id,
            "assistant",
            json.dumps({"task": task[:200], "outcome": outcome[:200], "steps": steps}),
        )

    def get_recent_tasks(self, limit: int = 5) -> list[dict]:
        """Get recent task outcomes."""
        conv_id = self._get_or_create_conv()
        msgs = self._store.get_messages(conv_id, limit=limit * 2)
        tasks = []
        for msg in msgs:
            if msg["role"] == "assistant":
                try:
                    tasks.append(json.loads(msg["content"]))
                except json.JSONDecodeError:
                    continue
        return tasks[-limit:]

    def _get_or_create_conv(self) -> int:
        """Get or create the memory conversation for this project."""
        convs = self._store.get_conversations(limit=100)
        project_name = self.project_dir.name
        for conv in convs:
            if conv["title"] == f"memory:{project_name}":
                return conv["id"]
        return self._store.create_conversation(f"memory:{project_name}")

    def close(self):
        self._store.close()
