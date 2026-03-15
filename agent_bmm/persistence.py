# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Persistence — SQLite-backed conversation storage.

Saves and restores conversation history, tool results,
and agent state across sessions.
"""

from __future__ import annotations

import json
import sqlite3
import time


class ConversationStore:
    """
    SQLite-backed conversation persistence.

    Usage:
        store = ConversationStore("conversations.db")
        conv_id = store.create_conversation("My chat")
        store.add_message(conv_id, "user", "Hello")
        store.add_message(conv_id, "assistant", "Hi there!")

        # Later...
        messages = store.get_messages(conv_id)
    """

    def __init__(self, db_path: str = "agent_bmm.db"):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                tool_name TEXT,
                tool_result TEXT,
                expert_id INTEGER,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS agent_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                step INTEGER NOT NULL,
                routing_decisions TEXT,
                expert_distribution TEXT,
                timestamp REAL NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_state_conv
                ON agent_state(conversation_id);
        """)
        conn.commit()

    def create_conversation(self, title: str = "Untitled", metadata: dict | None = None) -> int:
        conn = self._get_conn()
        now = time.time()
        cursor = conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at, metadata) VALUES (?, ?, ?, ?)",
            (title, now, now, json.dumps(metadata or {})),
        )
        conn.commit()
        return cursor.lastrowid

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        tool_name: str | None = None,
        tool_result: str | None = None,
        expert_id: int | None = None,
        metadata: dict | None = None,
    ) -> int:
        conn = self._get_conn()
        now = time.time()
        cursor = conn.execute(
            """INSERT INTO messages
            (conversation_id, role, content, timestamp, tool_name, tool_result, expert_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                conversation_id,
                role,
                content,
                now,
                tool_name,
                tool_result,
                expert_id,
                json.dumps(metadata or {}),
            ),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        conn.commit()
        return cursor.lastrowid

    def save_agent_state(
        self,
        conversation_id: int,
        step: int,
        routing_decisions: list[int],
        expert_distribution: dict[str, int],
    ):
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO agent_state
            (conversation_id, step, routing_decisions, expert_distribution, timestamp)
            VALUES (?, ?, ?, ?, ?)""",
            (
                conversation_id,
                step,
                json.dumps(routing_decisions),
                json.dumps(expert_distribution),
                time.time(),
            ),
        )
        conn.commit()

    def get_messages(self, conversation_id: int, limit: int = 100) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_conversations(self, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: int) -> dict | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return dict(row) if row else None

    def delete_conversation(self, conversation_id: int):
        conn = self._get_conn()
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM agent_state WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()

    def search_messages(self, query: str, limit: int = 20) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM messages WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
