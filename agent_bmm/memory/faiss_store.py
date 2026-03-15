# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
FAISS Long-Term Memory — Persistent vector memory across conversations.

Stores conversation snippets as embeddings in a FAISS index.
Retrieves relevant memories for new queries.

Requires: pip install agent-bmm[gpu]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FAISSMemory:
    """Persistent vector memory using FAISS."""

    def __init__(self, db_path: str = ".agent-bmm-memory", dim: int = 384):
        self.db_path = Path(db_path)
        self.dim = dim
        self._index: Any = None
        self._encoder: Any = None
        self._entries: list[dict] = []
        self._load()

    def _load(self):
        """Load existing index and entries from disk."""
        entries_path = self.db_path / "entries.json"
        if entries_path.exists():
            self._entries = json.loads(entries_path.read_text())

        index_path = self.db_path / "index.faiss"
        if index_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(index_path))
            except ImportError:
                pass

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._encoder

    def _get_index(self):
        if self._index is None:
            import faiss
            self._index = faiss.IndexFlatIP(self.dim)  # Inner product (cosine after normalization)
        return self._index

    def add(self, text: str, metadata: dict | None = None):
        """Add a memory entry."""
        import numpy as np

        encoder = self._get_encoder()
        index = self._get_index()

        embedding = encoder.encode([text], normalize_embeddings=True)
        index.add(np.array(embedding, dtype=np.float32))
        self._entries.append({"text": text, "metadata": metadata or {}})

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for relevant memories."""
        import numpy as np

        if not self._entries:
            return []

        encoder = self._get_encoder()
        index = self._get_index()

        query_emb = encoder.encode([query], normalize_embeddings=True)
        scores, indices = index.search(np.array(query_emb, dtype=np.float32), min(top_k, len(self._entries)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._entries):
                entry = self._entries[idx].copy()
                entry["score"] = float(score)
                results.append(entry)
        return results

    def save(self):
        """Persist index and entries to disk."""
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Save entries
        (self.db_path / "entries.json").write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=2)
        )

        # Save FAISS index
        if self._index is not None:
            import faiss
            faiss.write_index(self._index, str(self.db_path / "index.faiss"))

    def clear(self):
        """Clear all memories."""
        self._entries = []
        self._index = None
        if self.db_path.exists():
            for f in self.db_path.iterdir():
                f.unlink()

    @property
    def size(self) -> int:
        return len(self._entries)
