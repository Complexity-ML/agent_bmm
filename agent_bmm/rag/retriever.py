# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
RAG Retriever — FAISS-based vector search.

Chunks documents, embeds them with sentence-transformers,
and retrieves relevant context for the agent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """A document chunk with its embedding."""

    text: str
    source: str = ""
    chunk_id: int = 0


class Retriever:
    """
    FAISS-based retriever for RAG.

    Uses sentence-transformers for embedding and FAISS for search.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self._index = None
        self._chunks: list[Chunk] = []
        self._embedder = None

    def _load_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.model_name)

    def add_documents(self, texts: list[str], sources: list[str] | None = None):
        """Add documents to the index."""
        import numpy as np

        self._load_embedder()
        sources = sources or [""] * len(texts)

        # Chunk (simple split for now)
        new_chunks = []
        for text, source in zip(texts, sources):
            words = text.split()
            for i in range(0, len(words), 200):  # ~200 word chunks
                chunk_text = " ".join(words[i : i + 200])
                new_chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=source,
                        chunk_id=len(self._chunks) + len(new_chunks),
                    )
                )

        # Embed
        embeddings = self._embedder.encode(
            [c.text for c in new_chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Index
        if self._index is None:
            import faiss

            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)  # inner product (cosine on normalized)

        self._index.add(embeddings.astype(np.float32))
        self._chunks.extend(new_chunks)

    def search(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Search for relevant chunks."""
        import numpy as np

        if self._index is None or self._index.ntotal == 0:
            return []

        self._load_embedder()
        k = top_k or self.top_k

        query_emb = self._embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

        scores, indices = self._index.search(query_emb, k)
        return [self._chunks[i] for i in indices[0] if i < len(self._chunks)]
