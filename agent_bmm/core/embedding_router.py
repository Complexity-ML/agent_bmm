# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Embedding-based Tool Routing — Use sentence-transformers to route queries.

Encodes tool descriptions and queries into embeddings, routes via
cosine similarity. More accurate than keyword matching.

Requires: pip install agent-bmm[gpu] (for sentence-transformers)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class EmbeddingRouter:
    """Route queries to tools using semantic similarity."""

    def __init__(
        self,
        tool_descriptions: list[str],
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.tool_descriptions = tool_descriptions
        self._model: Any = None
        self._model_name = model_name
        self._tool_embeddings: torch.Tensor | None = None

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            # Pre-compute tool description embeddings
            embeddings = self._model.encode(self.tool_descriptions, convert_to_tensor=True)
            self._tool_embeddings = F.normalize(embeddings, dim=-1)

    def route(self, query: str, top_k: int = 1) -> list[tuple[int, float]]:
        """
        Route a query to the best tool(s).

        Returns:
            List of (tool_index, similarity_score) sorted by score.
        """
        self._load_model()
        query_emb = self._model.encode([query], convert_to_tensor=True)
        query_emb = F.normalize(query_emb, dim=-1)
        scores = (query_emb @ self._tool_embeddings.T).squeeze(0)
        top_scores, top_ids = scores.topk(min(top_k, len(self.tool_descriptions)))
        return [(idx.item(), score.item()) for idx, score in zip(top_ids, top_scores)]

    def route_batch(self, queries: list[str], top_k: int = 1) -> list[list[tuple[int, float]]]:
        """Route multiple queries at once (batched)."""
        self._load_model()
        query_embs = self._model.encode(queries, convert_to_tensor=True)
        query_embs = F.normalize(query_embs, dim=-1)
        scores = query_embs @ self._tool_embeddings.T  # (N, num_tools)
        results = []
        for i in range(len(queries)):
            top_scores, top_ids = scores[i].topk(min(top_k, len(self.tool_descriptions)))
            results.append([(idx.item(), s.item()) for idx, s in zip(top_ids, top_scores)])
        return results
