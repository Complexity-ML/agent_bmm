# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
LRU Cache for tool results (#44) + Query deduplication (#42).

Avoids re-executing identical tool calls. Configurable TTL and max size.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CacheEntry:
    result: str
    timestamp: float
    hits: int = 0


class ToolResultCache:
    """LRU cache for tool execution results."""

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def _key(self, tool_name: str, query: str) -> str:
        raw = f"{tool_name}:{query}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, tool_name: str, query: str) -> str | None:
        """Get cached result, or None if miss/expired."""
        key = self._key(tool_name, query)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() - entry.timestamp > self.ttl:
            del self._cache[key]
            return None
        entry.hits += 1
        self._cache.move_to_end(key)
        return entry.result

    def put(self, tool_name: str, query: str, result: str):
        """Cache a tool result."""
        key = self._key(tool_name, query)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = CacheEntry(result=result, timestamp=time.time())
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def invalidate(self, tool_name: str | None = None):
        """Invalidate cache entries. None = clear all."""
        if tool_name is None:
            self._cache.clear()
        else:
            prefix = hashlib.md5(f"{tool_name}:".encode()).hexdigest()[:8]
            to_delete = [k for k in self._cache if k.startswith(prefix)]
            for k in to_delete:
                del self._cache[k]

    @property
    def stats(self) -> dict:
        total_hits = sum(e.hits for e in self._cache.values())
        return {"size": len(self._cache), "max_size": self.max_size, "total_hits": total_hits}


class QueryDeduplicator:
    """Detect and deduplicate identical queries in a batch."""

    def deduplicate(self, queries: list[str]) -> tuple[list[str], dict[int, int]]:
        """
        Remove duplicate queries from a batch.

        Returns:
            (unique_queries, index_map) where index_map[original_idx] = unique_idx
        """
        seen: dict[str, int] = {}
        unique: list[str] = []
        index_map: dict[int, int] = {}

        for i, query in enumerate(queries):
            if query in seen:
                index_map[i] = seen[query]
            else:
                idx = len(unique)
                seen[query] = idx
                unique.append(query)
                index_map[i] = idx

        return unique, index_map

    def expand_results(self, results: list[str], index_map: dict[int, int], total: int) -> list[str]:
        """Expand deduplicated results back to original indices."""
        return [results[index_map[i]] for i in range(total)]
