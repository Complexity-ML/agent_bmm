# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Smart Context — Only send relevant files to the LLM.

Instead of indexing all files, uses TF-IDF similarity to find
files most relevant to the current task. Falls back to simple
keyword matching if scikit-learn is not installed.
"""

from __future__ import annotations

import re


def rank_files_by_relevance(
    query: str,
    files: dict[str, str],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Rank files by relevance to the query.

    Args:
        query: The user's task description.
        files: Dict of {path: content}.
        top_k: Number of files to return.

    Returns:
        List of (path, score) sorted by relevance.
    """
    if not files or not query:
        return list((k, 0.0) for k in list(files.keys())[:top_k])

    try:
        return _rank_tfidf(query, files, top_k)
    except ImportError:
        return _rank_keyword(query, files, top_k)


def _rank_tfidf(
    query: str, files: dict[str, str], top_k: int
) -> list[tuple[str, float]]:
    """Rank using TF-IDF cosine similarity (requires scikit-learn)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    paths = list(files.keys())
    contents = [files[p] for p in paths]

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf = vectorizer.fit_transform(contents + [query])

    query_vec = tfidf[-1]
    doc_vecs = tfidf[:-1]
    scores = cosine_similarity(query_vec, doc_vecs).flatten()

    ranked = sorted(zip(paths, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _rank_keyword(
    query: str, files: dict[str, str], top_k: int
) -> list[tuple[str, float]]:
    """Simple keyword matching fallback."""
    keywords = set(re.findall(r"\w+", query.lower()))
    if not keywords:
        return list((k, 0.0) for k in list(files.keys())[:top_k])

    scored = []
    for path, content in files.items():
        content_lower = content.lower()
        # Score = fraction of keywords found in file
        hits = sum(1 for kw in keywords if kw in content_lower)
        score = hits / len(keywords)
        # Bonus for path matches
        path_lower = path.lower()
        path_hits = sum(1 for kw in keywords if kw in path_lower)
        score += path_hits * 0.5
        scored.append((path, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
