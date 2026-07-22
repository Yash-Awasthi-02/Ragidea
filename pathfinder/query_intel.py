"""
PATHFINDER — Query Intelligence Layer (Section 4.4).

Lightweight intent classifier + semantic cache + query rewriting.
Routes queries to appropriate processing paths before traversal.
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from dataclasses import dataclass

from pathfinder.config import ALPHA, BETA, GAMMA, DELTA, EPSILON, CACHE_THETA


class QueryType(Enum):
    FACTUAL    = "factual"
    RELATIONAL = "relational"
    PROCEDURAL = "procedural"
    TEMPORAL   = "temporal"
    FALSE_PREM = "false_premise"
    MULTI_INT  = "multi_intent"


@dataclass
class QueryRoute:
    """Routing decision for a query."""
    query_type: QueryType
    weights: tuple[float, float, float, float, float]  # α, β, γ, δ, ε
    frontier_restriction: int  # 0 = none, 1 = 1-hop only
    cache_hit: bool = False
    cached_result: object = None


def classify_query(query: str) -> QueryType:
    """
    Lightweight intent classifier (keyword-based heuristic).
    In production this would be a trained classifier or LLM call.
    """
    q_lower = query.lower()

    # Temporal keywords
    temporal_words = ["when", "year", "date", "recent", "latest", "before",
                      "after", "since", "until", "old", "new", "first", "last"]
    if any(w in q_lower for w in temporal_words):
        return QueryType.TEMPORAL

    # Procedural keywords
    proc_words = ["how to", "how do", "how can", "steps", "procedure",
                  "process", "method", "way to"]
    if any(w in q_lower for w in proc_words):
        return QueryType.PROCEDURAL

    # False premise detection (simple heuristic: "is it true that X" patterns)
    false_prem_words = ["is it true that", "isn't it", "doesn't", "wasn't"]
    if any(w in q_lower for w in false_prem_words):
        return QueryType.FALSE_PREM

    # Multi-intent: multiple questions
    if " and " in q_lower and "?" in q_lower:
        # Check if there are multiple question marks
        if q_lower.count("?") > 1:
            return QueryType.MULTI_INT

    # Relational: comparison or connection
    rel_words = ["compare", "difference", "relation", "connect", "between",
                 "versus", "vs", "same", "similar", "both"]
    if any(w in q_lower for w in rel_words):
        return QueryType.RELATIONAL

    # Default: relational (full PATHFINDER traversal)
    return QueryType.RELATIONAL


def route_query(
    query: str,
    q_emb: np.ndarray,
    cache_embeddings: list[np.ndarray] = None,
    cache_results: list[object] = None,
    cache_theta: float = CACHE_THETA,
) -> QueryRoute:
    """
    Route a query through the Query Intelligence Layer.

    1. Check semantic cache (cosine ≥ θ=0.92)
    2. Classify intent
    3. Adjust weights / frontier restriction per query type
    """
    # ── Semantic cache check ────────────────────────────────────────────────
    if cache_embeddings and cache_results:
        for i, cached_emb in enumerate(cache_embeddings):
            cos_sim = float(np.dot(q_emb, cached_emb))
            if cos_sim >= cache_theta:
                return QueryRoute(
                    query_type=QueryType.RELATIONAL,
                    weights=(ALPHA, BETA, GAMMA, DELTA, EPSILON),
                    frontier_restriction=0,
                    cache_hit=True,
                    cached_result=cache_results[i],
                )

    # ── Intent classification ───────────────────────────────────────────────
    qtype = classify_query(query)

    # ── Weight adjustment per query type ────────────────────────────────────
    if qtype == QueryType.FACTUAL:
        # Elevate α, restrict to 1-hop
        weights = _rescale_weights(alpha=0.65, beta=0.10, gamma=0.10,
                                   delta=0.08, epsilon=0.07)
        return QueryRoute(qtype, weights, frontier_restriction=1)

    elif qtype == QueryType.TEMPORAL:
        # Elevate β (temporal weight)
        weights = _rescale_weights(alpha=0.40, beta=0.30, gamma=0.12,
                                   delta=0.09, epsilon=0.09)
        return QueryRoute(qtype, weights, frontier_restriction=0)

    elif qtype == QueryType.PROCEDURAL:
        # Check skill store first (not implemented here)
        # PATHFINDER only on miss
        return QueryRoute(qtype, (ALPHA, BETA, GAMMA, DELTA, EPSILON), 0)

    elif qtype == QueryType.FALSE_PREM:
        # Flag; return structured error; skip traversal
        return QueryRoute(qtype, (ALPHA, BETA, GAMMA, DELTA, EPSILON), 0)

    elif qtype == QueryType.MULTI_INT:
        # Decompose into sub-queries (not implemented here)
        return QueryRoute(qtype, (ALPHA, BETA, GAMMA, DELTA, EPSILON), 0)

    # RELATIONAL: full PATHFINDER traversal (default)
    return QueryRoute(qtype, (ALPHA, BETA, GAMMA, DELTA, EPSILON), 0)


def _rescale_weights(alpha, beta, gamma, delta, epsilon) -> tuple:
    """Rescale weights to sum to 1 (Section 4.4)."""
    total = alpha + beta + gamma + delta + epsilon
    return (alpha/total, beta/total, gamma/total, delta/total, epsilon/total)
