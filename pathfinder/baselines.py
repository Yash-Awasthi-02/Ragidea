"""
PATHFINDER — Baseline retrieval systems (Section 7.4).

All baselines operate on the same per-query graph as PATHFINDER.
Same embedding model, same token budget (K_tok=2048).

Baselines:
  naive_rag            — top-k nodes by cosine(q_emb, φ_sem(v)), no graph
  bfs_2hop             — BFS from entry node v₀ up to depth 2
  spreading_activation — single-source SA from v₀ with edge-weight decay
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from collections import deque

from pathfinder.graph import KnowledgeGraph
from pathfinder.config import K_TOK


def _entry_node(kg: KnowledgeGraph) -> int:
    """v₀: highest cosine similarity to query."""
    phi_sem_q = kg.embeddings @ kg.q_emb
    return int(np.argmax(phi_sem_q))


def _tok(v: int, kg: KnowledgeGraph) -> int:
    return kg.token_count(v)


# ── Naive RAG — top-k dense retrieval, no graph ─────────────────────────────
def naive_rag(kg: KnowledgeGraph, k: int = 5) -> list[int]:
    """
    Dense top-k retrieval — no graph structure.
    Returns the k nodes with highest cosine similarity to query.
    """
    if kg.N == 0:
        return []
    phi_sem_q = kg.embeddings @ kg.q_emb
    ranked = np.argsort(phi_sem_q)[::-1]
    return ranked[:k].tolist()


# ── BFS 2-hop — all nodes within 2 hops of entry node ───────────────────────
def bfs_2hop(kg: KnowledgeGraph, max_depth: int = 2,
             k_tok: int = K_TOK) -> list[int]:
    """
    BFS from entry node v₀, up to max_depth hops.
    Adds nodes in BFS order until token budget is exhausted.
    """
    G = kg.G
    if kg.N == 0:
        return []

    v0 = _entry_node(kg)
    visited = {v0}
    queue = deque([(v0, 0)])
    result = [v0]
    tok = _tok(v0, G if False else kg)

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nb in kg.out_neighbors(node):
            if nb in visited:
                continue
            nb_tok = _tok(nb, kg)
            if tok + nb_tok > k_tok:
                continue
            visited.add(nb)
            queue.append((nb, depth + 1))
            result.append(nb)
            tok += nb_tok

    return result


# ── Spreading Activation RAG ────────────────────────────────────────────────
def spreading_activation(
    kg: KnowledgeGraph,
    decay: float = 0.5,
    k_tok: int = K_TOK,
    min_activation: float = 0.01,
) -> list[int]:
    """
    Single-source spreading activation from entry node v₀.
    Activation propagates along directed edges multiplied by:
        activation_gain = edge_weight * decay
    Nodes with activation below min_activation are pruned.
    Final ranking by activation score; nodes added until token budget exhausted.
    """
    if kg.N == 0:
        return []

    v0 = _entry_node(kg)
    G = kg.G

    # Initialize activation
    activation = {v0: 1.0}
    # BFS-like propagation
    queue = deque([v0])

    while queue:
        node = queue.popleft()
        cur_act = activation[node]
        if cur_act < min_activation:
            continue
        for nb in kg.out_neighbors(node):
            w = kg.edge_weight(node, nb)
            new_act = cur_act * w * decay
            if nb not in activation or new_act > activation[nb]:
                activation[nb] = new_act
                if new_act >= min_activation:
                    queue.append(nb)

    # Rank by activation score
    ranked = sorted(activation.keys(), key=lambda v: activation[v], reverse=True)

    # Add nodes until token budget exhausted
    result = []
    tok = 0
    for v in ranked:
        v_tok = _tok(v, kg)
        if tok + v_tok > k_tok:
            continue
        result.append(v)
        tok += v_tok

    return result
