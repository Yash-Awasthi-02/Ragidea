"""
PATHFINDER Experiment — Step 3: Baseline Retrieval Systems
===========================================================
All baselines operate on the same per-query graph as PATHFINDER.
Same embedding model (all-MiniLM-L6-v2), same token budget (K_tok=2048).

Baselines:
  naive_rag            — top-k nodes by cosine(q_emb, phi_sem(v)), no graph
  bfs_2hop             — BFS from entry node v₀ up to depth 2
  spreading_activation — single-source SA from v₀ with edge-weight decay
"""

import numpy as np
import networkx as nx
from collections import deque

K_TOK_DEFAULT = 2048

# ── Shared utility ─────────────────────────────────────────────────────────────
def _tok(node_idx: int, G: nx.DiGraph) -> int:
    """Word count of node text as token proxy."""
    return max(1, len(G.nodes[node_idx].get("text", "").split()))

def _entry_node(graph_data: dict) -> int:
    """v₀: highest cosine similarity to query."""
    phi_sem_q = graph_data["embeddings"] @ graph_data["q_emb"]
    return int(np.argmax(phi_sem_q))


# ── Naive RAG ─────────────────────────────────────────────────────────────────
def naive_rag(graph_data: dict, k: int = 5) -> list[int]:
    """
    Dense top-k retrieval — no graph structure.
    Returns the k nodes with highest cosine similarity to query.
    """
    if graph_data["N"] == 0:
        return []
    phi_sem_q = graph_data["embeddings"] @ graph_data["q_emb"]
    ranked = np.argsort(phi_sem_q)[::-1]
    return ranked[:k].tolist()


# ── BFS 2-hop ─────────────────────────────────────────────────────────────────
def bfs_2hop(graph_data: dict, max_depth: int = 2,
             k_tok: int = K_TOK_DEFAULT) -> list[int]:
    """
    BFS from entry node v₀, up to max_depth hops.
    Adds nodes in BFS order until token budget is exhausted.
    """
    G = graph_data["G"]
    N = graph_data["N"]
    if N == 0:
        return []

    v0 = _entry_node(graph_data)

    visited = {v0}
    queue   = deque([(v0, 0)])
    result  = [v0]
    tok     = _tok(v0, G)

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nb in G.successors(node):
            if nb in visited:
                continue
            nb_tok = _tok(nb, G)
            if tok + nb_tok > k_tok:
                continue
            visited.add(nb)
            queue.append((nb, depth + 1))
            result.append(nb)
            tok += nb_tok

    return result


# ── Spreading Activation ──────────────────────────────────────────────────────
def spreading_activation(graph_data: dict,
                         decay: float = 0.5,
                         k_tok: int = K_TOK_DEFAULT,
                         min_activation: float = 0.01) -> list[int]:
    """
    Single-source spreading activation from entry node v₀.

    Activation propagates along directed edges multiplied by:
        activation_gain = edge_weight * decay

    Nodes with activation below min_activation are pruned (no further spread).
    Final selection: nodes sorted by descending activation, added greedily
    until token budget is exhausted.
    """
    G = graph_data["G"]
    N = graph_data["N"]
    if N == 0:
        return []

    v0 = _entry_node(graph_data)

    # BFS-style activation spread
    activation: dict[int, float] = {v0: 1.0}
    queue = deque([(v0, 1.0)])

    while queue:
        node, act = queue.popleft()
        for nb in G.successors(node):
            edge_w   = G[node][nb].get("weight", 0.5)
            new_act  = act * decay * edge_w
            if new_act < min_activation:
                continue
            if nb not in activation or activation[nb] < new_act:
                activation[nb] = new_act
                queue.append((nb, new_act))

    # Select by descending activation within token budget
    sorted_nodes = sorted(activation.keys(), key=lambda v: -activation[v])
    result, tok = [], 0
    for v in sorted_nodes:
        v_tok = _tok(v, G)
        if tok + v_tok <= k_tok:
            result.append(v)
            tok += v_tok

    return result


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pickle, sys
    path = "data/hotpotqa_graphs.pkl"
    try:
        with open(path, "rb") as f:
            records = pickle.load(f)
    except FileNotFoundError:
        print(f"No cache at {path}. Run 01_build_kg.py first.")
        sys.exit(1)

    print(f"Testing baselines on first 3 queries...\n")
    for rec in records[:3]:
        gd = rec["graph"]
        q  = rec["question"][:60]
        print(f"Q: {q}...")
        print(f"  NaiveRAG    : {len(naive_rag(gd))} nodes")
        print(f"  BFS 2-hop   : {len(bfs_2hop(gd))} nodes")
        print(f"  Spreading A : {len(spreading_activation(gd))} nodes")
        print()
