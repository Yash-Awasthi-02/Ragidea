"""
Synthetic graph fixtures for unit testing PATHFINDER.

Builds small KnowledgeGraph objects with hand-crafted embeddings and facets
so we can test formal properties without an embedding model.
"""

import numpy as np
import networkx as nx
from pathfinder.graph import KnowledgeGraph
from pathfinder.config import ALPHA, BETA, GAMMA, DELTA, EPSILON


def make_synthetic_kg(
    embeddings: np.ndarray,
    q_emb: np.ndarray,
    edges: list[tuple[int, int, float]] | None = None,
    phi_temp: list[float] | None = None,
    phi_imp: list[float] | None = None,
    phi_conf: list[float] | None = None,
    phi_dom: np.ndarray | None = None,
    texts: list[str] | None = None,
) -> KnowledgeGraph:
    """
    Build a synthetic KnowledgeGraph for testing.

    Args:
        embeddings: (N, D) L2-normalised node embeddings
        q_emb:      (D,) L2-normalised query embedding
        edges:      list of (u, v, weight) directed edges
        phi_temp:   list of N temporal scores (default 1.0)
        phi_imp:    list of N importance scores (default 0.5)
        phi_conf:   list of N confidence scores (default 0.70)
        phi_dom:    (N, K) domain embeddings (default zeros)
        texts:      list of N node texts (default "node_{i}")
    """
    N, D = embeddings.shape

    # Default facets
    if phi_temp is None:
        phi_temp = [1.0] * N
    if phi_imp is None:
        phi_imp = [0.5] * N
    if phi_conf is None:
        phi_conf = [0.70] * N
    if phi_dom is None:
        phi_dom = np.zeros((N, 1))
    if texts is None:
        texts = [f"node_{i} " + "word " * 10 for i in range(N)]  # ~11 tokens each

    # Compute domain via PCA if not provided (or use zeros)
    if phi_dom is not None and phi_dom.shape[0] == N:
        phi_dom_matrix = phi_dom
        K = phi_dom_matrix.shape[1]
    else:
        K = 1
        phi_dom_matrix = np.zeros((N, 1))

    W_dom = np.eye(K, D) if K <= D else np.zeros((K, D))
    q_dom = W_dom @ q_emb if W_dom.shape[1] == D else np.zeros(K)
    q_norm = np.linalg.norm(q_dom)
    if q_norm > 1e-12:
        q_dom = q_dom / q_norm

    # Similarity to query
    phi_sem_q = embeddings @ q_emb  # (N,)

    # Build graph
    G = nx.DiGraph()
    for i in range(N):
        G.add_node(
            i,
            text=texts[i],
            doc_title=f"doc_{i}",
            sent_idx=0,
            phi_conf=float(phi_conf[i]),
            phi_temp=float(phi_temp[i]),
            phi_imp=float(phi_imp[i]),
            phi_dom=phi_dom_matrix[i],
            sim_to_query=float(max(0.0, phi_sem_q[i])),
        )

    # Add edges
    if edges:
        for u, v, w in edges:
            G.add_edge(u, v, weight=float(w))

    phi_imp_arr = np.array(phi_imp, dtype=float)

    return KnowledgeGraph(
        G=G,
        nodes=[{"text": texts[i], "doc_title": f"doc_{i}", "sent_idx": 0,
                "phi_conf": float(phi_conf[i])} for i in range(N)],
        embeddings=embeddings,
        phi_dom_matrix=phi_dom_matrix,
        W_dom=W_dom,
        q_emb=q_emb,
        q_dom=q_dom,
        phi_imp=phi_imp_arr,
        N=N,
    )


def make_chain_graph(n: int = 5, sim_values: list[float] | None = None) -> KnowledgeGraph:
    """
    Build a simple chain graph: 0 → 1 → 2 → ... → n-1.
    Each node has a specified similarity to the query.
    Edge weights are 0.85 between consecutive nodes.
    """
    if sim_values is None:
        # Generate n decreasing similarity values from 0.9 to 0.1
        sim_values = list(np.linspace(0.9, 0.1, n))

    D = 16
    embeddings = np.zeros((n, D))
    for i in range(n):
        embeddings[i, 0] = sim_values[i]  # first dim = sim to query
    # L2-normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    embeddings = embeddings / norms

    q_emb = np.zeros(D)
    q_emb[0] = 1.0
    q_emb = q_emb / np.linalg.norm(q_emb)

    edges = [(i, i + 1, 0.85) for i in range(n - 1)]
    # Also add reverse edges for undirected connectivity
    edges += [(i + 1, i, 0.85) for i in range(n - 1)]

    return make_synthetic_kg(embeddings, q_emb, edges=edges)


def make_star_graph(n_branches: int = 3, branch_len: int = 2) -> KnowledgeGraph:
    """
    Build a star graph: node 0 is center, with n_branches branches of length branch_len.
    Node 0 has highest similarity to query; branch nodes have decreasing similarity.
    """
    total = 1 + n_branches * branch_len
    D = 16
    embeddings = np.zeros((total, D))
    embeddings[0, 0] = 0.95  # center node
    idx = 1
    sim = 0.8
    for b in range(n_branches):
        for j in range(branch_len):
            embeddings[idx, 0] = sim - j * 0.15
            idx += 1
    # L2-normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    embeddings = embeddings / norms

    q_emb = np.zeros(D)
    q_emb[0] = 1.0
    q_emb = q_emb / np.linalg.norm(q_emb)

    edges = []
    idx = 1
    for b in range(n_branches):
        prev = 0
        for j in range(branch_len):
            edges.append((prev, idx, 0.80))
            edges.append((idx, prev, 0.80))
            prev = idx
            idx += 1

    return make_synthetic_kg(embeddings, q_emb, edges=edges)


def make_random_graph(n: int = 10, seed: int = 42, edge_prob: float = 0.3) -> KnowledgeGraph:
    """
    Build a random graph with n nodes, random embeddings, and random edges.
    """
    rng = np.random.RandomState(seed)
    D = 16
    embeddings = rng.randn(n, D)
    # L2-normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    embeddings = embeddings / norms

    q_emb = rng.randn(D)
    q_emb = q_emb / np.linalg.norm(q_emb)

    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < edge_prob:
                w = float(max(0.0, np.dot(embeddings[i], embeddings[j])))
                if w > 0.3:  # θ_edge threshold
                    edges.append((i, j, w))

    phi_conf = rng.uniform(0.5, 0.95, n).tolist()
    phi_imp = rng.uniform(0.0, 1.0, n).tolist()

    return make_synthetic_kg(
        embeddings, q_emb, edges=edges,
        phi_conf=phi_conf, phi_imp=phi_imp,
    )
