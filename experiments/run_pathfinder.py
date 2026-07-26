"""
PATHFINDER Experiment — Step 2: Algorithm 1 Implementation
===========================================================
Greedy submodular coverage maximization on multidimensional knowledge graphs.

Implements exactly the pseudocode from §4.2 of the paper:
  - Guard conditions 0a / 0b / 3b
  - FEASIBLE pre-filter (line 10b) replacing old hard-break
  - Marginal gain Δ_full with precomputed product_factor (O(1) per step)
  - σ̃ running lower bound active at line 15b
  - Frontier as dict → disc_parent (handles cycles correctly)
  - Bounded re-traversal protocol (MAX_RETRIES = 3)
  - True σ(S) computed at line 17 via parent[] tree traversal
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional

# ── Default hyperparameters (§3.4, §4.3) ─────────────────────────────────────
ALPHA       = 1.00    # semantic coverage weight (default semantic-only for high retrieval recall)
BETA        = 0.00    # temporal recency weight
GAMMA       = 0.00    # structural importance weight
DELTA       = 0.00    # domain alignment weight
EPSILON     = 0.00    # epistemic confidence weight

K_TOK       = 2048    # default token budget
TAU_HIGH    = 0.50    # σ ≥ τ_high → proceed
TAU_LOW     = 0.30    # σ < τ_low  → re-traverse
MAX_RETRIES = 3
EPSILON_MAX = 0.40
EPSILON_STEP= 0.05
SUFFICIENCY_THRESHOLD = 0.99   # coverage score for sufficiency_check (0.99 = effectively disabled for Recall@5)

# ── Teleportation parameters (Phase 2, Task 2.1) ───────────────────────────
THETA_TELEPORT    = 0.01   # min marginal gain threshold to trigger teleportation
TELEPORT_TOPK     = 5      # number of global dense nodes to inject on teleport
MAX_TELEPORTS     = 3      # cap teleportation jumps per traversal

# Approximate tokens per sentence (proxy for c̄)
def tok_count(node_idx: int, G: nx.DiGraph) -> int:
    text = G.nodes[node_idx].get("text", "")
    return max(1, len(text.split()))


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class TraversalResult:
    S:               list[int]       # selected node indices in selection order
    F:               float           # F(S, q) — full coverage score
    sigma:           float           # σ(S) — path confidence
    confidence_flag: str             # "HIGH" | "HEDGE" | "LOW"
    retries:         int             # number of re-traversal attempts made
    parent:          dict            # disc_parent map for downstream use


# ── σ(S) — true path confidence & bottleneck confidence ──────────────────────
def compute_sigma_min(S: list[int], parent: dict[int, Optional[int]], G: nx.DiGraph) -> float:
    """
    σ_min(S) = min_{v∈S} min_{edges/nodes on path} (W(e) · φ_conf(u))
    Bottleneck confidence aggregation (fuzzy logic AND) to prevent path-product decay.
    Teleportation nodes (parent=None, non-root) are treated as fresh entries.
    """
    if not S:
        return 1.0

    min_conf = 1.0
    for v in S:
        cur = v
        while cur is not None:
            min_conf = min(min_conf, G.nodes[cur].get("phi_conf", 0.7))
            par = parent.get(cur)
            if par is not None and G.has_edge(par, cur):
                min_conf = min(min_conf, max(0.0, G[par][cur].get("weight", 0.0)))
            cur = par
    return min_conf


def compute_sigma_product(S: list[int], parent: dict[int, Optional[int]],
                          G: nx.DiGraph) -> float:
    """
    σ_prod(S) = min_{v∈S} Π_{edges/nodes on path} (W(e) · φ_conf(u))
    Original path-product confidence (no geometric mean normalization).
    Teleportation nodes reset the product to φ_conf(v).
    """
    if not S:
        return 1.0

    min_sigma = 1.0
    for v in S:
        sigma_v = 1.0
        cur = v
        while cur is not None:
            sigma_v *= G.nodes[cur].get("phi_conf", 0.7)
            par = parent.get(cur)
            if par is not None and G.has_edge(par, cur):
                sigma_v *= max(0.0, G[par][cur].get("weight", 0.0))
            cur = par
        min_sigma = min(min_sigma, sigma_v)

    return min_sigma


# Alias for consistency with paper terminology
compute_sigma_bottleneck = compute_sigma_min


def compute_sigma(S: list[int], parent: dict[int, Optional[int]],
                  G: nx.DiGraph,
                  use_geometric_mean: bool = True) -> float:
    """
    σ(S) = min_{v∈S} σ(v₀→v)

    If use_geometric_mean=True:
      σ(v₀→v) = (Π φ_conf(u) × Π W(e)) ^ (1 / path_length)
    Else (original paper formula):
      σ(v₀→v) = Π φ_conf(u) × Π W(e)

    Uses disc_parent pointers to trace each root-to-node path.
    Teleportation nodes (parent=None but not root) are treated as fresh
    entry points: σ(v) = φ_conf(v) for that node.
    """
    if not S:
        return 1.0

    min_sigma = 1.0
    for v in S:
        sigma_v = 1.0
        path_length = 0  # number of edges on path
        cur = v
        while cur is not None:
            sigma_v *= G.nodes[cur].get("phi_conf", 0.7)
            par = parent.get(cur)
            if par is not None and G.has_edge(par, cur):
                sigma_v *= max(0.0, G[par][cur].get("weight", 0.0))
                path_length += 1
            cur = par
        if use_geometric_mean and path_length > 0:
            sigma_v = sigma_v ** (1.0 / path_length)
        min_sigma = min(min_sigma, sigma_v)

    return min_sigma


# ── F(S, q) — full objective ──────────────────────────────────────────────────
def compute_F(S: list[int], G: nx.DiGraph, q_dom: np.ndarray,
              phi_sem_q: np.ndarray, weights: tuple) -> float:
    """
    F(S,q) = α·f(S,q) + Σ_{v∈S}[β·φ_temp(v) + γ·φ_imp(v)
                                   + δ·max(0,cos(φ_dom(v),q_dom))
                                   + ε·φ_conf(v)]
    """
    if not S:
        return 0.0
    alpha, beta, gamma, delta, epsilon = weights

    # Coverage term f(S,q) = 1 - Π(1 - sim(v,q))
    product = 1.0
    for v in S:
        product *= (1.0 - max(0.0, float(phi_sem_q[v])))
    f_cov = 1.0 - product

    # Additive facet sums
    temp_sum, imp_sum, dom_sum, conf_sum = 0.0, 0.0, 0.0, 0.0
    q_dom_norm = np.linalg.norm(q_dom) if q_dom is not None else 0.0
    for v in S:
        nd = G.nodes[v]
        temp_sum += nd.get("phi_temp", 1.0)
        imp_sum  += nd.get("phi_imp",  0.5)
        conf_sum += nd.get("phi_conf", 0.7)
        if q_dom_norm > 0:
            pd = nd.get("phi_dom", np.zeros_like(q_dom))
            pd_norm = np.linalg.norm(pd)
            if pd_norm > 0:
                dom_sum += max(0.0, float(np.dot(pd, q_dom) / (pd_norm * q_dom_norm)))

    return (alpha * f_cov + beta * temp_sum + gamma * imp_sum
            + delta * dom_sum + epsilon * conf_sum)


# ── Δ_full — marginal gain (fast, O(1) with product_factor) ──────────────────
def delta_full_fast(v: int, G: nx.DiGraph, q_dom: np.ndarray,
                    phi_sem_q: np.ndarray, weights: tuple,
                    product_factor: float) -> float:
    """
    Δ_full(v|S,q) using precomputed product_factor = Π_{u∈S}(1-sim(u,q)).
    Δ_coverage(v|S,q) = sim(v,q) · product_factor  (O(1))
    """
    alpha, beta, gamma, delta, epsilon = weights
    nd = G.nodes[v]

    sim_v = max(0.0, float(phi_sem_q[v]))
    d_cov = sim_v * product_factor

    phi_temp = nd.get("phi_temp", 1.0)
    phi_imp  = nd.get("phi_imp",  0.5)
    phi_conf = nd.get("phi_conf", 0.7)

    q_dom_norm = np.linalg.norm(q_dom) if q_dom is not None else 0.0
    dom_score = 0.0
    if q_dom_norm > 0:
        pd = nd.get("phi_dom", np.zeros_like(q_dom))
        pd_norm = np.linalg.norm(pd)
        if pd_norm > 0:
            dom_score = max(0.0, float(np.dot(pd, q_dom) / (pd_norm * q_dom_norm)))

    return (alpha * d_cov + beta * phi_temp + gamma * phi_imp
            + delta * dom_score + epsilon * phi_conf)


# ── Sufficiency check (line 15) ───────────────────────────────────────────────
def sufficiency_check(product_factor: float,
                      threshold: float = SUFFICIENCY_THRESHOLD) -> bool:
    """f(S,q) = 1 - product_factor ≥ threshold."""
    if threshold >= 1.0:
        return False  # disabled
    return (1.0 - product_factor) >= threshold


# ── Single traversal pass (Algorithm 1) ──────────────────────────────────────
def _single_pass(G: nx.DiGraph, q_emb: np.ndarray, q_dom: np.ndarray,
                 phi_sem_q: np.ndarray, N: int,
                 weights: tuple, k_tok: int,
                 enable_teleport: bool = True,
                 always_dense: bool = False,
                 teleport_gate: str = "gain") -> tuple:
    """
    One complete run of Algorithm 1.
    Returns (S, F, sigma, parent).

    Teleportation modes (Phase 2, Task 2.1 / PLAN §2.5):
      - enable_teleport=True, teleport_gate="gain"  : inject TopK dense nodes only
        when max frontier marginal gain < θ_teleport (original behaviour).
      - always_dense=True : add TopK global dense nodes to the frontier at EVERY
        greedy step (teleportation becomes the default — "PATHFINDER-Flat").
      - teleport_gate="connectivity" : only allow teleportation when the entry
        node's connected component cannot already reach the dense candidates.

    always_dense weakens the graph-coherence story; report as PATHFINDER-Flat vs
    PATHFINDER-Coherent in the paper.
    """
    import networkx as _nx  # local alias to avoid shadowing
    alpha, beta, gamma, delta, epsilon = weights

    # Line 0a — empty graph
    if N == 0:
        return [], 0.0, 1.0, {}

    # Line 0b — zero query embedding
    if np.linalg.norm(q_emb) < 1e-8:
        return [], 0.0, 1.0, {}

    # Line 3 — entry node: argmax cosine similarity to query
    v0 = int(np.argmax(phi_sem_q))

    # Line 3b — entry node budget guard
    if tok_count(v0, G) > k_tok:
        return [], 0.0, 1.0, {}

    # Lines 4-7 — initialise
    S      = [v0]
    S_set  = {v0}
    parent = {v0: None}

    tok            = tok_count(v0, G)
    sigma_tilde    = G.nodes[v0].get("phi_conf", 0.7)   # line 5: σ̃ init
    sigma_tilde_edges = 0  # number of edges in σ̃ product (for geometric mean)
    product_factor = 1.0 - max(0.0, float(phi_sem_q[v0]))

    # Frontier dict: node → disc_parent
    frontier: dict[int, int] = {}
    for u in G.successors(v0):
        if u not in S_set:
            frontier[u] = v0

    teleport_count = 0  # track teleportation jumps

    # Precompute global dense ranking for teleportation
    global_dense_ranked = None
    if enable_teleport:
        global_dense_ranked = np.argsort(phi_sem_q)[::-1]

    # Line 8 — main greedy loop
    while frontier and tok < k_tok:

        # Line 10b — FEASIBLE: frontier nodes that fit in remaining budget
        rem = k_tok - tok
        feasible = {v: p for v, p in frontier.items()
                    if tok_count(v, G) <= rem}

        if not feasible:        # budget exhausted for all frontier candidates
            break

        # Line 10c — v* = argmax Δ_full over FEASIBLE
        best_v, best_gain = None, -1.0
        for v in feasible:
            g = delta_full_fast(v, G, q_dom, phi_sem_q, weights, product_factor)
            if g > best_gain:
                best_gain, best_v = g, v

        if best_v is None:
            break

        # ── Teleportation jump (Phase 2, Task 2.1 / PLAN §2.5) ───────────
        # Decide whether to inject global dense nodes into the frontier now.
        #   always_dense           → every step (PATHFINDER-Flat)
        #   teleport_gate="gain"   → only when frontier gain < θ_teleport
        #   teleport_gate="connectivity" → only when top dense nodes are NOT
        #                            already reachable from S ∪ frontier
        teleport_allowed = False
        if enable_teleport and teleport_count < MAX_TELEPORTS \
                and global_dense_ranked is not None:
            if always_dense:
                teleport_allowed = True
            elif teleport_gate == "connectivity":
                reachable = set(S_set) | set(frontier.keys())
                top_dense = [int(c) for c in global_dense_ranked[:TELEPORT_TOPK]]
                teleport_allowed = not any(c in reachable for c in top_dense)
            else:  # "gain" gate (default)
                teleport_allowed = best_gain < THETA_TELEPORT

        if teleport_allowed:
            injected = 0
            for cand in global_dense_ranked:
                cand = int(cand)
                if cand in S_set or cand in frontier:
                    continue
                if tok_count(cand, G) > rem:
                    continue
                # Inject into frontier with no graph parent (teleportation entry)
                frontier[cand] = None  # None parent = teleportation jump
                injected += 1
                if injected >= TELEPORT_TOPK:
                    break
            if injected > 0:
                teleport_count += 1
                # Re-evaluate feasible with new teleport candidates
                feasible = {v: p for v, p in frontier.items()
                            if tok_count(v, G) <= rem}
                if not feasible:
                    break
                # Re-select best from expanded frontier
                best_v, best_gain = None, -1.0
                for v in feasible:
                    g = delta_full_fast(v, G, q_dom, phi_sem_q, weights, product_factor)
                    if g > best_gain:
                        best_gain, best_v = g, v
                if best_v is None:
                    break

        v_star      = best_v
        disc_parent = feasible[v_star]

        # Line 12 — add to S
        S.append(v_star)
        S_set.add(v_star)
        parent[v_star] = disc_parent  # None if teleportation, else graph parent

        # Update coverage product factor (O(1))
        product_factor *= (1.0 - max(0.0, float(phi_sem_q[v_star])))

        # Line 13 — update σ̃ (skip edge weight for teleportation jumps)
        if disc_parent is not None and G.has_edge(disc_parent, v_star):
            w_edge = G[disc_parent][v_star].get("weight", 0.0)
            sigma_tilde *= max(0.0, w_edge) * G.nodes[v_star].get("phi_conf", 0.7)
            sigma_tilde_edges += 1
        else:
            # Teleportation jump: reset σ̃ to node confidence (fresh entry point)
            sigma_tilde = min(sigma_tilde, G.nodes[v_star].get("phi_conf", 0.7))

        # Line 14 — update token count
        tok += tok_count(v_star, G)

        # Line 15 — sufficiency check
        if sufficiency_check(product_factor, SUFFICIENCY_THRESHOLD):
            break

        # Line 15b — σ̃ low → early exit (caller will re-traverse)
        # Apply geometric mean normalization to sigma_tilde for the check
        sigma_tilde_check = sigma_tilde ** (1.0 / sigma_tilde_edges) if sigma_tilde_edges > 0 else sigma_tilde
        if sigma_tilde_check < TAU_LOW:
            del frontier[v_star]
            for u in G.successors(v_star):
                if u not in S_set and u not in frontier:
                    frontier[u] = v_star
            break

        # Remove v_star from frontier; expand with its successors
        del frontier[v_star]
        for u in G.successors(v_star):
            if u not in S_set and u not in frontier:
                frontier[u] = v_star

    # Line 17 — compute true σ(S)
    sigma = compute_sigma(S, parent, G)

    # Compute F(S, q)
    F = compute_F(S, G, q_dom, phi_sem_q, weights)

    return S, F, sigma, parent


# ── Public API: run_pathfinder ────────────────────────────────────────────────
def run_pathfinder(graph_data: dict,
                   weights: tuple = None,
                   k_tok: int = K_TOK,
                   tau_low: float = TAU_LOW,
                   tau_high: float = TAU_HIGH,
                   max_retries: int = MAX_RETRIES,
                   auto_detect_uniform_temp: bool = True,
                   enable_teleport: bool = True) -> TraversalResult:
    """
    Run PATHFINDER Algorithm 1 with bounded re-traversal protocol.

    Args:
        graph_data : output of KGBuilder.build()
        weights    : (α, β, γ, δ, ε). Defaults to paper values.
        k_tok      : token budget K_tok.
        enable_teleport : if True, enables dynamic dense-frontier teleportation
                          jumps (Phase 2, Task 2.1).

    Returns:
        TraversalResult
    """
    if weights is None:
        weights = (ALPHA, BETA, GAMMA, DELTA, EPSILON)

    G         = graph_data["G"]
    q_emb     = graph_data["q_emb"]
    q_dom     = graph_data["q_dom"]
    embs      = graph_data["embeddings"]
    N         = graph_data["N"]
    phi_sem_q = embs @ q_emb   # (N,) cosine similarities to query

    # FIX 3: Auto-detect uniform φ_temp → set β=0, renormalize remaining weights
    if auto_detect_uniform_temp:
        phi_temps = [G.nodes[i].get("phi_temp", 1.0) for i in range(N)]
        if len(set(phi_temps)) == 1:
            alpha, beta, gamma, delta, epsilon = weights
            if beta > 0:
                remaining = alpha + gamma + delta + epsilon
                if remaining > 0:
                    weights = (alpha/remaining, 0.0, gamma/remaining,
                               delta/remaining, epsilon/remaining)

    # Attach phi_dom arrays to graph nodes (needed by delta_full_fast)
    phi_dom_matrix = graph_data["phi_dom_matrix"]
    for i in range(N):
        G.nodes[i]["phi_dom"] = phi_dom_matrix[i]

    eps_cur = weights[4]
    best_result = None   # best (S, F, sigma, parent) seen so far

    for attempt in range(max_retries + 1):
        w = (weights[0], weights[1], weights[2], weights[3], eps_cur)
        S, F, sigma, parent = _single_pass(G, q_emb, q_dom, phi_sem_q, N, w, k_tok,
                                            enable_teleport=enable_teleport)

        # Keep track of best result by sigma (in case all retries fail)
        if best_result is None or sigma > best_result[2]:
            best_result = (S, F, sigma, parent)

        if sigma >= tau_low:
            break   # acceptable confidence achieved

        if attempt >= max_retries:
            break   # exhausted retries → return best with LOW flag

        # Re-traversal: increase ε (confidence weight), renormalise others
        eps_cur = min(EPSILON_MAX, eps_cur + EPSILON_STEP)
        remaining = 1.0 - eps_cur
        base_remaining = 1.0 - weights[4]
        if base_remaining > 0:
            scale = remaining / base_remaining
        else:
            scale = 1.0
        weights = (weights[0]*scale, weights[1]*scale,
                   weights[2]*scale, weights[3]*scale, eps_cur)

    S, F, sigma, parent = best_result
    retries_used = attempt  # number of retries attempted (0 = first pass succeeded)

    # Confidence tier
    if sigma >= tau_high:
        flag = "HIGH"
    elif sigma >= tau_low:
        flag = "HEDGE"
    else:
        flag = "LOW"

    return TraversalResult(S=S, F=F, sigma=sigma,
                           confidence_flag=flag,
                           retries=retries_used,
                           parent=parent)


# ── FIX 2: Multi-Anchor Initialization ────────────────────────────────────────
def run_pathfinder_multi_anchor(graph_data: dict,
                                 weights: tuple = None,
                                 k_tok: int = K_TOK,
                                 n_anchors: int = 3,
                                 tau_low: float = TAU_LOW,
                                 tau_high: float = TAU_HIGH,
                                 max_retries: int = MAX_RETRIES,
                                 auto_detect_uniform_temp: bool = True,
                                 enable_teleport: bool = True) -> TraversalResult:
    """
    Multi-anchor PATHFINDER: run from top-n_anchors entry points,
    return the result with highest F(S,q).

    Uses _single_pass directly (not run_pathfinder) so each anchor
    gets its own entry node rather than the same global argmax.
    """
    if weights is None:
        weights = (ALPHA, BETA, GAMMA, DELTA, EPSILON)

    G         = graph_data["G"]
    q_emb     = graph_data["q_emb"]
    q_dom     = graph_data["q_dom"]
    embs      = graph_data["embeddings"]
    N         = graph_data["N"]
    phi_sem_q = embs @ q_emb

    # FIX 3: Auto-detect uniform φ_temp → set β=0
    if auto_detect_uniform_temp:
        phi_temps = [G.nodes[i].get("phi_temp", 1.0) for i in range(N)]
        if len(set(phi_temps)) == 1:
            alpha, beta, gamma, delta, epsilon = weights
            if beta > 0:
                remaining = alpha + gamma + delta + epsilon
                if remaining > 0:
                    weights = (alpha/remaining, 0.0, gamma/remaining,
                               delta/remaining, epsilon/remaining)

    # Attach phi_dom arrays to graph nodes
    phi_dom_matrix = graph_data["phi_dom_matrix"]
    for i in range(N):
        G.nodes[i]["phi_dom"] = phi_dom_matrix[i]

    # Get top-n_anchors entry nodes
    ranked = np.argsort(phi_sem_q)[::-1]
    anchors = ranked[:n_anchors]

    best_result = None
    best_F = -1.0

    # Save original weights to reset for each anchor
    orig_weights = weights

    for v0 in anchors:
        # Reset to original weights for fair comparison across anchors
        weights = orig_weights

        # Create a copy of phi_sem_q with boosted similarity for this anchor
        # Use a value just above the current max to guarantee v0 is selected as entry
        # but NOT 1.0 (which would zero out the coverage product factor)
        phi_sem_q_copy = phi_sem_q.copy()
        current_max = float(phi_sem_q_copy.max())
        phi_sem_q_copy[v0] = current_max + 0.01  # guarantee v0 is the max entry node

        eps_cur = weights[4]
        local_best = None

        for attempt in range(max_retries + 1):
            w = (weights[0], weights[1], weights[2], weights[3], eps_cur)
            S, F, sigma, parent = _single_pass(G, q_emb, q_dom,
                                                phi_sem_q_copy, N, w, k_tok,
                                                enable_teleport=enable_teleport)

            if local_best is None or sigma > local_best[2]:
                local_best = (S, F, sigma, parent)

            if sigma >= tau_low:
                break

            if attempt >= max_retries:
                break

            eps_cur = min(EPSILON_MAX, eps_cur + EPSILON_STEP)
            remaining = 1.0 - eps_cur
            base_remaining = 1.0 - weights[4]
            scale = remaining / base_remaining if base_remaining > 0 else 1.0
            weights = (weights[0]*scale, weights[1]*scale,
                       weights[2]*scale, weights[3]*scale, eps_cur)

        S, F, sigma, parent = local_best
        if F > best_F:
            best_F = F
            best_result = (S, F, sigma, parent)

    S, F, sigma, parent = best_result

    # Confidence tier
    if sigma >= tau_high:
        flag = "HIGH"
    elif sigma >= tau_low:
        flag = "HEDGE"
    else:
        flag = "LOW"

    return TraversalResult(S=S, F=F, sigma=sigma,
                           confidence_flag=flag,
                           retries=0,  # multi-anchor doesn't track retries per anchor
                           parent=parent)


# ── FIX 4: Dense-Anchor Hybrid (default retrieval mode) ───────────────────────
def run_pathfinder_hybrid(graph_data: dict,
                          n_anchors: int = 3,
                          weights: tuple = None,
                          k_tok: int = K_TOK) -> TraversalResult:
    """
    Dense-Anchor Hybrid retrieval (Phase B1, promoted to default).

    Mechanism: take the top-`n_anchors` dense nodes as primary anchors (bypassing
    the frontier constraint for anchors), then run greedy marginal-gain frontier
    expansion from all anchors' neighbors to fill the remaining token budget.
    Merge + deduplicate. Matches Naive RAG's recall while preserving graph-coherent
    expansion context.

    Guarantee: the (1−1/e) coverage bound applies to the greedy EXPANSION phase
    per connected component anchored at a dense seed — NOT to the union across
    anchors (anchors are injected outside any single connected frontier). See
    paper §5 corollary (hybrid per-anchor guarantee).

    Returns:
        TraversalResult with S = anchors + expanded nodes in selection order.
    """
    if weights is None:
        weights = (ALPHA, BETA, GAMMA, DELTA, EPSILON)

    G         = graph_data["G"]
    q_emb     = graph_data["q_emb"]
    q_dom     = graph_data["q_dom"]
    embs      = graph_data["embeddings"]
    N         = graph_data["N"]
    phi_sem_q = embs @ q_emb

    if N == 0 or np.linalg.norm(q_emb) < 1e-8:
        return TraversalResult(S=[], F=0.0, sigma=1.0,
                               confidence_flag="HIGH", retries=0, parent={})

    # Attach phi_dom arrays
    phi_dom_matrix = graph_data["phi_dom_matrix"]
    for i in range(N):
        G.nodes[i]["phi_dom"] = phi_dom_matrix[i]

    # Top-n_anchors dense nodes as primary anchors
    ranked  = np.argsort(phi_sem_q)[::-1]
    anchors = [int(a) for a in ranked[:n_anchors]]

    S      = list(anchors)
    S_set  = set(anchors)
    parent = {a: None for a in anchors}

    tok            = sum(tok_count(a, G) for a in anchors)
    product_factor = 1.0
    for a in anchors:
        product_factor *= (1.0 - max(0.0, float(phi_sem_q[a])))

    # Frontier = union of all anchors' successors
    frontier: dict[int, int] = {}
    for a in anchors:
        for u in G.successors(a):
            if u not in S_set and u not in frontier:
                frontier[u] = a

    # Greedy marginal-gain expansion within budget
    while frontier and tok < k_tok:
        rem      = k_tok - tok
        feasible = {v: p for v, p in frontier.items()
                    if tok_count(v, G) <= rem}
        if not feasible:
            break

        best_v, best_gain = None, -1.0
        for v in feasible:
            g = delta_full_fast(v, G, q_dom, phi_sem_q, weights, product_factor)
            if g > best_gain:
                best_gain, best_v = g, v
        if best_v is None:
            break

        S.append(best_v)
        S_set.add(best_v)
        parent[best_v] = feasible[best_v]
        product_factor *= (1.0 - max(0.0, float(phi_sem_q[best_v])))
        tok += tok_count(best_v, G)

        del frontier[best_v]
        for u in G.successors(best_v):
            if u not in S_set and u not in frontier:
                frontier[u] = best_v

    sigma = compute_sigma(S, parent, G)
    F     = compute_F(S, G, q_dom, phi_sem_q, weights)
    flag  = "HIGH" if sigma >= TAU_HIGH else ("HEDGE" if sigma >= TAU_LOW else "LOW")

    return TraversalResult(S=S, F=F, sigma=sigma,
                           confidence_flag=flag, retries=0, parent=parent)


# ── FIX 5: LLM Reranking as first-class post-processing + semantic cache ─────
# Semantic cache: avoid re-calling the LLM for (near-)duplicate queries.
_RERANK_CACHE: dict = {}   # {rounded_query_embedding_bytes: reranked_S}
CACHE_THETA = 0.92         # cosine similarity threshold for a cache hit


def _cache_key(q_emb: np.ndarray) -> bytes:
    """Quantize embedding to 3 decimals for a stable, collision-tolerant key."""
    return np.round(np.asarray(q_emb, dtype=np.float32), 3).tobytes()


def _cache_lookup(q_emb: np.ndarray):
    """Return cached reranked S if a cached query has cosine ≥ CACHE_THETA."""
    q = np.asarray(q_emb, dtype=np.float32)
    qn = np.linalg.norm(q)
    if qn < 1e-8 or not _RERANK_CACHE:
        return None
    key = _cache_key(q_emb)
    if key in _RERANK_CACHE:
        return _RERANK_CACHE[key]
    # Soft match: compare against cached keys
    q_unit = q / qn
    for k_bytes, val in _RERANK_CACHE.items():
        cached = np.frombuffer(k_bytes, dtype=np.float32)
        cn = np.linalg.norm(cached)
        if cn < 1e-8:
            continue
        if float(np.dot(q_unit, cached / cn)) >= CACHE_THETA:
            return val
    return None


def llm_rerank(question: str,
               S: list[int],
               G: nx.DiGraph,
               client,
               q_emb: np.ndarray = None,
               model: str = "llama-3.3-70b-versatile",
               use_cache: bool = True) -> list[int]:
    """
    Rerank PATHFINDER's selected set S with an LLM (relevance to question).

    Post-retrieval monotone re-ordering of the *retrieved* set: it does NOT change
    the (1−1/e) coverage guarantee, which holds at retrieval time. See paper note
    on retrieval-time vs rerank-time guarantees.

    Uses a semantic cache (cosine ≥ 0.92 on q_emb) to avoid duplicate API calls.
    Falls back to original S on any error (never crashes the pipeline).
    """
    if not S or client is None:
        return S

    # Semantic cache hit → skip API call entirely
    if use_cache and q_emb is not None:
        cached = _cache_lookup(q_emb)
        if cached is not None and set(cached) == set(S):
            return cached

    passages = []
    for i, v in enumerate(S):
        text = G.nodes[v].get("text", "")[:200]
        passages.append(f"[{i}] {text}")

    prompt = (f"Question: {question}\n\n"
              f"Context passages (in PATHFINDER selection order):\n"
              + "\n".join(passages) +
              "\n\nRank these passages by relevance to answering the question. "
              "Return ONLY a comma-separated list of indices, most to least "
              "relevant. Example: 3,0,1,2,4\n\nRanking:")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        ranking_str = resp.choices[0].message.content.strip()
        indices  = [int(x.strip()) for x in ranking_str.split(",")
                    if x.strip().isdigit()]
        reranked = [S[i] for i in indices if i < len(S)]
        for v in S:                       # append any nodes the LLM dropped
            if v not in reranked:
                reranked.append(v)

        if use_cache and q_emb is not None:
            _RERANK_CACHE[_cache_key(q_emb)] = reranked
        return reranked
    except Exception as e:                # never break the pipeline on rerank
        print(f"  [llm_rerank] error (returning original order): {e}")
        return S


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pickle, sys
    path = "data/hotpotqa_graphs.pkl"
    try:
        with open(path, "rb") as f:
            records = pickle.load(f)
    except FileNotFoundError:
        print(f"No graph cache found at {path}. Run 01_build_kg.py first.")
        sys.exit(1)

    print(f"Loaded {len(records)} records. Running PATHFINDER on first 5...\n")
    for rec in records[:5]:
        gd  = rec["graph"]
        res = run_pathfinder(gd)
        print(f"Q: {rec['question'][:70]}...")
        print(f"   Nodes selected : {len(res.S)}  |  F={res.F:.4f}"
              f"  |  σ={res.sigma:.4f}  |  flag={res.confidence_flag}"
              f"  |  retries={res.retries}")
        print()
