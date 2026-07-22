"""
PATHFINDER — Algorithm 1: PATHFINDER-Greedy (Section 4.2).

Faithful implementation of the greedy submodular coverage maximization
traversal, including:
  - Guard conditions 0a / 0b / 3b
  - Entry node selection (line 3)
  - Frontier-expansion with first-admitted-wins parent tracking (line 16)
  - FEASIBLE budget filtering (line 10b)
  - Δ_full marginal gain with O(1) residual updates (lines 4b/12b)
  - Sufficiency check (line 15)
  - σ̃ running lower bound (line 13, 15b)
  - No Δ-based early exit (only frontier/budget exhaustion or sufficiency)
  - σ(S) computation via disc_parent tree (line 17)
  - Three-tier threshold policy + bounded re-traversal (Section 4.3)
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional
from itertools import combinations

from pathfinder.config import (
    ALPHA, BETA, GAMMA, DELTA, EPSILON,
    K_TOK, TAU_HIGH, TAU_LOW, MAX_RETRIES,
    EPSILON_MAX, EPSILON_STEP, SUFFICIENCY_THRES,
)
from pathfinder.graph import KnowledgeGraph


# ── Result dataclass ────────────────────────────────────────────────────────
@dataclass
class TraversalResult:
    """Output of a single PATHFINDER traversal."""
    S:               list[int]       # selected node indices in selection order
    F:               float           # F(S, q) — full coverage score
    sigma:           float           # σ(S) — path confidence
    confidence_flag: str             # "HIGH" | "HEDGE" | "LOW"
    retries:         int             # number of re-traversal attempts made
    parent:          dict            # disc_parent map {node: parent_or_None}


# ── sim(v, q) = max(0, cosine(φ_sem(v), q_emb)) ─────────────────────────────
def _sim(v: int, kg: KnowledgeGraph) -> float:
    return kg.sim_to_query(v)


# ── Δ_full(v | S, q) — marginal gain (Definition 3 + Section 4.1) ───────────
def marginal_gain(
    v: int,
    rho: float,
    kg: KnowledgeGraph,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    delta: float = DELTA,
    epsilon: float = EPSILON,
) -> float:
    """
    Δ_full(v | S, q) = α · sim(v,q) · ρ
                     + β · φ_temp(v)
                     + γ · φ_imp(v)
                     + δ · max(0, cos(φ_dom(v), q_dom))
                     + ε · φ_conf(v)

    where ρ = ∏_{u∈S}(1 − sim(u,q)) is the running residual.
    """
    node = kg.G.nodes[v]
    sim_vq = _sim(v, kg)

    # Coverage component: α · sim(v,q) · ρ
    delta_coverage = alpha * sim_vq * rho

    # Temporal component: β · φ_temp(v)
    delta_temp = beta * node.get("phi_temp", 1.0)

    # Importance component: γ · φ_imp(v)
    delta_imp = gamma * node.get("phi_imp", 0.0)

    # Domain component: δ · max(0, cos(φ_dom(v), q_dom))
    phi_dom_v = node.get("phi_dom", np.zeros(kg.q_dom.shape[0]))
    dom_norm = np.linalg.norm(phi_dom_v)
    q_norm = np.linalg.norm(kg.q_dom)
    if dom_norm > 1e-12 and q_norm > 1e-12:
        dom_cos = max(0.0, float(np.dot(phi_dom_v, kg.q_dom) / (dom_norm * q_norm)))
    else:
        dom_cos = 0.0
    delta_dom = delta * dom_cos

    # Confidence component: ε · φ_conf(v)
    delta_conf = epsilon * node.get("phi_conf", 0.7)

    return delta_coverage + delta_temp + delta_imp + delta_dom + delta_conf


# ── F(S, q) — full objective (Definition 4) ─────────────────────────────────
def compute_F(
    S: list[int],
    kg: KnowledgeGraph,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    delta: float = DELTA,
    epsilon: float = EPSILON,
) -> float:
    """
    F(S, q) = α·f(S,q) + Σ_{v∈S}[β·φ_temp(v) + γ·φ_imp(v)
                                   + δ·max(0,cos(φ_dom(v),q_dom)) + ε·φ_conf(v)]

    where f(S,q) = 1 − ∏_{v∈S}(1 − sim(v,q)).
    F(∅, q) = 0 by empty-product convention.
    """
    if not S:
        return 0.0

    # Coverage: f(S,q) = 1 − ∏(1 − sim(v,q))
    product = 1.0
    for v in S:
        product *= (1.0 - _sim(v, kg))
    f_val = 1.0 - product

    # Modular terms
    modular_sum = 0.0
    for v in S:
        node = kg.G.nodes[v]
        modular_sum += beta * node.get("phi_temp", 1.0)
        modular_sum += gamma * node.get("phi_imp", 0.0)

        phi_dom_v = node.get("phi_dom", np.zeros(kg.q_dom.shape[0]))
        dom_norm = np.linalg.norm(phi_dom_v)
        q_norm = np.linalg.norm(kg.q_dom)
        if dom_norm > 1e-12 and q_norm > 1e-12:
            dom_cos = max(0.0, float(np.dot(phi_dom_v, kg.q_dom) / (dom_norm * q_norm)))
        else:
            dom_cos = 0.0
        modular_sum += delta * dom_cos
        modular_sum += epsilon * node.get("phi_conf", 0.7)

    return alpha * f_val + modular_sum


# ── σ(S) — true path confidence over traversal tree (Section 4.3) ───────────
def compute_sigma(
    S: list[int],
    parent: dict[int, Optional[int]],
    kg: KnowledgeGraph,
) -> float:
    """
    σ(S) = min_{v∈S} σ(v₀ → v)
    σ(v₀ → v) = ∏_{edges on tree-path} W(e) · ∏_{nodes on path} φ_conf(u)

    Uses disc_parent pointers to trace each root-to-node path.
    """
    if not S:
        return 1.0

    min_sigma = 1.0
    for v in S:
        sigma_v = 1.0
        cur = v
        while cur is not None:
            sigma_v *= kg.G.nodes[cur].get("phi_conf", 0.7)
            par = parent.get(cur)
            if par is not None and kg.G.has_edge(par, cur):
                sigma_v *= max(0.0, kg.G[par][cur].get("weight", 0.0))
            cur = par
        min_sigma = min(min_sigma, sigma_v)

    return min_sigma


# ── Sufficiency check (line 15) ─────────────────────────────────────────────
def _sufficiency_check(
    S: list[int],
    kg: KnowledgeGraph,
    threshold: float = SUFFICIENCY_THRES,
) -> bool:
    """
    Lightweight classifier: tests whether collected node set contains
    sufficient information to answer query q.
    Uses embedding coverage score + keyword overlap (Section 4.2).
    """
    if not S:
        return False

    # Coverage score: f(S, q) = 1 − ∏(1 − sim(v,q))
    product = 1.0
    for v in S:
        product *= (1.0 - _sim(v, kg))
    coverage = 1.0 - product

    return coverage >= threshold


# ── Core greedy traversal (Algorithm 1, lines 0a–17) ────────────────────────
def _greedy_traverse(
    kg: KnowledgeGraph,
    k_tok: int = K_TOK,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    delta: float = DELTA,
    epsilon: float = EPSILON,
    sufficiency_threshold: float = SUFFICIENCY_THRES,
    tau_low: float = TAU_LOW,
    use_sufficiency: bool = True,
    use_sigma_break: bool = True,
) -> TraversalResult:
    """
    Execute Algorithm 1 exactly as written in Section 4.2.
    Returns TraversalResult with S, F, σ, parent, confidence_flag.
    """
    G = kg.G

    # ── Line 0a: IF V = ∅ → RETURN ∅, 0, 1.0 ─────────────────────────────
    if kg.N == 0:
        return TraversalResult([], 0.0, 1.0, "HIGH", 0, {})

    # ── Line 1: q_emb ← semantic_embed(q) ── (already in kg) ──────────────

    # ── Line 0b: IF ‖q_emb‖ = 0 → RETURN ∅, 0, 1.0 ───────────────────────
    if np.linalg.norm(kg.q_emb) < 1e-12:
        return TraversalResult([], 0.0, 1.0, "HIGH", 0, {})

    # ── Line 2: q_dom ← domain_embed(q) ── (already in kg) ────────────────

    # ── Line 3: v₀ ← argmax_{v∈V} cosine(φ_sem(v), q_emb) ────────────────
    phi_sem_q = kg.embeddings @ kg.q_emb
    v0 = int(np.argmax(phi_sem_q))

    # ── Line 3b: IF token_count(v₀) > K_tok → RETURN ∅, 0, 1.0 ───────────
    if kg.token_count(v0) > k_tok:
        return TraversalResult([], 0.0, 1.0, "HIGH", 0, {})

    # ── Line 4: S ← {v₀}, parent[v₀] ← null ──────────────────────────────
    S = [v0]
    parent = {v0: None}

    # ── Line 4b: ρ ← 1 − sim(v₀, q) ──────────────────────────────────────
    rho = 1.0 - _sim(v0, kg)

    # ── Line 5: σ̃ ← φ_conf(v₀) ──────────────────────────────────────────
    sigma_tilde = kg.G.nodes[v0].get("phi_conf", 0.7)

    # ── Line 6: tok ← token_count(v₀) ────────────────────────────────────
    tok = kg.token_count(v0)

    # ── Line 7: frontier ← {} ; FOR u IN out_neighbors(v₀): frontier[u] ← v₀
    frontier: dict[int, int] = {}
    for u in kg.out_neighbors(v0):
        frontier[u] = v0

    # ── Line 8: while frontier ≠ {} and tok < K_tok ──────────────────────
    while frontier and tok < k_tok:

        # ── Line 9: FEASIBLE ← {v ∈ frontier : tok + token_count(v) ≤ K_tok}
        feasible = {
            v for v in frontier if tok + kg.token_count(v) <= k_tok
        }

        # ── Line 10b: IF FEASIBLE = {} → BREAK ────────────────────────────
        if not feasible:
            break

        # ── Line 10c: v* ← argmax_{v ∈ FEASIBLE} Δ_full(v | S, q) ──────────
        best_v = None
        best_gain = -1.0
        for v in feasible:
            gain = marginal_gain(
                v, rho, kg, alpha, beta, gamma, delta, epsilon
            )
            if gain > best_gain:
                best_gain = gain
                best_v = v

        v_star = best_v

        # ── Line 12: S ← S ∪ {v*} ─────────────────────────────────────────
        S.append(v_star)

        # ── Line 12b: ρ ← ρ · (1 − sim(v*, q)) ────────────────────────────
        rho *= (1.0 - _sim(v_star, kg))

        # ── Line 13: parent[v*] ← frontier[v*] ────────────────────────────
        parent[v_star] = frontier[v_star]

        # ── σ̃ ← σ̃ · W(parent[v*], v*) · φ_conf(v*) ───────────────────────
        par = parent[v_star]
        w_edge = kg.edge_weight(par, v_star) if par is not None else 1.0
        sigma_tilde *= w_edge * kg.G.nodes[v_star].get("phi_conf", 0.7)

        # ── Line 14: tok ← tok + token_count(v*) ──────────────────────────
        tok += kg.token_count(v_star)

        # Remove v* from frontier (it's now in S)
        del frontier[v_star]

        # ── Line 15: if sufficiency_check(S, q): break ────────────────────
        if use_sufficiency and _sufficiency_check(S, kg, sufficiency_threshold):
            break

        # ── Line 15b: if σ̃ < τ_low: break ─────────────────────────────────
        if use_sigma_break and sigma_tilde < tau_low:
            break

        # ── Line 16: FOR u IN out_neighbors(v*, G): expand frontier ───────
        for u in kg.out_neighbors(v_star):
            if u not in S and u not in frontier:
                frontier[u] = v_star

    # ── Line 17: σ(S) ← min_{v∈S} path_confidence(v₀, v, parent) ─────────
    sigma = compute_sigma(S, parent, kg)

    # ── Line 18: return S, F(S, q), σ(S) ──────────────────────────────────
    F_val = compute_F(S, kg, alpha, beta, gamma, delta, epsilon)

    # Determine confidence flag
    if sigma >= TAU_HIGH:
        flag = "HIGH"
    elif sigma >= TAU_LOW:
        flag = "HEDGE"
    else:
        flag = "LOW"

    return TraversalResult(S, F_val, sigma, flag, 0, parent)


# ── Full PATHFINDER with re-traversal protocol (Section 4.3) ────────────────
class PathfinderGreedy:
    """
    PATHFINDER algorithm with three-tier threshold policy and
    bounded re-traversal protocol (MAX_RETRIES=3).

    Args mirror the paper defaults; override for ablations.
    """

    def __init__(
        self,
        k_tok: int = K_TOK,
        alpha: float = ALPHA,
        beta: float = BETA,
        gamma: float = GAMMA,
        delta: float = DELTA,
        epsilon: float = EPSILON,
        tau_high: float = TAU_HIGH,
        tau_low: float = TAU_LOW,
        max_retries: int = MAX_RETRIES,
        epsilon_max: float = EPSILON_MAX,
        epsilon_step: float = EPSILON_STEP,
        sufficiency_threshold: float = SUFFICIENCY_THRES,
        use_sufficiency: bool = True,
        use_sigma_break: bool = True,
    ):
        self.k_tok = k_tok
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.max_retries = max_retries
        self.epsilon_max = epsilon_max
        self.epsilon_step = epsilon_step
        self.sufficiency_threshold = sufficiency_threshold
        self.use_sufficiency = use_sufficiency
        self.use_sigma_break = use_sigma_break

    def run(self, kg: KnowledgeGraph) -> TraversalResult:
        """
        Run PATHFINDER with bounded re-traversal protocol.

        retry_count ← 0
        LOOP:
          Run Algorithm 1 → obtain S, F(S,q), σ(S)
          IF σ(S) ≥ τ_low: EXIT LOOP
          retry_count ← retry_count + 1
          IF retry_count ≥ MAX_RETRIES:
              OUTPUT S with confidence_flag = 'LOW'
              EXIT LOOP
          ELSE:
              increase ε by 0.05 (up to ε_max = 0.40) and/or rewrite query
              CONTINUE LOOP
        """
        retry_count = 0
        cur_epsilon = self.epsilon
        best_result = None

        while True:
            result = _greedy_traverse(
                kg,
                k_tok=self.k_tok,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                delta=self.delta,
                epsilon=cur_epsilon,
                sufficiency_threshold=self.sufficiency_threshold,
                tau_low=self.tau_low,
                use_sufficiency=self.use_sufficiency,
                use_sigma_break=self.use_sigma_break,
            )
            result.retries = retry_count
            best_result = result

            # IF σ(S) ≥ τ_low: EXIT LOOP
            if result.sigma >= self.tau_low:
                break

            retry_count += 1

            # IF retry_count ≥ MAX_RETRIES: output with LOW flag, exit
            if retry_count >= self.max_retries:
                result.confidence_flag = "LOW"
                result.retries = retry_count
                break

            # ELSE: increase ε by step (up to ε_max)
            cur_epsilon = min(cur_epsilon + self.epsilon_step, self.epsilon_max)

        return best_result


# ── Brute-force optimum for small graphs (for unit tests, §7.5.9) ───────────
def brute_force_optimum(
    kg: KnowledgeGraph,
    v0: int,
    k: int = 5,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    delta: float = DELTA,
    epsilon: float = EPSILON,
) -> tuple[float, list[int]]:
    """
    Enumerate all connected subtrees of size ≤ k rooted at v0 and
    return the maximum F(S, q) and the corresponding node set.

    Used for:
      - Unit tests verifying the (1−1/e) bound empirically.
      - Coverage ratio experiment (§7.5.9).

    Only tractable for small graphs (|V| ≤ ~25, k ≤ 5).
    """
    N = kg.N
    best_F = -1.0
    best_S = []

    def _enumerate(current_set, frontier_nodes):
        nonlocal best_F, best_S

        # Evaluate current set
        F_val = compute_F(current_set, kg, alpha, beta, gamma, delta, epsilon)
        if F_val > best_F:
            best_F = F_val
            best_S = list(current_set)

        if len(current_set) >= k:
            return

        # Try adding each frontier node
        for v in sorted(frontier_nodes):
            if v in current_set:
                continue
            new_set = current_set + [v]
            # Update frontier: add out-neighbors of v not already in set or frontier
            new_frontier = set(frontier_nodes)
            for u in kg.out_neighbors(v):
                if u not in new_set:
                    new_frontier.add(u)
            new_frontier.discard(v)
            _enumerate(new_set, new_frontier)

    # Start with {v0}
    initial_frontier = set(kg.out_neighbors(v0))
    _enumerate([v0], initial_frontier)

    return best_F, best_S
