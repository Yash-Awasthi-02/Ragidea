"""
PATHFINDER — Online Feedback Loop (Section 4.5).

After each LLM generation, PATHFINDER evaluates answer grounding and updates:
  - Edge weights W via online gradient descent
  - Node confidences φ_conf via online gradient descent
  - Path crystallization into procedural skills (optional)
  - W_dom adaptation (optional, infrequent)
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from collections import defaultdict, deque

from pathfinder.config import ETA_EDGE, MU_CONF, ETA_W_DOM
from pathfinder.graph import KnowledgeGraph


@dataclass
class SkillStore:
    """
    Procedural memory: crystallized traversal patterns.
    When a path structural pattern successfully resolves a query class
    in ≥ N=5 of the last M=8 instances, store as a named skill.
    """
    skills: dict = field(default_factory=dict)
    # Rolling window: {query_class: deque of (success, path_pattern)}
    history: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=8)))

    N_CRYSTALLIZE = 5  # min successes
    M_WINDOW = 8       # rolling window size

    def record(self, query_class: str, path_pattern: tuple, success: bool):
        """Record a traversal outcome for potential crystallization."""
        self.history[query_class].append((success, path_pattern))

        # Check crystallization condition
        window = self.history[query_class]
        successes = sum(1 for s, _ in window if s)
        if successes >= self.N_CRYSTALLIZE and len(window) >= self.N_CRYSTALLIZE:
            # Crystallize
            successful_patterns = [p for s, p in window if s]
            if successful_patterns:
                # Most common pattern
                from collections import Counter
                pattern_counts = Counter(successful_patterns)
                most_common = pattern_counts.most_common(1)[0][0]
                sigmas = [1.0]  # placeholder; would track actual σ values
                skill = {
                    "name": f"{query_class} → {most_common}",
                    "entry_type": most_common[0] if most_common else None,
                    "hop_pattern": most_common,
                    "avg_sigma": float(np.mean(sigmas)),
                    "use_count": successes,
                }
                self.skills[query_class] = skill

    def lookup(self, query_class: str) -> dict | None:
        """Check if a crystallized skill exists for this query class."""
        return self.skills.get(query_class)


class FeedbackLoop:
    """
    Online feedback loop for edge weight and confidence updates.

    After each LLM generation, evaluates answer grounding and updates
    the knowledge graph's edge weights and node confidences.
    """

    def __init__(
        self,
        eta_edge: float = ETA_EDGE,
        mu_conf: float = MU_CONF,
        eta_w_dom: float = ETA_W_DOM,
    ):
        self.eta_edge = eta_edge
        self.mu_conf = mu_conf
        self.eta_w_dom = eta_w_dom
        self.skill_store = SkillStore()
        self.update_count = 0

    def update(
        self,
        kg: KnowledgeGraph,
        S: list[int],
        parent: dict[int, int | None],
        grounding_score: float,
        query_class: str = "default",
        path_pattern: tuple | None = None,
    ):
        """
        Update edge weights and node confidences based on grounding score g.

        Edge weight update (Section 4.5):
            W'(vᵢ, vᵢ₊₁) ← W(vᵢ, vᵢ₊₁) · (1−η) + g·η    η = 0.05

        Confidence update (Section 4.5):
            φ'_conf(v) ← φ_conf(v) · (1−μ) + g·μ            μ = 0.03
        """
        g = max(0.0, min(1.0, grounding_score))

        # ── Edge weight updates ────────────────────────────────────────────
        for v in S:
            par = parent.get(v)
            if par is not None and kg.G.has_edge(par, v):
                old_w = kg.G[par][v].get("weight", 0.0)
                new_w = old_w * (1 - self.eta_edge) + g * self.eta_edge
                kg.G[par][v]["weight"] = max(0.0, min(1.0, new_w))

        # ── Confidence updates ─────────────────────────────────────────────
        for v in S:
            old_conf = kg.G.nodes[v].get("phi_conf", 0.7)
            new_conf = old_conf * (1 - self.mu_conf) + g * self.mu_conf
            kg.G.nodes[v]["phi_conf"] = max(0.0, min(1.0, new_conf))

        # ── Path crystallization ───────────────────────────────────────────
        if path_pattern is not None:
            success = g >= 0.5
            self.skill_store.record(query_class, path_pattern, success)

        self.update_count += 1

    def update_w_dom(
        self,
        kg: KnowledgeGraph,
        q_emb: np.ndarray,
        v_responsible: int,
        grounding_score: float,
    ):
        """
        Optional W_dom adaptation via projected gradient descent (Section 4.5).
        Updated at most once per N=100 traversals.
        """
        if self.update_count % 100 != 0:
            return

        g = max(0.0, min(1.0, grounding_score))
        q_dom_error = kg.q_dom - kg.W_dom @ q_emb  # reconstruction error
        phi_sem_v = kg.embeddings[v_responsible]

        # Gradient step
        grad = g * np.outer(q_dom_error, phi_sem_v)
        W_dom_new = kg.W_dom + self.eta_w_dom * grad

        # Project onto Stiefel manifold (orthonormal rows) via QR
        # Simple approximation: SVD-based projection
        U, _, Vt = np.linalg.svd(W_dom_new, full_matrices=False)
        kg.W_dom = U @ Vt  # closest matrix with orthonormal rows

        # Recompute domain embeddings
        from pathfinder.facets import compute_phi_dom, compute_q_dom
        # Actually just reproject
        phi_dom_new = kg.W_dom @ kg.embeddings.T  # (K, N)
        kg.phi_dom_matrix = phi_dom_new.T  # (N, K)
        # L2-normalise
        row_norms = np.linalg.norm(kg.phi_dom_matrix, axis=1, keepdims=True)
        row_norms[row_norms < 1e-12] = 1.0
        kg.phi_dom_matrix = kg.phi_dom_matrix / row_norms

        # Update q_dom
        kg.q_dom = compute_q_dom(kg.W_dom, q_emb)

        # Update node phi_dom in graph
        for i in range(kg.N):
            kg.G.nodes[i]["phi_dom"] = kg.phi_dom_matrix[i]


def compute_grounding_score(
    answer: str,
    S: list[int],
    kg: KnowledgeGraph,
) -> float:
    """
    Compute grounding score g ∈ [0, 1] — whether key answer spans
    are supported by nodes on the traversal path.

    Simple implementation: fraction of answer tokens that appear
    in the retrieved node texts.
    """
    if not answer or not S:
        return 0.0

    answer_tokens = set(answer.lower().split())
    if not answer_tokens:
        return 0.0

    context_tokens = set()
    for v in S:
        text = kg.G.nodes[v].get("text", "").lower()
        context_tokens.update(text.split())

    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens) if answer_tokens else 0.0
