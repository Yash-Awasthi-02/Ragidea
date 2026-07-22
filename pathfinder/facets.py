"""
PATHFINDER — Five-facet node representation (Section 3.3).

Each node v ∈ V is characterized by five independent facets:
  φ_sem(v)  ∈ ℝ^D    Semantic embedding
  φ_temp(v) ∈ (0,1]  Temporal recency score
  φ_imp(v)  ∈ [0,1]  Structural importance
  φ_dom(v)  ∈ ℝ^K    Domain classification embedding
  φ_conf(v) ∈ [0,1]  Epistemic confidence
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from pathfinder.config import (
    LAMBDA_TEMP, PR_DAMPING, PR_MAX_ITER, PHI_CONF_INIT, PCA_K,
)


@dataclass
class NodeFacets:
    """Container for the five facets of a single node."""
    phi_sem:  np.ndarray   # (D,) L2-normalised embedding
    phi_temp: float = 1.0  # default when timestamps unavailable (§7.3)
    phi_imp:  float = 0.0  # set by PageRank normalisation
    phi_dom:  np.ndarray = field(default_factory=lambda: np.zeros(PCA_K))
    phi_conf: float = PHI_CONF_INIT

    def domain_cos(self, q_dom: np.ndarray) -> float:
        """max(0, cos(φ_dom(v), q_dom)) — floored domain alignment (Definition 4)."""
        dom_norm = np.linalg.norm(self.phi_dom)
        q_norm   = np.linalg.norm(q_dom)
        if dom_norm < 1e-12 or q_norm < 1e-12:
            return 0.0
        cos_val = float(np.dot(self.phi_dom, q_dom) / (dom_norm * q_norm))
        return max(0.0, cos_val)

    def sim_to_query(self, q_emb: np.ndarray) -> float:
        """sim(v, q) = max(0, cosine(φ_sem(v), q_emb)) ∈ [0,1] (Definition 2)."""
        sem_norm = np.linalg.norm(self.phi_sem)
        q_norm   = np.linalg.norm(q_emb)
        if sem_norm < 1e-12 or q_norm < 1e-12:
            return 0.0
        cos_val = float(np.dot(self.phi_sem, q_emb) / (sem_norm * q_norm))
        return max(0.0, cos_val)


def compute_phi_temp(age_days: float, lam: float = LAMBDA_TEMP) -> float:
    """
    φ_temp(v) = exp(-λ · age_days(v))  (Section 3.3).
    Half-life = ln(2)/λ ≈ 13.86 days at default λ=0.05.
    Returns 1.0 when age_days is None (no timestamp available, §7.3).
    """
    if age_days is None:
        return 1.0
    return float(np.exp(-lam * age_days))


def compute_phi_imp(
    pagerank_scores: np.ndarray,
) -> np.ndarray:
    """
    φ_imp(v) = (PR(v) - min_u) / (max_u - min_u)  (Section 3.3).
    Edge case: when max_u = min_u, φ_imp(v) = 0.5 for all v.
    """
    pr_min = pagerank_scores.min()
    pr_max = pagerank_scores.max()
    if pr_max - pr_min < 1e-15:
        return np.full_like(pagerank_scores, 0.5)
    return (pagerank_scores - pr_min) / (pr_max - pr_min)


def compute_phi_dom(
    embeddings: np.ndarray,
    k: int = PCA_K,
) -> tuple[np.ndarray, np.ndarray]:
    """
    φ_dom(v) = W_dom · φ_sem(v), where W_dom is the top-K PCA components
    of the node embedding matrix E (Section 3.3).

    Returns:
        W_dom          : (K, D) PCA component matrix
        phi_dom_matrix : (N, K) domain embeddings (L2-normalised per row)
    """
    from sklearn.decomposition import PCA

    N, D = embeddings.shape
    k_actual = min(k, N - 1, D)  # can't have more components than min(N-1, D)
    k_actual = max(1, k_actual)

    pca = PCA(n_components=k_actual)
    pca.fit(embeddings)
    W_dom = pca.components_                    # (k_actual, D)
    phi_dom_matrix = pca.transform(embeddings)  # (N, k_actual)

    # L2-normalise each row
    row_norms = np.linalg.norm(phi_dom_matrix, axis=1, keepdims=True)
    row_norms[row_norms < 1e-12] = 1.0
    phi_dom_matrix = phi_dom_matrix / row_norms

    return W_dom, phi_dom_matrix


def compute_q_dom(W_dom: np.ndarray, q_emb: np.ndarray) -> np.ndarray:
    """q_dom = W_dom · q_emb, L2-normalised (Section 3.3)."""
    q_dom_raw = W_dom @ q_emb
    q_norm = np.linalg.norm(q_dom_raw)
    if q_norm < 1e-12:
        return q_dom_raw
    return q_dom_raw / q_norm
