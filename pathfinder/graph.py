"""
PATHFINDER — Knowledge graph construction (Sections 3.2, 7.2).

Builds a directed weighted knowledge graph G = (V, E, W, Φ) from a document set:
  - Sentence-level node segmentation
  - Semantic edges (cosine similarity ≥ θ_edge=0.3)
  - Entity co-mention edges (spaCy NER, W_entity=0.70)
  - Five-facet node representation (Section 3.3)
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional, Callable

from pathfinder.config import (
    THETA_EDGE, W_ENTITY, PHI_CONF_INIT, PCA_K, PR_DAMPING, PR_MAX_ITER,
)
from pathfinder.facets import (
    NodeFacets,
    compute_phi_temp,
    compute_phi_imp,
    compute_phi_dom,
    compute_q_dom,
)


@dataclass
class KnowledgeGraph:
    """
    Directed weighted knowledge graph G = (V, E, W, Φ).

    Attributes:
        G              : networkx.DiGraph with node/edge attributes
        nodes          : list of node metadata dicts {text, doc_title, sent_idx, ...}
        embeddings     : (N, D) L2-normalised semantic embeddings
        phi_dom_matrix : (N, K) domain embeddings
        W_dom          : (K, D) PCA projection matrix
        q_emb          : (D,) query embedding (L2-normalised)
        q_dom          : (K,) query domain embedding (L2-normalised)
        phi_imp        : (N,) structural importance scores
        N              : number of nodes
    """
    G:              nx.DiGraph
    nodes:          list[dict]
    embeddings:     np.ndarray
    phi_dom_matrix: np.ndarray
    W_dom:          np.ndarray
    q_emb:          np.ndarray
    q_dom:          np.ndarray
    phi_imp:        np.ndarray
    N:              int

    def sim_to_query(self, v: int) -> float:
        """sim(v, q) = max(0, cosine(φ_sem(v), q_emb)) for node v."""
        return float(self.G.nodes[v].get("sim_to_query", 0.0))

    def edge_weight(self, u: int, v: int) -> float:
        """W(u, v) for edge (u→v), floored at 0 (Section 3.2)."""
        if self.G.has_edge(u, v):
            return max(0.0, float(self.G[u][v].get("weight", 0.0)))
        return 0.0

    def out_neighbors(self, v: int) -> list[int]:
        """Out-neighbors of v in G."""
        return list(self.G.successors(v))

    def token_count(self, v: int) -> int:
        """tok(v) = token_count(v.content) — word count proxy (Section 3.4)."""
        text = self.G.nodes[v].get("text", "")
        return max(1, len(text.split()))


class KGBuilder:
    """
    Builds per-query knowledge graphs from a document set.

    Args:
        embedder       : callable(texts: list[str]) -> np.ndarray (N, D), L2-normalised
        nlp            : spaCy language pipeline for NER (or None to skip entity edges)
        theta_edge     : semantic edge admission threshold (default 0.3)
        w_entity       : entity co-mention edge weight (default 0.70)
        pca_k          : domain embedding dimensionality (default 16)
        phi_conf_init  : initial epistemic confidence (default 0.70)
        min_sent_len   : minimum word count for a sentence to become a node
    """

    def __init__(
        self,
        embedder: Callable[[list[str]], np.ndarray],
        nlp=None,
        theta_edge: float = THETA_EDGE,
        w_entity: float = W_ENTITY,
        pca_k: int = PCA_K,
        phi_conf_init: float = PHI_CONF_INIT,
        min_sent_len: int = 8,
    ):
        self.embedder = embedder
        self.nlp = nlp
        self.theta_edge = theta_edge
        self.w_entity = w_entity
        self.pca_k = pca_k
        self.phi_conf_init = phi_conf_init
        self.min_sent_len = min_sent_len

    def _segment(self, text: str) -> list[str]:
        """Sentence-tokenize a document; filter very short sentences."""
        from nltk.tokenize import sent_tokenize
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                pass
        sents = sent_tokenize(text.strip())
        return [s.strip() for s in sents if len(s.split()) >= self.min_sent_len]

    def _entities(self, text: str) -> set[str]:
        """Extract lowercased named entity strings (spaCy NER)."""
        if self.nlp is None:
            return set()
        doc = self.nlp(text[:512])
        return {ent.text.lower().strip() for ent in doc.ents
                if len(ent.text.strip()) > 2}

    def build(
        self,
        query: str,
        docs: list[dict],
        timestamps: Optional[list[float]] = None,
    ) -> Optional[KnowledgeGraph]:
        """
        Build per-query KG.

        Args:
            query      : question string
            docs       : list of {"title": str, "sentences": list[str]}
            timestamps : optional list of age_days per node (None → φ_temp=1.0)

        Returns:
            KnowledgeGraph or None if graph is empty.
        """
        # ── 1. Collect sentence nodes ──────────────────────────────────────
        nodes = []
        raw_texts = []

        for doc in docs:
            title = doc["title"]
            for sent_idx, sent in enumerate(doc["sentences"]):
                sent = sent.strip()
                if len(sent.split()) < self.min_sent_len:
                    continue
                nodes.append({
                    "text":      sent,
                    "doc_title": title,
                    "sent_idx":  sent_idx,
                    "phi_conf":  self.phi_conf_init,
                })
                raw_texts.append(sent)

        N = len(nodes)
        if N == 0:
            return None

        # ── 2. Embeddings ──────────────────────────────────────────────────
        embs = self.embedder(raw_texts)  # (N, D), should be L2-normalised
        q_emb = self.embedder([query])[0]

        # ── 3. Domain embeddings (PCA K) ───────────────────────────────────
        W_dom, phi_dom_matrix = compute_phi_dom(embs, k=self.pca_k)
        q_dom = compute_q_dom(W_dom, q_emb)

        # ── 4. Similarity matrix ───────────────────────────────────────────
        sim = embs @ embs.T          # (N, N) cosine similarities (L2-normalised)
        phi_sem_q = embs @ q_emb     # (N,) cosine sim to query

        # ── 5. Build DiGraph ───────────────────────────────────────────────
        G = nx.DiGraph()
        for i in range(N):
            G.add_node(
                i,
                text=nodes[i]["text"],
                doc_title=nodes[i]["doc_title"],
                sent_idx=nodes[i]["sent_idx"],
                phi_conf=nodes[i]["phi_conf"],
                phi_temp=compute_phi_temp(
                    timestamps[i] if timestamps else None
                ),
                phi_dom=phi_dom_matrix[i],
                sim_to_query=float(max(0.0, phi_sem_q[i])),
            )

        # Semantic edges: W(u,v) = max(0, cosine(φ_sem(u), φ_sem(v))) > θ_edge
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                w = float(max(0.0, sim[i, j]))
                if w > self.theta_edge:
                    if G.has_edge(i, j):
                        if G[i][j]["weight"] < w:
                            G[i][j]["weight"] = w
                    else:
                        G.add_edge(i, j, weight=w)

        # Entity co-mention edges: shared NER entity → W = max(existing, 0.70)
        if self.nlp is not None:
            entity_sets = [self._entities(n["text"]) for n in nodes]
            for i in range(N):
                for j in range(i + 1, N):
                    if entity_sets[i] & entity_sets[j]:
                        for u, v in [(i, j), (j, i)]:
                            if G.has_edge(u, v):
                                G[u][v]["weight"] = max(
                                    G[u][v]["weight"], self.w_entity
                                )
                            else:
                                G.add_edge(u, v, weight=self.w_entity)

        # ── 6. PageRank → φ_imp ────────────────────────────────────────────
        if G.number_of_edges() > 0:
            try:
                pr = nx.pagerank(
                    G, alpha=PR_DAMPING, max_iter=PR_MAX_ITER, weight="weight"
                )
            except (nx.PowerIterationFailedConvergence, Exception):
                pr = {i: 1.0 / N for i in range(N)}
        else:
            pr = {i: 1.0 / N for i in range(N)}

        pr_arr = np.array([pr[i] for i in range(N)])
        phi_imp = compute_phi_imp(pr_arr)

        for i in range(N):
            G.nodes[i]["phi_imp"] = float(phi_imp[i])

        return KnowledgeGraph(
            G=G,
            nodes=nodes,
            embeddings=embs,
            phi_dom_matrix=phi_dom_matrix,
            W_dom=W_dom,
            q_emb=q_emb,
            q_dom=q_dom,
            phi_imp=phi_imp,
            N=N,
        )
