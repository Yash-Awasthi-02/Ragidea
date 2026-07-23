# PATHFINDER — Future Research & Engineering Extensions

This document outlines key future work, open theoretical problems, and planned empirical benchmarks for the **PATHFINDER** retrieval framework.

---

## 1. Multi-Benchmark Evaluation & Scale

- **Full Multi-Hop Suite Evaluation:**
  While full HotpotQA validation evaluation ($N=7,405$) is complete, full benchmark loader support for **2WikiMultihopQA** ($N=12,576$) and **MuSiQue** ($N=2,417$) is now implemented (`experiments/02_load_2wiki_musique.py`) for cross-dataset evaluation. Phase 2 introduces multi-granularity metrics (Recall@10, Recall@20, Paragraph-Recall@k) to fairly evaluate MuSiQue's 4-hop paragraph retrieval.
- **Heterogeneous Generator LLMs:**
  Evaluate PATHFINDER's retrieved context subtrees across a broader spectrum of open-source and proprietary generator models (e.g., Llama-3-70B, Qwen-2.5-72B, Claude 3.5 Sonnet, GPT-4o).
- **Teleportation Hybrid Evaluation:**
  Full evaluation of the dynamic dense-frontier teleportation operator (§4.2) across all three benchmarks, comparing Pure Graph vs Teleportation Hybrid vs Naive RAG. The teleportation operator is expected to close the gap on HotpotQA and 2Wiki where graph connectivity is sparse.

---

## 2. Advanced Path Confidence ($\sigma$) Calibration

- **Bottleneck / Min-Edge Confidence ($\sigma_{\text{min}}$):**
  Implemented fuzzy bottleneck confidence aggregation ($\sigma_{\text{min}}(S) = \min_{v \in S} \min_{e,u \in \text{path}} \{W(e), \phi_{\text{conf}}(u)\}$ in `run_pathfinder.py`) alongside path-product confidence to prevent exponential path decay over deep multi-hop traversals.
- **Geometric Mean Confidence ($\sigma_{\text{geom}}$):**
  Implemented depth-normalized confidence ($\sigma_{\text{geom}}(S) = \min_{v \in S} (\prod W(e) \cdot \prod \phi_{\text{conf}}(u))^{1/L}$) that prevents exponential decay from penalizing legitimate multi-hop reasoning chains.
- **NLI & Entailment Verification:**
  Incorporate lightweight Natural Language Inference (NLI) step verification along traversal branches during early-exit gating. This would replace the current heuristic sufficiency check (line 15) with a learned entailment model that verifies whether the selected context actually supports answering the query.
- **Confidence Model Selection Framework:**
  Develop a principled framework for selecting between product, geometric mean, and bottleneck confidence models based on query characteristics (hop depth, graph connectivity, domain). The grid search results (§7.6.5) provide initial empirical guidance, but an automated selection mechanism is needed.

---

## 3. Theoretical Open Problems

- **Tightness of the Frontier-Constrained $(1 - 1/e)$ Bound:**
  Formally establish matching lower bounds for frontier-constrained submodular connected-subtree selection on general non-tree graphs, analogous to the Feige (1998) construction for set cover.
- **Teleportation Operator Formal Analysis:**
  Formally prove that the teleportation operator preserves the $(1-1/e)$ approximation guarantee. The key insight is that teleportation only *expands* the candidate frontier; the greedy selection criterion and budget constraint remain unchanged. However, a rigorous proof showing that $S^*_{\text{frontier}}$ (the optimal connected-subtree solution) remains a valid comparator under the expanded frontier requires careful treatment of the teleportation nodes' parent assignments.
- **Joint Correlation Modeling:**
  Extend the independent coverage model $f(S,q) = 1 - \prod(1 - \text{sim}(v,q))$ to explicitly incorporate positive edge correlations via Markov Random Fields (MRFs) or graph neural message passing.

---

## 4. Architectural & System Extensions

- **Cross-Domain Facet Learning:**
  Replace static weight vectors $(\alpha, \beta, \gamma, \delta, \epsilon)$ with online per-domain gradient descent learning driven by downstream answer verification feedback. The grid search (§7.6.5) provides the search space; the feedback loop (§4.5) provides the online learning mechanism. The foundation is the `pathfinder/feedback.py` module, which already implements online edge weight and confidence updates. The extension requires:
  1. A domain classifier to partition queries into domain buckets.
  2. Per-domain weight vectors with gradient updates from grounding scores.
  3. A multi-armed bandit or Thompson sampling mechanism to explore weight configurations during cold-start.
- **Multi-Vector Dense-Frontier Teleportation:**
  Formally integrate single-vector ANN retrieval (e.g., FAISS / HNSW) with submodular frontier expansion to allow dynamic "teleportation" jumps across disconnected document clusters when knowledge graph edges are sparse or missing. The Phase 2 teleportation operator (§4.2) provides the algorithmic foundation; the extension requires:
  1. A FAISS/HNSW index over all node embeddings, built at index time.
  2. O(log|V|) ANN lookup during traversal to find teleportation candidates.
  3. A learned threshold for when to trigger teleportation (currently fixed at θ_teleport = 0.01).
  4. Multi-vector teleportation using domain embeddings (φ_dom) in addition to semantic embeddings (φ_sem) for more targeted jumps.
- **Dynamic Edge Synthesis:**
  When teleportation reveals that two disconnected clusters are both relevant to the query, synthesize a new graph edge between them. This would gradually improve graph connectivity over time, reducing the need for teleportation in future queries on similar topics. The feedback loop (§4.5) provides the online update mechanism; the extension requires an edge creation policy that balances exploration (adding edges) with exploitation (using existing structure).
- **LLM-in-the-Loop Reranking:**
  After PATHFINDER retrieves a candidate set S, use a lightweight LLM to rerank nodes based on multi-hop reasoning chain quality. This combines the structural guarantees of submodular coverage with the semantic understanding of LLMs. The reranking could be implemented as a secondary scoring function that adjusts the greedy selection order.

---

## 5. Empirical Findings & Next Steps

- **Dense Retrieval Dominance on Disconnected Graphs:** Naive RAG outperforms pure graph traversal at Recall@5 on HotpotQA (0.310 vs 0.268) and 2Wiki (0.304 vs 0.226) because inter-document entity links are sparse. However, PATHFINDER surpasses Naive RAG at Recall@10 on both datasets (HotpotQA: 0.350 vs 0.310, 2Wiki: 0.334 vs 0.304), showing graph traversal discovers relevant nodes that dense retrieval misses with a slightly larger budget. The teleportation operator is designed to close the k=5 gap.
- **MuSiQue Metric Challenge:** Sentence-level Recall@5 produces near-zero scores for MuSiQue due to 4-hop queries with many supporting sentences. Paragraph-level Recall@5 (0.617) and Fractional Recall@5 (0.269) provide more meaningful signal, confirming relevant content is being retrieved.
- **Confidence Decay in Deep Multi-Hop:** Product confidence σ_prod collapses to 0.0077 on deep paths, confirming exponential decay. Geometric mean (min=0.2718) and bottleneck (min=0.3001) models address this (Phase 2, Task 2.3).
