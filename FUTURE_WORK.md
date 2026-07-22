# PATHFINDER — Future Research & Engineering Extensions

This document outlines key future work, open theoretical problems, and planned empirical benchmarks for the **PATHFINDER** retrieval framework.

---

## 1. Multi-Benchmark Evaluation & Scale

- **Full Multi-Hop Suite Evaluation:**
  While full HotpotQA validation evaluation ($N=7,405$) is complete, full benchmark evaluations for **2WikiMultihopQA** ($N=12,576$) and **MuSiQue** ($N=2,417$) are planned to assess cross-dataset generalizability.
- **Heterogeneous Generator LLMs:**
  Evaluate PATHFINDER's retrieved context subtrees across a broader spectrum of open-source and proprietary generator models (e.g., Llama-3-70B, Qwen-2.5-72B, Claude 3.5 Sonnet, GPT-4o).

---

## 2. Advanced Path Confidence ($\sigma$) Calibration

- **Bottleneck / Min-Edge Confidence ($\sigma_{\text{min}}$):**
  Replace path-product confidence aggregation with fuzzy bottleneck confidence ($\sigma_{\text{min}}(S) = \min_{v \in S} \text{conf}(v)$) or temperature-scaled softmax normalization to avoid exponential path decay over deep multi-hop traversals.
- **NLI & Entailment Verification:**
  Incorporate lightweight Natural Language Inference (NLI) step verification along traversal branches during early-exit gating.

---

## 3. Theoretical Open Problems

- **Tightness of the Frontier-Constrained $(1 - 1/e)$ Bound:**
  Formally establish matching lower bounds for frontier-constrained submodular connected-subtree selection on general non-tree graphs, analogous to the Feige (1998) construction for set cover.
- **Joint Correlation Modeling:**
  Extend the independent coverage model $f(S,q) = 1 - \prod(1 - \text{sim}(v,q))$ to explicitly incorporate positive edge correlations via Markov Random Fields (MRFs) or graph neural message passing.

---

## 4. Architectural & System Extensions

- **Cross-Domain Facet Learning:**
  Replace static weight vectors $(\alpha, \beta, \gamma, \delta, \epsilon)$ with online per-domain gradient descent learning driven by downstream answer verification feedback.
- **Multi-Vector Dense-Frontier Teleportation:**
  Formally integrate single-vector ANN retrieval (e.g., FAISS / HNSW) with submodular frontier expansion to allow dynamic "teleportation" jumps across disconnected document clusters when knowledge graph edges are sparse or missing.
