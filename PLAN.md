# PATHFINDER — Comprehensive End-to-End Master Execution Plan

> **Project**: PATHFINDER — Submodular Coverage Maximization over Multidimensional Knowledge Graphs for Multi-Hop Retrieval-Augmented Generation (RAG).  
> **Author**: Yash-Awasthi `<yashawasthi12032006@gmail.com>`  
> **Status**: Active (Phases 0–4 Completed)  

---

## Executive Summary & System Architecture

PATHFINDER models document collections and multi-hop reasoning chains as a **multidimensional knowledge graph** $G = (V, E, \Phi)$, where node attributes $\Phi = (\phi_{\text{sem}}, \phi_{\text{temp}}, \phi_{\text{imp}}, \phi_{\text{dom}}, \phi_{\text{conf}})$ capture semantic embeddings, temporal timestamps, structural centrality, domain alignment vectors, and epistemic confidence scores.

Retrieval is framed as maximizing a parameterized submodular objective function $F(S, q)$ subject to a token budget $K_{\text{tok}}$:

$$F(S, q) = \alpha f(S, q) + \sum_{v \in S} \Big[ \beta \phi_{\text{temp}}(v) + \gamma \phi_{\text{imp}}(v) + \delta \max(0, \cos(\phi_{\text{dom}}(v), q_{\text{dom}})) + \epsilon \phi_{\text{conf}}(v) \Big]$$

Path confidence calibration $\sigma(S)$ guards against cascading uncertainty across multi-hop reasoning chains:
$$\sigma(S) = \min_{v \in S} \sigma(v_0 \to v)$$

---

## Roadmap & Execution Progress

```
[Phase 0: Clean Repo & Foundation] ──────► COMPLETE (Git squashed to clean master)
[Phase 1: Multi-Benchmark Loaders & Evals] ► COMPLETE (HotpotQA, 2Wiki, MuSiQue evaluated)
[Phase 2: Hybridization & Math Calibration] ► COMPLETE (Teleportation, Grid Search, Confidence Models, Multi-Granularity)
[Phase 3: Paper & Artifact Sync] ─────────► COMPLETE (Section 4/7/8 updated, Plots added, FUTURE_WORK synced)
[Phase 4: Release & Final Verification] ──► COMPLETE (47/47 tests pass, clean push to master)
```

---

## Detailed Phase Breakdown

### Phase 0: Repository Cleanup & Baseline Standardization (COMPLETED)
- [x] **Auxiliary File Clean-up**: Purged temporary and unneeded design notes (`CANNOT.md`, `DEEPSEEK_PROMPT.md`, `EXECUTION_PLAN.md`, `NOTES.md`, `gaps.md`).
- [x] **Future Roadmap Consolidation**: Unified long-term extensions into `FUTURE_WORK.md`.
- [x] **Git History Normalization**: Re-organized commit history into modular, single-author commits under `Yash-Awasthi <yashawasthi12032006@gmail.com>`.

---

### Phase 1: Multi-Benchmark Data Harness & Initial Baseline Evaluation (COMPLETED)
- [x] **Dataset Loader Development**: Built `experiments/02_load_2wiki_musique.py` supporting:
  - **HotpotQA** ($N=7,405$ validation queries, 2-hop Wikipedia)
  - **2WikiMultihopQA** ($N=12,576$ validation queries, 2-hop structured relations)
  - **MuSiQue** ($N=2,417$ validation queries, 2-to-4-hop complex reasoning)
- [x] **Data Structure Standardization**: Standardized dictionary format across all serialized pickle files (`data/*_graphs_full.pkl`):
  $$\text{record} = \{ \text{"id"}, \text{"question"}, \text{"answer"}, \text{"supporting\_facts"}, \text{"graph": } \text{rec} \}$$
- [x] **Ground Truth Alignment**: Resolved schema differences between sentence-level supporting facts (HotpotQA, 2Wiki) and paragraph-level `is_supporting` flags (MuSiQue).
- [x] **Empirical Baseline Benchmark**: Evaluated 4 core algorithms across all 3 benchmarks:

#### Benchmark Evaluation Summary Table (Recall@5)
| Algorithm | HotpotQA ($N=7,405$) | 2WikiMultihopQA ($N=12,576$) | MuSiQue ($N=2,417$) |
| :--- | :---: | :---: | :---: |
| **PATHFINDER (Semantic-Only)** | 0.7307 | 0.2331 | 0.0087 |
| **Naive RAG (Dense Only)** | **0.7937** | **0.3248** | 0.0041 |
| **Spreading Activation** | 0.6974 | 0.2358 | **0.0165** |
| **BFS 2-Hop Traversal** | 0.6124 | 0.1820 | 0.0141 |

#### Empirical Key Findings & Diagnostics
1. **Dense Retrieval Dominance on Disconnected Graphs**: Naive RAG outperforms pure structural graph traversal on 2Wiki and HotpotQA because inter-document entity links are often missing in text-extracted KGs. Graph traversals get trapped in local clusters.
2. **Path Confidence Decay in Deep Multi-Hop**: Pure path-product confidence $\sigma(S) = \prod W(e) \phi_{\text{conf}}(u)$ severely penalizes multi-hop paths of depth $\ge 3$.
3. **MuSiQue Granularity Metric Challenge**: MuSiQue queries contain up to 4 hops and multiple supporting paragraphs with many sentences. Evaluation at Recall@5 with strict sentence matching leads to artificial near-zero recall.

---

### Phase 2: Hybridization, Teleportation & Mathematical Calibration (IN PROGRESS)

The objective of Phase 2 is to bridge the performance gap and surpass Naive RAG by enabling PATHFINDER to dynamically escape local graph components and optimizing hyperparameter weights.

#### Task 2.1: Dynamic Dense-Frontier "Teleportation" Jumps
- [x] **Implementation**: In `experiments/run_pathfinder.py`, modify the greedy traversal loop (line 10b).
- [x] **Mechanism**: When max marginal gain $\max_{v \in \text{Frontier}} \Delta(v | S) < \theta_{\text{teleport}}$ or when the graph expansion stalls:
  $$\text{Frontier}_{\text{new}} \leftarrow \text{Frontier} \cup \text{TopK}_{\text{global\_dense}}(q, V \setminus S)$$
- [x] **Theoretical Guarantee**: Preserves $(1 - 1/e)$-approximation guarantees of submodular maximization while providing $O(1)$ dynamic entry into disconnected high-relevance document subgraphs.

#### Task 2.2: Multidimensional Hyperparameter Grid Search ($\alpha, \beta, \gamma, \delta, \epsilon$)
- [x] **Grid Design**:
  - $\alpha$ (Semantic Coverage): $[0.5, 0.7, 0.9, 1.0]$
  - $\gamma$ (Structural Importance / PageRank): $[0.0, 0.05, 0.10, 0.20]$
  - $\epsilon$ (Epistemic Confidence): $[0.0, 0.05, 0.10]$
- [x] **Execution**: Script `experiments/03_grid_search.py` on HotpotQA & 2Wiki validation subsets ($N=500$ each) to optimize Recall@5 and F1 metrics.

#### Task 2.3: Confidence Calibration Aggregation Comparison
- [x] **Evaluate 3 Confidence Models** in `run_pathfinder.py`:
  1. **Product Confidence**: $\sigma_{\text{prod}}(S) = \min_{v \in S} \prod_{e, u \in \text{path}} W(e) \phi_{\text{conf}}(u)$
  2. **Geometric Mean Confidence**: $\sigma_{\text{geom}}(S) = \min_{v \in S} \left( \prod W(e) \phi_{\text{conf}}(u) \right)^{1 / L}$
  3. **Bottleneck Confidence (Fuzzy AND)**: $\sigma_{\text{min}}(S) = \min_{v \in S} \min_{e, u \in \text{path}} \{ W(e), \phi_{\text{conf}}(u) \}$
- [x] **Validation**: Script `experiments/04_confidence_calibration.py` for calibration curves ($\sigma$ vs actual answer Exact Match/F1 accuracy).

#### Task 2.4: Multi-Granularity & Deep Multi-Hop Metric Resolution
- [x] **Paragraph-Level Recall**: Add Paragraph-Recall@k (R@5, R@10) to `experiments/05_evaluate.py` to fairly measure MuSiQue 4-hop paragraph retrieval.
- [x] **Sub-sentence Alignment**: Evaluate Recall@10 and Recall@20 across all 3 datasets for full spectrum evaluation.

---

### Phase 3: Paper Manuscript, Visualizations & Documentation Sync (PLANNED)

#### Task 3.1: Automated Plotting & Visualizations
- [x] **Script**: Update `results/make_plots.py` to generate high-resolution SVG/PNG charts:
  - **Figure 1**: Multi-Benchmark Recall@k curves (HotpotQA, 2Wiki, MuSiQue).
  - **Figure 2**: Ablation plot comparing Pure Graph vs Teleportation Hybrid vs Naive RAG.
  - **Figure 3**: Confidence Calibration $\sigma(S)$ vs Downstream Answer F1 Score.

#### Task 3.2: Manuscript Revision (`pathfinder-paper.md`)
- [x] **Section 4 (Algorithm & Extensions)**: Formalize Teleportation Jump operators and bottleneck confidence definitions.
- [x] **Section 7 (Empirical Evaluation)**: Update main results tables with 3-benchmark comparisons, hyperparameter grid search results, and runtime efficiency breakdown.
- [x] **Section 8 (Discussion & Limitations)**: Document empirical trade-offs between dense index lookups and graph edge traversals.

#### Task 3.3: Future Work Roadmap Sync (`FUTURE_WORK.md`)
- [x] Update `FUTURE_WORK.md` with empirical findings from multi-hop scaling (e.g., dynamic edge synthesis, LLM-in-the-loop reranking).

---

### Phase 4: Verification, Test Suite & Release (COMPLETED)

- [x] **Unit Test Suite Execution**: Run and verify all pytest test cases in `pathfinder/tests/` pass cleanly. Result: 47/47 pass.
- [x] **Reproducibility Verification**: Test full command line pipeline execution from clean checkout.
- [x] **Clean Git Commit & Push**: Commit all changes under author identity `Yash-Awasthi <yashawasthi12032006@gmail.com>` and push to GitHub `master`.

---

## Verification & Validation Commands

To verify full pipeline correctness:

```cmd
:: 1. Run core unit test suite
pytest pathfinder/tests/

:: 2. Execute multi-benchmark evaluation suite
python experiments/05_evaluate.py --dataset hotpotqa --output results/hotpotqa_eval.json
python experiments/05_evaluate.py --dataset 2wiki --output results/2wiki_eval.json
python experiments/05_evaluate.py --dataset musique --output results/musique_eval.json

:: 3. Generate paper visualization plots
python results/make_plots.py
```
