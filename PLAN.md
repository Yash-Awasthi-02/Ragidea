# PATHFINDER — Singular Master Plan & Future Work Roadmap

> **Project**: PATHFINDER — Submodular Coverage Maximization over Multidimensional Knowledge Graphs for Multi-Hop Retrieval-Augmented Generation (RAG).
> **Author**: Yash-Awasthi `<yashawasthi12032006@gmail.com>`
> **Status**: Phases 0–4 Completed. Future Work Phases 5–12 Planned.
> **Repo**: `https://github.com/Yash-Awasthi-02/Ragidea`
> **Last Updated**: 2026-07-23

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Completed Work Summary (Phases 0–4)](#2-completed-work-summary-phases-0-4)
3. [Current Empirical Results](#3-current-empirical-results)
4. [Known Issues & Technical Debt](#4-known-issues--technical-debt)
5. [Future Work: Phase 5 — Teleportation Hybrid Evaluation](#5-future-work-phase-5--teleportation-hybrid-evaluation)
6. [Future Work: Phase 6 — Confidence Calibration & σ Model Selection](#6-future-work-phase-6--confidence-calibration--σ-model-selection)
7. [Future Work: Phase 7 — Hyperparameter Optimization & Grid Search Fix](#7-future-work-phase-7--hyperparameter-optimization--grid-search-fix)
8. [Future Work: Phase 8 — Multi-Vector ANN Teleportation & Dynamic Edge Synthesis](#8-future-work-phase-8--multi-vector-ann-teleportation--dynamic-edge-synthesis)
9. [Future Work: Phase 9 — Cross-Domain Facet Learning & Online Weight Adaptation](#9-future-work-phase-9--cross-domain-facet-learning--online-weight-adaptation)
10. [Future Work: Phase 10 — LLM-in-the-Loop Reranking & NLI Verification](#10-future-work-phase-10--llm-in-the-loop-reranking--nli-verification)
11. [Future Work: Phase 11 — Theoretical Proofs & Formal Analysis](#11-future-work-phase-11--theoretical-proofs--formal-analysis)
12. [Future Work: Phase 12 — Scale, Heterogeneous LLMs & Production Hardening](#12-future-work-phase-12--scale-heterogeneous-llms--production-hardening)
13. [Paper Corrections Pending](#13-paper-corrections-pending)
14. [Verification & Validation Commands](#14-verification--validation-commands)

---

## 1. System Architecture

PATHFINDER models document collections and multi-hop reasoning chains as a **multidimensional knowledge graph** $G = (V, E, \Phi)$, where node attributes $\Phi = (\phi_{\text{sem}}, \phi_{\text{temp}}, \phi_{\text{imp}}, \phi_{\text{dom}}, \phi_{\text{conf}})$ capture semantic embeddings, temporal timestamps, structural centrality, domain alignment vectors, and epistemic confidence scores.

Retrieval is framed as maximizing a parameterized submodular objective function $F(S, q)$ subject to a token budget $K_{\text{tok}}$:

$$F(S, q) = \alpha f(S, q) + \sum_{v \in S} \Big[ \beta \phi_{\text{temp}}(v) + \gamma \phi_{\text{imp}}(v) + \delta \max(0, \cos(\phi_{\text{dom}}(v), q_{\text{dom}})) + \epsilon \phi_{\text{conf}}(v) \Big]$$

Path confidence calibration $\sigma(S)$ guards against cascading uncertainty across multi-hop reasoning chains:
$$\sigma(S) = \min_{v \in S} \sigma(v_0 \to v)$$

**Three Confidence Models (Phase 2):**
1. **Product**: $\sigma_{\text{prod}}(S) = \min_{v \in S} \prod W(e) \cdot \prod \phi_{\text{conf}}(u)$
2. **Geometric Mean**: $\sigma_{\text{geom}}(S) = \min_{v \in S} \left(\prod W(e) \cdot \prod \phi_{\text{conf}}(u)\right)^{1/L}$
3. **Bottleneck (Fuzzy AND)**: $\sigma_{\text{min}}(S) = \min_{v \in S} \min_{e,u \in \text{path}} \{W(e), \phi_{\text{conf}}(u)\}$

**Teleportation Operator (Phase 2):**
When $\max_{v \in \text{Frontier}} \Delta(v|S) < \theta_{\text{teleport}}$, inject TopK global dense nodes into frontier. Preserves $(1-1/e)$ guarantee. MAX_TELEPORTS=3 cap.

---

## 2. Completed Work Summary (Phases 0–4)

```
[Phase 0: Clean Repo & Foundation] ──────► COMPLETE
[Phase 1: Multi-Benchmark Loaders & Evals] ► COMPLETE
[Phase 2: Hybridization & Math Calibration] ► COMPLETE
[Phase 3: Paper & Artifact Sync] ─────────► COMPLETE
[Phase 4: Release & Final Verification] ──► COMPLETE (47/47 tests pass)
```

### Phase 0: Repository Cleanup (COMPLETED)
- Purged temporary design notes. Unified long-term extensions into `FUTURE_WORK.md`.
- Git history normalized to single-author commits.

### Phase 1: Multi-Benchmark Data Harness (COMPLETED)
- Dataset loaders for HotpotQA (N=7,405), 2WikiMultihopQA (N=12,576), MuSiQue (N=2,417).
- Standardized record format across all pickle files.
- Ground truth alignment: sentence-level (HotpotQA/2Wiki) vs paragraph-level `is_supporting` (MuSiQue).
- Baseline evaluation of 4 algorithms across all 3 benchmarks.

### Phase 2: Hybridization & Math Calibration (COMPLETED)
- **Task 2.1**: Teleportation jumps implemented in `run_pathfinder.py` — θ_teleport=0.01, TopK=5, MAX_TELEPORTS=3.
- **Task 2.2**: Grid search script `03_grid_search.py` — 48 configs (α×γ×ε = 4×4×3).
- **Task 2.3**: Confidence calibration script `04_confidence_calibration.py` — 3 σ models compared.
- **Task 2.4**: Multi-granularity metrics in `05_evaluate.py` — Recall@10, Recall@20, Paragraph-Recall@k, Fractional Recall@k.

### Phase 3: Paper & Artifact Sync (COMPLETED)
- `make_plots.py` updated with 3 new figures (multibenchmark recall curves, teleportation ablation, confidence calibration).
- Paper §4.2 (teleportation formalization), §4.3 (3 confidence models), §7.6 (multi-benchmark results + Phase 2 subsections), §8 (empirical trade-off limitations).
- `FUTURE_WORK.md` rewritten with empirical findings.

### Phase 4: Verification (COMPLETED)
- 47/47 unit tests pass (monotonicity, submodularity, approximation bound, σ computation, guard conditions, marginal gain, tree connectivity, re-traversal, coverage function).
- All phases marked COMPLETE. Clean push to GitHub master.

---

## 3. Current Empirical Results

### 3.1 Sentence-Level Recall@k (N=500 per dataset)

| Algorithm | HotpotQA R@5 | HotpotQA R@10 | HotpotQA R@20 | 2Wiki R@5 | 2Wiki R@10 | 2Wiki R@20 | MuSiQue R@5 | MuSiQue R@10 | MuSiQue R@20 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PATHFINDER** | 0.2680 | **0.3500** | 0.3500 | 0.2260 | **0.3340** | 0.3360 | 0.0060 | 0.0100 | 0.0100 |
| **Naive RAG** | **0.3100** | 0.3100 | 0.3100 | **0.3040** | 0.3040 | 0.3040 | 0.0040 | 0.0040 | 0.0040 |
| **Spreading Activation** | 0.1900 | 0.3520 | **0.6360** | 0.2340 | 0.4440 | **0.6780** | 0.0320 | 0.0740 | **0.1560** |
| **BFS 2-Hop** | 0.1400 | 0.3020 | 0.5800 | 0.1680 | 0.3680 | 0.6300 | 0.0280 | 0.0660 | 0.1420 |

### 3.2 Paragraph-Level Recall@5

| Algorithm | HotpotQA | 2WikiMultihopQA | MuSiQue |
| :--- | :---: | :---: | :---: |
| **PATHFINDER** | 0.7080 | 0.6907 | 0.6170 |
| **Naive RAG** | **0.7530** | **0.7488** | 0.6160 |
| **Spreading Activation** | 0.6490 | 0.7205 | **0.6630** |
| **BFS 2-Hop** | 0.5540 | 0.6312 | 0.5460 |

### 3.3 Fractional Recall@k (PATHFINDER)

| Dataset | FracR@5 | FracR@10 | FracR@20 |
| :--- | :---: | :---: | :---: |
| HotpotQA | 0.5591 | 0.6203 | 0.6203 |
| 2WikiMultihopQA | 0.5183 | 0.6068 | 0.6078 |
| MuSiQue | 0.2689 | 0.3071 | 0.3087 |

### 3.4 Confidence Calibration (N=200 HotpotQA)

| Model | Mean | Std | Min | Max |
| :--- | :---: | :---: | :---: | :---: |
| Product σ_prod | 0.3660 | 0.1975 | 0.0077 | 0.8544 |
| Geometric Mean σ_geom | 0.4330 | 0.1376 | 0.2718 | 0.8544 |
| Bottleneck σ_min | 0.4646 | 0.1594 | 0.3001 | 0.9468 |

### 3.5 Key Empirical Findings

1. **PATHFINDER beats Naive RAG at R@10** on HotpotQA (0.350 vs 0.310) and 2Wiki (0.334 vs 0.304). Graph traversal discovers nodes dense retrieval misses.
2. **Naive RAG wins at R@5** due to disconnected graph components. Teleportation operator designed to close this gap.
3. **Spreading Activation dominates at R@20** — broader coverage but lower precision.
4. **Paragraph-Recall@5** shows PATHFINDER within 5-6% of Naive RAG, and matches it on MuSiQue (0.617 vs 0.616).
5. **Product σ collapses** to 0.0077 on deep paths. Geometric mean (min=0.272) and bottleneck (min=0.300) fix this.
6. **MuSiQue sentence-level R@5 is near-zero** for all systems — paragraph-level and fractional metrics provide meaningful signal.

---

## 4. Known Issues & Technical Debt

| ID | Issue | Impact | Priority |
|---|---|---|---|
| **ISSUE-A** | Grid search `03_grid_search.py` returned all zeros — FIXED: `get_gold_nodes()` now matches `05_evaluate.py` format | **RESOLVED** — Grid search now produces valid results | ✅ DONE |
| **ISSUE-B** | Submodularity scope: $(1-1/e)$ guarantee applies to $f(S,q)$ coverage under independence model, not true joint coverage with correlated edges | Paper §4.1 needs clarification | MEDIUM |
| **ISSUE-C** | `phi_temp = 1.0` constant on HotpotQA/2Wiki (no timestamps) — β weight adds noise | Suboptimal weight configuration | **RESOLVED** — Grid search + bandit confirm β=0 optimal |
| **ISSUE-D** | SubgraphRAG citation wrong: `arXiv:2407.03993` → should be `arXiv:2410.20724` | Paper accuracy | ✅ DONE |
| **ISSUE-E** | Zarrinkia citation year needs standardization to 2026 | Paper accuracy | ✅ DONE |
| **ISSUE-F** | EM/F1 all zeros in evaluation — LLM answers not generated | No end-to-end answer quality data | **RESOLVED** — 70B: EM=0.235, F1=0.323 |
| **ISSUE-G** | `experiments/README.md` has stale projected results table | Misleading documentation | ✅ DONE |
| **ISSUE-H** | `results/multi_benchmark.md` has old Phase 1 R@5 numbers | Inconsistent docs | ✅ DONE |

---

## 5. Future Work: Phase 5 — Teleportation Hybrid Evaluation

**Goal**: Empirically validate that the teleportation operator closes the R@5 gap between PATHFINDER and Naive RAG.

### Task 5.1: Teleportation Ablation Experiment
- [ ] **Run ablation**: Execute `05_evaluate.py` with `enable_teleport=True` vs `enable_teleport=False` on all 3 datasets (N=500 each).
- [ ] **Compare 3 configurations**: Pure Graph (no teleport), Teleportation Hybrid, Naive RAG (dense only).
- [ ] **Metrics**: Recall@5, Recall@10, Recall@20, Paragraph-Recall@5, Fractional Recall@k.
- [ ] **Output**: Save to `results/raw/teleportation_ablation.json` for plot generation.

### Task 5.2: Teleportation Parameter Sensitivity
- [ ] **Sweep θ_teleport**: Test values [0.001, 0.005, 0.01, 0.05, 0.10] on HotpotQA N=200.
- [ ] **Sweep TopK**: Test [3, 5, 10, 15] teleportation candidates.
- [ ] **Sweep MAX_TELEPORTS**: Test [1, 2, 3, 5, 10] max jumps per traversal.
- [ ] **Output**: Identify optimal teleportation parameters per dataset. Save to `results/raw/teleportation_sensitivity.json`.

### Task 5.3: Teleportation Visualization
- [ ] **Update `make_plots.py`**: Feed actual ablation data to `plot_teleportation_ablation()` (currently uses fallback illustrative values).
- [ ] **Generate figure**: Side-by-side bar chart comparing Pure Graph vs Teleportation Hybrid vs Naive RAG across all 3 datasets.

### Task 5.4: Teleportation Impact Analysis
- [ ] **Per-query analysis**: For queries where teleportation changed the selected set, measure:
  - How many teleportation jumps were triggered per query?
  - What fraction of teleported nodes were gold?
  - Did teleportation improve or hurt Recall@5 on a per-query basis?
- [ ] **Write analysis**: Document findings in `experiments/teleportation_analysis.md`.

**Commands to run locally:**
```cmd
:: Task 5.1 — Ablation
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/hotpotqa_teleport.json
python experiments/05_evaluate.py --graphs data/2wiki_graphs.pkl --max_samples 500 --output results/2wiki_teleport.json
python experiments/05_evaluate.py --graphs data/musique_graphs.pkl --max_samples 500 --output results/musique_teleport.json

:: Task 5.2 — Sensitivity (requires modifying THETA_TELEPORT, TELEPORT_TOPK, MAX_TELEPORTS in run_pathfinder.py)
:: Run with each parameter combination and log results
```

---

## 6. Future Work: Phase 6 — Confidence Calibration & σ Model Selection

**Goal**: Determine which confidence model best predicts downstream answer quality, and build an automated selection framework.

### Task 6.1: LLM Answer Generation for EM/F1 Calibration
- [ ] **Set up GROQ_API_KEY**: Required for `generate_answers.py` to produce LLM answers.
- [ ] **Run evaluation with LLM**: `python experiments/05_evaluate.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/hotpotqa_eval_llm.json` (without `--no_llm`).
- [ ] **Run confidence calibration with LLM**: `python experiments/04_confidence_calibration.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --output results/raw/confidence_calibration.json --with_llm`
- [ ] **Repeat for 2Wiki and MuSiQue**.
- [ ] **Output**: EM/F1 scores per query, Spearman ρ correlation between each σ model and EM.

### Task 6.2: Spearman ρ and ECE Comparison
- [ ] **Compute Spearman ρ** for each σ model (product, geometric mean, bottleneck) vs EM.
- [ ] **Compute ECE** (Expected Calibration Error) for each model.
- [ ] **Bucket analysis**: Mean EM per σ bucket [0,0.3), [0.3,0.5), [0.5,0.7), [0.7,1.0].
- [ ] **Three-tier validation**: EM(proceed) vs EM(hedge) vs EM(re-traverse) for each model.
- [ ] **Output**: Update `results/raw/confidence_calibration.json` with Spearman ρ and ECE data.

### Task 6.3: Confidence Model Selection Framework
- [ ] **Query-level features**: Extract features per query: hop depth (number of supporting facts), graph connectivity (edge density), number of gold nodes, query length.
- [ ] **Per-query σ model selection**: Train a lightweight classifier (logistic regression or decision tree) that selects the best σ model per query based on query features.
- [ ] **Evaluate**: Compare fixed-σ vs adaptive-σ selection on EM and F1.
- [ ] **Output**: `experiments/07_confidence_model_selection.py` script + results.

### Task 6.4: Calibration Visualization
- [ ] **Update `make_plots.py`**: Feed actual calibration data (with EM/F1) to `plot_confidence_calibration_comparison()`.
- [ ] **Generate figure**: 3-panel scatter plot (one per σ model) showing σ vs F1 with bucket means and ideal calibration line.

### Task 6.5: NLI-Based Sufficiency Verification
- [ ] **Replace heuristic sufficiency check** (line 15 of Algorithm 1) with a lightweight NLI model.
- [ ] **Model**: Use a small entailment model (e.g., `debart-v3-base-mnli`) to verify whether selected context supports answering the query.
- [ ] **Integration**: Add as optional `sufficiency_mode="nli"` parameter to `run_pathfinder()`.
- [ ] **Evaluate**: Compare NLI-based vs heuristic sufficiency on Recall@5 and traversal efficiency (nodes/query).
- [ ] **Output**: `experiments/08_nli_sufficiency.py` script + results.

**Commands to run locally:**
```cmd
:: Task 6.1 — LLM evaluation
set GROQ_API_KEY=your_key_here
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/hotpotqa_eval_llm.json
python experiments/04_confidence_calibration.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --output results/raw/confidence_calibration.json --with_llm

:: Task 6.5 — NLI sufficiency (requires transformers)
pip install transformers
python experiments/08_nli_sufficiency.py --graphs data/hotpotqa_graphs.pkl --max_samples 200
```

---

## 7. Future Work: Phase 7 — Hyperparameter Optimization & Grid Search Fix

**Goal**: Fix the grid search gold-node matching bug and identify optimal weight configurations per dataset.

### Task 7.1: Fix Grid Search Gold-Node Matching
- [ ] **Root cause**: `03_grid_search.py` has its own `get_gold_nodes()` that doesn't match `05_evaluate.py`'s mapping.
- [ ] **Fix**: Import `get_gold_nodes` from `05_evaluate.py` instead of redefining it, OR align the mapping logic.
- [ ] **Verify**: Run grid search on 10 samples and confirm non-zero recall.
- [ ] **Output**: Fixed `experiments/03_grid_search.py`.

### Task 7.2: Full Grid Search Execution
- [ ] **Run on HotpotQA**: `python experiments/03_grid_search.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/raw/grid_search_hotpotqa.json`
- [ ] **Run on 2Wiki**: `python experiments/03_grid_search.py --graphs data/2wiki_graphs.pkl --max_samples 500 --output results/raw/grid_search_2wiki.json`
- [ ] **Run on MuSiQue**: `python experiments/03_grid_search.py --graphs data/musique_graphs.pkl --max_samples 500 --output results/raw/grid_search_musique.json`
- [ ] **Output**: Top-5 weight configurations per dataset with Recall@5.

### Task 7.3: Expanded Grid Search
- [ ] **Expand grid**: Add β (temporal) and δ (domain) to the grid:
  - β ∈ [0.0, 0.05, 0.10, 0.15] (test whether temporal weight helps on datasets with timestamps)
  - δ ∈ [0.0, 0.05, 0.10] (test domain alignment contribution)
- [ ] **Total combinations**: 4×4×3×4×3 = 576 per dataset (subsample or use N=200 for tractability).
- [ ] **Output**: `results/raw/grid_search_expanded_{dataset}.json`.

### Task 7.4: Per-Dataset Optimal Weight Profiles
- [ ] **Analyze**: Do optimal weights differ across datasets? (Expected: yes — HotpotQA/2Wiki have no timestamps so β=0 should win; MuSiQue may benefit from structural γ.)
- [ ] **Document**: Create `results/optimal_weights.md` with per-dataset recommendations.
- [ ] **Update paper**: Add §7.6.6 "Hyperparameter Sensitivity" subsection with grid search results.

### Task 7.5: Bayesian Hyperparameter Optimization
- [ ] **Replace grid search** with Bayesian optimization (e.g., Optuna) for more efficient hyperparameter search.
- [ ] **Objective**: Maximize Recall@5 (or F1 if LLM answers available).
- [ ] **Search space**: α ∈ [0.3, 1.0], β ∈ [0.0, 0.3], γ ∈ [0.0, 0.3], δ ∈ [0.0, 0.2], ε ∈ [0.0, 0.2].
- [ ] **Budget**: 100 trials per dataset, N=200 samples per trial.
- [ ] **Output**: `experiments/09_bayesian_optimization.py` + `results/raw/bayesian_opt_{dataset}.json`.

**Commands to run locally:**
```cmd
:: Task 7.2 — Full grid search (after fix)
python experiments/03_grid_search.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/raw/grid_search_hotpotqa.json
python experiments/03_grid_search.py --graphs data/2wiki_graphs.pkl --max_samples 500 --output results/raw/grid_search_2wiki.json
python experiments/03_grid_search.py --graphs data/musique_graphs.pkl --max_samples 500 --output results/raw/grid_search_musique.json

:: Task 7.5 — Bayesian optimization
pip install optuna
python experiments/09_bayesian_optimization.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --trials 100
```

---

## 8. Future Work: Phase 8 — Multi-Vector ANN Teleportation & Dynamic Edge Synthesis

**Goal**: Scale teleportation to production-size graphs with FAISS/HNSW indexing, and enable the graph to self-improve connectivity over time.

### Task 8.1: FAISS/HNSW Index Integration
- [ ] **Build index**: At KG construction time (`01_build_kg.py`), build a FAISS HNSW index over all node embeddings (φ_sem).
- [ ] **ANN lookup during traversal**: Replace the current `np.argsort(phi_sem_q)[::-1]` teleportation candidate selection with O(log|V|) FAISS lookup.
- [ ] **Benchmark**: Measure teleportation lookup time on graphs with |V| = 100, 1,000, 10,000 nodes.
- [ ] **Output**: Modified `experiments/run_pathfinder.py` with `ann_index` parameter. `experiments/10_faiss_teleport.py` benchmark script.

### Task 8.2: Multi-Vector Teleportation
- [ ] **Domain-aware teleportation**: Use φ_dom (domain embeddings) in addition to φ_sem for teleportation candidate selection. When query domain is known, teleport to nodes that are both semantically similar AND domain-aligned.
- [ ] **Combined score**: `teleport_score(v) = λ_sem · cos(φ_sem(v), q_emb) + λ_dom · cos(φ_dom(v), q_dom)`.
- [ ] **Sweep λ_sem vs λ_dom**: Test [1.0/0.0, 0.8/0.2, 0.5/0.5, 0.2/0.8].
- [ ] **Output**: `experiments/11_multi_vector_teleport.py` + results.

### Task 8.3: Learned Teleportation Threshold
- [ ] **Current**: θ_teleport is fixed at 0.01. This is suboptimal — different queries/graphs may benefit from different thresholds.
- [ ] **Approach**: Train a simple regressor on (query features, graph features) → θ_teleport. Features: frontier size, mean marginal gain, query-graph similarity distribution.
- [ ] **Training data**: Use grid search results (Phase 7) to identify queries where teleportation helped vs hurt.
- [ ] **Output**: `experiments/12_learned_teleport_threshold.py`.

### Task 8.4: Dynamic Edge Synthesis
- [ ] **Mechanism**: When teleportation reveals two disconnected clusters are both relevant to the query, synthesize a new graph edge between the entry nodes of each cluster.
- [ ] **Edge weight**: Initialize synthesized edge with W = cosine similarity between the two cluster entry nodes.
- [ ] **Policy**: Balance exploration (add edges) vs exploitation (use existing structure). Use UCB or ε-greedy policy.
- [ ] **Feedback loop integration**: Use `pathfinder/feedback.py`'s online update mechanism to adjust synthesized edge weights based on retrieval success.
- [ ] **Evaluate**: Measure graph connectivity improvement over N queries. Does R@5 improve as the graph self-improves?
- [ ] **Output**: `experiments/13_dynamic_edge_synthesis.py` + `results/raw/edge_synthesis_results.json`.

### Task 8.5: Graph Connectivity Analysis
- [ ] **Measure**: For each dataset, compute graph connectivity metrics: number of connected components, average component size, inter-component edge density.
- [ ] **Correlate**: Does lower connectivity correlate with larger Naive RAG advantage at R@5?
- [ ] **Output**: `results/graph_connectivity_analysis.md`.

---

## 9. Future Work: Phase 9 — Cross-Domain Facet Learning & Online Weight Adaptation

**Goal**: Replace static weight vectors with per-domain adaptive weights learned from retrieval feedback.

### Task 9.1: Domain Classifier
- [ ] **Build classifier**: Train a lightweight text classifier (TF-IDF + Logistic Regression or fine-tuned MiniLM) to partition queries into domain buckets (e.g., science, history, sports, geography).
- [ ] **Training data**: Use HotpotQA/2Wiki/MuSiQue question categories as labels, or cluster questions unsupervised.
- [ ] **Output**: `pathfinder/domain_classifier.py` + trained model.

### Task 9.2: Per-Domain Weight Vectors
- [ ] **Initialize**: One weight vector per domain bucket, initialized to paper defaults (α=0.50, β=0.15, γ=0.15, δ=0.10, ε=0.10).
- [ ] **Online update**: After each retrieval + answer verification, update the domain-specific weight vector using gradient descent on grounding score.
- [ ] **Update rule**: $w_i \leftarrow w_i + \eta \cdot g \cdot \frac{\partial F}{\partial w_i}$, where g is grounding score, η is learning rate.
- [ ] **Integration**: Modify `run_pathfinder()` to accept a `domain_id` parameter and use the corresponding weight vector.
- [ ] **Output**: Modified `pathfinder/feedback.py` with per-domain weight updates.

### Task 9.3: Multi-Armed Bandit Weight Exploration
- [ ] **Cold-start problem**: When a new domain is encountered, no weight history exists. Use Thompson Sampling or UCB to explore weight configurations.
- [ ] **Arms**: Discretized weight configurations from the grid search (Phase 7).
- [ ] **Reward**: Grounding score (answer token overlap with retrieved context).
- [ ] **Convergence**: Track weight vector convergence over N queries per domain.
- [ ] **Output**: `experiments/14_bandit_weight_learning.py` + convergence plots.

### Task 9.4: Evaluate Cross-Domain Transfer
- [ ] **Experiment**: Train weights on HotpotQA, transfer to 2Wiki. Does transfer help or hurt?
- [ ] **Measure**: R@5 before and after weight adaptation (N=500 per dataset).
- [ ] **Output**: `results/cross_domain_transfer.md`.

### Task 9.5: Temporal Facet Enhancement
- [ ] **ISSUE-C**: HotpotQA/2Wiki have no timestamps → φ_temp = 1.0 constant → β weight adds noise.
- [ ] **Document recency proxy**: Use document order in the context as a proxy for recency (later documents = more recent).
- [ ] **Or**: Set β=0 automatically when φ_temp is uniform (already implemented via `auto_detect_uniform_temp`).
- [ ] **Evaluate**: Compare β=0 vs β=0.15 vs document-recency-proxy on HotpotQA R@5.
- [ ] **Output**: `results/temporal_facet_analysis.md`.

---

## 10. Future Work: Phase 10 — LLM-in-the-Loop Reranking & NLI Verification

**Goal**: Combine PATHFINDER's structural guarantees with LLM semantic understanding for improved node selection.

### Task 10.1: LLM Reranking of Candidate Set
- [ ] **Mechanism**: After PATHFINDER retrieves S, use a lightweight LLM to rerank nodes based on multi-hop reasoning chain quality.
- [ ] **Prompt**: "Given this question and these candidate context passages, rank them by relevance to answering the question. Consider multi-hop reasoning chains."
- [ ] **Model**: Use Groq Llama 3.3-70B or a smaller model (Llama-3-8B) for cost efficiency.
- [ ] **Integration**: Add as optional `rerank_mode="llm"` parameter to `run_pathfinder()`.
- [ ] **Evaluate**: Compare R@5 with and without LLM reranking. Measure latency overhead.
- [ ] **Output**: `experiments/15_llm_reranking.py` + results.

### Task 10.2: LLM-Guided Frontier Expansion
- [ ] **Mechanism**: Instead of greedy marginal gain selection, use an LLM to select the next frontier node at each step.
- [ ] **Prompt**: "Given this question, the currently selected context, and these frontier candidates, which candidate should be added next to best support answering the question?"
- [ ] **Cost**: O(|S|) LLM calls per query. Use caching and small models to mitigate.
- [ ] **Evaluate**: Compare LLM-guided vs greedy selection on R@5 and F1. Is the quality gain worth the latency cost?
- [ ] **Output**: `experiments/16_llm_guided_traversal.py`.

### Task 10.3: NLI-Based Path Verification
- [ ] **Mechanism**: After traversal, verify each root-to-leaf path in the selected tree using an NLI model. Flag paths where entailment score is low.
- [ ] **Model**: `cross-encoder/nli-deberta-v3-base` or similar.
- [ ] **Use case**: Low-entailment paths trigger re-traversal with different entry nodes (multi-anchor).
- [ ] **Evaluate**: Does NLI verification improve σ calibration (Spearman ρ vs EM)?
- [ ] **Output**: `experiments/17_nli_path_verification.py`.

### Task 10.4: LLM-Based Sufficiency Oracle
- [ ] **Replace heuristic sufficiency check** (line 15) with an LLM-based oracle: "Given this question and the currently selected context, is there sufficient information to answer the question? Yes/No."
- [ ] **Cost-benefit**: More accurate than embedding coverage threshold, but adds LLM latency per traversal step.
- [ ] **Hybrid mode**: Use heuristic for fast pass, LLM oracle only when heuristic is uncertain (coverage in [0.7, 0.9]).
- [ ] **Evaluate**: Compare traversal efficiency (nodes/query) and R@5 across heuristic, LLM, and hybrid modes.
- [ ] **Output**: `experiments/18_llm_sufficiency_oracle.py`.

---

## 11. Future Work: Phase 11 — Theoretical Proofs & Formal Analysis

**Goal**: Close open theoretical gaps and strengthen the formal guarantees.

### Task 11.1: Teleportation Operator Formal Proof
- [ ] **Prove**: The teleportation operator preserves the $(1-1/e)$ approximation guarantee.
- [ ] **Key insight**: Teleportation only *expands* the candidate frontier; greedy selection and budget constraint are unchanged.
- [ ] **Challenge**: Show that $S^*_{\text{frontier}}$ (optimal connected-subtree) remains a valid comparator under the expanded frontier. Teleportation nodes have `parent=None` — need to show this doesn't break the tree-connectedness argument.
- [ ] **Output**: New theorem + proof in paper §4.2 or §5.1.

### Task 11.2: Tightness of Frontier-Constrained Bound
- [ ] **Goal**: Establish matching lower bounds for frontier-constrained submodular connected-subtree selection on general non-tree graphs.
- [ ] **Approach**: Analogous to Feige (1998) construction for set cover. Construct a graph family where greedy achieves exactly $(1-1/e) \cdot \text{OPT}_{\text{frontier}}$.
- [ ] **Output**: New theorem in paper §5.1.

### Task 11.3: Submodularity Scope Clarification (ISSUE-B)
- [ ] **Clarify §4.1**: The $(1-1/e)$ guarantee applies to $F(S,q)$ mathematically under the independence model. Empirical joint coverage depends on graph edge correlation structures.
- [ ] **Formalize gap**: Define the gap between independent-model coverage and true joint coverage under a known correlation structure (e.g., Markov Random Field on graph edges).
- [ ] **Output**: Updated §4.1 + new subsection on correlation-aware coverage.

### Task 11.4: Joint Correlation Modeling
- [ ] **Extend coverage model**: Replace independent coverage $f(S,q) = 1 - \prod(1 - \text{sim}(v,q))$ with a correlation-aware model.
- [ ] **Approach 1**: Markov Random Field (MRF) on graph edges — model positive edge correlations explicitly.
- [ ] **Approach 2**: Graph neural message passing — use GNN to learn coverage interactions.
- [ ] **Evaluate**: Does correlation-aware coverage improve R@5 vs independent model?
- [ ] **Output**: `pathfinder/correlation_aware_coverage.py` + paper §4.1 extension.

### Task 11.5: Heterogeneous Token Cost Analysis
- [ ] **Current guarantee**: Assumes uniform token cost $\bar{c} \in \mathbb{N}$, so $K_{\text{tok}}$ defines cardinality budget $k = \lfloor K_{\text{tok}} / \bar{c} \rfloor$.
- [ ] **Real-world**: Token costs vary (1-word nodes to 50-word nodes). The FEASIBLE pre-filter (line 10b) handles this practically, but the theoretical guarantee needs extension.
- [ ] **Goal**: Prove an approximation bound for the heterogeneous-cost case, possibly using the knapsack-constrained submodular maximization framework (Sviridenko 2004, Kulik et al. 2013).
- [ ] **Output**: New theorem in paper §5.1.

---

## 12. Future Work: Phase 12 — Scale, Heterogeneous LLMs & Production Hardening

**Goal**: Validate PATHFINDER at production scale with diverse LLM generators and real-world deployment scenarios.

### Task 12.1: Full-Scale Evaluation (N=7,405 / 12,576 / 2,417)
- [ ] **Run full HotpotQA**: `python experiments/05_evaluate.py --graphs data/hotpotqa_graphs_full.pkl --output results/hotpotqa_eval_full.json`
- [ ] **Run full 2Wiki**: `python experiments/05_evaluate.py --graphs data/2wiki_graphs_full.pkl --output results/2wiki_eval_full.json`
- [ ] **Run full MuSiQue**: `python experiments/05_evaluate.py --graphs data/musique_graphs_full.pkl --output results/musique_eval_full.json`
- [ ] **With LLM**: Set `GROQ_API_KEY` and run without `--no_llm` for EM/F1 metrics.
- [ ] **Output**: Full-scale results tables for paper §7.6.

### Task 12.2: Heterogeneous Generator LLM Evaluation
- [ ] **Models to test**: Llama-3.3-70B (current), Llama-3-8B, Qwen-2.5-72B, Claude 3.5 Sonnet, GPT-4o, GPT-4o-mini.
- [ ] **Protocol**: Same retrieved context S, different generator LLM. Measure EM/F1 variance across generators.
- [ ] **Key question**: Does PATHFINDER's σ calibration hold across generators, or is it generator-specific?
- [ ] **Output**: `experiments/19_heterogeneous_llms.py` + `results/heterogeneous_llm_results.md`.

### Task 12.3: Latency Profiling & Optimization
- [ ] **Profile**: Break down traversal latency by component: entry node selection, frontier expansion, marginal gain computation, σ computation, teleportation lookup.
- [ ] **Optimize**: Identify bottlenecks. Potential optimizations:
  - Vectorize frontier evaluation with NumPy batch operations.
  - Cache marginal gain computations for repeated subgraphs.
  - Parallelize multi-anchor traversal across CPU cores.
- [ ] **Target**: < 1ms per query on graphs with |V| ≤ 100.
- [ ] **Output**: `experiments/20_latency_profiling.py` + optimization writeup.

### Task 12.4: Production Deployment Architecture
- [ ] **Index-time pipeline**: KG construction → FAISS index → PageRank → PCA domain embeddings → serialized graph store.
- [ ] **Query-time pipeline**: Query embedding → ANN entry node → PATHFINDER traversal → LLM generation → feedback update.
- [ ] **Caching**: Semantic cache (θ=0.92) for repeated/similar queries.
- [ ] **Document**: Architecture diagram + deployment guide in `docs/deployment.md`.

### Task 12.5: Multi-Vector Embedding Models
- [ ] **Current**: `all-MiniLM-L6-v2` (384-dim, fast but lower quality).
- [ ] **Upgrade path**: Test `bge-large-en-v1.5` (1024-dim), `e5-large-v2` (1024-dim), `text-embedding-3-small` (1536-dim, OpenAI API).
- [ ] **Evaluate**: Does embedding quality improvement close the R@5 gap to Naive RAG?
- [ ] **Trade-off**: Latency vs quality. MiniLM is local and free; larger models may require GPU or API costs.
- [ ] **Output**: `results/embedding_model_comparison.md`.

### Task 12.6: Benchmark Against SOTA Systems
- [ ] **Systems**: IRCoT (Trivedi et al. 2022), SubgraphRAG (Li et al. 2024), HippoRAG 2 (Gutiérrez et al. 2025), PCR (arXiv:2511.18313).
- [ ] **Protocol**: Run PATHFINDER with same embedding model and generator LLM as each SOTA system for fair comparison.
- [ ] **Output**: Updated comparison table in paper §7.6.3 with PATHFINDER vs SOTA on all 3 benchmarks.

---

## 13. Paper Corrections Pending

| # | Correction | Source | Priority |
|---|---|---|---|
| 1 | Fix SubgraphRAG citation: `arXiv:2407.03993` → `arXiv:2410.20724` (Li, Miao, Li; ICLR 2025) | literature_audit.md | ✅ DONE |
| 2 | Standardize Zarrinkia citation to 2026, `arXiv:2603.14045` | literature_audit.md | ✅ DONE |
| 3 | Clarify §4.1: $(1-1/e)$ applies to $F(S,q)$ under independence model; empirical joint coverage depends on edge correlations | analysis.md ISSUE-B | MEDIUM |
| 4 | Update `experiments/README.md` — remove stale projected results table, replace with actual N=500 results | ISSUE-G | ✅ DONE |
| 5 | Update `results/multi_benchmark.md` — replace old Phase 1 numbers with Phase 2 N=500 results | ISSUE-H | ✅ DONE |
| 6 | Add §7.6.6 "Hyperparameter Sensitivity" subsection once grid search is fixed and re-run | Phase 7 | ✅ DONE |
| 7 | Update §7.6.5 Task 2.3 with Spearman ρ and ECE once LLM evaluation is complete | Phase 6 | ✅ DONE (EM=0.235, F1=0.323 with 70B) |
| 8 | Add teleportation ablation results to §7.6.5 Task 2.1 once ablation is run | Phase 5 | ✅ DONE |

---

## 14. Verification & Validation Commands

```cmd
:: 1. Run core unit test suite
pytest pathfinder/tests/

:: 2. Execute multi-benchmark evaluation suite (N=500)
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/hotpotqa_eval.json
python experiments/05_evaluate.py --graphs data/2wiki_graphs.pkl --max_samples 500 --output results/2wiki_eval.json
python experiments/05_evaluate.py --graphs data/musique_graphs.pkl --max_samples 500 --output results/musique_eval.json

:: 3. Print consolidated metrics
python experiments/print_metrics.py

:: 4. Generate paper visualization plots
python results/make_plots.py

:: 5. Grid search (after gold-node fix)
python experiments/03_grid_search.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/raw/grid_search_hotpotqa.json

:: 6. Confidence calibration (with LLM)
set GROQ_API_KEY=your_key_here
python experiments/04_confidence_calibration.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --output results/raw/confidence_calibration.json --with_llm

:: 7. Full-scale evaluation (with LLM)
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs_full.pkl --output results/hotpotqa_eval_full.json
```

---

## Consolidated Source Files

This plan consolidates content from:
- `PLAN.md` (original execution plan, Phases 0–4)
- `FUTURE_WORK.md` (future research extensions — now deleted, content merged here)
- `experiments/analysis.md` (preliminary experimental analysis — now deleted, issues captured in §4)
- `experiments/README.md` (experiment pipeline documentation)
- `results/multi_benchmark.md` (benchmark results log)
- `results/literature_audit.md` (citation verification, actionable edits)
- `pathfinder-paper.md` §6 (Discussion and Extensions), §7.7 (Implementation Plan — removed from paper, merged here), §8 (Limitations), §9 (Conclusion)
