# PATHFINDER — Singular Master Plan & Optimization Pathway

> **Project**: PATHFINDER — Submodular Coverage Maximization over Multidimensional Knowledge Graphs for Multi-Hop RAG
> **Author**: Yash-Awasthi `<yashawasthi12032006@gmail.com>`
> **Repo**: `https://github.com/Yash-Awasthi-02/Ragidea`
> **Last Updated**: 2026-07-23

---

## Table of Contents

1. [Current State](#1-current-state)
2. [Empirical Results Summary](#2-empirical-results-summary)
3. [Honest Assessment](#3-honest-assessment)
4. [Optimization Pathway — Phase A: Graph Construction](#4-optimization-pathway--phase-a-graph-construction)
5. [Optimization Pathway — Phase B: Hybrid Retrieval](#5-optimization-pathway--phase-b-hybrid-retrieval)
6. [Optimization Pathway — Phase C: LLM Integration](#6-optimization-pathway--phase-c-llm-integration)
7. [Optimization Pathway — Phase D: Embedding & Scale](#7-optimization-pathway--phase-d-embedding--scale)
8. [Optimization Pathway — Phase E: SOTA Comparison](#8-optimization-pathway--phase-e-sota-comparison)
9. [Theoretical Work Remaining](#9-theoretical-work-remaining)
10. [Repo Cleanup Checklist](#10-repo-cleanup-checklist)
11. [Verification Commands](#11-verification-commands)

---

## 1. Current State

```
Phases 0–4:  COMPLETE (repo, benchmarks, teleportation, confidence, paper, tests)
Phases 5–12: COMPLETE (18 experiment scripts, all evaluations run, results pushed)
Issues:      8/8 RESOLVED
Tests:       47/47 PASS
Paper:       §4.1 submodularity scope clarified, §4.2 teleportation proof added,
             §7.6.4–7.6.6 empirical results, §8 limitations updated
```

### What Works
- Theoretical foundation is solid: proven submodularity, (1−1/e) guarantee, teleportation corollary
- 47 unit tests verify all formal properties
- Latency is production-viable (3.5ms mean, 7.1ms p95)
- LLM reranking improves R@5 by +33.3% (0.24→0.32)
- LLM-guided traversal improves R@5 by +50% (0.20→0.30)
- Bandit weight learning converges (0.18→0.30)
- 70B LLM achieves EM=0.235, F1=0.323 on retrieved context

### What Doesn't Work (Yet)
- Naive RAG beats PATHFINDER at R@5 on all datasets (0.31 vs 0.27 on HotpotQA)
- Teleportation triggers on only 9.8% of queries, helps 0 at R@5
- 8B LLM completely fails on sentence-level context (EM=F1=0)
- NLI sufficiency (lexical fallback) agrees with heuristic only 8.5% of the time
- Graph connectivity doesn't explain the R@5 gap (all p > 0.36)
- Dynamic edge synthesis synthesized 0 edges (teleport nodes already reachable)

---

## 2. Empirical Results Summary

### Sentence-Level Recall@k (N=500)

| System | HotpotQA R@5 | R@10 | 2Wiki R@5 | R@10 | MuSiQue R@5 | R@10 |
|---|---|---|---|---|---|---|
| PATHFINDER | 0.268 | **0.350** | 0.226 | **0.334** | 0.006 | 0.010 |
| Naive RAG | **0.310** | 0.310 | **0.304** | 0.304 | 0.004 | 0.004 |
| Spreading Act. | 0.190 | 0.352 | 0.234 | 0.444 | **0.032** | 0.074 |
| BFS 2-Hop | 0.140 | 0.302 | 0.168 | 0.368 | 0.028 | 0.066 |

### Key Intervention Results

| Intervention | R@5 Before | R@5 After | Delta | Cost |
|---|---|---|---|---|
| LLM Reranking (70B) | 0.240 | 0.320 | **+33.3%** | 1 LLM call/query |
| LLM-Guided Traversal | 0.200 | 0.300 | **+50.0%** | O(\|S\|) LLM calls/query |
| Bandit Weight Learning | 0.180 | 0.300 | **+66.7%** | Online, no inference cost |
| Teleportation (R@10) | 0.330 | 0.350 | +6.1% | Free |
| Hybrid LLM Sufficiency | 0.200 | 0.150 | **−25.0%** | LLM calls during traversal |

### Generator LLM Comparison (N=200)

| Model | EM | F1 | R@5 |
|---|---|---|---|
| Llama 3.3-70B | **0.235** | **0.323** | 0.240 |
| Llama 3-8B | 0.000 | 0.000 | 0.240 |

---

## 3. Honest Assessment

**The theory is publishable. The empirical results are not yet SOTA-competitive.**

The core problem: sentence-level graph traversal on sparse text-extracted KGs cannot beat dense retrieval at small k. The graph is too disconnected, the nodes are too granular, and the frontier constraint excludes relevant nodes that dense retrieval finds trivially.

**The path to competitiveness requires 4 changes (Phases A–D below):**
1. **Better graphs** — entity linking, cross-document edges, passage-level nodes
2. **Hybrid retrieval** — dense top-k as anchors, graph traversal to expand
3. **LLM reranking** — biggest single win (+33.3%), should be default
4. **Better embeddings** — MiniLM is weak; BGE/E5 would improve both graph and dense

---

## 4. Optimization Pathway — Phase A: Graph Construction

**Goal**: Build better knowledge graphs with richer connectivity so PATHFINDER's frontier expansion actually reaches relevant nodes.

### Task A1: Entity Linking with spaCy NER
- [ ] **Current state**: Entity co-mention edges use spaCy NER but only within the same query's context documents. Cross-document entity links are missing.
- [ ] **Fix**: During KG construction (`01_build_kg.py`), build a global entity index across all documents in the corpus. When two nodes in different documents share an entity, create an edge with W = W_ENTITY (0.70).
- [ ] **Expected impact**: More inter-document edges → frontier can cross document boundaries → R@5 improves on multi-hop queries.
- [ ] **Output**: Modified `experiments/01_build_kg.py` with `--global_entities` flag.

### Task A2: Passage-Level Node Segmentation
- [ ] **Current state**: Each sentence is a node. This creates ~50 nodes/query with ~15 tokens each. The graph is large and sparse.
- [ ] **Alternative**: Each passage/paragraph is a node. This creates ~10 nodes/query with ~100 tokens each. The graph is smaller and denser.
- [ ] **Implementation**: Add `--node_granularity passage` flag to `01_build_kg.py`. Map supporting facts to passage-level by checking if any sentence in the passage is a gold sentence.
- [ ] **Expected impact**: Fewer nodes → denser graph → frontier reaches more relevant nodes within budget. Paragraph-Recall@5 is already 0.708 (vs sentence R@5 of 0.268), suggesting passage-level is the right granularity.
- [ ] **Output**: Modified `01_build_kg.py`, new eval comparing sentence vs passage granularity.

### Task A3: Cross-Document Semantic Edges
- [ ] **Current state**: Semantic edges (cosine ≥ 0.30) are only created within the same query's context. No edges between documents from different queries.
- [ ] **Fix**: For each query, also add semantic edges between nodes from different context documents if cosine ≥ 0.30. This creates inter-document bridges.
- [ ] **Expected impact**: More edges → better connectivity → frontier can reach gold nodes in other documents.
- [ ] **Output**: Modified `01_build_kg.py` with `--cross_doc_edges` flag.

### Task A4: Graph Construction Quality Metrics
- [ ] **Measure**: After building graphs, compute and log: n_components, edge_density, avg_degree, inter_doc_edge_fraction.
- [ ] **Correlate**: Compare these metrics before and after Tasks A1–A3 to quantify improvement.
- [ ] **Output**: `experiments/22_graph_quality.py` script.

**Commands to run locally:**
```cmd
:: Rebuild graphs with improvements (after implementing A1-A3)
python experiments/01_build_kg.py --max_samples 500 --output data/hotpotqa_graphs_v2.pkl --global_entities --cross_doc_edges
python experiments/02_load_2wiki_musique.py --dataset 2wiki --max_samples 500 --output data/2wiki_graphs_v2.pkl
python experiments/02_load_2wiki_musique.py --dataset musique --max_samples 500 --output data/musique_graphs_v2.pkl

:: Compare graph quality
python experiments/22_graph_quality.py --graphs data/hotpotqa_graphs.pkl --label "v1_original" --output results/raw/graph_quality_v1.json
python experiments/22_graph_quality.py --graphs data/hotpotqa_graphs_v2.pkl --label "v2_improved" --output results/raw/graph_quality_v2.json

:: Re-evaluate on improved graphs
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs_v2.pkl --max_samples 500 --output results/hotpotqa_eval_v2.json
```

---

## 5. Optimization Pathway — Phase B: Hybrid Retrieval

**Goal**: Combine dense retrieval's broad coverage with graph traversal's structural coherence.

### Task B1: Dense-Anchor Hybrid (NR-First)
- [ ] **Mechanism**: Take Naive RAG's top-k as primary anchors. Use PATHFINDER's frontier expansion to fill remaining budget slots with graph-connected nodes.
- [ ] **Implementation**: New function `run_pathfinder_hybrid()` in `run_pathfinder.py`:
  1. Get top-3 dense nodes by cosine similarity
  2. Add all 3 to S immediately (bypass frontier constraint for anchors)
  3. Run frontier expansion from each anchor for remaining budget
  4. Merge and deduplicate
- [ ] **Expected impact**: Matches Naive RAG's R@5 (0.31) while adding graph-coherent context. Previous "NR-First Hybrid" achieved R@5=0.2932 on full HotpotQA.
- [ ] **Output**: `experiments/23_hybrid_retrieval.py`.

### Task B2: Dense-Graph Interleaving
- [ ] **Mechanism**: At each greedy step, consider BOTH frontier nodes AND global dense nodes as candidates. Select by Δ_full regardless of source.
- [ ] **Implementation**: Modify `_single_pass()` to add top-5 global dense nodes to the frontier at each step (not just when teleportation triggers). This makes teleportation the default, not the exception.
- [ ] **Expected impact**: Eliminates the R@5 gap by always considering dense candidates. May hurt structural coherence but improves recall.
- [ ] **Output**: Modified `run_pathfinder.py` with `--always_dense` mode.

### Task B3: Two-Stage Retrieval (Dense → Graph Rerank)
- [ ] **Mechanism**: Stage 1: Dense retrieval gets top-20 candidates. Stage 2: PATHFINDER's submodular selection picks the best k=5 from those 20 using graph structure.
- [ ] **Implementation**: Build a subgraph from the top-20 dense nodes, then run PATHFINDER on that subgraph.
- [ ] **Expected impact**: Combines dense recall with graph-based diversity. The submodular coverage function ensures the selected k=5 are non-redundant.
- [ ] **Output**: `experiments/24_two_stage_retrieval.py`.

### Task B4: Adaptive k Selection
- [ ] **Mechanism**: Instead of fixed k=5, use σ(S) to decide how many nodes to return. High σ → return fewer (confident). Low σ → return more (hedge).
- [ ] **Implementation**: After traversal, if σ ≥ τ_high, return top-5. If σ ∈ [τ_low, τ_high), return top-10. If σ < τ_low, return top-20.
- [ ] **Expected impact**: Better R@10/R@20 when PATHFINDER is uncertain, without hurting R@5 when confident.
- [ ] **Output**: `experiments/25_adaptive_k.py`.

**Commands to run locally:**
```cmd
python experiments/23_hybrid_retrieval.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/raw/hybrid_retrieval.json
python experiments/24_two_stage_retrieval.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/raw/two_stage_retrieval.json
python experiments/25_adaptive_k.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/raw/adaptive_k.json
```

---

## 6. Optimization Pathway — Phase C: LLM Integration

**Goal**: Make LLM reranking the default and explore deeper LLM integration.

### Task C1: LLM Reranking as Default
- [ ] **Current state**: LLM reranking is a separate experiment (15_llm_reranking.py). It improved R@5 by +33.3%.
- [ ] **Action**: Integrate LLM reranking into the main `run_pathfinder()` function as an optional post-processing step.
- [ ] **Implementation**: Add `rerank=True` parameter. After traversal, if GROQ_API_KEY is set, rerank S using LLM. Cache reranking results by query embedding (semantic cache).
- [ ] **Expected impact**: R@5 0.24→0.32 by default. With better graphs (Phase A), potentially 0.35+.
- [ ] **Output**: Modified `run_pathfinder.py`.

### Task C2: LLM-Guided Entry Node Selection
- [ ] **Current state**: Entry node is argmax cosine similarity. This may not be the best starting point for multi-hop reasoning.
- [ ] **Action**: Ask LLM to select the best entry node from top-5 dense candidates.
- [ ] **Implementation**: Present top-5 nodes' texts to LLM, ask "Which passage should we start exploring from to answer this question?"
- [ ] **Expected impact**: Better entry node → better traversal → higher R@5. Low cost (1 LLM call).
- [ ] **Output**: `experiments/26_llm_entry_node.py`.

### Task C3: LLM-Based Edge Weighting
- [ ] **Mechanism**: After graph construction, use LLM to reweight edges based on semantic relevance to common query types.
- [ ] **Implementation**: For each edge (u,v), ask LLM: "How relevant are these two passages to each other?" Score ∈ [0,1]. Replace cosine-similarity edge weights with LLM-judged weights.
- [ ] **Expected impact**: Better edge weights → better frontier expansion → higher R@5. High cost (O(|E|) LLM calls at index time).
- [ ] **Output**: `experiments/27_llm_edge_weighting.py`.

### Task C4: Chain-of-Thought Path Scoring
- [ ] **Mechanism**: After traversal, score each root-to-leaf path using LLM CoT: "Given this question, does this reasoning chain (passage1 → passage2 → passage3) support answering it?"
- [ ] **Implementation**: Extract paths from parent tree, send each to LLM for CoT scoring, rerank S by path scores.
- [ ] **Expected impact**: Better reranking than flat passage reranking. Medium cost (O(paths) LLM calls).
- [ ] **Output**: `experiments/28_cot_path_scoring.py`.

**Commands to run locally:**
```cmd
set GROQ_API_KEY=your_key_here
python experiments/26_llm_entry_node.py --graphs data/hotpotqa_graphs.pkl --max_samples 50 --output results/raw/llm_entry_node.json
python experiments/27_llm_edge_weighting.py --graphs data/hotpotqa_graphs.pkl --max_samples 50 --output results/raw/llm_edge_weighting.json
python experiments/28_cot_path_scoring.py --graphs data/hotpotqa_graphs.pkl --max_samples 50 --output results/raw/cot_path_scoring.json
```

---

## 7. Optimization Pathway — Phase D: Embedding & Scale

**Goal**: Upgrade embedding model and run full-scale evaluation.

### Task D1: Embedding Model Upgrade
- [ ] **Current state**: `all-MiniLM-L6-v2` (384-dim, fast but weak).
- [ ] **Upgrade path**: Test `BAAI/bge-large-en-v1.5` (1024-dim), `intfloat/e5-large-v2` (1024-dim), `OpenAI/text-embedding-3-small` (1536-dim).
- [ ] **Implementation**: Add `--encoder_model` flag to `01_build_kg.py`. Re-embed all nodes with the new model.
- [ ] **Expected impact**: Better embeddings → better cosine similarity → better graph edges AND better dense retrieval. This lifts ALL systems, but PATHFINDER benefits more because graph edges also improve.
- [ ] **Output**: Modified `01_build_kg.py`, comparison eval.

### Task D2: Full-Scale Evaluation (N=7,405)
- [ ] **Run full HotpotQA**: All 7,405 validation queries with LLM answers.
- [ ] **Run full 2Wiki**: All 12,576 validation queries.
- [ ] **Run full MuSiQue**: All 2,417 validation queries.
- [ ] **Output**: Full-scale results for paper §7.6.

### Task D3: Multi-Vector FAISS Index
- [ ] **Build FAISS index** at KG construction time for O(log|V|) entry node selection and teleportation lookup.
- [ ] **Benchmark**: Compare latency on |V| = 100, 1,000, 10,000.
- [ ] **Output**: Production-ready indexing pipeline.

### Task D4: Batch Query Processing
- [ ] **Vectorize**: Process multiple queries in parallel using batched numpy operations.
- [ ] **Target**: 100 queries/second on CPU for production deployment.
- [ ] **Output**: `experiments/29_batch_processing.py`.

**Commands to run locally:**
```cmd
:: D1 — Rebuild with better embeddings
python experiments/01_build_kg.py --max_samples 500 --output data/hotpotqa_graphs_bge.pkl --encoder_model BAAI/bge-large-en-v1.5
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs_bge.pkl --max_samples 500 --output results/hotpotqa_eval_bge.json

:: D2 — Full-scale eval (LONG RUN)
set GROQ_API_KEY=your_key_here
python experiments/01_build_kg.py --output data/hotpotqa_graphs_full.pkl
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs_full.pkl --output results/hotpotqa_eval_full.json

python experiments/02_load_2wiki_musique.py --dataset 2wiki --output data/2wiki_graphs_full.pkl
python experiments/05_evaluate.py --graphs data/2wiki_graphs_full.pkl --output results/2wiki_eval_full.json

python experiments/02_load_2wiki_musique.py --dataset musique --output data/musique_graphs_full.pkl
python experiments/05_evaluate.py --graphs data/musique_graphs_full.pkl --output results/musique_eval_full.json
```

---

## 8. Optimization Pathway — Phase E: SOTA Comparison

**Goal**: Direct comparison against published systems on the same benchmarks.

### SOTA Systems to Compare

| System | Paper | Repo | Expected HotpotQA EM |
|---|---|---|---|
| IRCoT | Trivedi et al. 2022 | `github.com/stonybrooknlp/ircot` | 56.5–61.2 |
| SubgraphRAG | Li et al. 2024 (ICLR 2025) | `github.com/Graph-COM/SubgraphRAG` | 41.2–47.8 |
| HippoRAG 2 | Gutiérrez et al. 2025 | `github.com/OSU-NLP-Group/HippoRAG` | 49.8–54.2 |
| PCR | arXiv:2511.18313 | (check paper) | 45.0–50.3 |
| GraphRAG | Edge et al. 2024 | `github.com/microsoft/graphrag` | N/A |
| LightRAG | Guo et al. 2024 | `github.com/HKUDS/LightRAG` | N/A |

### Task E1: Download and Setup SOTA Systems

```cmd
:: Create a siblings directory for SOTA baselines
mkdir C:\Users\yasha\.gemini\antigravity\scratch\baselines
cd C:\Users\yasha\.gemini\antigravity\scratch\baselines

:: IRCoT
git clone https://github.com/stonybrooknlp/ircot.git
cd ircot
pip install -r requirements.txt
cd ..

:: SubgraphRAG (ICLR 2025)
git clone https://github.com/Graph-COM/SubgraphRAG.git
cd SubgraphRAG
pip install -r requirements.txt
cd ..

:: HippoRAG 2
git clone https://github.com/OSU-NLP-Group/HippoRAG.git
cd HippoRAG
pip install -r requirements.txt
cd ..

:: LightRAG
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
pip install -r requirements.txt
cd ..

:: GraphRAG (Microsoft)
git clone https://github.com/microsoft/graphrag.git
cd graphrag
pip install -r requirements.txt
cd ..

cd C:\Users\yasha\.gemini\antigravity\scratch\Ragidea
```

### Task E2: Run SOTA Baselines on HotpotQA

```cmd
:: IRCoT — uses GPT-3.5/4, needs OpenAI API key
set OPENAI_API_KEY=your_openai_key
cd C:\Users\yasha\.gemini\antigravity\scratch\baselines\ircot
python run.py --dataset hotpotqa --split validation --max_samples 500 --output ../../Ragidea/results/raw/ircot_hotpotqa.json
cd C:\Users\yasha\.gemini\antigravity\scratch\Ragidea

:: SubgraphRAG — check their README for exact run commands
cd C:\Users\yasha\.gemini\antigravity\scratch\baselines\SubgraphRAG
:: Typically: python main.py --dataset hotpotqa --max_samples 500
cd C:\Users\yasha\.gemini\antigravity\scratch\Ragidea

:: HippoRAG 2 — check their README
cd C:\Users\yasha\.gemini\antigravity\scratch\baselines\HippoRAG
:: Typically: python eval.py --dataset hotpotqa --max_samples 500
cd C:\Users\yasha\.gemini\antigravity\scratch\Ragidea

:: LightRAG
cd C:\Users\yasha\.gemini\antigravity\scratch\baselines\LightRAG
:: Typically: python evaluate.py --dataset hotpotqa --max_samples 500
cd C:\Users\yasha\.gemini\antigravity\scratch\Ragidea
```

### Task E3: Fair Comparison Protocol
- [ ] **Same embedding model**: Run all systems with `all-MiniLM-L6-v2` for fair comparison. Some SOTA systems use stronger embeddings by default — note this.
- [ ] **Same generator LLM**: Use Llama 3.3-70B via Groq for all systems. Some SOTA systems use GPT-4 — note this.
- [ ] **Same N**: Evaluate all systems on the same 500-query subset.
- [ ] **Same metrics**: EM, F1, Recall@5, Recall@10.
- [ ] **Output**: `results/sota_comparison.md` with unified comparison table.

### Task E4: Paper Comparison Table
- [ ] **Update §7.6.3**: Replace "not run" entries with actual numbers.
- [ ] **Add PATHFINDER + LLM reranking** as a separate row (best configuration).
- [ ] **Add caveat**: PATHFINDER uses sentence-level nodes + MiniLM + Groq 70B; SOTA systems use passage-level + stronger embeddings + GPT-4. Fair comparison requires same-stack evaluation.

---

## 9. Theoretical Work Remaining

| Task | Status | Notes |
|---|---|---|
| Teleportation (1−1/e) proof | ✅ DONE | Corollary added to §5.1 |
| Submodularity scope clarification | ✅ DONE | Remarks added to §4.1 |
| Bound tightness (Feige construction) | TODO | Phase 11, Task 11.2 — requires constructing a graph family where greedy achieves exactly (1−1/e) |
| Joint correlation modeling | TODO | Phase 11, Task 11.4 — MRF or GNN-based coverage model |
| Heterogeneous token cost bound | TODO | Phase 11, Task 11.5 — knapsack-constrained submodular maximization |

---

## 10. Repo Cleanup Checklist

- [x] Delete `FUTURE_WORK.md` (consolidated into PLAN.md)
- [x] Delete `experiments/analysis.md` (issues captured in PLAN.md)
- [x] Delete `experiments/evaluate_fixes.py` (stale one-off script)
- [x] Delete `experiments/results/results_full.json` (14MB stale data)
- [x] Fix SubgraphRAG citation in paper
- [x] Standardize Zarrinkia citation
- [x] Remove §7.7 Implementation Plan from paper (merged into PLAN.md)
- [x] Remove TODO placement note from §6
- [x] Update `experiments/README.md` with actual results
- [x] Update `results/multi_benchmark.md` with Phase 2 data
- [ ] Clean stale JSON files in `results/raw/` from Phase 1 (anchor_quality_200.json, hotpotqa_200_*.json, root_cause_analysis.json, sigma_calibration_200.json, coverage_ratio_*.json)
- [ ] Add `results/raw/` to `.gitignore` for future evals (keep current results, ignore new ones)
- [ ] Add `README.md` at repo root with quickstart guide

---

## 11. Verification Commands

```cmd
:: Unit tests
pytest pathfinder/tests/

:: Quick eval (N=500, no LLM)
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/hotpotqa_eval.json

:: Print metrics
python experiments/print_metrics.py

:: Generate plots
python results/make_plots.py

:: Full eval with LLM (LONG RUN)
set GROQ_API_KEY=your_key_here
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs_full.pkl --output results/hotpotqa_eval_full.json
```

---

## Priority Order for Maximum Impact

1. **Phase A2 (passage-level nodes)** — easiest, biggest expected impact
2. **Phase B1 (dense-anchor hybrid)** — second biggest impact, moderate effort
3. **Phase C1 (LLM reranking as default)** — already proven (+33.3%), just integrate
4. **Phase D1 (better embeddings)** — lifts all systems, moderate effort
5. **Phase B3 (two-stage retrieval)** — novel contribution, moderate effort
6. **Phase E (SOTA comparison)** — needed for paper, requires external codebases
7. **Phase A1 (entity linking)** — moderate impact, moderate effort
8. **Phase C2 (LLM entry node)** — low cost, moderate impact
