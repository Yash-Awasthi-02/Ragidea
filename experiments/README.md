# PATHFINDER Experiments

No-cost experiment pipeline for evaluating PATHFINDER on HotpotQA, 2WikiMultihopQA, and MuSiQue.

## Stack

| Component | Tool | Cost |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Free, local |
| NER | spaCy `en_core_web_sm` | Free, local |
| Graph ops + PageRank | networkx | Free, local |
| Domain embeddings | scikit-learn PCA | Free, local |
| LLM generation | Groq API (Llama 3.3-70B) | Free tier |
| Datasets | HuggingFace `datasets` | Free |

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Quick Start

```bash
# Build sentence-level KGs (HotpotQA, 500 samples)
python experiments/01_build_kg.py --max_samples 500 --output data/hotpotqa_graphs.pkl

# Build passage-level KGs (Phase A2 — recommended)
python experiments/01b_build_kg_passage.py --max_samples 500 --output data/hotpotqa_graphs_passage.pkl

# Load 2Wiki / MuSiQue
python experiments/02_load_2wiki_musique.py --dataset 2wiki --max_samples 500 --output data/2wiki_graphs.pkl
python experiments/02_load_2wiki_musique.py --dataset musique --max_samples 500 --output data/musique_graphs.pkl

# Evaluate (no LLM needed)
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/hotpotqa_eval.json

# Print consolidated metrics
python experiments/print_metrics.py

# Generate plots
python results/make_plots.py
```

## Experiment Scripts

| File | Phase | Purpose | API Key? |
|---|---|---|---|
| `01_build_kg.py` | — | Build sentence-level KGs from HotpotQA | No |
| `01b_build_kg_passage.py` | A2 | Build passage-level KGs (~13 nodes/query) | No |
| `02_load_2wiki_musique.py` | — | Load 2Wiki/MuSiQue datasets | No |
| `03_grid_search.py` | 7 | Hyperparameter grid search (48 configs) | No |
| `04_confidence_calibration.py` | 6 | Compare 3 σ models (product, geom, bottleneck) | Optional |
| `05_evaluate.py` | — | Full evaluation (all metrics, multi-granularity) | Optional |
| `05b_teleportation_ablation.py` | 5 | Pure Graph vs Teleport vs Naive RAG | No |
| `05c_teleportation_sensitivity.py` | 5 | Sweep θ/TopK/MAX_TELEPORTS | No |
| `05d_teleportation_impact.py` | 5 | Per-query teleportation impact | No |
| `06_plot_sigma.py` | — | σ calibration plots | No |
| `07_confidence_model_selection.py` | 6 | Train classifier to select σ model per query | No |
| `08_nli_sufficiency.py` | 6 | NLI-based sufficiency check | No |
| `09_bayesian_optimization.py` | 7 | Optuna TPE weight optimization | No |
| `10_faiss_teleport.py` | 8 | FAISS vs numpy teleportation lookup | No |
| `11_multi_vector_teleport.py` | 8 | Domain-aware teleportation | No |
| `12_learned_teleport_threshold.py` | 8 | Learn when to trigger teleportation | No |
| `13_dynamic_edge_synthesis.py` | 8 | Synthesize edges between clusters | No |
| `14_bandit_weight_learning.py` | 9 | Thompson Sampling weight exploration | No |
| `15_llm_reranking.py` | 10 | LLM reranking of PATHFINDER candidates | **Yes** |
| `16_llm_guided_traversal.py` | 10 | LLM selects frontier nodes | **Yes** |
| `17_nli_path_verification.py` | 10 | NLI entailment on root-to-leaf paths | No |
| `18_llm_sufficiency_oracle.py` | 10 | LLM-based sufficiency check | **Yes** |
| `19_heterogeneous_llms.py` | 12 | Evaluate with different Groq models | **Yes** |
| `20_latency_profiling.py` | 12 | Fine-grained timing breakdown | No |
| `21_graph_connectivity.py` | 8 | Correlate connectivity with R@5 gap | No |
| `23_hybrid_retrieval.py` | B | Dense-anchor, interleaving, two-stage | No |
| `25_rerank_eval.py` | C | Full eval with LLM reranking integrated | **Yes** |
| `run_pathfinder.py` | — | Algorithm 1 implementation | No |
| `run_baselines.py` | — | Naive RAG, BFS, Spreading Activation | No |
| `generate_answers.py` | — | Groq LLM generation + EM/F1 | **Yes** |
| `print_metrics.py` | — | Print consolidated metrics | No |

## Key Results

| Configuration | R@5 | vs Naive RAG |
|---|---|---|
| PATHFINDER (sentence, original) | 0.257 | -12.3% |
| PATHFINDER (passage, original) | 0.644 | -9.0% |
| Dense-Anchor Hybrid (passage) | **0.708** | **0.0%** (matches) |
| Naive RAG (passage) | 0.708 | — |
| PATHFINDER + LLM Rerank (sentence) | 0.320 | +3.2% |

## Data Validity Notice (2026-07-26)

Four LLM-dependent result files were **deleted** because they were corrupted by
**Groq free-tier rate-limit exhaustion**: `generate_answers.py` silently returned
empty predictions after retries, which were scored as EM=0/F1=0. Affected files:
`heterogeneous_llm_8b.json`, `heterogeneous_llm_70b_passage.json`,
`heterogeneous_llm_70b_v2.json`, `rerank_passage.json`. Their **R@5 values were valid**
(retrieval needs no LLM); only EM/F1 were garbage. They are **regenerated in PLAN.md
§6.2** with a fresh API key. The generator now refuses to silently zero-out results —
it flags `"quota_exhausted": true` and counts rate-limited queries instead.
