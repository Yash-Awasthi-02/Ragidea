# PATHFINDER Experiments

No-cost experiment pipeline for evaluating PATHFINDER on HotpotQA, 2WikiMultihopQA, and MuSiQue.

## Stack

| Component | Tool | Cost |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Free, local |
| NER | spaCy `en_core_web_sm` | Free, local |
| Graph ops + PageRank | networkx | Free, local |
| Domain embeddings | scikit-learn PCA | Free, local |
| LLM generation | Groq API (Llama 3.3-70B) | Free tier, no credit card |
| Datasets | HuggingFace `datasets` | Free |

**Estimated compute cost: < $0** (Groq free tier covers all queries)
**Estimated disk: ~1.1 GB packages + ~650 MB data/output**

## Setup

```bash
# 1. Install PyTorch CPU-only (saves ~4 GB vs default)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Get Groq API key (free, no credit card)
# https://console.groq.com/keys
export GROQ_API_KEY=gsk_...
```

## Run

```bash
# Smoke test — 100 queries, ~5 min
bash run_all.sh --fast

# No LLM — Recall@5 only, no Groq needed
bash run_all.sh --no_llm

# Full run — 7405 queries, ~3 hours
bash run_all.sh

# Visualise σ calibration after run
python 06_plot_sigma.py --results results/results.json
```

## Files

| File | Purpose |
|---|---|
| `01_build_kg.py` | Build per-query KGs from HotpotQA candidate documents |
| `02_load_2wiki_musique.py` | Load 2WikiMultihopQA and MuSiQue datasets |
| `03_grid_search.py` | Hyperparameter grid search (α, γ, ε) — 48 configs |
| `04_confidence_calibration.py` | Compare 3 σ models (product, geometric mean, bottleneck) |
| `run_pathfinder.py` | Algorithm 1 — greedy submodular traversal with teleportation |
| `run_baselines.py` | Naive RAG, BFS 2-hop, Spreading Activation |
| `generate_answers.py` | Groq LLM generation + HotpotQA EM/F1 |
| `05_evaluate.py` | All metrics (§7.5.1–7.5.9) with multi-granularity recall |
| `06_plot_sigma.py` | σ calibration plots (bucket analysis + calibration curve) |
| `print_metrics.py` | Print consolidated metrics from all benchmark evals |
| `run_all.sh` | End-to-end orchestrator |

## Metrics

| Metric | What it validates |
|---|---|
| Recall@5 | Retrieval coverage vs. baselines |
| EM / F1 | End-to-end answer quality |
| Node Expansion Rate | Traversal efficiency |
| Latency p50/p95 | Production viability |
| **σ Calibration** | **Core claim: path confidence predicts answer correctness** |
| Post-feedback Δ Recall@5 | Online learning effectiveness |
| Anchor quality rank | Entry node selection quality |
| Weight ablation | Validates α>β=γ>δ=ε ordering |

## Expected Results (empirical — N=500 per dataset)

| System | HotpotQA Recall@5 | HotpotQA Recall@10 |
|---|---|---|
| Naive RAG | 0.3100 | 0.3100 |
| BFS 2-hop | 0.1400 | 0.3020 |
| Spreading Activation | 0.1900 | 0.3520 |
| **PATHFINDER** | **0.2680** | **0.3500** |

σ calibration: Product σ collapses to min=0.0077; geometric mean (min=0.272) and bottleneck (min=0.300) models address decay. Spearman ρ pending LLM evaluation.
