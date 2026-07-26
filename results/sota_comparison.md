# PATHFINDER vs SOTA — Comparison Setup & Results

> Status: **SETUP ONLY (PLAN §5)**. No runs performed yet. All result cells are
> **pending Phase 5 (§6)** and will be filled by the user after the eval block.
> Adapter: `experiments/30_sota_adapter.py` (uniform EM/F1/R@5/R@10 scoring,
> HotpotQA-official normalisation shared across every system).

## 1. Fair-comparison protocol (PLAN §5.3)

| Rule | Value |
|------|-------|
| Query subset | Same **500-query** HotpotQA `distractor` validation subset for **every** system |
| Generator | **Llama-3.3-70B-versatile via Groq** for all systems (unless a system ships its own) |
| Metrics | EM, F1 (HotpotQA official normalisation), R@5, R@10 |
| Scoring | Single adapter `30_sota_adapter.py::score_system` — identical code path for all |
| Embeddings | PATHFINDER default `all-MiniLM-L6-v2`; BGE variant noted separately |
| Token budget | K_tok = 2048 for PATHFINDER variants |

> **Honesty note** — where a SOTA system uses a **stronger generator (GPT-4)** or
> **stronger embeddings** than our default, that is flagged in the Notes column so
> the comparison is not misrepresented as strictly equal-capacity.

## 2. Systems & how to run each (PLAN §5.2)

### PATHFINDER variants (ours, no external code)
```cmd
:: Coherent (default graph traversal)
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs_passage_bge.pkl --max_samples 500 --output results/raw/sota_pathfinder_coherent.json
:: Flat (always-dense teleportation)
:: set always_dense=True via run_pathfinder in an eval wrapper
:: Hybrid (dense-anchor)
python experiments/23_hybrid_retrieval.py --graphs data/hotpotqa_graphs_passage_bge.pkl --max_samples 500 --output results/raw/sota_pathfinder_hybrid.json
```

### IRCoT (Trivedi et al. 2023)
- Interleaves chain-of-thought steps with retrieval; each CoT sentence guides the next query.
- Repo: `https://github.com/StonyBrookNLP/ircot`
- Setup: needs an LLM (originally GPT-3.5/4). For parity, drive with Llama-3.3-70B via Groq.
- Adapter: wrap its per-question retrieved passage list into `score_system(..., predict_fn=...)`.
- **Pending §6.**

### SubgraphRAG (Ma et al. 2024)
- Retrieves a compact subgraph then generates; uses a learned subgraph scorer.
- Repo: `https://github.com/Graph-COM/SubgraphRAG`
- Setup: ships its own KG (WebQSP/CWQ focus); requires mapping HotpotQA → its triple format.
- **Pending §6** (KG-format adapter needed).

### HippoRAG 2 (Gutiérrez et al. 2025)
- Personalised-PageRank over an OpenIE knowledge graph; strong multi-hop recall.
- Repo: `https://github.com/OSU-NLP-Group/HippoRAG`
- Setup: OpenIE extraction (uses an LLM) — flag generator + extraction cost in Notes.
- **Pending §6.**

### LightRAG (Guo et al. 2024)
- Dual-level (entity + community) graph retrieval with LLM keyword extraction.
- Repo: `https://github.com/HKUDS/LightRAG`
- Setup: requires an LLM for entity/keyword extraction; note generator used.
- **Pending §6.**

## 3. Results (PENDING Phase 5)

| System | EM | F1 | R@5 | R@10 | Notes |
|--------|----|----|-----|------|-------|
| PATHFINDER-Coherent (passage) | pending | pending | pending | pending | ours, MiniLM/BGE |
| PATHFINDER-Flat (always-dense) | pending | pending | pending | pending | ours |
| PATHFINDER-Hybrid (dense-anchor) | pending | pending | pending | pending | ours |
| Naive RAG (dense top-k) | pending | pending | pending | pending | baseline |
| IRCoT | pending | pending | pending | pending | generator: Llama-3.3-70B (parity) |
| SubgraphRAG | pending | pending | pending | pending | KG-format adapter |
| HippoRAG 2 | pending | pending | pending | pending | OpenIE extraction cost noted |
| LightRAG | pending | pending | pending | pending | LLM keyword extraction noted |

> Fill via `experiments/30_sota_adapter.py::score_system` so every number shares one
> normalisation and metric implementation. Do **not** hand-transcribe from each
> system's own eval script (different normalisation → unfair comparison).
