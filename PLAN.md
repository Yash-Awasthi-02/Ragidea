# PATHFINDER — Research Master Plan (v3)

| Field | Value |
|-------|-------|
| **Project** | PATHFINDER — Submodular Coverage Maximization over Multidimensional KGs for Multi-Hop RAG |
| **Repo** | https://github.com/Yash-Awasthi-02/Ragidea |
| **Paper** | `pathfinder-paper.md` |
| **Plan version** | **v3** (supersedes v2) |
| **Last updated** | 2026-07-26 |
| **Owner** | Yash Vaibhav Awasthi |
| **Status** | Phases 0–D largely complete · Generator-quality + external SOTA pending |

---

## How to use this document

This is a **research operating system**, not a scratchpad.

1. **§0** — non-negotiable rules (read first).
2. **§1** — scientific framing: claims, hypotheses, success criteria.
3. **§2** — locked evidence ledger (what is already proven; do not re-run).
4. **§3** — open scientific problems (ranked by expected paper impact).
5. **§4** — experimental design protocol (how every future run must be done).
6. **§5** — optimization pathway (ordered workstreams A→F).
7. **§6** — paper & theory deliverables.
8. **§7** — SOTA comparison protocol.
9. **§8** — evaluation runbook (user-only; long jobs).
10. **§9** — milestone tracker & decision gates.
11. **§10** — risk register & anti-patterns.

Inspired by practices from strong multi-hop RAG work (HippoRAG, IRCoT, SubgraphRAG) and recent controlled ablation protocols (e.g. agentic-RAG component ablations): **pre-register hypotheses, change one factor at a time, fix budgets, share one scorer, log every run**.

---

## §0. Operating rules (non-negotiable)

| # | Rule | Why |
|---|------|-----|
| R1 | **Never run full evaluations from the agent.** Jobs take 9–18 h and burn API quota. | Cost / time |
| R2 | **Never call** `05*.py`, `15–19*.py`, `25*.py`, `generate_answers.py`, or any script that loads datasets / hits Groq — unless the **user** is running it. | Same |
| R3 | **Never push, force-push, or rewrite history** unless the user explicitly provides a PAT and asks. | Repo safety |
| R4 | Validation = print the **exact command** + **expected output**, then **stop** and wait for the user paste. | Division of labor |
| R5 | Do all no-eval work first: cleaning, code, tests, theory, paper, adapters. | Throughput |
| R6 | Before any code change: state **WHY → expected impact → theoretical implication → files touched**. | Scientific rigor |
| R7 | **One variable per experiment.** No multi-factor confounds. | Causal claims |
| R8 | **One scorer for all systems** (`30_sota_adapter.py`). Never mix vendor EM/F1. | Fair comparison |
| R9 | Every run writes: config snapshot + git SHA + seed + wall time + `quota_exhausted` flag. | Reproducibility |
| R10 | Corrupted / rate-limited EM files are **deleted, not averaged**. | Data integrity |

**Safe local checks (seconds, no API):**
```cmd
pytest pathfinder/tests/          :: expect 58 passed
python -c "import ast,glob;[ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('experiments/*.py')]"
```

---

## §1. Scientific framing

### 1.1 Core claim (what the paper must defend)

> Under a **fixed token budget**, greedy submodular coverage maximization over a multidimensional knowledge graph retrieves a **graph-coherent** evidence set with a **provable (1 − 1/e)** approximation to the optimal connected subtree, and this coherence **improves multi-hop answer quality** relative to flat dense retrieval — especially when node granularity and embeddings are strong enough that the graph is a useful inductive bias.

### 1.2 What is *not* claimed

- PATHFINDER does **not** claim to beat every agentic multi-step system that spends many LLM calls per query.
- PATHFINDER does **not** claim sentence-level MiniLM graphs are competitive (empirically they are not).
- The (1 − 1/e) guarantee is w.r.t. the **optimal graph-coherent connected subtree**, not w.r.t. the unrestricted flat top-k optimum.

### 1.3 Pre-registered hypotheses (test in this order)

| ID | Hypothesis | Primary metric | Gate |
|----|------------|----------------|------|
| **H1** | Passage nodes ≫ sentence nodes for multi-hop R@k | R@5, R@10 | ✅ Confirmed (locked) |
| **H2** | Stronger bi-encoders lift both PATHFINDER and RAG; relative gap shrinks | R@5 Δ(PF−RAG) | ✅ Confirmed (locked) |
| **H3** | Dense-anchor hybrid matches RAG R@5 while preserving path structure | R@5, path coherence | ✅ Confirmed on BGE-passage N=500 |
| **H4** | LLM listwise/cross-encoder rerank improves R@5 over greedy order | R@5 Δ | ⚠ Partial (sentence only; passage rerun needed) |
| **H5** | Graph-coherent context improves **EM/F1** over equal-R@k flat context | EM, F1 | 🔴 Open (blocked by generator quality / quota) |
| **H6** | Teleportation helps when graph is sparse; hurts when dense & well-connected | R@5 by connectivity quartile | ⚠ Partial ablations exist |
| **H7** | PATHFINDER is competitive with HippoRAG / IRCoT / LightRAG under **matched generator + budget** | EM, F1, R@5, latency | 🔴 Not started |
| **H8** | Confidence σ is calibrated (Spearman ρ with correctness > 0.3) | Spearman ρ | ⚠ Weak with heuristic; LLM-σ partial |

### 1.4 Success criteria (paper-ready)

| Tier | Bar | Status |
|------|-----|--------|
| **Minimum publishable** | Passage-level full Hotpot R@5 ≥ 0.65; theory section complete; ≥3 external baselines under fair protocol | R@5 ✅ · theory ~80% · SOTA ❌ |
| **Strong** | EM/F1 gap vs naive RAG statistically significant (bootstrap p < 0.05) on Hotpot N≥500 with matched generator | ❌ |
| **Outstanding** | Match or beat HippoRAG-class R@5 on Hotpot under matched encoder; positive transfer on 2Wiki + MuSiQue | ❌ |

---

## §2. Locked evidence ledger

> **Do not re-run these.** Cite them. Re-running wastes days and risks quota corruption.

### 2.1 Primary retrieval results (HotpotQA distractor)

| Setting | N | Encoder | PF R@5 | RAG R@5 | PF R@10 | Notes |
|---------|---|--------:|-------:|--------:|--------:|-------|
| Sentence full | 7405 | MiniLM | **0.257** | 0.293 | **0.335** | Paper baseline |
| Passage N=500 | 500 | MiniLM | **0.644** | 0.708 | 0.764 | Phase A breakthrough |
| Passage full | 7405 | MiniLM | **0.667** | 0.705 | **0.771** | Phase D2 |
| Sentence N=500 | 500 | BGE | **0.466** | 0.484 | 0.480 | Phase D1 |
| Passage N=500 | 500 | BGE | **0.864** | 0.870 | 0.866 | Phase D1 |
| Hybrid dense-anchor (BGE pass.) | 500 | BGE | **0.870** | 0.870 | 0.964 | = RAG @5; path structure kept |

**Files:**  
`results/hotpotqa_eval_full.json` · `hotpotqa_eval_passage.json` · `hotpotqa_eval_passage_full.json` · `hotpotqa_eval_bge.json` · `hotpotqa_eval_bge_passage.json` · `results/raw/hybrid_bge_passage.json`

### 2.2 Cross-benchmark (sentence MiniLM, full)

| Dataset | N | PF R@5 | RAG R@5 | PF R@10 |
|---------|--:|-------:|--------:|--------:|
| 2WikiMultihopQA | 12576 | 0.235 | 0.325 | 0.373 |
| MuSiQue | 2417 | 0.008 | 0.004 | 0.012 |

> MuSiQue near-zero is a **known open problem** (graph construction / hop structure mismatch) — see §3.P3.

### 2.3 Component studies (valid R@k / systems metrics)

| Study | Key finding | File |
|-------|-------------|------|
| Teleportation ablation | Hybrid teleport Hotpot R@10 0.350 vs pure graph 0.330 | `raw/teleportation_ablation.json` |
| Bandit weight learning | late_recall 0.30 > early 0.18 | `raw/bandit_weight_learning.json` |
| LLM rerank (sentence) | R@5 0.24 → 0.32 (Δ+0.08) | `raw/llm_reranking.json` |
| LLM-guided traversal | greedy 0.20 → LLM 0.30 | `raw/llm_guided_traversal.json` |
| Latency | total mean 3.5 ms, p95 7.1 ms (retrieval only) | `raw/latency_profiling.json` |
| NLI sufficiency | agreement 0.085 — **miscalibrated, do not trust** | `raw/nli_sufficiency.json` |
| σ calibration (LLM) | see file; weak–moderate | `raw/confidence_calibration_llm.json` |

### 2.4 Intentionally deleted (invalid EM/F1)

Silent Groq rate-limit → empty predictions recorded as EM=0:

- ~~`raw/rerank_passage.json`~~
- ~~`raw/heterogeneous_llm_70b_v2.json`~~
- ~~`raw/heterogeneous_llm_70b_passage.json`~~
- ~~`raw/heterogeneous_llm_8b.json`~~

**Root cause fixed in code:** `generate_answers.py` now returns `RATE_LIMITED` sentinel and writes `quota_exhausted` / `n_rate_limited`. **Regenerate only under §8 with a fresh key.**

### 2.5 Engineering baseline

| Item | Value |
|------|------:|
| Unit tests | **58/58 pass** |
| Experiment scripts | 33 |
| Core algorithm | `run_pathfinder.py` (teleport, submodular F, ALPHA) |
| SOTA adapter | `30_sota_adapter.py` (setup only) |
| HEAD | track `origin/master` |

---

## §3. Open scientific problems (ranked)

| Pri | ID | Problem | Why it matters | Workstream |
|----:|----|---------|----------------|------------|
| 1 | **P1** | **Generator quality / EM-F1 validity** | Without clean EM/F1, H5 and paper QA tables are empty | E |
| 2 | **P2** | **Fair external SOTA** | Reviewers require HippoRAG / IRCoT / LightRAG-class comparisons | F |
| 3 | **P3** | **MuSiQue collapse** | Cross-benchmark story is incomplete; risks “Hotpot-only method” critique | C |
| 4 | **P4** | **When does graph help? (error analysis)** | Need connectivity / hop / bridge-entity slices proving *mechanism* | C |
| 5 | **P5** | **Rerank + hybrid composition** | H4 incomplete at passage+BGE; composition may be the production default | D |
| 6 | **P6** | **Teleportation theory ↔ data** | Guarantee remarks need empirical regime map (sparse vs dense graphs) | B + theory |
| 7 | **P7** | **σ / sufficiency calibration** | Stopping rules and confidence reporting currently weak | D |
| 8 | **P8** | **End-to-end latency with LLM** | Retrieval is ms; generation dominates — report both | E |

---

## §4. Experimental design protocol

### 4.1 Fixed evaluation contract (every future run)

| Knob | Default (unless hypothesis says otherwise) |
|------|--------------------------------------------|
| Query split | HotpotQA **distractor validation** |
| Pilot N | **500** fixed subset (hash-stable; document IDs) |
| Full N | **7405** (Hotpot), official full for 2Wiki/MuSiQue |
| Token budget K_tok | **2048** |
| Top-k report | R@5, R@10, R@20 + paragraph-R@5 |
| QA metrics | Hotpot **official** EM/F1 via `30_sota_adapter.score_system` |
| Generator (parity) | **Llama-3.3-70B-versatile @ Groq** |
| Default encoder | **BGE** for new work; MiniLM kept for legacy tables |
| Default nodes | **Passage-level** |
| Default retrieval | **Dense-anchor hybrid** (k_anchor ∈ {3,5}) |
| Seeds | Record `seed` in JSON; default 42 |
| Stats | Paired bootstrap 10k resamples on query-level scores; report 95% CI |

### 4.2 Ablation discipline (one factor)

Hold constant: dataset subset, encoder, K_tok, generator, scorer.  
Vary **exactly one** of: node granularity · encoder · hybrid mode · teleport · rerank · weights · sufficiency.

Name outputs:
```
results/raw/{experiment}_{factor}_{value}_n{N}_{encoder}.json
```

### 4.3 Run card (paste into every result JSON `meta`)

```json
{
  "git_sha": "...",
  "script": "experiments/XX_....py",
  "hypothesis_id": "H5",
  "n": 500,
  "encoder": "BAAI/bge-base-en-v1.5",
  "node_level": "passage",
  "retrieval_mode": "dense_anchor_5",
  "k_tok": 2048,
  "generator": "llama-3.3-70b-versatile",
  "seed": 42,
  "quota_exhausted": false,
  "n_rate_limited": 0,
  "wall_time_s": 0
}
```

### 4.4 Decision rule

- **Promote** a change to default only if pilot N=500 improves primary metric with CI not covering 0 **and** no regression >1 pt on secondary metric.
- **Full-scale** only after pilot promote (except paper-required full tables already locked).

---

## §5. Optimization pathway (workstreams)

```
 A. Repo hygiene          ──► done / maintain
 B. Algorithm defaults    ──► code complete; verify flags
 C. Retrieval science     ──► error analysis, MuSiQue, teleport regimes
 D. Calibration & rerank  ──► compose best stack on pilot
 E. Generator quality     ──► clean EM/F1 (USER eval)
 F. External SOTA         ──► adapters + matched runs (USER eval)
        │
        ▼
 G. Paper lock + camera-ready tables
```

### Workstream A — Hygiene & reproducibility ✅ mostly done

| Task | Status | Notes |
|------|--------|-------|
| Delete corrupted EM JSONs | ✅ | §2.4 |
| Rate-limit sentinel + `quota_exhausted` | ✅ | `generate_answers.py` |
| `.gitignore` raw dumps / logs | ✅ | maintain |
| Formal property tests | ✅ | 58 tests |
| Config/meta in eval outputs | ⬜ | add `meta` block to `05_evaluate` / adapter writers |
| Single `print_metrics.py` dashboard | ⬜ | include BGE + hybrid + passage full |

### Workstream B — Algorithm defaults (no full eval)

**Goal:** production path = passage + BGE + dense-anchor + safe teleport flags.

| Task | WHY | Expected impact | Files |
|------|-----|-----------------|-------|
| B1. Default `node_level=passage` in eval entrypoints | H1 locked | Prevent accidental sentence runs | `05_evaluate.py`, README |
| B2. Default hybrid `dense_anchor_5` behind flag | H3 locked | Match RAG R@5 with paths | `run_pathfinder.py`, `23_hybrid_retrieval.py` |
| B3. Encoder CLI `--encoder {minilm,bge}` | H2 | Reproducible D1 | `01c_*`, `05_*` |
| B4. Teleport: document regimes; default off on dense BGE-passage, on for sparse sentence | H6 | Avoid silent regressions | `run_pathfinder.py`, paper § |
| B5. Weight vector freeze + load path from bandit best | reproducibility | Stable F | `14_*`, `run_pathfinder.py` |

**Promote criteria:** unit tests green; dry-run on 3 queries prints expected mode string.

### Workstream C — Retrieval science (mechanism)

**Goal:** explain *when* and *why* PATHFINDER wins/loses — reviewer-proof analysis.

| Exp | Design | Metric | N | Owner |
|-----|--------|--------|---|-------|
| C1. **Error taxonomy** | Slice by #hops, bridge-entity degree, graph component size, dense-rank of gold | conditional R@5 | 500 BGE-pass | code now / run user |
| C2. **Connectivity quartiles × teleport** | 2×4 ablation | R@5 | 500 | user |
| C3. **MuSiQue root-cause** | Compare gold support overlap vs retrieved; inspect KG density | diagnostic report | 200 | code+user |
| C4. **Passage-full BGE** | Same protocol as D2 with BGE graphs | R@5/R@10 | 7405 | user long |
| C5. **2Wiki passage+BGE pilot** | Transfer test | R@5 | 500 | user |

**Deliverable:** `results/analysis/error_taxonomy.md` + one paper figure (win/loss by hop).

### Workstream D — Stack composition (rerank · σ · sufficiency)

**Goal:** best *system* under fixed budget — not more isolated gadgets.

| Exp | Design | Gate |
|-----|--------|------|
| D1. Cross-encoder / LLM rerank on **BGE-passage hybrid** candidates | R@5 Δ on N=500 | promote if Δ≥+0.02 |
| D2. Two-stage: retrieve 20 → rerank 5 (already in hybrid file) | EM pilot with clean generator | after E1 |
| D3. Recalibrate NLI threshold on labeled sufficiency subset | agreement ≥0.6 | else drop NLI from default |
| D4. σ model: logistic on {coverage gap, frontier entropy, rerank margin} | Spearman ρ≥0.3 | else report uncalibrated |

**Anti-goal:** do not ship LLM-guided traversal as default (latency/cost) unless D1 fails and H5 needs it.

### Workstream E — Generator quality (USER eval)

**Goal:** valid EM/F1 for H5.

| Step | Action |
|------|--------|
| E0 | Fresh Groq key; verify `RATE_LIMITED` path with 5 queries |
| E1 | Pilot N=100 passage+BGE+hybrid → EM/F1; abort if `quota_exhausted` |
| E2 | Pilot N=500 same stack; bootstrap CI vs naive RAG **same contexts budget** |
| E3 | Matched-context study: give RAG and PF **identical token count**; isolate coherence |
| E4 | Heterogeneous LLMs 8B vs 70B **only after** E2 clean |
| E5 | Report retrieval ms vs generation s separately |

**Context-build rules:** `max_chars` cap; never silently empty-answer; drop queries with `RATE_LIMITED` from EM denominator **or** mark separately (state choice in paper).

### Workstream F — External SOTA (USER eval)

See **§7**. Code adapters first; runs last.

### Suggested default stack (post D/E)

```
Passage nodes + BGE encoder
  → Dense-anchor hybrid (k=5)
  → optional cross-encoder rerank top-5
  → Llama-3.3-70B generation
  → official EM/F1
```

---

## §6. Theory & paper deliverables

### 6.1 Theory checklist

| Item | Status | Notes |
|------|--------|-------|
| Submodularity of coverage F | ✅ | scope clarified |
| Greedy (1−1/e) under cardinality | ✅ | |
| Heterogeneous token costs → Sviridenko remark | ✅ | |
| Teleportation vs guarantee (when teleport breaks connected-subtree premise) | ⚠ | finish proof remark + empirical map |
| Multidimensional facet independence assumptions | ⚠ | state limitations |
| Formal tests for diminishing returns / monotonicity | ✅ | `test_formal_properties.py` |

### 6.2 Paper structure targets (`pathfinder-paper.md`)

| Section | Action |
|---------|--------|
| Abstract | Lead with passage R@5 0.67 / BGE 0.86; theory guarantee; hybrid parity |
| Method | Default = passage · hybrid; sentence MiniLM as ablation |
| Theory | Guarantee scope box; teleport caveat |
| Experiments | Tables from §2 ledger + new E/F only |
| Ablations | One-factor table (granularity, encoder, hybrid, teleport, rerank) |
| Analysis | Error taxonomy figure (C1) |
| Baselines | §7 fair protocol; honesty notes on generator/encoder mismatch |
| Limitations | MuSiQue, quota, guarantee≠flat-optimum |
| Reproducibility | scripts, seeds, hardware, API model IDs |

### 6.3 Claims hygiene

- Never cite deleted EM files.
- Never compare PF-MiniLM EM to GPT-4 HippoRAG without a **Notes** flag.
- Always report N and encoder beside every headline number.

---

## §7. SOTA comparison protocol

### 7.1 Principles (from HippoRAG / IRCoT-style reporting)

1. **Same questions** (fixed 500-id subset).  
2. **Same scorer** (`30_sota_adapter.score_system`).  
3. **Same generator** when the method allows (Llama-3.3-70B).  
4. **Flag** stronger official generators/encoders in Notes — do not hide.  
5. **Report cost**: LLM calls/query, index time, latency.  
6. **No cherry-pick**: pre-register systems below before seeing numbers.

### 7.2 System card

| System | Role | Repo / entry | Adapter status |
|--------|------|--------------|----------------|
| PATHFINDER-Coherent | ours | `05_evaluate.py` | ready |
| PATHFINDER-Hybrid | ours | `23_hybrid_retrieval.py` | ready |
| PATHFINDER-Flat (always dense teleport) | ours / upper ref | flag in `run_pathfinder` | ready |
| Naive RAG top-k | baseline | `run_baselines.py` | ready |
| IRCoT | iterative multi-hop | https://github.com/StonyBrookNLP/ircot | ⬜ wrap retrieves → `score_system` |
| HippoRAG / HippoRAG2 | PPR + OpenIE graph | https://github.com/OSU-NLP-Group/HippoRAG | ⬜ |
| LightRAG | dual-level graph | https://github.com/HKUDS/LightRAG | ⬜ |
| SubgraphRAG | learned subgraph | https://github.com/Graph-COM/SubgraphRAG | ⬜ KG map hard |
| (Optional) RAPTOR | tree summary | official repo | ⬜ |

Fill live numbers only in `results/sota_comparison.md`.

### 7.3 Minimal fair table (paper)

| System | EM | F1 | R@5 | R@10 | LLM calls/q | Notes |
|--------|----|----|-----|------|-------------|-------|
| … | | | | | | |

### 7.4 Adapter implementation order (code, no full run)

1. Unify PATHFINDER variants → adapter schema.  
2. IRCoT retrieve-only dump → score.  
3. HippoRAG index on same Hotpot corpus subset → score.  
4. LightRAG same.  
5. SubgraphRAG only if KG mapping feasible in <1 day; else “out of scope” in limitations.

---

## §8. Evaluation runbook (USER ONLY)

> Agent must **not** execute this section. Copy-paste locally / Docker.

### 8.1 Environment

```cmd
cd Ragidea
pip install -r experiments/requirements.txt
python -m spacy download en_core_web_sm
pip install optuna faiss-cpu transformers sentence-transformers
set GROQ_API_KEY=...
pytest pathfinder/tests/
```

### 8.2 Phase order (do not skip gates)

```
[Gate 0] pytest 58/58
[Gate 1] 5-query generation smoke (no RATE_LIMITED)
[Pilot]  N=500 retrieval stacks (no LLM) if new code
[Pilot]  N=100 EM/F1  →  N=500 EM/F1
[Full]   N=7405 only for paper final tables
[SOTA]   external systems on same 500 ids
[Figures] print_metrics + plots
```

### 8.3 Copy-paste: regenerate clean LLM pilots

```cmd
:: smoke
python experiments/generate_answers.py --smoke 5

:: E1 pilot EM (example flags — adjust to your CLI)
python experiments/25_rerank_eval.py --graphs data/hotpotqa_graphs_passage_bge.pkl --max_samples 100 --output results/raw/rerank_bge_pass_n100.json
python experiments/19_heterogeneous_llms.py --graphs data/hotpotqa_graphs_passage_bge.pkl --max_samples 100 --model llama-3.3-70b-versatile --output results/raw/llm70b_bge_pass_n100.json
```

Abort if JSON `summary.quota_exhausted == true`.

### 8.4 Copy-paste: retrieval-only (no API)

```cmd
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs_passage_bge.pkl --max_samples 500 --output results/hotpotqa_eval_bge_passage.json
python experiments/23_hybrid_retrieval.py --graphs data/hotpotqa_graphs_passage_bge.pkl --max_samples 500 --output results/raw/hybrid_bge_passage.json
python experiments/05b_teleportation_ablation.py --graphs data/hotpotqa_graphs_passage_bge.pkl --max_samples 500 --output results/raw/teleport_bge_pass.json
```

### 8.5 Copy-paste: full-scale (LONG)

```cmd
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs_passage.pkl --output results/hotpotqa_eval_passage_full.json
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs_full.pkl --output results/hotpotqa_eval_full.json
python experiments/05_evaluate.py --graphs data/2wiki_graphs_full.pkl --output results/2wiki_eval_full.json
python experiments/05_evaluate.py --graphs data/musique_graphs_full.pkl --output results/musique_eval_full.json
```

### 8.6 SOTA external (after clone + their install)

```cmd
:: example pattern — per-repo README first
python experiments/30_sota_adapter.py --system pathfinder_hybrid --pred results/raw/sota_pf_hybrid.json --gold data/hotpot_500_ids.json
python experiments/30_sota_adapter.py --system ircot --pred results/raw/ircot_500.json --gold data/hotpot_500_ids.json
:: then edit results/sota_comparison.md
```

### 8.7 Figures

```cmd
python experiments/print_metrics.py
python results/make_plots.py
```

---

## §9. Milestone tracker & decision gates

| Milestone | Exit criterion | Status |
|-----------|----------------|--------|
| **M0** Repo clean + rate-limit safe | corrupted JSONs gone; sentinel shipped; 58 tests | ✅ |
| **M1** Passage + BGE + hybrid locked | §2.1 table complete | ✅ |
| **M2** Mechanism analysis | error taxonomy + teleport regime note in paper | ⬜ |
| **M3** Best stack on pilot | D1–D2 decided; defaults merged | ⬜ |
| **M4** Clean EM/F1 N=500 | H5 accept/reject with CI; no quota flag | ⬜ |
| **M5** ≥3 external SOTA scored | `sota_comparison.md` filled via adapter | ⬜ |
| **M6** Paper v_next | abstract/experiments/limitations synced to ledger | ⬜ |
| **M7** Camera-ready | full-scale tables + figures + repro appendix | ⬜ |

**Kill criteria**

- If clean EM/F1 shows PF ≪ RAG with identical R@5 → pivot paper to **retrieval-structure / guarantee** contribution; downweight QA SOTA claim.
- If MuSiQue stays ~0 after passage+BGE → confine claims to Hotpot/2Wiki; explain MuSiQue as negative result.

---

## §10. Risk register & anti-patterns

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Groq silent quota | high | invalid EM | sentinel + abort; never average empties |
| Confounded multi-changes | med | false discoveries | R7 one-factor |
| Unfair SOTA (GPT-4 vs 70B) | high | reviewer reject | Notes column + matched re-run |
| Guarantee overclaim | med | theory pushback | connected-subtree scope box |
| MuSiQue zero | high | “not general” | P3 diagnostic; honest limitations |
| Agent runs 18h eval | med | blocked machine | R1–R2 |
| Result file sprawl | med | wrong file cited | ledger §2 only |

**Anti-patterns (banned)**

- Re-running locked full evals “to be sure.”
- Reporting EM from files with empty predictions.
- Tuning on full test sets.
- Changing encoder and hybrid and rerank in one commit claiming a win.
- Hand-copied metrics from upstream READMEs into our tables.

---

## §11. Immediate next actions (priority queue)

### For the agent (no eval)

1. Keep defaults/flags aligned with §5.B (passage, hybrid, encoder CLI).  
2. Implement C1 error-taxonomy **script** (writes report path; user runs).  
3. Extend `30_sota_adapter.py` loaders for IRCoT / HippoRAG prediction dumps.  
4. Paper pass: abstract + experiments tables ← §2 ledger; remove any cite to deleted EM.  
5. Finish teleportation–guarantee remark (§6.1).  
6. Add `meta` run-card fields to eval JSON writers.

### For the user (eval machine)

1. Gate 0–1 smoke with fresh `GROQ_API_KEY`.  
2. E1→E2 clean EM/F1 on BGE-passage hybrid N=100 then 500.  
3. D1 rerank composition pilot.  
4. Clone IRCoT + HippoRAG; dump preds on same 500 ids; score via adapter.  
5. Only then consider full-scale BGE passage (C4).

---

## §12. Quick command sheet

```cmd
:: always
pytest pathfinder/tests/

:: metrics dashboard
python experiments/print_metrics.py

:: syntax
python -c "import ast,glob;[ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('experiments/*.py')]"
```

---

## Document history

| Ver | Date | Summary |
|-----|------|---------|
| v1 | 2026-07 | Early phase 5–12 experiment list |
| v2 | 2026-07-26 | Cleaning + no-silent-eval rules; rate-limit crisis |
| **v3** | **2026-07-26** | Professional research OS: locked ledger, pre-registered hypotheses, one-factor protocol, workstreams A–F, SOTA fair-compare, milestones/kill criteria |

---

*End of PLAN.md v3 — execute top-down; promote only through gates; never sacrifice the evidence ledger for speed.*
