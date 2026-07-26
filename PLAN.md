# PATHFINDER — Research Master Plan (v4)

| Field | Value |
|-------|-------|
| **Project** | PATHFINDER — Submodular Coverage Maximization over Multidimensional KGs for Multi-Hop RAG |
| **Repo** | https://github.com/Yash-Awasthi-02/Ragidea |
| **Paper** | `pathfinder-paper.md` |
| **Plan version** | **v4** (supersedes v3) |
| **Last updated** | 2026-07-26 |
| **Status** | Thesis locked · offline work first · API/evals last |

---

## How to use this document

This is the **research operating system** for PATHFINDER. It encodes:

1. What the literature says is safe vs a wall  
2. What our locked numbers already prove  
3. What the paper may claim (and must not claim)  
4. An **offline-first** execution order (code → theory → paper → local retrieval evals → API last)

Read **§0–§2** before changing code or the paper.

---

## §0. Non-negotiable operating policy

### 0.1 Division of labor

| Who | Does | Never does |
|-----|------|------------|
| **Agent** | Cleaning, code, tests, theory, paper edits, adapters, analysis scripts, docs | Full evals, Groq/API calls, multi-hour dataset runs, git force-push |
| **User (local)** | Retrieval evals, graph builds that need data, long jobs | — |
| **User (API last)** | `generate_answers.py`, rerank/LLM scripts, heterogeneous LLMs, anything with `GROQ_API_KEY` | Running API work “in the middle” of offline tasks |

### 0.2 Hard rules

| # | Rule | Why |
|---|------|-----|
| R1 | **Offline work first.** Everything that needs no dataset dump and no API is finished before any eval queue. | Throughput + no quota waste |
| R2 | **Evals run locally on the user’s machine** (or Docker). Agent prints commands + expected output, then **stops**. | 9–18 h jobs; agent must not block |
| R3 | **API / Groq work is last** (or only when a gate explicitly requires it and offline options are exhausted). No “quick API check” mid-refactor. | Rate-limit corruption already cost us 4 result files |
| R4 | **Never run** from the agent: `05*.py`, `15–19*.py`, `25*.py`, `generate_answers.py`, full graph builds over full datasets, or any script that loads Hotpot/2Wiki/MuSiQue dumps or hits an LLM API. | Same |
| R5 | **Never push / rewrite history** unless the user explicitly provides a PAT and asks. | Repo safety |
| R6 | **One variable per experiment.** No multi-factor confounds. | Causal claims |
| R7 | **One scorer for all systems** (`30_sota_adapter.py`). Never mix vendor EM/F1 into our tables. | Fair comparison |
| R8 | Every run JSON must carry a **run card** (`meta`: git SHA, script, N, encoder, mode, seed, `quota_exhausted`). | Reproducibility |
| R9 | Corrupted / rate-limited EM files are **deleted, not averaged**. | Data integrity |
| R10 | Before any code change: **WHY → expected impact → theoretical implication → files touched**. | Scientific rigor |

### 0.3 Safe agent checks (seconds, no API, no full dataset)

```cmd
pytest pathfinder/tests/
python -c "import ast,glob;[ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('experiments/*.py')]"
python experiments/print_metrics.py
```

### 0.4 Work order (never invert)

```
Phase 0  Hygiene & correctness (offline)
Phase 1  Code defaults & adapters (offline)
Phase 2  Theory + paper surgery (offline)
Phase 3  Analysis scripts only (offline; user may run on existing JSONs)
Phase 4  Local retrieval evals (USER, no API)     ← only after 0–3 gates
Phase 5  API / generator / LLM rerank (USER, last)
Phase 6  External SOTA dumps + score (USER; API only if that system needs it)
Phase 7  Paper lock + figures
```

---

## §1. Thesis lock (from literature + our evidence)

### 1.1 Defensible core claim

> Under a **fixed token budget**, greedy submodular coverage maximization over a **passage-level** knowledge graph selects a **graph-coherent** evidence set with a **provable (1 − 1/e)** approximation to the optimal **frontier-feasible / connected-subtree** solution. Combined with **dense anchors** (hybrid), retrieval R@k **matches strong dense top-k** while preserving path structure. The contribution is a **principled coherent selector** for multi-hop RAG — not a claim of unrestricted flat-optimum optimality, and not a claim of agentic QA SOTA without matched runs.

### 1.2 What the field already established (do not fight this)

| Finding | Sources (representative) | Implication for us |
|--------|---------------------------|--------------------|
| Dense and GraphRAG are **complementary**; hybrid/routing wins | RAG vs GraphRAG systematic evals; adaptive GraphRAG routing | Default = **dense-anchor hybrid**, not pure graph |
| Graph helps most on **true multi-hop**; dense often wins single-hop / R@5 | Same + HippoRAG notes Hotpot has weaker multi-hop signal | Error slices by hop/type matter more than average R@5 bragging |
| **Passage-first** is standard; sentence is auxiliary | Hotpot pipelines, HopRetriever, PRISM | Passage = primary; sentence = ablation |
| Iterative LLM retrieval (**IRCoT**) is strong and **expensive** | IRCoT; ColBERT/HippoRAG+IRCoT F1 often mid-60s–70s class | We compete on **single-pass / few-call** coherent retrieval + theory, not call-count wars unless we opt in |
| Submodular selection for RAG **already exists** | **S-RAG** (flat knapsack coverage); **COSMOS** (connected subgraph submodular) | Soften “first ever”; position **difference** (Hotpot-style passage KG, frontier greedy, hybrid, multi-hop RAG stack) |
| MuSiQue is built to kill shortcuts; large human–model gap | MuSiQue (Trivedi et al.) | Near-zero R@5 is a **construction/regime** problem until diagnosed — not a quiet table cell |

### 1.3 Pre-registered hypotheses

| ID | Hypothesis | Primary metric | Status |
|----|------------|----------------|--------|
| **H1** | Passage nodes ≫ sentence nodes for multi-hop R@k | R@5, R@10 | ✅ Locked |
| **H2** | Stronger bi-encoders lift PF and RAG; relative gap shrinks | R@5 Δ(PF−RAG) | ✅ Locked |
| **H3** | Dense-anchor hybrid matches RAG R@5 and keeps path structure | R@5, tree connectivity | ✅ Locked (BGE-pass N=500) |
| **H4** | Rerank (CE/LLM) helps on top of hybrid passage+BGE | R@5 Δ | ⬜ Offline prep; **API last** |
| **H5** | Graph-coherent context improves EM/F1 vs equal-budget flat context | EM, F1 + bootstrap CI | ⬜ **API last**; kill criterion applies |
| **H6** | Teleport helps sparse graphs; neutral/hurts dense well-linked graphs | R@5 by connectivity | ⚠ Partial; finish offline analysis on existing JSON |
| **H7** | Under **matched** encoder/generator/budget, PF-hybrid is competitive with listed SOTA on retrieval and/or QA | R@5, EM, F1, calls/q | ⬜ Adapters offline; runs USER |
| **H8** | σ correlates with correctness (Spearman ρ > 0.3) | ρ | ⚠ Weak today; demote from headline until fixed |

### 1.4 Success bars

| Tier | Bar | Status |
|------|-----|--------|
| **Minimum publishable** | Passage R@5 ≥ 0.65; hybrid parity with dense; theory scope correct; related work cites S-RAG/COSMOS; no poisoned EM in tables; ≥ internal graph baselines | Retrieval ✅ · writing ⚠ · SOTA ❌ |
| **Strong** | Clean H5: EM/F1 PF ≥ RAG at matched R@5/budget, bootstrap p < 0.05, N≥500 | ❌ API last |
| **Outstanding** | Fair external SOTA (HippoRAG/IRCoT/LightRAG class) under matched protocol + MuSiQue not ≈0 after fix | ❌ |

### 1.5 Kill criteria (pivot, don’t deny)

| If… | Then… |
|-----|--------|
| Clean H5 shows PF EM ≪ RAG at **matched** R@5 | Pivot paper to **structure + guarantee + hybrid parity**; downweight QA SOTA |
| MuSiQue stays ~0 after passage+BGE + construction fix attempt | Confine claims to Hotpot/2Wiki; MuSiQue = negative/limitation |
| Multidimensional non-semantic weights never help on wiki distractors | Keep facets as **interface**; default **semantic-only** on these benches |

### 1.6 Banned claims (reviewer walls)

| Do **not** claim | Why |
|------------------|-----|
| Pure graph beats dense @ R@5 as the main result | Our data + literature disagree |
| (1−1/e) vs **unrestricted flat** top-k optimum | False; only frontier/connected-subtree OPT |
| “First graph-coherent submodular retrieval” without COSMOS/S-RAG | Overclaim |
| Multidimensional facets **drive** Hotpot gains | Bandit/grid: semantic-only wins |
| QA SOTA vs HippoRAG/IRCoT from literature rows next to our EM≈0 | Unfair + poisoned optics |
| σ “knows when uncertain” as a headline | Full-scale ρ ≈ 0 |
| Works on MuSiQue “in general” at R@5≈0.008 | Not true yet |

---

## §2. Locked evidence ledger

> **Do not re-run these.** Cite them. Re-running wastes days.

### 2.1 Primary retrieval (HotpotQA distractor)

| Setting | N | Encoder | PF R@5 | RAG R@5 | PF R@10 | Notes |
|---------|--:|---------|-------:|--------:|--------:|-------|
| Sentence full | 7405 | MiniLM | 0.257 | 0.293 | **0.335** | Ablation / legacy |
| Passage N=500 | 500 | MiniLM | **0.644** | 0.708 | 0.764 | Phase A |
| Passage full | 7405 | MiniLM | **0.667** | 0.705 | **0.771** | Phase D2 |
| Sentence N=500 | 500 | BGE | **0.466** | 0.484 | 0.480 | Phase D1 |
| Passage N=500 | 500 | BGE | **0.864** | 0.870 | 0.866 | Phase D1 |
| Hybrid dense-anchor BGE pass. | 500 | BGE | **0.870** | 0.870 | 0.964 | **Default stack target** |

**Files:**  
`results/hotpotqa_eval_full.json` · `hotpotqa_eval_passage.json` · `hotpotqa_eval_passage_full.json` · `hotpotqa_eval_bge.json` · `hotpotqa_eval_bge_passage.json` · `results/raw/hybrid_bge_passage.json`

### 2.2 Cross-benchmark (sentence MiniLM full) — diagnostic, not headline

| Dataset | N | PF R@5 | RAG R@5 | PF R@10 |
|---------|--:|-------:|--------:|--------:|
| 2WikiMultihopQA | 12576 | 0.235 | 0.325 | 0.373 |
| MuSiQue | 2417 | 0.008 | 0.004 | 0.012 |

### 2.3 Valid component studies (R@k / systems; no poisoned EM)

| Study | Finding | File |
|-------|---------|------|
| Teleportation | Hybrid teleport helps R@10 modestly (e.g. Hotpot 0.330→0.350) | `raw/teleportation_ablation.json` |
| Bandit weights | Semantic-only wins; late_recall > early | `raw/bandit_weight_learning.json` |
| LLM rerank (sentence, small N) | R@5 0.24→0.32 | `raw/llm_reranking.json` |
| Latency | ~3.5 ms mean retrieval | `raw/latency_profiling.json` |
| NLI sufficiency | agreement 0.085 — **do not ship** | `raw/nli_sufficiency.json` |

### 2.4 Deleted / invalid (never cite EM)

Silent Groq exhaustion → empty predictions:

- ~~`raw/rerank_passage.json`~~  
- ~~`raw/heterogeneous_llm_70b_v2.json`~~  
- ~~`raw/heterogeneous_llm_70b_passage.json`~~  
- ~~`raw/heterogeneous_llm_8b.json`~~  

**Code fix status:** `generate_answers.py` exposes `RATE_LIMITED` / `quota_exhausted` / `n_rate_limited`.  
**Regenerate only in Phase 5 (API last).**

### 2.5 EM numbers currently in big eval JSONs

Most `hotpotqa_eval_*.json` EM fields are **not** trustworthy end-to-end QA (missing/empty generation path on full runs).  
**Passage N=500 EM=0.106** is the only semi-usable pilot-scale EM in the ledger — still re-validate in Phase 5 under run cards before paper QA tables.

**Paper rule:** split **Retrieval tables** vs **QA tables**. Never put literature IRCoT/HippoRAG EM in the same row block as our EM≈0.

### 2.6 Engineering baseline

| Item | Value |
|------|------:|
| Tests | **58/58** |
| Experiment scripts | 33 |
| SOTA adapter | `30_sota_adapter.py` (setup) |
| Algorithm | `run_pathfinder.py` |

---

## §3. Paper correctness checklist (offline)

Priority order for `pathfinder-paper.md` surgery:

| # | Issue | Action |
|---|--------|--------|
| P1 | Protocol says `text-embedding-3-small`; results are MiniLM/BGE | Align §7.3 to **actual** encoders |
| P2 | Abstract / mixed headlining of sentence 0.26 vs passage 0.64+ | Lead **passage + hybrid**; sentence = ablation |
| P3 | “First graph-coherent submodular…” | Cite **S-RAG**, **COSMOS**; state delta |
| P4 | SOTA table mixes literature EM 50–60 with our EM≈0 | Split tables; Notes on generator/encoder |
| P5 | Passage EM=0.000 “LLM can’t use passages” narrative | Attribute to **rate limit / invalid run**, not science |
| P6 | Llama-8B EM=0 as model failure | Quarantine until clean regen |
| P7 | Multidimensional as contribution #1 | Reframe: interface; **semantic-only default** on wiki |
| P8 | σ / four open problems as central | Demote until calibrated |
| P9 | Guarantee scope | Keep connected-subtree / frontier box; hybrid corollary only per-component |
| P10 | MuSiQue | Limitation + diagnostic plan; not a silent win |

### 3.1 Target paper pitch (safe)

1. Multi-hop needs **coherent** evidence, not only cosine neighbors.  
2. Method: greedy submodular coverage on **passage** KG + optional **dense anchors**.  
3. Theory: (1−1/e) vs **OPT_frontier**; related to S-RAG/COSMOS; state differences.  
4. Empirics: granularity, encoder, hybrid parity, internal graph baselines, mechanism slices.  
5. QA: only **clean matched** generator results.  
6. Limits: MuSiQue, σ, facets without metadata, teleport weak, no free lunch vs multi-call IRCoT.

---

## §4. Experimental protocol (when USER runs)

### 4.1 Fixed contract

| Knob | Default |
|------|---------|
| Split | HotpotQA distractor validation |
| Pilot N | **500** hash-stable IDs |
| Full N | 7405 / official full only for final tables |
| K_tok | 2048 |
| Metrics | R@5, R@10, R@20, paragraph-R@5; EM/F1 via **one** scorer |
| Generator (QA only) | Llama-3.3-70B-versatile @ Groq |
| Encoder (new work) | **BGE**; MiniLM legacy |
| Nodes | **Passage** |
| Retrieval default | **Dense-anchor hybrid** (k∈{3,5}) |
| Weights default (wiki) | **Semantic-only** α=1 unless metadata exists |
| Stats | Paired bootstrap 10k; 95% CI |

### 4.2 Run card (`meta` in every new JSON)

```json
{
  "git_sha": "...",
  "script": "experiments/XX_....py",
  "hypothesis_id": "H3",
  "n": 500,
  "encoder": "BAAI/bge-base-en-v1.5",
  "node_level": "passage",
  "retrieval_mode": "dense_anchor_5",
  "k_tok": 2048,
  "generator": null,
  "seed": 42,
  "quota_exhausted": false,
  "n_rate_limited": 0,
  "wall_time_s": 0
}
```

### 4.3 Promote rule

Pilot N=500 improves primary metric with CI not covering 0, and no >1 pt regression on secondary → then consider full-scale.  
**Full-scale never runs “to peek.”**

---

## §5. Workstreams (offline → local → API)

```
 A Hygiene              offline     maintain
 B Defaults & code      offline     agent
 C Paper + theory       offline     agent
 D Analysis tooling     offline     agent; user runs on disk JSONs
 E Local retrieval      USER local  no API
 F Generator / LLM      USER API    LAST
 G External SOTA        USER        adapters offline first
 H Paper lock           offline     after E/F gates
```

### A — Hygiene ✅ / maintain

- [x] Delete poisoned EM JSONs  
- [x] Rate-limit sentinel in `generate_answers.py`  
- [x] Formal tests 58  
- [ ] `meta` run cards on eval writers  
- [ ] `print_metrics.py` includes BGE + hybrid + passage full  
- [ ] `.gitignore` keeps new raw dumps from accidental commit  

### B — Code defaults (offline, agent)

| Task | WHY | Files |
|------|-----|-------|
| B1 Default passage node path in CLI/docs | H1 | `05_evaluate.py`, README |
| B2 Hybrid `dense_anchor_5` flag as recommended default | H3 | `run_pathfinder.py`, `23_hybrid_retrieval.py` |
| B3 `--encoder {minilm,bge}` consistency | H2 | build + eval scripts |
| B4 Semantic-only default weights on wiki benches | bandit evidence | `run_pathfinder.py` |
| B5 Teleport: off by default on dense BGE-passage; document regimes | H6 | flags + docs |
| B6 Extend `30_sota_adapter.py` loaders (IRCoT/HippoRAG/LightRAG dump → `score_system`) | H7 | `30_sota_adapter.py` |
| B7 Error-taxonomy **script** (reads existing preds; no API) | mechanism | new `experiments/31_error_taxonomy.py` |

**Gate B:** `pytest` 58/58; syntax check; dry-run help/CLI only (no full data).

### C — Theory + paper surgery (offline, agent) — **highest ROI**

| Task | Output |
|------|--------|
| C1 Related work: S-RAG, COSMOS, HippoRAG, IRCoT, LightRAG | paper §2 |
| C2 Guarantee scope box + hybrid corollary wording | paper §5 |
| C3 Abstract + contributions rewrite per §1.1 / §3.1 | paper |
| C4 Split retrieval vs QA tables; remove poisoned narratives | paper §7 |
| C5 Encoder protocol = MiniLM/BGE reality | paper §7.3 |
| C6 Limitations: MuSiQue, σ, facets, teleport, unfair SOTA | paper §8 |
| C7 Finish teleport vs connected-subtree remark | paper + optional proof note |

**Gate C:** paper contains no literature-EM-vs-our-EM single table; no “first” without COSMOS/S-RAG; abstract leads passage/hybrid.

### D — Offline analysis on **existing** artifacts

| Task | Needs API? | Who runs |
|------|------------|----------|
| D1 Error taxonomy from locked JSONs (hop, bridge, connectivity) | No | Script agent; exec user if heavy |
| D2 Teleport help/hurt slices from `teleportation_ablation.json` | No | same |
| D3 MuSiQue diagnostic plan + code to measure gold-support vs graph overlap | No code API | agent write; user run local |
| D4 Coverage-ratio interpretation note (synthetic tightness ≠ real hardness) | No | paper |

### E — Local retrieval evals (USER, **no API**)

Only after Gates B–C (and D scripts ready). Examples:

```cmd
pytest pathfinder/tests/

:: retrieval-only pilots (no GROQ)
python experiments/23_hybrid_retrieval.py --graphs data/hotpotqa_graphs_passage_bge.pkl --max_samples 500 --output results/raw/hybrid_bge_passage.json
python experiments/05b_teleportation_ablation.py --graphs data/hotpotqa_graphs_passage_bge.pkl --max_samples 500 --output results/raw/teleport_bge_pass.json
python experiments/31_error_taxonomy.py --eval results/hotpotqa_eval_bge_passage.json --out results/analysis/error_taxonomy.md
```

Full-scale retrieval only for paper-final locked tables not already in §2.

**Gate E:** mechanism figure/table exists; no new retrieval claim without N and encoder.

### F — API / generator (USER, **LAST**)

Prerequisites: Gate E done or explicitly waived; fresh key; smoke 5 queries.

```cmd
set GROQ_API_KEY=...
python experiments/generate_answers.py --smoke 5
:: abort if RATE_LIMITED / quota_exhausted

:: H5 pilot ladder
:: N=100 then N=500; matched budget PF-hybrid vs RAG; same generator
:: write quota_exhausted into summary; drop or separately report limited queries
```

Order inside F:

1. Smoke  
2. H5 N=100  
3. H5 N=500 + bootstrap  
4. Optional rerank composition (H4)  
5. Heterogeneous 8B/70B only if H5 clean  
6. Never average empty strings into EM  

**Gate F:** `quota_exhausted=false` on reported QA tables; CI reported.

### G — External SOTA (USER; offline adapter first)

| Step | Offline? |
|------|----------|
| G1 Adapter schema + PATHFINDER dump → score | Yes (agent) |
| G2 Clone IRCoT / HippoRAG / LightRAG; follow **their** install | User local |
| G3 Same 500 IDs; dump preds; `30_sota_adapter.score_system` | User |
| G4 Fill `results/sota_comparison.md` | User/agent edit |
| G5 Honesty Notes: generator, encoder, LLM calls/q | Required |

Systems pre-registered: PATHFINDER-Coherent, PATHFINDER-Hybrid, Naive RAG, IRCoT, HippoRAG(2), LightRAG; SubgraphRAG only if KG map <1 day else limitations.

### H — Paper lock

Sync abstract, §7, limitations, repro appendix to ledger + F/G outcomes. Figures via `print_metrics` / `make_plots` **after** numbers freeze.

---

## §6. Default production stack (target)

```
Passage nodes
  + BGE encoder
  + Dense-anchor hybrid (k=5)
  + semantic-only weights (wiki distractors)
  + optional CE/LLM rerank (Phase F only if H4 holds)
  + Llama-3.3-70B generation (Phase F)
  + official EM/F1 via 30_sota_adapter
```

Not default: LLM-guided traversal, NLI sufficiency, multidimensional weights on Hotpot, pure sentence MiniLM.

---

## §7. SOTA fair-compare principles

1. Same question IDs  
2. Same scorer  
3. Same generator when the method allows  
4. Flag stronger official stacks in Notes — never hide  
5. Report LLM calls/query and index cost  
6. No cherry-pick; systems listed before numbers  
7. **Retrieval-only** and **QA** reported separately  

Live table lives in `results/sota_comparison.md` (setup only until Phase G).

---

## §8. Milestone tracker

| ID | Milestone | Depends on | Status |
|----|-----------|------------|--------|
| M0 | Hygiene + rate-limit safe + 58 tests | — | ✅ |
| M1 | Passage + BGE + hybrid locked in ledger | — | ✅ |
| M2 | Paper surgery §3 (offline) | M0 | ⬜ **next** |
| M3 | Defaults + adapter + error-taxonomy script (offline) | M0 | ⬜ |
| M4 | Mechanism analysis from disk / local retrieval | M3, USER E | ⬜ |
| M5 | Clean H5 EM/F1 N=500 | M4 optional, USER F | ⬜ |
| M6 | ≥3 external systems scored fairly | M3, USER G | ⬜ |
| M7 | Camera-ready paper + figures | M5–M6 decisions | ⬜ |

---

## §9. Risk register

| Risk | Mitigation |
|------|------------|
| Groq silent quota | API last; sentinel; abort; never cite empties |
| Chasing dense R@5 past parity | Thesis = coherence + hybrid parity, not dense deathmatch |
| Novelty clash with S-RAG/COSMOS | Cite and differentiate early (Phase C) |
| MuSiQue zero | Diagnostic offline; honest limitation |
| Unfair SOTA tables | Split metrics; one scorer; Notes |
| Agent runs 18h eval | R2–R4 |
| Multidimensional story collapses | Semantic-only default; facets optional |
| Guarantee overclaim | Frontier/connected-subtree only |

---

## §10. Immediate next actions

### Agent (now — offline only)

1. Paper surgery (C1–C7) — highest leverage  
2. B1–B7 code/docs/adapter/taxonomy script  
3. Keep PLAN/paper/ledger consistent  
4. No evals, no API  

### User (later — local, no API)

1. Gate: `pytest` 58/58  
2. Phase E retrieval / taxonomy commands as needed  
3. Only then Phase F with fresh `GROQ_API_KEY`  
4. Phase G external clones after adapters ready  

### Explicitly deferred (do not start early)

- Full Hotpot/2Wiki/MuSiQue re-evals already in §2  
- Any Groq batch  
- Claiming H5/H7 in the abstract  

---

## §11. Quick command sheet

```cmd
:: offline always
pytest pathfinder/tests/
python experiments/print_metrics.py

:: local retrieval only (USER) — no API key
python experiments/23_hybrid_retrieval.py --help

:: API last (USER)
set GROQ_API_KEY=...
python experiments/generate_answers.py --smoke 5
```

---

## §12. Document history

| Ver | Date | Summary |
|-----|------|---------|
| v1 | 2026-07 | Phase 5–12 experiment list |
| v2 | 2026-07-26 | Cleaning + no-silent-eval; rate-limit crisis |
| v3 | 2026-07-26 | Research OS: ledger, H1–H8, workstreams A–F |
| **v4** | **2026-07-26** | **Literature-backed thesis lock; banned claims; S-RAG/COSMOS; offline→local→API order; kill criteria; paper surgery first** |

---

*End of PLAN.md v4 — defend the coherent hybrid thesis; finish offline work; evals local; API last.*
