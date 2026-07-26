# PATHFINDER — Research Master Plan (v6)

| Field | Value |
|-------|-------|
| **Project** | PATHFINDER — Submodular Coverage Maximization over Multidimensional KGs for Multi-Hop RAG |
| **Repo** | https://github.com/Yash-Awasthi-02/Ragidea |
| **Paper** | `pathfinder-paper.md` |
| **Plan version** | **v6** (supersedes v5; reconciles v5 skeleton with the v3 consolidation pass) |
| **Last updated** | 2026-07-27 |
| **Status** | Prior-art map locked · consolidation pass done (100 tests, theory T1–T5, fairness gate, unified EM path) · offline → local → API-last · copy-paste eval runbook below |

---

## How to use this document

1. **§0** — operating policy (offline first; evals local; API last; push rules).  
2. **§1** — thesis lock (what we may claim).  
3. **§2** — **prior-art map** (who already tried the idea; where we sit).  
4. **§3** — locked evidence ledger + **consolidation completed** (§3.6).  
5. **§4** — **stronger development directions** (D1–D10) + **theory extensions T1–T5** (§4.0).  
6. **§5–§11** — protocol, workstreams, milestones, risks, next actions.  
7. **§12** — **copy-paste eval runbook (Phases 0–6, Windows CMD)** — user runs.  
8. **§15** — **git & sync protocol / conflict-handling note**.

> **v6 = v5 (research-mature skeleton) + the v3 consolidation pass** (10 redundant files deleted, 58→100 tests, theory T1–T5 shipped, fairness gate + unified EM path shipped, paper surgery largely done) **+ a verified eval runbook + a git-conflict-handling note.** v5's literature sweep (§2), evidence ledger (§3), and directions (§4) are preserved verbatim; only status markers, the eng row, and §12/§14/§15 are new. The two `D#` schemes are kept distinct: **T1–T5 = theory extensions** (§4.0); **D1–D10 = development directions** (§4).

---

## §0. Non-negotiable operating policy

### 0.1 Division of labor

| Who | Does | Never does |
|-----|------|------------|
| **Agent** | Cleaning, code, tests, theory, paper, adapters, offline analysis scripts, docs | Full evals, Groq/API, multi-hour dataset runs, force-push |
| **User (local)** | Retrieval evals, graph builds, long offline jobs | — |
| **User (API last)** | Generation, LLM rerank, heterogeneous LLMs, any `GROQ_API_KEY` work | API work mid-refactor “just to check” |

### 0.2 Hard rules

| # | Rule |
|---|------|
| R1 | **Offline first** — finish all no-dataset / no-API work before any eval queue. |
| R2 | **Evals run locally** on the user’s machine. Agent prints command + expected output, then **stops**. |
| R3 | **API / Groq is last** (or only when a gate requires it and offline options are exhausted). |
| R4 | Agent never runs: `05*.py`, `15–19*.py`, `25*.py`, `generate_answers.py`, full-dataset graph builds, or LLM APIs. |
| R5 | No push/history rewrite unless user provides PAT and asks. *(One authorized push executed under v6 — see §15. Standing rule otherwise unchanged.)* |
| R6 | One variable per experiment. |
| R7 | One scorer for all systems (`30_sota_adapter.py`). |
| R8 | Every new JSON has a **run card** (`meta`). |
| R9 | Rate-limited / empty-pred EM files are deleted, not averaged. |
| R10 | Before code changes: WHY → impact → theory implication → files. |

### 0.3 Safe agent checks

```cmd
python -m pytest pathfinder\tests -q --rootdir=.
python -c "import ast,glob;[ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('experiments/*.py')]"
python experiments\print_metrics.py
```

### 0.4 Work order (never invert)

```
Phase 0  Hygiene (offline)
Phase 1  Code defaults, adapters, diagnostics (offline)
Phase 2  Theory + paper surgery + related-work rewrite (offline)
Phase 3  Offline analysis on existing JSONs / gold labels
Phase 4  Local retrieval evals (USER, no API)
Phase 5  API / generator / LLM (USER, LAST)
Phase 6  External SOTA dumps + score (USER)
Phase 7  Paper lock + figures
```

---

## §1. Thesis lock

### 1.1 Defensible claim

> Under a **fixed token budget**, greedy submodular coverage maximization over a **passage-level text graph**, optionally **seeded by dense anchors**, selects a **graph-coherent** evidence set with a **provable (1 − 1/e)** guarantee w.r.t. the optimal **frontier-feasible / connected-subtree** solution. On HotpotQA-style distractor multi-hop RAG, this yields **strong internal gains vs BFS/SA**, **hybrid parity with dense top-k at R@5**, and a **certificate dense top-k lacks** — without LLM calls during traversal.

### 1.2 What we do **not** claim

| Banned | Why |
|--------|-----|
| “First submodular multi-hop / graph-coherent retrieval” | **S-RAG, GeoRAG, COSMOS, Bala’26 packing, Lin–Bilmes lineage** |
| (1−1/e) vs unrestricted flat top-k OPT | Guarantee is frontier/connected only |
| Pure graph beats dense @ R@5 as headline | Our data + RAG-vs-GraphRAG surveys disagree |
| Multidimensional facets drive Hotpot gains | Semantic-only wins empirically |
| QA SOTA vs IRCoT/HippoRAG without matched runs | Unfair + our EM mostly invalid |
| Works on MuSiQue generally at R@5≈0 | Construction/regime failure until fixed |
| σ is calibrated | Full-scale ρ≈0 |

### 1.3 Hypotheses

| ID | Hypothesis | Status |
|----|------------|--------|
| H1 | Passage ≫ sentence | ✅ |
| H2 | Stronger encoder lifts both; gap shrinks | ✅ |
| H3 | Dense-anchor hybrid matches RAG R@5 | ✅ |
| H4 | Packing/rerank improves **answer-in-context** / R@k on hybrid pool | ⬜ |
| H5 | Coherent context improves EM/F1 at matched budget | ⬜ API last |
| H6 | Teleport helps sparse graphs only | ⚠ partial |
| H7 | Fair external SOTA competitive under matched protocol | ⬜ |
| H8 | σ calibrates | ⚠ product-σ null (demoted); ✅ conformal/bottleneck replacement shipped (T8, §5.9); coverage validation = eval |
| **H9 (new)** | **Answer-in-context** predicts F1 better than R@5 on our packs | ⬜ offline metric |
| **H10 (new)** | Cost-aware greedy (ΔF / tokens) beats cardinality greedy under heterogeneous passage lengths | ⬜ **code-ready** (flag + theorem shipped; run pending) |
| **H11 (new)** | Entity/OpenIE edges (HippoRAG-style) fix MuSiQue more than encoder alone | ⬜ local graph rebuild |

### 1.4 Kill criteria

| If | Then |
|----|------|
| Clean H5: PF EM ≪ RAG at matched R@5 | Pivot to structure/guarantee/hybrid-parity paper |
| MuSiQue stays ~0 after H11 attempt | Hotpot/2Wiki-only claims |
| H9 fails (answer-in-context uncorrelated) | Do not center packing narrative; stick to retrieval R@k + theory |

---

## §2. Prior-art map — people already tried pieces of this

> **Conclusion of the sweep:** the *ingredients* are not new. The **combination + setting + honesty of scope** can still be a paper — only if we cite neighbors and pick deltas they do not cover.

### 2.1 Lineage table (closest → farther)

| Work | What they did | Graph? | Submodular? | Multi-hop RAG text? | Guarantee | Overlap with us | Our delta |
|------|---------------|--------|-------------|---------------------|-----------|-----------------|-----------|
| **Lin & Bilmes (2011)** | Coverage+diversity summarization | No | Yes | No | greedy | Product/coverage DNA | Apply under **graph frontier** |
| **S-RAG** (OpenReview / ARR’26) | Knapsack submodular **doc** selection for RAG; beats top-k/MMR; Hotpot multi-hop gains | No (flat) | Yes | Yes | (1−1/e) knapsack | Same objective family | We add **connectivity / frontier** constraint |
| **GeoRAG** (arXiv:2606.29328) | Information-demand coverage; facility-location style; Sinkhorn; +EM on Hotpot | No | Yes | Yes | (1−1/e) | Flat multi-aspect coverage | We use **passage graph structure**, not only sub-queries |
| **Bala (arXiv:2607.00725)** | Budgeted **submodular evidence packing**; **answer-in-context** diagnostic; when packing helps (4 conditions); Hotpot +3B reader | Optional ACE graph heuristic | Yes (packer) | Yes | budgeted greedy | **Closest concurrent packing paper** | We are **retriever+traversal**, not post-hoc packer only; different F; graph-coherent tree |
| **COSMOS** (ACL 2026) | Connectivity-oriented submodular **subgraph** retrieval; seed-greedy + MaxST; SACT | Yes (KB triples) | Yes | KBQA (WebQSP/CWQ/M3GQA) | (1−e^(−γ)) style | **Closest theory cousin** | We target **Hotpot-style text passage graphs + RAG**, not Freebase-style KBQA |
| **G-Retriever** (NeurIPS 2024) | Connected subgraph via **PCST** (Steiner); GNN+LLM | Yes (textual graphs) | No (PCST) | GraphQA | opt. approx for PCST | Connected evidence | Different objective (prizes/costs vs coverage F); we stay LLM-free at retrieve |
| **SubgraphRAG** | Learned triple scorer → size-bounded subgraph → LLM | Yes (KG) | No (MLP) | KGQA | None | Subgraph evidence | We are **training-free greedy** with certificate |
| **SeedER** (arXiv:2605.23753) | Dense **seed** + **RL expand** on KG frontier | Yes | Notes greedy limits | KG / STaRK | None (learned) | Seed+expand skeleton | We keep **greedy submodular** (certificate) not RL |
| **HippoRAG** | OpenIE KG + synonymy + **PPR** single-step multi-hop | Yes (entity) | No | Yes (MuSiQue/2Wiki strong) | None | Graph multi-hop RAG | Different graph (OpenIE vs passage); PPR vs submodular greedy |
| **IRCoT** | Interleave CoT + retrieval | No | No | Yes (very strong) | None | Multi-hop SOTA class | Many LLM calls; we are single-pass / few-call |
| **PathRetriever / HopRetriever / MDR / BeamDR / Beam Retrieval** | Learned multi-hop **paths** over wiki | Hyperlink/dense | No | Yes | None | Path evidence | No submodular F; usually supervised |
| **LightRAG / Microsoft GraphRAG** | Entity/community graphs, summaries | Yes | No | Mixed | None | GraphRAG family | Different construction & no (1−1/e) coverage claim |
| **MMR / DPP / ScalDPP** | Diversity subset selection | No | DPP related | Sometimes | DPP/MAP | Redundancy control | No connectivity; we should **beat them as packing baselines** |
| **RECOMP** | Compress retrieved docs for reader | No | No | QA | None | Context budget | Orthogonal; can sit **after** us |
| **DuP-RAG / STEM / HybRAG** | Hybrid structural+semantic paths, pruning | Yes | No | Yes | None | Hybrid theme | We formalize selection as submodular F |

### 2.2 What the field already proved (do not re-discover)

1. **Dense ⊥ Graph are complementary** → hybrid/routing wins (RAG vs GraphRAG evals).  
2. **Passage/chunk granularity** dominates sentence-only for multi-hop QA pipelines.  
3. **Submodular packing helps multi-hop only conditionally** (Bala’26):  
   (i) complementary multi-hop structure,  
   (ii) retrieval already surfaces evidence,  
   (iii) binding but not extreme budget,  
   (iv) reader weak enough that **evidence density** is the bottleneck — gains **shrink/reverse** as reader scales (3B→14B).  
4. **Answer-in-context ≫ recall** as predictor of budgeted QA F1 (Bala’26).  
5. **MuSiQue** kills shortcut multi-hop; retrieval + reader both hard.  
6. **Entity/OpenIE graphs** (HippoRAG) help 2Wiki/MuSiQue more than Hotpot’s weaker multi-hop signal.  
7. **Greedy frontier can miss delayed connectors** (SeedER motivation) — teleport/dense anchors are the practical fix (we already hybridize).  
8. **Learned multi-hop retrievers** (MDR/Beam) are strong but **supervised** and certificate-free.

### 2.3 Honest novelty niches still open for PATHFINDER

| Niche | Why still open | Risk |
|-------|----------------|------|
| **N1. Frontier-constrained product-coverage greedy on Hotpot *passage* graphs** with explicit OPT_frontier certificate | COSMOS is KB triples; S-RAG/GeoRAG/Bala are mostly flat packers | Must cite all four |
| **N2. Dense-anchor hybrid that preserves tree coherence + per-anchor corollary** | Many hybrids exist; few with submodular certificate language | Don’t overclaim uniqueness of hybrid |
| **N3. Training-free coherent selector vs learned path/PPR/RL** under **matched encoder/budget** | Fair compute story (ms traversal, 0 LLM calls) | Must measure calls/q for IRCoT/HippoRAG |
| **N4. Mechanism: when connectivity helps** (bridge degree, hop, component) on *same* graphs | Surveys call for this; few papers do it on their own ablations | Needs error taxonomy (offline) |
| **N5. Answer-in-context + R@k joint reporting for graph-coherent packs** | Bala did packing; we can apply diagnostic to **graph vs dense packs** | Offline-friendly metric |
| **N6. Cost-aware (Δ/token) greedy on heterogeneous passage lengths** | Theory remark exists; empirical underused in our repo | Easy offline code + local pilot |

### 2.4 Crowded / low-EV ideas (do **not** pour months here)

| Idea | Why low EV now |
|------|----------------|
| More multidimensional weight grids on Hotpot | Already lost to α=1 |
| LLM-guided traversal as default | ToG/PRISM/IRCoT own this; cost |
| Claiming SOTA F1 from literature tables | Fatal review pattern |
| Re-running locked full MiniLM sentence evals | Ledger frozen |
| NLI sufficiency as product feature | 8.5% agreement |
| Beating ColBERT+IRCoT with MiniLM single-pass | Wrong arena |
| Synthetic coverage-ratio ≈1.0 as main theory result | Too easy instances |

### 2.5 Required related-work rewrite (paper §2)

Must cite and position against at minimum:

**Submodular / packing:** Lin & Bilmes 2011; S-RAG; GeoRAG; Bala arXiv:2607.00725; (optional) DRAG/DPP/ScalDPP.  
**Connected subgraph:** COSMOS; G-Retriever (PCST); SubgraphRAG; SR (ACL’22).  
**Multi-hop retrieval:** PathRetriever; MDR; Beam Retrieval; IRCoT; HopRetriever.  
**GraphRAG systems:** HippoRAG(+2); LightRAG; GraphRAG; SeedER.  
**Diagnostics / when graph helps:** RAG vs GraphRAG surveys; Bala conditions; entity vs passage graph analyses.

**One-sentence positioning (paper-ready):**

> Prior submodular RAG methods optimize *flat* context packing (S-RAG, GeoRAG, Bala); connectivity-oriented submodular retrieval (COSMOS) targets *KB triples*. PATHFINDER applies coverage-style greedy selection under a **frontier/tree constraint on Hotpot-style passage graphs**, with **dense-anchor hybridization** and a **matched multi-hop RAG** evaluation protocol.

---

## §3. Locked evidence ledger

> Do not re-run. Cite.

### 3.1 Hotpot retrieval

| Setting | N | Enc | PF R@5 | RAG R@5 | PF R@10 |
|---------|--:|-----|-------:|--------:|--------:|
| Sent full | 7405 | MiniLM | 0.257 | 0.293 | **0.335** |
| Pass 500 | 500 | MiniLM | **0.644** | 0.708 | 0.764 |
| Pass full | 7405 | MiniLM | **0.667** | 0.705 | **0.771** |
| Sent 500 | 500 | BGE | 0.466 | 0.484 | 0.480 |
| Pass 500 | 500 | BGE | **0.864** | 0.870 | 0.866 |
| Hybrid DA BGE pass | 500 | BGE | **0.870** | 0.870 | 0.964 |

### 3.2 Cross-bench (sent MiniLM) — diagnostic

| Set | PF R@5 | RAG R@5 |
|-----|-------:|--------:|
| 2Wiki full | 0.235 | 0.325 |
| MuSiQue full | 0.008 | 0.004 |

### 3.3 Valid components

Teleport R@10 modest ↑ · Bandit → semantic-only · Latency ~3.5 ms · LLM sentence rerank small-N ↑ · NLI sufficiency invalid.

### 3.4 Invalid EM (deleted / untrusted)

Poisoned Groq files deleted. Most eval-JSON EM fields are **not** paper-grade QA. Only clean Phase-5 runs may enter QA tables.  
**Split retrieval vs QA tables always.**

### 3.5 Eng

**100 tests** (58 formal-property + 42 theory) · 33 experiment scripts · `30_sota_adapter.py` with **fairness gate** (`fairness_check` / `enforce_fairness`, matched encoder + generator) · `RATE_LIMITED` sentinel shipped · `answer_helper.py::retrieve_then_answer` unified EM context path · `cost_benefit` flag in `algorithm.py` + `cost_benefit_greedy` (Sviridenko three-phase) in `theory.py` · coverage-ratio > 1.0 bug fixed in `05_evaluate.py`.

### 3.6 Consolidation completed (v6 pass — offline, no evals)

| Item | Outcome |
|------|---------|
| 10 redundant `.md` files deleted | Final state = `pathfinder-paper.md` + `PLAN.md` + `README.md` (standard repo readme kept). Per the user's "paper + plan .md only" directive, every redundant planning/inspection `.md` (incl. `pathfinder/THEORY.md`, `prompt.md`, `results/literature_audit.md`, `results/multi_benchmark.md`, `results/sota_comparison.md`, `experiments/README.md`, `results/em_discrepancy_analysis.md`) was swept — theory lives in `pathfinder/theory.py` + paper §5. Commit `118763e`. |
| Theory extensions T1–T8 | Shipped in `pathfinder/theory.py` + 42 tests in `pathfinder/tests/test_theory.py`; proofs in paper §5 (§5.2 tightness, §5.9 conformal σ, §5.10 tree-DP). (See §4.0.) |
| Fairness gate | `experiments/30_sota_adapter.py`: `fairness_check` (FAIR / FAIR_WITH_NOTE / UNFAIR) + `enforce_fairness` (raises on UNFAIR). Blocks confounded comparisons before scoring. |
| Unified EM path | `experiments/answer_helper.py::retrieve_then_answer` — one (retriever, node set, `build_context` chars, prompt) across `05`/`19`/`25`/`30`. EM now comparable across scripts. |
| Cost-aware greedy (Thm 3) | `cost_benefit=True` in `algorithm.py` (line 10c) + `cost_benefit_greedy` in `theory.py`. Shipped code now satisfies the theorem it cites. |
| Coverage-ratio > 1.0 fix | `05_evaluate.py::coverage_ratio`: teleportation forced OFF, cardinality matched at k, identical `compute_F` both sides, ratio clamped ≤ 1+ε; `method_note` records the comparison is teleport-free + cardinality-matched. |
| EM discrepancy | RESOLVED as *not a code bug*: full-scale runs had 98.2% empty predictions (quota exhaustion, pre-§2.1 guard). Findings integrated into paper §7.6.10. |
| Paper surgery | Theorems T1–T5 inserted (§5); limitations (σ null, MuSiQue near-zero, MiniLM dependence, NLI miscalibration, anchor sensitivity); retrieval-time vs rerank-time guarantee clarification; data-validity note extended (§7.6.10); related-work positioning sentence (§2.5). Citation-existence verification of the 7 questionable arXiv IDs remains USER/web (§5 item 11). |
| Tests | 58 → **100, all passing**. |

---

## §4. Stronger development directions (high EV only)

Ranked by **expected paper impact / effort**, given prior art.  
**All start offline unless marked LOCAL or API.**

### 4.0 Theory extensions shipped this pass — T1–T5 (renamed from the v3 D-labels to avoid collision with development D1–D10)

| ID | (was v3) | Contribution | Status | Location |
|----|----------|--------------|--------|----------|
| **T1** | D1 | **Depth-gain regularity** — replaces unverifiable Condition FC with computable ρ_d and bound `F ≥ (1−1/e)(ρ_d − O(d/k))F(S*)`. Highest-value theory contribution. | ✅ impl+test+paper | `theory.py::depth_gain_regularity` |
| **T2** | D3 | **Correlated coverage (pairwise MRF)** — fixes independence bias; monotone + submodular; independence is the corr→0 limit. | ✅ impl+test+paper | `theory.py::correlated_coverage` |
| **T3** | D4 | **Knapsack cost (Theorem 3)** — Sviridenko three-phase; shipped code == theorem. | ✅ impl+test+paper+flag | `theory.py::cost_benefit_greedy`, `algorithm.py` cost_benefit |
| **T4** | D6 | **Multi-anchor per-component** — subsumes teleportation + dense-anchor hybrid under one bounded forest algorithm; `F(∪Sᵢ) ≥ (1−1/e)F(∪S*_{Cᵢ})` for disjoint components. | ✅ impl+test+paper | `theory.py::multi_anchor_greedy` |
| **T5** | D7 | **Hierarchical granularity** — partition matroid over sentence+passage; 1/2 vs (1−1/e) regimes. Makes the strongest empirical finding an algorithmic contribution. | ✅ impl+test+paper | `theory.py::hierarchical_greedy` |

**Theory also shipped this pass — T6–T8 (renamed from v3 §3.6 / §3.7 / D5; all impl+test+paper; only eval-dependent empirical validation remains):**

| ID | (was v3) | Contribution | Status | Location |
|----|----------|---------------|--------|----------|
| **T6** | §3.6 | **Tightness (Feige-style, Prop. 1)** — explicit construction + F values + limit + matching hardness; proves the (1−1/e) bound is tight. | ✅ impl+test+paper | `theory.py::make_tightness_instance` (L778); paper §5.2 / Prop. 1; 2 tests (`test_ratio_approaches_bound`, `test_expected_ratio_formula`) |
| **T7** | §3.7 | **Tree-DP exact optimum** — O(\|V\|·k²) exact S* on near-tree graphs; computes the TRUE approximation ratio (flushes the ratio>1 class of bug at scale). | ✅ impl+test+paper | `theory.py::tree_dp_optimum` (L676); paper §5.10; 4 tests (`test_chain_exact`, `test_matches_bruteforce_on_star`, `test_budget_never_exceeded`, `test_rejects_cyclic_graph`) |
| **T8** | D5 | **Conformal σ replacement** — split-conformal predictor (distribution-free coverage ≥ 1−α) + bottleneck (weakest-link) confidence; converts the σ null result into a real uncertainty theorem. | ✅ impl+test+paper | `theory.py::ConformalSigma` (L608) + `bottleneck_sigma` (L540); paper §5.9; 11 tests (`test_calibrate_and_certify`, `test_coverage_level`, `test_weakest_link`, `test_length_invariant`, …) |

> **All theory T1–T8 is shipped, tested (42 theory tests / 100 total), and in paper §5.** Only eval-dependent empirical validation remains: (a) measure the TRUE ratio via T7 on real near-tree graphs, (b) validate conformal coverage ≥ 1−α on held-out correctness labels — both are Phase-3 / Phase-5 items in the §12 runbook (USER runs).

### D1. Answer-in-context diagnostic on *our* packs ⭐⭐⭐ (offline → local)

**Why:** Bala’26 shows this predicts F1 better than recall and explains when submodular packing wins. We currently optimize/report R@k only — reviewers will ask for the packing-relevant metric.  
**What:** For each system’s packed context string, measure whether gold answer span survives (Hotpot/2Wiki). Correlate with EM/F1 when available; even without EM, report answer-in-context@budget vs R@5.  
**Diff vs Bala:** Apply to **graph-coherent vs dense vs hybrid** packs from PATHFINDER, not only flat packers.  
**Effort:** Small script on existing retrieval outputs + gold answers (no API).  
**Gate:** If ρ(answer-in-context, F1) weak when F1 exists, do not center this metric.

### D2. Packing baselines: MMR, focused top-k, cost-aware submodular, DPP-lite ⭐⭐⭐ (offline code, LOCAL run)

**Why:** S-RAG/Bala/GeoRAG all beat top-k/MMR with flat submodular. If we only compare BFS/SA, reviewers say “wrong baselines.”  
**What:** Same candidate pool & budget: (a) dense top-k, (b) MMR, (c) cardinality greedy coverage, (d) **ΔF/token** greedy, (e) PATHFINDER frontier greedy, (f) dense-anchor hybrid.  
**Diff:** Adds **connectivity constraint** comparison the flat papers lack.  
**Effort:** Medium; mostly offline implementation.

### D3. Cost-aware / knapsack greedy (H10) ⭐⭐ (offline + local) — **code-ready**

**Why:** Passages have heterogeneous token length; paper already cites Sviridenko-style remark — empirics missing. Bala uses marginal-gain-per-token.  
**What:** Implement density greedy; pilot N=500 BGE-passage. **Code + theorem already shipped (T3, `cost_benefit` flag); only the pilot run remains.**  
**Promote if:** answer-in-context or R@k↑ at same token budget.

### D4. Mechanism / error taxonomy ⭐⭐⭐ (offline on locked JSON)

**Why:** Surveys and SeedER/HippoRAG all say *when* structure helps is underspecified. This is our best “not just another hybrid” story.  
**Slices:** hop type (bridge/compare), bridge-entity degree, gold dense-rank, graph component size, teleport help/hurt.  
**Output:** `results/analysis/error_taxonomy.md` + one paper figure.  
**No API.**

### D5. Graph construction upgrade for MuSiQue / 2Wiki (H11) ⭐⭐ (local, no API)

**Why:** HippoRAG wins on MuSiQue/2Wiki via **OpenIE entity graph + PPR**, not passage cosine graphs. Our MuSiQue R@5≈0 is likely **construction**, not greedy.  
**What (in order):**  
1. Diagnostics: gold support overlap, edge density, component connectivity on MuSiQue graphs.  
2. Add **entity co-occurrence / coref / OpenIE-lite edges** (spaCy first; LLM OpenIE only if needed → API last).  
3. Optional: PPR baseline on *same* new graph (fair).  
**Kill:** if still ~0 after entity edges, limitation section — don’t hide.

### D6. Two-stage pipeline: wide dense retrieve → PATHFINDER pack ⭐⭐ (local)

**Why:** Bala separates retriever vs packer; G-Retriever/SubgraphRAG assume a graph already. Production RAG is two-stage.  
**What:** Pool top-20/50 dense passages → build mini-graph or run frontier greedy / cost-aware pack under K_tok.  
**Already partial:** `two_stage_20_5` in hybrid JSON — promote, ablate pool size, report answer-in-context.

### D7. Fair efficiency table ⭐⭐ (local + API only for others if needed)

**Why:** Our real advantage vs IRCoT/ToG may be **0 LLM calls @ retrieve, ~ms latency**, not F1.  
**Columns:** R@5, answer-in-context, EM/F1 (if clean), **LLM calls/q**, index time, p50 latency.  
**Systems:** PF-hybrid, dense, MMR pack, HippoRAG, IRCoT (matched generator when possible).

### D8. Theory sharpening vs COSMOS / PCST ⭐ (offline writing) — **partially done via T1–T5**

**Why:** Reviewers who know COSMOS/G-Retriever will ask.  
**What:**  
- Table: objective, constraint (frontier tree vs MaxST vs PCST), graph type, approx ratio assumptions (γ alignment in COSMOS).  
- Clarify teleport/hybrid **breaks** pure tree certificate → per-anchor corollary only (T4).  
- Independence assumption limits (T2; already in limitations).

### D9. Paper surgery (mandatory, offline) ⭐⭐⭐ — **largely done; see §5**

See §5. Without this, new experiments won’t save the paper.

### D10. Clean H5 EM only after D1–D4 ⭐ (API last)

Matched generator, matched budget, run cards, bootstrap. Accept kill criterion.

### Explicitly deferred

Learned RL expansion (SeedER clone) · full DPP training (ScalDPP) · community summaries (GraphRAG) · training PathRetriever-class models · multidimensional revival on Hotpot.

---

## §5. Paper surgery checklist (offline, do before new evals)

| # | Fix | Status |
|---|-----|--------|
| 1 | Related work = §2.5 list; kill “first graph-coherent submodular” | ✅ positioning sentence + neighbors cited |
| 2 | Abstract leads passage + hybrid + certificate scope | ✅ repositioned (parity + guarantee + structure) |
| 3 | Encoder protocol = MiniLM/BGE (not text-embedding-3-small) | ✅ |
| 4 | Split retrieval vs QA; no literature EM beside our EM≈0 | ✅ |
| 5 | Quarantine poisoned EM narratives (passage EM=0 ≠ “LLM can’t read passages”) | ✅ §7.6.10 data-validity note |
| 6 | Semantic-only default; facets = optional interface | ✅ demoted to generalization |
| 7 | Demote σ / four-open-problems centrality | ✅ σ null result in limitations + ✅ conformal/bottleneck replacement (T8, §5.9) shipped |
| 8 | Add answer-in-context to metrics protocol | ⬜ **(metric not yet implemented — D1/B2)** |
| 9 | Add MMR/cost-aware as baselines in protocol | ⬜ cost-aware code ready (T3); MMR packer script pending (D2/B4) |
| 10 | Limitations: MuSiQue, reader-scale interaction (cite Bala), COSMOS/S-RAG overlap | ✅ |
| 11 | **Verify the 7 questionable arXiv citation IDs** (PCR `2511.18313`, reasoning-failure `2603.14045`, …) — BLOCKS submission | ⬜ **USER/web** (cannot verify offline) |

---

## §6. Experimental protocol

| Knob | Default |
|------|---------|
| Data | Hotpot distractor; pilot N=500 stable IDs |
| Nodes | Passage |
| Encoder (new) | BGE |
| Retrieval | Dense-anchor hybrid k=5 |
| Weights (wiki) | Semantic-only |
| K_tok | 2048 (also report tight budget e.g. 160–512 for packing story) |
| Metrics | R@5/10/20, para-R@5, **answer-in-context**, EM/F1 (clean only), latency, LLM calls/q |
| Packing baselines | top-k, MMR, cost-aware submodular, PF frontier, hybrid |
| Stats | paired bootstrap 10k |
| Generator | Llama-3.3-70B @ Groq (**API last**) |

Run card `meta`: git_sha, script, hypothesis_id, n, encoder, node_level, retrieval_mode, k_tok, generator, seed, quota_exhausted, n_rate_limited, wall_time_s.

---

## §7. Workstreams (mapped to §4)

```
A Hygiene              ✅ / maintain
B Code: defaults, cost-aware, MMR, answer-in-context metric, adapter   OFFLINE
C Paper + theory + related work                                        OFFLINE  (✅ theory T1–T5; paper §5 largely done)
D Mechanism taxonomy + MuSiQue diagnostics on disk                     OFFLINE
E Local retrieval / packing pilots                                     USER LOCAL
F Generator EM/F1                                                      USER API LAST
G External SOTA fair score                                             USER
H Paper lock                                                           OFFLINE
```

### B — Offline engineering (agent)

| ID | Task | Status |
|----|------|--------|
| B1 | Passage + hybrid + encoder CLI defaults | ✅ |
| B2 | `answer_in_context(context, gold_answers)` utility + unit tests | ⬜ |
| B3 | Cost-aware greedy (ΔF/tokens) flag in `run_pathfinder` | ✅ (`cost_benefit` flag + `cost_benefit_greedy`, T3) |
| B4 | MMR packer baseline script | ⬜ |
| B5 | `31_error_taxonomy.py` over existing eval JSON | ⬜ |
| B6 | `32_packing_baselines.py` (top-k / MMR / cost-aware / PF / hybrid) | ⬜ |
| B7 | Extend `30_sota_adapter` for external dumps | ✅ fairness gate shipped; external `retrieve` callables PENDING §6 |
| B8 | `print_metrics` includes BGE, hybrid, answer-in-context when present | ⬜ (BGE/hybrid labels present; answer-in-context pending B2) |
| — | `answer_helper.py::retrieve_then_answer` unified EM path | ✅ (infra, not in original B list) |

**Gate B:** pytest ≥ 100 tests passing; no dataset full scan required for CI. **Currently 100/100.**

### C — Paper (agent)

§5 checklist + §2 positioning paragraph. **Theory T1–T5 done; paper surgery items 1–7,10 done; 8,9,11 pending (8,9 need B2/B4 code; 11 needs USER/web citation verification).**

### D — Offline analysis (agent scripts; user may run)

Taxonomy; teleport slices; MuSiQue gold-vs-graph overlap report.

### E — Local (user, no API)

```cmd
python -m pytest pathfinder\tests -q --rootdir=.
python experiments\32_packing_baselines.py --graphs data\hotpotqa_graphs_passage_bge.pkl --max_samples 500 ...
python experiments\31_error_taxonomy.py --eval results\hotpotqa_eval_bge_passage.json ...
:: optional graph rebuild with entity edges for MuSiQue
```

### F — API last (user)

Smoke 3 (generate_answers.py) → H5 N=100 → N=500 → optional rerank → never average empties.

### G — SOTA (user)

Adapters first; same 500 IDs; one scorer (`30_sota_adapter.py` with `enforce_fairness`); efficiency columns.

---

## §8. Default stack (target)

```
Passage nodes + BGE
  → dense pool (top-20) optional
  → dense-anchor hybrid / frontier submodular (semantic-only)
  → cost-aware tie-break under K_tok
  → report R@k + answer-in-context
  → (API last) Llama-3.3-70B → EM/F1
```

Not default: LLM traversal, NLI stop, multi-facet weights on wiki, sentence MiniLM headline.

---

## §9. Milestones

| ID | Milestone | Status |
|----|-----------|--------|
| M0 | Hygiene + sentinel + tests | ✅ (**100 tests**) |
| M1 | Passage/BGE/hybrid ledger | ✅ |
| M2 | Prior-art map + thesis lock | ✅ |
| M2.5 | **Consolidation pass** (10 files deleted, T1–T5, fairness gate, unified EM, coverage fix, paper surgery) | ✅ v6 |
| M3 | Paper surgery + related work | ✅ (items 8,9,11 pending) |
| M4 | Offline metrics: answer-in-context, MMR, cost-aware code | ◐ cost-aware ready (T3); theory code fully shipped (T1–T8); answer-in-context utility + MMR packer pending (B2/B4) |
| M5 | Mechanism taxonomy from disk | ⬜ |
| M6 | Local packing/retrieval pilots | ⬜ USER |
| M7 | MuSiQue construction attempt | ⬜ USER |
| M8 | Clean H5 | ⬜ USER API |
| M9 | Fair SOTA + efficiency table | ⬜ USER |
| M10 | Camera-ready | ⬜ |

---

## §10. Risks

| Risk | Mitigation |
|------|------------|
| Novelty clash with S-RAG/COSMOS/Bala | §2 citations + N1–N6 deltas only |
| Submodular packing gains vanish at 70B reader | Cite Bala scale ladder; emphasize certificate + efficiency + R@k |
| MuSiQue zero | H11 entity edges; else limitation |
| API corruption | API last; sentinel; drop limited queries |
| Wrong baselines (only BFS) | D2 MMR/cost-aware |
| Chasing dense R@5 | Thesis = coherence + parity + certificate |
| SeedER critique (greedy misses connectors) | Hybrid anchors + teleport regime analysis |

---

## §11. Immediate next actions

### Agent (now — offline only)

1. **B2 `answer_in_context` utility + tests** (unblocks §5 item 8, D1, H9).  
2. **B4 MMR packer + B6 `32_packing_baselines.py`** (unblocks §5 item 9, D2).  
3. **B5 `31_error_taxonomy.py`** (D4 mechanism story; offline on locked JSON).  
4. **No evals. No API.** (Theory T1–T8 already shipped + tested (42 theory tests) + in paper §5 — §5.2 tightness, §5.9 conformal σ, §5.10 tree-DP. Only eval-dependent empirical validation remains; see §12 runbook.)

### User (later — §12 runbook)

1. Phase 0 pre-flight (`pytest` 100/100 + AST parse).  
2. Phase 1 graph builds (`data/` is currently empty).  
3. Phase 2 retrieval evals (`--no_llm`, no key).  
4. Phase 4 generation **only after** Phase 2 + a fresh `GROQ_API_KEY`.  
5. Phase 6 external SOTA through `30_sota_adapter` (fairness gate enforced).

### Do not start early

Full re-eval of locked tables · Groq batches · RL SeedER clone · “first submodular” branding.

---

## §12. Eval runbook (USER runs — copy-paste, Windows CMD)

> **Agent does not run these (R2/R3/R4).** Commands are verified against each script's argparse.  
> **Windows CMD** uses `\` paths and `set VAR=`. **Git Bash / WSL** users: swap `\`→`/` and `set`→`export`.  
> **Order matters:** build graphs first (Phase 1) because `data/` starts empty → retrieval (Phase 2) → ablations (Phase 3, optional) → **API generation LAST (Phase 4)** → metrics/plots (Phase 5) → external SOTA (Phase 6).  
> **Never average empty / rate-limited predictions (R9):** re-run with a fresh `GROQ_API_KEY` instead.

### Phase 0 — Pre-flight (seconds, no data, no API) — MUST PASS first

```cmd
:: Expect: 100 passed in <N>s
python -m pytest pathfinder\tests -q --rootdir=.

:: Expect: no output (all experiment scripts parse)
python -c "import ast,glob;[ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('experiments/*.py')]"

:: Expect: coverage_ratio.mean_ratio <= 1.0  (was 1.0097 before the fix) and method_note present.
::          If mean_ratio > 1.0 -> the coverage fix regressed; STOP and re-open §3.6.
python experiments\05_evaluate.py --graphs data\hotpotqa_graphs.pkl --max_samples 20 --no_llm --no_ablation
```
> Phase 0's last command needs `data\hotpotqa_graphs.pkl` — run Phase 1 first if `data\` is empty, then come back.

### Phase 1 — Graph builds (USER, no API; data/ starts empty)

> Drop `--max_samples` for the **full** HotpotQA build (N=7405). `--max_samples 500` = the stable pilot subset.  
> **Note:** `01c`'s `--output` default is `data\hotpotqa_graphs_bge.pkl` regardless of `--passage_level`, so passage-BGE builds MUST pass `--output ...passage_bge.pkl --passage_level` explicitly (matches the repo convention).

```cmd
:: sentence-level MiniLM (the legacy / cross-bench graph)
python experiments\01_build_kg.py --max_samples 500 --output data\hotpotqa_graphs.pkl

:: passage-level MiniLM
python experiments\01b_build_kg_passage.py --max_samples 500 --output data\hotpotqa_graphs_passage.pkl

:: passage-level BGE  (MUST pass --output explicitly + --passage_level)
python experiments\01c_build_kg_bge.py --max_samples 500 --output data\hotpotqa_graphs_passage_bge.pkl --passage_level

:: optional: sentence-level BGE (for the §3.1 "Sent 500 BGE" row)
python experiments\01c_build_kg_bge.py --max_samples 500 --output data\hotpotqa_graphs_bge.pkl
```

### Phase 2 — Retrieval evals (USER, `--no_llm` = R@5/R@10 only, NO key)

```cmd
:: PATHFINDER coherent (BGE passage) — the headline config
python experiments\05_evaluate.py --graphs data\hotpotqa_graphs_passage_bge.pkl --max_samples 500 --no_llm --output results\raw\sota_pathfinder_coherent.json

:: dense-anchor hybrid (BGE passage) — the parity config
python experiments\23_hybrid_retrieval.py --graphs data\hotpotqa_graphs_passage_bge.pkl --max_samples 500 --output results\raw\sota_pathfinder_hybrid.json

:: passage MiniLM 500 + full-scale (drop --max_samples for N=7405)
python experiments\05_evaluate.py --graphs data\hotpotqa_graphs_passage.pkl --max_samples 500 --no_llm --output results\raw\hotpotqa_eval_passage_500.json
python experiments\05_evaluate.py --graphs data\hotpotqa_graphs_passage.pkl --no_llm --output results\raw\hotpotqa_eval_passage_full.json
```

### Phase 3 — Teleportation ablations (USER, optional, no API)

```cmd
python experiments\05b_teleportation_ablation.py --graphs data\hotpotqa_graphs_passage.pkl --max_samples 500
python experiments\05c_teleportation_sensitivity.py --graphs data\hotpotqa_graphs_passage.pkl --max_samples 200
python experiments\05d_teleportation_impact.py --graphs data\hotpotqa_graphs_passage.pkl --max_samples 500
```

### Phase 4 — Generation / EM-F1 (USER, API LAST — needs a fresh GROQ_API_KEY)

> **`generate_answers.py` has NO `--smoke` flag and NO argparse** — it runs a hardcoded 3-query smoke against `data\hotpotqa_graphs.pkl`. The correct invocation is just the script name. For N>3 use `19_heterogeneous_llms.py` or `25_rerank_eval.py`.  
> **R9:** if a run returns mostly `RATE_LIMITED` / empty predictions, DELETE that JSON and re-run with a fresh key — never average empties.

```cmd
set GROQ_API_KEY=<fresh_key>

:: 3-query smoke (hardcoded; reads data\hotpotqa_graphs.pkl, prints Q/Gold/Pred/EM/F1)
python experiments\generate_answers.py

:: scaled generation (N queries) — pick ONE per run:
python experiments\19_heterogeneous_llms.py --graphs data\hotpotqa_graphs_passage_bge.pkl --max_samples 100 --model llama-3.3-70b-versatile
python experiments\25_rerank_eval.py --graphs data\hotpotqa_graphs_passage_bge.pkl --max_samples 100 --model llama-3.3-70b-versatile
::  ^^ 25_rerank_eval.py: add --no_llm for the pre-rerank baseline only (no key needed).
```

### Phase 5 — Metrics + plots (USER, no API)

```cmd
:: prints the consolidated benchmark table (labels by filename: "passage" / "full" / else N=500)
python experiments\print_metrics.py

:: writes 7 PNGs to results\plots\
python results\make_plots.py
```

### Phase 6 — External SOTA (USER, fairness gate ENFORCED)

> Build/clone each external system (IRCoT, SubgraphRAG, HippoRAG 2, LightRAG), then score through **one scorer** — `experiments\30_sota_adapter.py` — with `enforce_fairness` called per system. Same 500 stable IDs. **Matched encoder + matched generator** or the row is blocked (UNFAIR) / flagged (FAIR_WITH_NOTE).

```cmd
:: adapter self-test (toy data, no dataset/LLM) — expect "Smoke test passed."
python experiments\30_sota_adapter.py
```
Then, per external system, wire a `retrieve(question, qid) -> list` callable + `predict_fn`, call `enforce_fairness(name, encoder, generator)` first, then `score_system(...)` and write `results\raw\sota_<name>.json` with a run-card `meta`. Pending adapters: `ircot`, `subgraphrag`, `hipporag2`, `lightrag` (see `30_sota_adapter.list_pending()`).

---

## §13. Literature anchors (for citations)

| Key | Ref |
|-----|-----|
| Submodular summarization | Lin & Bilmes, 2011 |
| Flat submodular RAG | S-RAG (OpenReview gtcOku1v2s); GeoRAG arXiv:2606.29328 |
| Budgeted packing + answer-in-context | Bala, arXiv:2607.00725 |
| Connected submodular KBQA | COSMOS, ACL 2026 (anthology 2026.acl-long.1662) |
| PCST subgraph RAG | G-Retriever, NeurIPS 2024 |
| Learned subgraph | SubgraphRAG |
| Seed+expand RL | SeedER, arXiv:2605.23753 |
| OpenIE+PPR | HippoRAG, NeurIPS 2024 |
| Iterative CoT retrieve | IRCoT |
| Path learning | PathRetriever; MDR; Beam Retrieval (NAACL 2024) |
| Dense vs Graph | RAG vs GraphRAG systematic evals (e.g. arXiv:2502.11371) |
| Hard multi-hop data | MuSiQue (Trivedi et al., TACL 2022) |
| Submodular maximization | NWF 1978 (greedy 1−1/e); Feige 1998 (tightness); Sviridenko 2004 (knapsack three-phase) |

---

## §14. Document history

| Ver | Summary |
|-----|---------|
| v1–v2 | Experiment lists; rate-limit crisis |
| v3 | Research OS skeleton + repository-inspection reconciliation (status of implemented fixes, theory D1/D3/D6/D7, 58→100 tests) |
| v4 | Thesis lock; offline→API policy |
| v5 | Max-effort prior-art map; COSMOS/S-RAG/Bala/GeoRAG/G-Retriever/SeedER/HippoRAG; honest niches N1–N6; stronger dirs D1–D10; answer-in-context + packing baselines; banned low-EV work |
| **v6** | **Merge of v5 (skeleton) + v3 (consolidation status). Theory D1/D3/D4/D6/D7 relabeled T1–T5 (§4.0) to avoid collision with development D1–D10. §3.6 consolidation-completed ledger (10 files deleted, 100 tests, fairness gate, unified EM path, cost-aware flag, coverage-ratio fix, EM discrepancy resolved, paper surgery largely done). §12 replaced with a verified copy-paste eval runbook (Phases 0–6, Windows CMD) — corrects the v5 `generate_answers.py --smoke 5` bug (no such flag) and the `01c --passage_level` output-naming gotcha. §15 git & sync protocol + conflict-handling note.** |

---

## §15. Git & sync protocol (and how the v6 conflict was handled)

### 15.1 Standing rule

**R5:** no push, no PR, no history rewrite unless the user provides a PAT and explicitly asks. The agent commits logical changes locally (offline work) but does not publish. One authorized push was executed under v6 at the user's explicit request with a PAT they supplied.

### 15.2 The v6 divergence & conflict (the note the user asked for)

- **merge-base:** `42265e9`. Local was ahead by 2 commits (consolidation work: `554b90f`, `118763e` — theory/tests, fairness gate, unified EM, paper surgery, 10 redundant-file deletions). Remote was ahead by 3 commits (`99cf437` v3, `d6d048a` v4, `a476c7d` v5 — the PLAN lineage).
- **What actually differed:** `git diff --name-status merge-base..origin/master` showed the remote changed **ONLY `PLAN.md`**. Every committed local-only change (paper, `theory.py`, `test_theory.py`, `30_sota_adapter.py`, `answer_helper.py`, `config.py`, `algorithm.py`, `make_plots.py`, `05_evaluate.py`, `generate_answers.py`, `README`) and every committed local deletion (`prompt.md`, `results/literature_audit.md`, `results/multi_benchmark.md`, `results/sota_comparison.md`, `experiments/README.md`) was **untouched on the remote**. (Two untracked scratch files — `pathfinder/THEORY.md`, `results/em_discrepancy_analysis.md` — were swept during consolidation but never entered git, so they are irrelevant to the merge.)
- **Therefore the merge was trivial in structure:** `git merge --no-commit --no-ff origin/master` produced **exactly one conflict — `PLAN.md` (both modified from the v3 base)**. All other files auto-merged cleanly; deletions auto-resolved as deleted. No modify/delete conflicts (the feared `experiments/README.md` / `results/sota_comparison.md` cases were clean local deletions — those files exist at the merge-base, so they resolve, not conflict).

### 15.3 How it was resolved (the "smart" handling)

A force-push would have destroyed v5's research-mature skeleton; discarding v3 would have lost the consolidation status. Instead:

1. `git merge --no-commit --no-ff origin/master` — paused on the single `PLAN.md` conflict.  
2. Resolved `PLAN.md` by writing a **synthesized v6** that takes v5 §0–§14 as the skeleton (preserved verbatim) and folds in v3's consolidation status (§3.6), the relabeled theory T1–T5 (§4.0), updated status markers, the corrected eval runbook (§12), and this git note (§15). Conflict markers removed; both lineages preserved.  
3. `git add PLAN.md` — staged the resolution.  
4. Verified `pytest 100/100` + AST parse before committing (so a broken merge never lands).  
5. `git commit` — the merge commit records that only `PLAN.md` conflicted and was resolved by v6 subsuming v5.

### 15.4 PAT handling (security)

- The push used the PAT **inline in the push URL only** — `git push https://<user>:<PAT>@github.com/.../Ragidea.git master` — so the token was **never written to `.git/config`** and `origin` stays at the non-token URL `https://github.com/Yash-Awasthi-02/Ragidea.git`.  
- The PAT was **not** committed into any tracked file.  
- ⚠️ **The PAT is exposed in plaintext in the chat history.** After the push lands, the user should **rotate/revoke it** at GitHub → Settings → Developer settings → Personal access tokens. Treat it as compromised regardless of the push succeeding.

### 15.5 Reusable conflict playbook for this repo

When local consolidation work and remote PLAN-only edits next diverge:

1. `git fetch origin && git diff --name-status merge-base..origin/master` — confirm the remote surface.  
2. If remote touches only `PLAN.md` (the usual case), `git merge --no-commit --no-ff origin/master`, resolve `PLAN.md` by re-synthesizing (keep the v6 skeleton, update status markers + history row), `git add PLAN.md`, verify `pytest` + AST, commit.  
3. If remote touches code the agent also touched, resolve each file by hand (prefer the side with the more recent fix; never silently drop a correctness fix), re-run `pytest`, then commit.  
4. Push only with an explicitly-provided PAT via an inline URL; never store the token; never force-push `master`.

---

*End of PLAN.md v6 — the idea is not empty space; win on combination, certificate, passage-graph multi-hop RAG, diagnostics, and honesty.*
