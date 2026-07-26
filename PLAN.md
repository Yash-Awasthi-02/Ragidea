# PATHFINDER — Research Master Plan (v5)

| Field | Value |
|-------|-------|
| **Project** | PATHFINDER — Submodular Coverage Maximization over Multidimensional KGs for Multi-Hop RAG |
| **Repo** | https://github.com/Yash-Awasthi-02/Ragidea |
| **Paper** | `pathfinder-paper.md` |
| **Plan version** | **v5** (supersedes v4) |
| **Last updated** | 2026-07-26 |
| **Status** | Deep prior-art map locked · offline → local → API-last · stronger deltas identified |

---

## How to use this document

1. **§0** — operating policy (offline first; evals local; API last).  
2. **§1** — thesis lock (what we may claim).  
3. **§2** — **prior-art map** (who already tried the idea; where we sit).  
4. **§3** — locked evidence ledger.  
5. **§4** — **stronger development directions** (only high-EV items).  
6. **§5–§11** — protocol, workstreams, milestones, risks, next actions.

This v5 is the result of an extensive literature sweep. If a direction is not listed in §4, it is either already done, low EV, or a known wall.

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
| R5 | No push/history rewrite unless user provides PAT and asks. |
| R6 | One variable per experiment. |
| R7 | One scorer for all systems (`30_sota_adapter.py`). |
| R8 | Every new JSON has a **run card** (`meta`). |
| R9 | Rate-limited / empty-pred EM files are deleted, not averaged. |
| R10 | Before code changes: WHY → impact → theory implication → files. |

### 0.3 Safe agent checks

```cmd
pytest pathfinder/tests/
python -c "import ast,glob;[ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('experiments/*.py')]"
python experiments/print_metrics.py
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
| H8 | σ calibrates | ⚠ weak — demote |
| **H9 (new)** | **Answer-in-context** predicts F1 better than R@5 on our packs | ⬜ offline metric |
| **H10 (new)** | Cost-aware greedy (ΔF / tokens) beats cardinality greedy under heterogeneous passage lengths | ⬜ offline+local |
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

58 tests · 33 experiment scripts · `30_sota_adapter.py` setup · `RATE_LIMITED` sentinel shipped.

---

## §4. Stronger development directions (high EV only)

Ranked by **expected paper impact / effort**, given prior art.  
**All start offline unless marked LOCAL or API.**

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

### D3. Cost-aware / knapsack greedy (H10) ⭐⭐ (offline + local)

**Why:** Passages have heterogeneous token length; paper already cites Sviridenko-style remark — empirics missing. Bala uses marginal-gain-per-token.  
**What:** Implement density greedy; pilot N=500 BGE-passage.  
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

### D8. Theory sharpening vs COSMOS / PCST ⭐ (offline writing)

**Why:** Reviewers who know COSMOS/G-Retriever will ask.  
**What:**  
- Table: objective, constraint (frontier tree vs MaxST vs PCST), graph type, approx ratio assumptions (γ alignment in COSMOS).  
- Clarify teleport/hybrid **breaks** pure tree certificate → per-anchor corollary only.  
- Independence assumption limits (already in limitations).

### D9. Paper surgery (mandatory, offline) ⭐⭐⭐

See §5. Without this, new experiments won’t save the paper.

### D10. Clean H5 EM only after D1–D4 ⭐ (API last)

Matched generator, matched budget, run cards, bootstrap. Accept kill criterion.

### Explicitly deferred

Learned RL expansion (SeedER clone) · full DPP training (ScalDPP) · community summaries (GraphRAG) · training PathRetriever-class models · multidimensional revival on Hotpot.

---

## §5. Paper surgery checklist (offline, do before new evals)

| # | Fix |
|---|-----|
| 1 | Related work = §2.5 list; kill “first graph-coherent submodular” |
| 2 | Abstract leads passage + hybrid + certificate scope |
| 3 | Encoder protocol = MiniLM/BGE (not text-embedding-3-small) |
| 4 | Split retrieval vs QA; no literature EM beside our EM≈0 |
| 5 | Quarantine poisoned EM narratives (passage EM=0 ≠ “LLM can’t read passages”) |
| 6 | Semantic-only default; facets = optional interface |
| 7 | Demote σ / four-open-problems centrality |
| 8 | Add answer-in-context to metrics protocol |
| 9 | Add MMR/cost-aware as baselines in protocol |
| 10 | Limitations: MuSiQue, reader-scale interaction (cite Bala), COSMOS/S-RAG overlap |

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
C Paper + theory + related work                                        OFFLINE
D Mechanism taxonomy + MuSiQue diagnostics on disk                     OFFLINE
E Local retrieval / packing pilots                                     USER LOCAL
F Generator EM/F1                                                      USER API LAST
G External SOTA fair score                                             USER
H Paper lock                                                           OFFLINE
```

### B — Offline engineering (agent)

| ID | Task |
|----|------|
| B1 | Passage + hybrid + encoder CLI defaults |
| B2 | `answer_in_context(context, gold_answers)` utility + unit tests |
| B3 | Cost-aware greedy (ΔF/tokens) flag in `run_pathfinder` |
| B4 | MMR packer baseline script |
| B5 | `31_error_taxonomy.py` over existing eval JSON |
| B6 | `32_packing_baselines.py` (top-k / MMR / cost-aware / PF / hybrid) |
| B7 | Extend `30_sota_adapter` for external dumps |
| B8 | `print_metrics` includes BGE, hybrid, answer-in-context when present |

**Gate B:** pytest 58+ new tests; no dataset full scan required for CI.

### C — Paper (agent)

§5 checklist + §2 positioning paragraph.

### D — Offline analysis (agent scripts; user may run)

Taxonomy; teleport slices; MuSiQue gold-vs-graph overlap report.

### E — Local (user, no API)

```cmd
pytest pathfinder/tests/
python experiments/32_packing_baselines.py --graphs data/hotpotqa_graphs_passage_bge.pkl --max_samples 500 ...
python experiments/31_error_taxonomy.py --eval results/hotpotqa_eval_bge_passage.json ...
:: optional graph rebuild with entity edges for MuSiQue
```

### F — API last (user)

Smoke 5 → H5 N=100 → N=500 → optional rerank → never average empties.

### G — SOTA (user)

Adapters first; same 500 IDs; one scorer; efficiency columns.

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
| M0 | Hygiene + sentinel + 58 tests | ✅ |
| M1 | Passage/BGE/hybrid ledger | ✅ |
| M2 | Prior-art map + thesis lock (this doc) | ✅ |
| M3 | Paper surgery + related work | ⬜ next |
| M4 | Offline metrics: answer-in-context, MMR, cost-aware code | ⬜ |
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

1. Paper surgery (§5) + related-work from §2.  
2. Implement B2–B6 (answer-in-context, cost-aware, MMR, taxonomy, packing baselines).  
3. Theory note vs COSMOS/PCST/S-RAG/Bala.  
4. **No evals. No API.**

### User (later)

1. `pytest`  
2. Phase E packing/taxonomy/MuSiQue graph diagnostics  
3. Phase F only after E gates  
4. Phase G external systems  

### Do not start early

Full re-eval of locked tables · Groq batches · RL SeedER clone · “first submodular” branding.

---

## §12. Quick commands

```cmd
:: offline
pytest pathfinder/tests/
python experiments/print_metrics.py

:: local retrieval/packing (USER, no key)
python experiments/32_packing_baselines.py --help

:: API last (USER)
set GROQ_API_KEY=...
python experiments/generate_answers.py --smoke 5
```

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

---

## §14. Document history

| Ver | Summary |
|-----|---------|
| v1–v2 | Experiment lists; rate-limit crisis |
| v3 | Research OS skeleton |
| v4 | Thesis lock; offline→API policy |
| **v5** | **Max-effort prior-art map; COSMOS/S-RAG/Bala/GeoRAG/G-Retriever/SeedER/HippoRAG; honest niches N1–N6; stronger dirs D1–D10; answer-in-context + packing baselines; banned low-EV work** |

---

*End of PLAN.md v5 — the idea is not empty space; win on combination, certificate, passage-graph multi-hop RAG, diagnostics, and honesty.*
