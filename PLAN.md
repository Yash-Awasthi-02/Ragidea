# PATHFINDER — Master Plan (v3, reconciled with repository inspection)

> **Project**: PATHFINDER — Submodular Coverage Maximization over Multidimensional Knowledge Graphs for Multi-Hop RAG
> **Repo**: `https://github.com/Yash-Awasthi-02/Ragidea`
> **Last Updated**: 2026-07-27
> **Supersedes**: PLAN.md v2. Reconciled with the external repository inspection review (v3).

---

## §0. CRITICAL OPERATING RULES — READ FIRST

| # | Rule | Reason |
|---|------|--------|
| 1 | **NEVER run evaluations.** Evals take **9–18 h**. They are the **ABSOLUTE LAST STEP** (§6), run manually by the user. | Time / quota cost |
| 2 | **NEVER run** `experiments/05*.py`, `15*.py`–`19*.py`, `25*.py`, `generate_answers.py`, or anything touching a dataset or the Groq API. | These are evals |
| 3 | **NEVER push, open PRs, or rewrite git history.** | Repo safety |
| 4 | When validation is needed: state the **exact command** + **expected output**, then **STOP and wait** for the user. | Division of labor |
| 5 | **No-eval work first** (cleaning, code fixes, theory, proofs, paper). Evals last. | Maximize progress |
| 6 | Before any code change: explain **WHY → expected impact → theoretical implication**. | Scientific rigor |
| 7 | **Stop competing on recall.** Reposition as *parity + certificate + structure* (§4). | Inspection §E2 |

**Safe-to-run (seconds, no API, no dataset):**
```cmd
python -m pytest pathfinder\tests -q --rootdir=.            :: 100 tests, ~1s
python -c "import ast,glob;[ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('experiments/*.py')]"
```

---

## §1. STATUS — WHAT THE INSPECTION CHANGED

The inspection (written **without repo access**) flagged items as "TODO / cannot verify."
With the actual source now read, the true state is:

### 1.1 Already implemented (verified in source) — NO action
- [x] §2.1 rate-limit guard (`RATE_LIMITED` sentinel, `quota_exhausted` flag, per-run counter) — `generate_answers.py`
- [x] §2.2 LLM rerank + semantic cache (`CACHE_THETA=0.92`) — `run_pathfinder.py::llm_rerank`
- [x] §2.4 dense-anchor hybrid `run_pathfinder_hybrid` — `run_pathfinder.py`
- [x] §2.5 teleportation flags (`always_dense`, `teleport_gate`) — `run_pathfinder.py::_single_pass`
- [x] Multi-anchor `run_pathfinder_multi_anchor` — `run_pathfinder.py`
- [x] `.gitignore` hardened (`results/raw/`, `*.log`)
- [x] 58 formal-property unit tests — `pathfinder/tests/test_formal_properties.py` (now 100 total with theory tests)

### 1.2 Fixed in THIS pass (inspection blockers + correctness)
- [x] **Coverage-ratio > 1 bug (C2-3)** — `05_evaluate.py::coverage_ratio` rewrote the comparison:
      teleportation forced OFF (stay inside T(v₀,G)), cardinality matched at k, identical
      `compute_F` objective on both sides, ratio clamped to ≤ 1+ε. Root cause documented in-line.
- [x] **Teleportation-during-measurement audit (Correctness)** — same fix; `method_note` now
      records the comparison is teleport-free and cardinality-matched.
- [x] **Cost-benefit greedy / Theorem 3 (C1-4, D4)** — `pathfinder/theory.py::cost_benefit_greedy`
      (Sviridenko three-phase: partial-enumeration prefix + gain-per-cost completion + max).
      Core change also behind `cost_benefit=True` in `algorithm.py` line 10c. **Shipped code now
      satisfies the theorem it cites.**
- [x] **EM discrepancy (C2-2)** — RESOLVED as *not a code bug*: full-scale runs had 98.2% empty
      predictions (quota exhaustion, pre-§2.1). Findings integrated into paper §7.6.10.

### 1.3 New theory added in THIS pass (inspection §D, ranked by publishability/effort)
- [x] **D1 — depth-gain regularity** replaces unverifiable Condition FC with a computable
      parameter ρ_d and a degradation bound `F ≥ (1−1/e)(ρ_d − O(d/k))F(S*)`. **Highest-value
      theory contribution.** `theory.py::depth_gain_regularity`.
- [x] **D3 — correlated coverage (pairwise MRF)** fixes the independence bias; monotone +
      submodular; independence is the corr→0 limit. `theory.py::correlated_coverage`.
- [x] **D6 — multi-anchor per-component guarantee** subsumes teleportation + dense-anchor hybrid
      under one bounded forest algorithm. `theory.py::multi_anchor_greedy`.
- [x] **D7 — hierarchical granularity** (partition matroid over sentence+passage) makes the
      strongest empirical finding an algorithmic contribution. `theory.py::hierarchical_greedy`.
- [x] **Proofs** integrated into paper §5; **tests** `pathfinder/tests/test_theory.py` (42 new).

**Test count: 58 → 100, all passing.**

---

## §2. REMAINING CODE / CORRECTNESS (no evals)

> Ordered by inspection severity. Each: WHY → change → impact → theory implication.

### 2.1 Unify the EM context path (blocks comparable EM across scripts)
- **WHY**: inspection C2-2 — every EM script feeds a
  different node set / `max_chars` to the same scorer, so EM is not comparable across scripts.
- **Change**: add `experiments/answer_helper.py::retrieve_then_answer(retriever, records, client,
  max_chars)` fixing (retriever, node set, `build_context` chars, prompt); route `05`, `19`,
  `25`, `30` through it.
- **Impact**: EM values become comparable; future discrepancies are attributable to retrieval,
  not plumbing.
- **Theory**: none (infra), but protects the primary answer metric's validity.

### 2.2 Passage-level context budget (minor EM confound)
- **WHY**: `build_context(max_chars=6000)` truncates passage nodes (≥80 words) to 1–2 nodes;
  sentence nodes fit many — confounds the granularity comparison at answer time.
- **Change**: per-node char budget or `--max_chars` scaled by granularity.
- **Impact**: fair passage-vs-sentence EM comparison.
- **Theory**: none.

### 2.3 σ calibration is a null result (C1-2) — replace, don't defend
- **WHY**: Spearman ρ ≈ −0.008, ECE 0.37–0.41, three-tier policy inverted. Two design pillars
  (routing + re-traversal) rest on an uncalibrated σ.
- **Change (D5, design now)**: (a) conformal predictor over traversal features (depth, min edge
  weight, residual ρ, frontier size at exit) → distribution-free coverage of answer correctness;
  or (b) reframe σ as bottleneck confidence and prove a *lower bound on worst-case path
  reliability* rather than predicting EM.
- **Impact**: converts a null result into a publishable uncertainty contribution.
- **Theory**: conformal gives a finite-sample coverage guarantee — a real theorem.

### 2.4 W_dom PCA well-posedness (Correctness)
- **WHY**: K=64 PCA fit on ~13–48 nodes/query fits fewer samples than dimensions ⇒ φ_dom may be noise.
- **Change**: cap `PCA_K ≤ min(N−1, D)` per query (already in `facets.py::compute_phi_dom`); add a
  `φ_dom` independent ablation (never isolated) to the eval design; assert in a unit test that
  `k_actual < N` always.
- **Impact**: honest φ_dom, or evidence to cut it.
- **Theory**: supports the §4 facet demotion.

---

## §3. THEORY STILL TO WRITE (pure proofs, no evals)

> Reordered per inspection §F: FC first (the hole a reviewer falls into), tightness last.

| # | Task | Status | Deliverable |
|---|------|--------|-------------|
| 3.1 | **Condition FC → depth-gain bound** (D1) | DONE (impl+test) | Full proof in paper §5; measure ρ_d on HotpotQA graphs in §6 |
| 3.2 | **Correlated coverage** (D3) | DONE (impl+test) | Write MRF submodularity proof; independence-as-limit remark |
| 3.3 | **Knapsack cost** (Thm 3 / D4) | DONE (impl+test) | Cite Sviridenko; state shipped code == theorem |
| 3.4 | **Multi-anchor per-component** (D6) | DONE (impl+test) | Corollary: F(∪Sᵢ) ≥ (1−1/e)F(∪S*_{Cᵢ}) for disjoint components |
| 3.5 | **Hierarchical granularity** (D7) | DONE (impl+test) | Partition-matroid submodularity; 1/2 vs (1−1/e) regimes |
| 3.6 | **Tightness (Feige-style, Prop. 1)** | TODO | Explicit construction + F values + limit + matching hardness |
| 3.7 | **Tree-DP exact optimum** | TODO | O(|V|·k) exact S* on near-tree graphs; report TRUE ratio (also flushes the ratio>1 class of bug at scale) |
| 3.8 | **Tests for 3.6/3.7** | TODO | Keep suite ≥ 80; add per-theorem property tests |

---

## §4. PAPER REVISION (writing only) — INSPECTION §E

> Reposition. Stop competing on recall. Lead with the certificate.

- [ ] **4.1 Reposition abstract + §1** around *parity + guarantee + structure*:
      "connectivity-constrained submodular selection achieves flat-retrieval parity while
      returning structured, budget-respecting evidence" (inspection §E2). Superiority is NOT
      the claim.
- [ ] **4.2 Lead §7 with passage-level** (R@5 0.667, hybrid 0.708); sentence-level → granularity ablation.
- [ ] **4.3 Demote the five-facet objective** to a *generalization*; report semantic-only as the
      method with an honest negative ablation (grid/Optuna/Thompson all → α=1; γ=0.15 costs 52% R@5;
      φ_temp ≡ 1.0 and φ_conf ≡ 0.70 are inert on these benchmarks).
- [ ] **4.4 Move teleportation to an appendix ablation (or cut)**; REMOVE the forest-admitting
      corollary from the main text. Present **D6 multi-anchor** as the principled mechanism instead.
- [ ] **4.5 Data-validity note (extend §7.6.10)** to cover the EM discrepancy — state the
      98.2%-empty-prediction cause and the §2.1 guard (content integrated into paper §7.6.10).
- [ ] **4.6 Insert new theorems** D1/D3/D4/D6/D7 (now integrated into paper §5).
- [ ] **4.7 Verify every citation** (BLOCKS submission). Confirm the 7 questionable arXiv IDs,
      esp. PCR `2511.18313` and reasoning-failure `2603.14045`; rewrite motivation if they don't exist.
- [ ] **4.8 Audit all five "to our knowledge, first" claims** — each needs a verified citation search.
- [ ] **4.9 Retrieval-time vs rerank-time guarantee** clarification (rerank is a monotone
      post-selection; the (1−1/e) bound holds at retrieval). Report both numbers.
- [ ] **4.10 Honest limitations**: σ null result, MuSiQue near-zero, MiniLM dependence, NLI
      miscalibration, anchor sensitivity.

### Venue ladder (inspection §E3)
| Target | Requires |
|---|---|
| Workshop (fastest) | Fix C1-4, C2-6, C2-7 only — **mostly DONE in §1.2** |
| ACL/EMNLP/NAACL | D2 connectivity experiment + controlled SOTA + all C1/C2 |
| SIGIR / ICTIR | Above; ICTIR takes D1 + D3 (theory-friendly) |
| ICLR / NeurIPS | Only with D1 + D3 landed (DONE) + strong empirics |

---

## §5. SOTA COMPARISON SETUP (code only, NO runs) — with a fairness gate

- [ ] **5.1** `experiments/30_sota_adapter.py` — uniform (question, passages) → (EM, F1, R@5, R@10)
      using our `exact_match`/`f1_score`. (Skeleton exists.)
- [ ] **5.2 Fairness gate (NEW, inspection §F)** — the adapter MUST enforce **matched encoder and
      matched generator** for every system, or the table repeats the confounded comparison
      (MiniLM-vs-GPT-4 is not a method gap). Flag any SOTA using GPT-4/stronger embeddings in Notes.
- [ ] **5.3** Document exact setup for IRCoT / SubgraphRAG / HippoRAG 2 / LightRAG in the
      controlled-SOTA protocol table (paper §7.6.11; setup pending Phase 5 runs).

---

## §6. EVALUATIONS (LAST — USER RUNS, 9–18 h) — with pre-flight

> **I do not run these.** Exact commands + expected output; user executes and pastes results.
> **Pre-flight gate (NEW, inspection §F): the two blocker validations must pass FIRST, or the
> 18-h block generates unpublishable numbers.**

### 6.0 Pre-flight (seconds) — MUST PASS before any long run
```cmd
set GROQ_API_KEY=<fresh_key>
python -m pytest pathfinder\tests -q --rootdir=.            :: expect 100/100 PASS
python experiments\05_evaluate.py --graphs data\hotpotqa_graphs.pkl --max_samples 20 --no_llm --no_ablation
::  ^^ EXPECT in output JSON: coverage_ratio.mean_ratio <= 1.0 (was 1.0097) and method_note present.
::     If mean_ratio > 1.0 → the coverage fix regressed; STOP and re-open §1.2.
```

### 6.1 Regenerate EM for the corrupted full-scale files (fresh key, small N first)
> `hotpotqa_eval_passage_full.json` + `hotpotqa_eval_full.json` are R@5-valid / EM-invalid
> (98.2% empty predictions). Re-answer with the §2.1 guard; confirm `"quota_exhausted": false`.

### 6.2 The missing-justification experiment (inspection D2) — design done, run here
**Connectivity-vs-shuffle at matched recall.** Fix the retrieved SET; present the same nodes as
(a) connected subtree vs (b) shuffled flat list; measure EM/F1 at matched recall. If connected
presentation wins → the graph framing is justified internally. If not → re-pitch as "provable
coverage at flat parity, with interpretable structure." **This is the experiment the paper lacks.**

### 6.3 New-theory measurements (non-LLM)
```cmd
:: Measure depth-gain regularity ρ_d on real HotpotQA graphs (D1 becomes empirical)
:: Tree-DP exact optimum on near-tree graphs → TRUE approximation ratio (§3.7)
:: φ_dom independent ablation (§2.4)
:: Passage + hybrid on MuSiQue (currently unvalidated — inspection C2-5)
```

### 6.4 Rebuild graphs (passage + BGE + hybrid), non-LLM — as v2 §6.3

### 6.5 Controlled SOTA runs (using §5 adapters, fairness gate enforced) — fill the
    controlled-SOTA protocol table (paper §7.6.11).

### 6.6 Full-scale runs (LONG — the 9–18 h block) — as v2 §6.5

### 6.7 Regenerate figures + consolidated metrics
```cmd
python experiments\print_metrics.py
python results\make_plots.py
```

---

## §7. RUNNING TODO TRACKER (reconciled with inspection §G)

**Blockers (must close before any long eval)**
- [x] Coverage ratio > 1.0 debug (C2-3) — FIXED
- [x] Teleportation-during-measurement audit — FIXED (teleport now OFF in ratio)
- [x] EM discrepancy resolve (C2-2) — RESOLVED (quota corruption; see analysis doc)
- [x] `generate_answers.py` RATE_LIMITED / quota_exhausted / counter — DONE (pre-existing)
- [ ] **Verify all 7 questionable arXiv citations** (PCR 2511.18313, 2603.14045, …) — USER/web
- [ ] Reconcile §7.6.3 (reports EM its footnote invalidates) — paper pass §4.5

**Correctness**
- [x] Cost-benefit greedy at line 10c + partial enumeration (Thm 3 applies to shipped code) — DONE
- [x] Teleportation-off during coverage measurement — DONE
- [ ] W_dom PCA well-posedness + φ_dom independent ablation — §2.4
- [ ] Sufficiency-check effect audit (8.5% NLI agreement) — eval design §6
- [ ] φ_imp PageRank `max_u=min_u → 0.5` edge case on small graphs — unit test
- [ ] Unify EM context path — §2.1
- [ ] Replace σ with conformal / bottleneck bound (D5) — §2.3

**Theory**
- [x] D1 depth-gain degradation bound — DONE
- [x] D3 correlated MRF coverage — DONE
- [x] D6 multi-anchor theorem — DONE
- [x] D7 hierarchical partition matroid — DONE
- [ ] Prop. 1 explicit (construction, F values, limit, hardness) — §3.6
- [ ] Tree-DP exact optimum + true ratio — §3.7

**Experiments (design now, run in §6)**
- [ ] D2 connectivity-vs-shuffle at matched recall — the missing justification
- [ ] Controlled SOTA (matched encoder + generator) — §5.2 gate
- [ ] Passage + hybrid on MuSiQue
- [ ] φ_dom independent ablation
- [ ] Conformal/learned σ vs product/geometric/bottleneck
- [ ] Encoder sweep: MiniLM vs BGE-large vs E5-large

**Paper**
- [ ] Reposition abstract/§1 (parity + certificate + structure)
- [ ] Demote five-facet to generalization + honest negative ablation
- [ ] Teleportation → appendix/cut; remove forest corollary from main text
- [ ] Extend §7.6.10 to EM discrepancy
- [ ] Audit five "first" claims
- [ ] Lead §7 with passage; sentence → granularity ablation

---

## §8. WHAT CHANGED vs v2 (for the record)

| v2 item | v3 disposition |
|---|---|
| §1 cleaning (delete 4 corrupted JSONs) | **Amended** — move to `results/quarantine/` (tracked), don't delete; keep the failure artifact (inspection §F). `.gitignore` already hardened. |
| §2.1 rate-limit guard | Already implemented; promoted to fix for EM discrepancy too |
| §2.5 teleportation flags | **Replaced by D6 multi-anchor** — don't invest in a mechanism that helps 0 queries and breaks the connectivity story |
| §3 theory order (tightness→knapsack→corr→hybrid) | **Reordered** — D1 (FC) first, tightness last (inspection §F) |
| §5 SOTA adapters | **+ fairness gate** (matched encoder + generator) |
| §6 evals | **+ pre-flight gate** (ratio ≤ 1.0 must hold first) |
| — (not in v2) | **+ D2 connectivity-vs-shuffle** — the missing justification experiment |
| — (not in v2) | **+ D1/D3/D6/D7 theory** — implemented, tested (100/100), documented |

---

## Verification (safe, seconds)
```cmd
python -m pytest pathfinder\tests -q --rootdir=.            :: 100/100 PASS
python -c "import ast,glob;[ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('experiments/*.py')]"
```
