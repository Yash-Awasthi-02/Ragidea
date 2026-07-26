# PATHFINDER — Master Plan (v2)

> **Project**: PATHFINDER — Submodular Coverage Maximization over Multidimensional Knowledge Graphs for Multi-Hop RAG
> **Repo**: `https://github.com/Yash-Awasthi-02/Ragidea`
> **Last Updated**: 2026-07-26
> **Supersedes**: previous PLAN.md optimization pathway

---

## §0. CRITICAL OPERATING RULES — READ FIRST

These rules override everything else in this document.

| # | Rule | Reason |
|---|------|--------|
| 1 | **NEVER run evaluations.** Evals take **9–18 hours**. They are the **ABSOLUTE LAST STEP** (§6), executed manually by the user. | Time / quota cost |
| 2 | **NEVER run** `experiments/05*.py`, `15*.py`–`19*.py`, `25*.py`, `generate_answers.py`, or anything touching a dataset or the Groq API. | These are evals |
| 3 | **NEVER push, open PRs, or rewrite git history.** Git PAT is provided by the user only when explicitly needed. | Repo safety |
| 4 | When validation is required: state the **exact command**, the **expected output**, then **STOP and wait** for the user to run it and paste results. | Division of labor |
| 5 | Everything that does **NOT** require an eval — cleaning, code fixes, theory, proofs, paper writing, refactors, test authoring — is done **FIRST** (§1–§5). | Maximize progress |
| 6 | Before any code change: explain **WHY**, the **expected impact**, and the **theoretical implication** if applicable. | Scientific rigor |

**Safe-to-run commands (seconds, no API, no dataset):**
```cmd
pytest pathfinder/tests/            :: 47 unit tests, ~5s
python -c "import ast,glob;[ast.parse(open(f).read()) for f in glob.glob('experiments/*.py')]"   :: syntax check
```

---

## §1. PHASE 0 — REPO CLEANING (DO FIRST)

> Goal: remove corrupted/stale artifacts so no future analysis reads poisoned data. All actions are file deletions / `.gitignore` edits — **no evals**.

### 1.1 Corrupted result files (silent Groq rate-limit → EM=0, predictions="")
Root cause: `generate_answers.py::generate_answer()` returns `""` after retries exhaust and callers record it as a genuine wrong answer. Four files were produced after the daily quota was already consumed and are **scientifically invalid** (their R@5 is fine, but EM/F1 are garbage).

- [ ] Delete `results/raw/heterogeneous_llm_8b.json` (EM=0.0, F1=0.0, all predictions empty)
- [ ] Delete `results/raw/heterogeneous_llm_70b_passage.json` (EM=0.0, F1=0.0, predictions empty)
- [ ] Delete `results/raw/heterogeneous_llm_70b_v2.json` (EM=0.02, F1=0.02, near-total failure)
- [ ] Delete `results/raw/rerank_passage.json` (pathfinder EM=0.08 / rerank EM=0.12 — partial rate-limit contamination)

> **Do NOT delete** the R@5-only files — they don't use the LLM and are valid. Only the 4 above embed corrupted EM/F1. They will be **regenerated** in §6.2 with a fresh key.

### 1.2 Stale Phase-1 JSONs (superseded by full runs)
- [ ] Delete `results/raw/anchor_quality_200.json`
- [ ] Delete `results/raw/hotpotqa_200_*.json` (all)
- [ ] Delete `results/raw/root_cause_analysis.json`
- [ ] Delete `results/raw/sigma_calibration_200.json`
- [ ] Delete `results/raw/coverage_ratio_*.json` (all)

### 1.3 `.gitignore` hardening
Current `.gitignore` already excludes `data/` and `__pycache__/`. Add result-dir ignores so future eval dumps never get committed accidentally:
```gitignore
# --- add below existing entries ---
results/raw/
*.log
```
> Keep the currently-tracked valid JSONs; the ignore only prevents *new* raw files from being staged. (If git still tracks existing ones, that is fine — we do not rewrite history.)

### 1.4 Documentation sync
- [ ] Update `experiments/README.md` — mark the 4 deleted files as "regenerated in Phase 5" and add a one-line warning about the rate-limit EM=0 failure mode.
- [ ] Verify cleaning didn't break anything:
```cmd
pytest pathfinder/tests/    :: expect 47/47 PASS
```

---

## §2. PHASE 1 — CODE FIXES & FEATURES (no evals)

> Each task lists WHY → expected impact → theoretical implication → files touched.

### 2.1 Fix the silent rate-limit bug (HIGHEST PRIORITY — blocks all LLM evals)
- **WHY**: 4 result files were silently zeroed. A reviewer reproducing our pipeline could hit the same corruption and conclude the method fails. Reproducibility is a stated core objective.
- **Change** in `experiments/generate_answers.py`:
  - `generate_answer()` returns a sentinel `RATE_LIMITED` (or raises `RateLimitExhausted`) instead of `""` when all retries fail due to 429/quota.
  - `evaluate_batch()` tracks `n_rate_limited`; if `> 0`, it prints a loud warning and writes `"quota_exhausted": true` + `n_rate_limited` into the output JSON summary.
  - Add a per-query API-call counter and an estimated-remaining-quota print every 25 queries.
- **Expected impact**: No future silent corruption; corrupted runs become detectable immediately.
- **Theoretical implication**: none (infra), but it protects the validity of every downstream empirical claim.

### 2.2 Make LLM reranking a first-class default
- **WHY**: LLM reranking gave the single biggest proven gain (+33.3% R@5, 0.24→0.32). It is currently an isolated script; it should be a toggle in the main pipeline.
- **Change** in `experiments/run_pathfinder.py` and/or `run_pathfinder()` wrapper:
  - Add `rerank: bool = False` and a `rerank_with_llm(S, question, client)` post-processing hook.
  - Add a **semantic cache** (cosine ≥ `CACHE_THETA=0.92` on query embeddings) so repeated/near-duplicate queries don't burn API calls.
  - Reuse the §2.1 rate-limit guard.
- **Expected impact**: R@5 0.24→0.32 by default; higher with better graphs (Phase A).
- **Theoretical implication**: reranking is a monotone post-selection on the *retrieved set* — it does **not** change the (1−1/e) coverage guarantee, which holds at retrieval time. State this explicitly in the paper to preempt a reviewer conflating the two stages.

### 2.3 Passage-level as the default node granularity
- **WHY**: Passage nodes give R@5=0.644 vs sentence 0.268, and the one successful passage LLM run (EM=0.106) beat full-scale sentence EM (0.0068). Sentence graphs are too sparse/disconnected.
- **Change**:
  - Promote `01b_build_kg_passage.py` to the default builder path in docs and eval scripts (`--node_granularity passage`).
  - Ensure supporting-fact → passage mapping (any gold sentence ⇒ passage is gold) is the canonical labeling.
- **Expected impact**: Primary metric jumps ~2.4×. This becomes the headline retrieval number.
- **Theoretical implication**: larger nodes ⇒ denser graph ⇒ the frontier constraint excludes fewer relevant nodes, so the connected-subtree feasible set is closer to the unconstrained optimum. Worth a short remark: the (1−1/e) bound is unchanged, but the *gap to Naive RAG* narrows empirically.

### 2.4 Dense-Anchor Hybrid as the default retrieval mode
- **WHY**: Dense-Anchor Hybrid already **matches** Naive RAG at passage level (R@5=0.708) while keeping graph-coherent subtrees. This is the strongest honest result we have.
- **Change**:
  - Promote `run_pathfinder_hybrid()` (dense top-3 anchors bypass frontier → greedy expansion fills budget → merge/dedupe) into `run_pathfinder.py` as `mode="hybrid"`.
- **Expected impact**: Neutralizes the "Naive RAG beats you" reviewer objection on the primary benchmark.
- **Theoretical implication**: anchors are injected *outside* the connected frontier; the (1−1/e) guarantee then applies to the **expansion phase per anchor**, not the union. Add a corollary clarifying the guarantee is per-connected-component anchored at a dense seed.

### 2.5 Fix / re-scope teleportation
- **WHY**: Teleportation fires on only 9.8% of queries and adds **0** at R@5 (helps only R@10: 0.330→0.350). As-is it is a weak claim reviewers will attack.
- **Change** (two options, implement BOTH behind flags, decide from §6 data):
  - `--always_dense`: add top-5 global dense nodes to the frontier at *every* greedy step (teleportation becomes default, not exception).
  - `--teleport_gate connectivity`: only allow teleportation when the entry node's component is provably disconnected from ≥1 dense top-k node.
- **Expected impact**: closes the R@5 gap on sparse sentence graphs.
- **Theoretical implication**: `--always_dense` weakens the "graph-coherent" story — the paper must frame this as "PATHFINDER-Flat" vs "PATHFINDER-Coherent" and report both, preserving the theoretical contribution on the coherent variant.

### 2.6 Recalibrate NLI sufficiency
- **WHY**: NLI sufficiency agrees with the σ-heuristic only 8.5% of the time (heuristic says sufficient 96%, NLI says 5.5%). Either the NLI threshold is miscalibrated or the lexical fallback is broken.
- **Change** in `experiments/08_nli_sufficiency.py`:
  - Sweep the entailment threshold; report agreement vs threshold.
  - Replace the lexical fallback with a calibrated score; log confusion matrix (heuristic × NLI).
- **Expected impact**: a sufficiency signal that actually tracks answerability → fewer wasted re-traversals.
- **Theoretical implication**: sufficiency is orthogonal to the coverage guarantee; it's the *stopping criterion*. A calibrated stopping rule improves the precision/recall trade-off, not the bound.

### 2.7 Embedding model upgrade flag
- **WHY**: `all-MiniLM-L6-v2` (384-d) is weak. BGE/E5 improve both graph edges *and* dense retrieval — lifts every system, PATHFINDER most (edges improve too).
- **Change**: add `--encoder_model` to `01_build_kg.py`, `01b`, `01c`; default remains MiniLM for backward-compat; support `BAAI/bge-large-en-v1.5`, `intfloat/e5-large-v2`.
- **Expected impact**: uniform lift; graph connectivity improves.
- **Theoretical implication**: none to the bound; improves the *constant-factor* empirical coverage.

---

## §3. PHASE 2 — THEORY (pure writing/proofs, no evals)

| Task | Status | Deliverable |
|------|--------|-------------|
| **3.1 Bound tightness (Feige-style)** | TODO | Construct a graph family where greedy achieves exactly (1−1/e); add as Proposition in §5 + unit test |
| **3.2 Heterogeneous token cost** | TODO | Extend remark → full theorem via knapsack-constrained submodular maximization (Sviridenko 2004), giving (1−1/e) for non-uniform `tok_count` |
| **3.3 Joint correlation modeling** | TODO | MRF/GNN coverage model replacing the independence assumption in `f(S,q)=1−Π(1−sim)`; discuss submodularity preservation |
| **3.4 Hybrid per-anchor guarantee** | TODO | Corollary formalizing §2.4's per-component (1−1/e) bound |
| **3.5 Tests** | TODO | Add formal-property tests for each new theorem to `pathfinder/tests/test_formal_properties.py` (keep 47/47 → target 55+) |

---

## §4. PHASE 3 — PAPER REVISION (writing only)

- [ ] **4.1** Lead §7 with **passage-level** results (0.644 / hybrid 0.708) as primary; demote sentence-level to an ablation.
- [ ] **4.2** Rewrite §7.6 with an explicit **data-validity note**: the 4 corrupted LLM runs, their cause (Groq quota), and that they are regenerated in Phase 5. (Transparency is a *strength* to reviewers.)
- [ ] **4.3** Sharpen §2 positioning vs HippoRAG 2, SubgraphRAG, IRCoT — one tight paragraph each on what we guarantee that they cannot.
- [ ] **4.4** Update §8 limitations honestly: teleportation weakness, NLI miscalibration, MuSiQue near-zero, MiniLM dependence.
- [ ] **4.5** Insert new theorems from §3 (tightness, heterogeneous cost, hybrid corollary).
- [ ] **4.6** Add the "retrieval-time vs rerank-time guarantee" clarification (§2.2) to preempt reviewer confusion.

---

## §5. PHASE 4 — SOTA COMPARISON SETUP (code only, NO runs)

> Write adapters + harness now; the actual runs happen in §6. Do **not** clone-and-run — only prepare scripts and docs.

- [ ] **5.1** `experiments/30_sota_adapter.py`: uniform interface that takes any system's (question, retrieved_passages) → (EM, F1, R@5, R@10) using our `exact_match`/`f1_score`.
- [ ] **5.2** Document exact setup/run commands for IRCoT, SubgraphRAG, HippoRAG 2, LightRAG in `results/sota_comparison.md` (skeleton table with "pending Phase 5" cells).
- [ ] **5.3** Fair-comparison protocol: same 500-query subset, same generator (Llama-3.3-70B via Groq), note where SOTA uses GPT-4/stronger embeddings.

---

## §6. PHASE 5 — EVALUATIONS (LAST STEP — USER RUNS, 9–18 hrs)

> **I do not run these.** I provide exact commands + expected output; the user executes and pastes results. Order matters: **quota-heavy LLM runs go FIRST with a fresh key**, then the long non-LLM rebuilds.

### 6.1 Pre-flight (seconds)
```cmd
set GROQ_API_KEY=<fresh_key>
pytest pathfinder/tests/                :: expect all PASS (target 55+)
```

### 6.2 Regenerate the 4 corrupted LLM results (fresh key, small N first)
```cmd
python experiments/19_heterogeneous_llms.py --model llama3-8b-8192 --max_samples 200 --output results/raw/heterogeneous_llm_8b.json
python experiments/19_heterogeneous_llms.py --model llama-3.3-70b-versatile --node_granularity passage --max_samples 50 --output results/raw/heterogeneous_llm_70b_passage.json
python experiments/19_heterogeneous_llms.py --model llama-3.3-70b-versatile --max_samples 50 --output results/raw/heterogeneous_llm_70b_v2.json
python experiments/25_rerank_eval.py --node_granularity passage --max_samples 50 --output results/raw/rerank_passage.json
```
**Expected**: non-zero EM/F1; each JSON summary contains `"quota_exhausted": false`. If `true`, STOP — quota is still exhausted; wait for reset.

### 6.3 Rebuild graphs (passage + BGE + hybrid), non-LLM
```cmd
python experiments/01b_build_kg_passage.py --max_samples 500 --encoder_model BAAI/bge-large-en-v1.5 --output data/hotpotqa_graphs_passage_bge.pkl
python experiments/02_load_2wiki_musique.py --dataset 2wiki --max_samples 500 --output data/2wiki_graphs.pkl
python experiments/02_load_2wiki_musique.py --dataset musique --max_samples 500 --output data/musique_graphs.pkl
```

### 6.4 Core retrieval evals (no LLM)
```cmd
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs_passage_bge.pkl --max_samples 500 --output results/hotpotqa_eval_passage_bge.json
python experiments/23_hybrid_retrieval.py --graphs data/hotpotqa_graphs_passage_bge.pkl --max_samples 500 --output results/raw/hybrid_passage_bge.json
```

### 6.5 Full-scale runs (LONG — the 9–18 hr block)
```cmd
python experiments/01b_build_kg_passage.py --output data/hotpotqa_graphs_full.pkl
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs_full.pkl --output results/hotpotqa_eval_full.json
python experiments/02_load_2wiki_musique.py --dataset 2wiki --output data/2wiki_graphs_full.pkl
python experiments/05_evaluate.py --graphs data/2wiki_graphs_full.pkl --output results/2wiki_eval_full.json
python experiments/02_load_2wiki_musique.py --dataset musique --output data/musique_graphs_full.pkl
python experiments/05_evaluate.py --graphs data/musique_graphs_full.pkl --output results/musique_eval_full.json
```

### 6.6 SOTA baseline runs (using §5 adapters) — fill `results/sota_comparison.md`

### 6.7 Regenerate all figures + consolidated metrics
```cmd
python experiments/print_metrics.py
python results/make_plots.py
```

---

## §7. RUNNING TODO TRACKER

- [ ] §1 cleaning (deletions + .gitignore + README)
- [ ] §2.1 rate-limit guard
- [ ] §2.2 rerank default + cache
- [ ] §2.3 passage default
- [ ] §2.4 hybrid default
- [ ] §2.5 teleportation flags
- [ ] §2.6 NLI recalibration
- [ ] §2.7 encoder flag
- [ ] §3 theorems + tests
- [ ] §4 paper revision
- [ ] §5 SOTA adapters
- [ ] §6 evals (USER, last)

---

## Verification (safe, seconds)
```cmd
pytest pathfinder/tests/
python -c "import ast,glob;[ast.parse(open(f).read()) for f in glob.glob('experiments/*.py')]"
```
