# PATHFINDER Preliminary Experimental Analysis

**Source:** DeepSeek CLI run on `yash-nit/dc` fork — 100-query fast run (HotpotQA distractor, validation split)
**Date:** 2026-07-21
**Status:** Preliminary — 100 queries only. Full run (7,405 queries) pending.

---

## Results Summary (100-query fast run)

| System | Recall@5 |
|---|---|
| NaiveRAG (top-k cosine) | 0.36 |
| BFS 2-hop | ~0.28 (est.) |
| Spreading Activation | ~0.25 (est.) |
| PATHFINDER (after σ bug fix) | 0.21 |
| PATHFINDER (original — σ collapse bug) | 0.09 |

### Weight Ablation
| Variant | Recall@5 |
|---|---|
| semantic_only (α=1.0) | 0.30 |
| full_default (α=0.50, β=0.15, γ=0.15, δ=0.10, ε=0.10) | 0.21 |

---

## Root Cause Analysis

### Bug 1 (CONFIRMED, FIXED): PHI_CONF_INIT = 0.70 caused σ collapse

The original code set `PHI_CONF_INIT = 0.70` for all nodes. With path-product σ:

```
σ(v0→v) = Π φ_conf(u) × Π W(e)
```

At depth 5 with mean edge weight 0.50:
```
σ ≈ 0.70^5 × 0.50^4 = 0.168 × 0.0625 ≈ 0.01
```

This is far below `τ_low = 0.30`, triggering re-traversal on **every query** from the first hop. The re-traversal loop ran 3 retries per query, distorting weights toward ε (confidence) — which was also constant at 0.70 across all nodes. Selection degraded to near-random within the graph frontier.

**Fix applied:** `PHI_CONF_INIT = 0.95`. This pushes first-collapse depth from ~2 nodes to ~5 nodes for semantic edges, allowing meaningful traversal before re-traversal fires.

**Note:** Even with 0.95, path-product σ still collapses exponentially for paths of length 7+ (0.95^7 × 0.50^6 ≈ 0.70 × 0.016 = 0.011). The path-product formula may require either geometric mean normalization or path-length scaling to be practically calibrated. See open issue below.

---

### Finding 1: PATHFINDER trails NaiveRAG by 15 pp (0.21 vs 0.36)

**Statistical reliability:** Moderate. At n=100 with binary Recall@5:
- PATHFINDER 95% CI: [0.13, 0.29]
- NaiveRAG 95% CI: [0.27, 0.45]
- Two-proportion z-test: z ≈ 2.6, p ≈ 0.009

The gap is statistically significant at this sample size and mechanically explained by the PHI_CONF_INIT bug distorting weight selection. With the bug fixed, the gap should narrow. Full 7,405-query run needed to confirm magnitude.

**Structural explanation:** The graph-connectivity constraint (PATHFINDER selects a connected subtree from v0) can exclude gold nodes that are not reachable from v0 via a high-gain path. NaiveRAG has no such constraint — it returns any top-k nodes regardless of graph structure. For HotpotQA two-hop queries where the second supporting fact is in a different document cluster, PATHFINDER's connectivity constraint may systematically miss it if the inter-cluster edge is weak.

---

### Finding 2: σ calibration shows negative (insignificant) correlation with EM

**Reliability: LOW — this is almost certainly a bug artifact, not a real finding.**

With the PHI_CONF_INIT=0.70 bug active, σ was near-zero for all queries. The `best_result` selection in the retry loop chose the pass with highest σ — which corresponds to the shallowest traversal (fewer path-product multiplications). Shallow traversals biased toward v0-proximity achieve high σ but miss the second hop in two-hop queries. This inverts the expected correlation.

**Action:** Re-run with PHI_CONF_INIT=0.95 on the full 7,405-query set before reporting σ calibration. Do not report the negative ρ as a finding about σ's validity.

---

### Finding 3: semantic_only (0.30) outperforms full multidimensional (0.21)

**Reliability: MEDIUM — directionally plausible, borderline significant at n=100.**

Three contributing factors:
1. **phi_temp = 1.0 for all nodes** — HotpotQA has no timestamps. The temporal facet is a constant and adds noise to the marginal gain ranking without contributing signal.
2. **phi_conf = 0.70 for all nodes** — Uniform confidence means the confidence facet also adds a constant, diluting the semantic coverage weight α from 0.50 to an effective lower value while contributing no discrimination.
3. **phi_imp (PageRank)** — on small per-query graphs (~48 nodes average), PageRank rewards structurally central nodes which may not contain the answer-supporting facts.

**Implication for paper:** The (1−1/e) guarantee applies strictly to the semantic coverage term f(S,q), which is submodular. The additive terms (β·φ_temp + γ·φ_imp + δ·φ_dom + ε·φ_conf) in F(S,q) are modular (linear), not submodular. The full F(S,q) is not guaranteed to be submodular, and the greedy selection on F may not inherit the (1−1/e) bound. This requires a clarification in §4.1 — either restrict the guarantee claim to f(S,q) coverage, or prove that the weighted combination remains submodular.

---

## Open Issues Raised by This Run

### ISSUE-A: Path-product σ still decays exponentially with PHI_CONF_INIT=0.95
Even with the fix, paths of depth 7+ collapse below τ_low. Consider:
- Geometric mean: `σ_geom(path) = (Π φ_conf × Π W(e))^(1/path_length)`
- Per-hop normalized: `σ_hop = φ_conf(v) × W(e)` averaged over the path

The geometric mean preserves the structural coherence semantics without exponential collapse.

### ISSUE-B: The (1−1/e) guarantee scope needs clarification in §4.1
F(S,q) = α·f(S,q) + Σ[β·φ_temp + γ·φ_imp + δ·φ_dom + ε·φ_conf] is not fully submodular if the additive terms are modular. The guarantee applies to f(S,q). Either:
(a) State the guarantee for f(S,q) specifically and note F is the selection objective, or
(b) Prove F is submodular (requires φ_imp and φ_dom to be submodular, which is not obvious).

### ISSUE-C: phi_temp = constant on HotpotQA
HotpotQA has no timestamps. The temporal facet should be zero-weighted (β=0) for this benchmark, or replaced with a document-recency proxy. Weight ablation should include a β=0 variant.

---

## Nodes/Query Before and After Fix

| Condition | Mean nodes/query |
|---|---|
| PHI_CONF_INIT=0.70 (original) | 2.23 |
| PHI_CONF_INIT=0.95 (fixed) | 7.18 |

The 3× increase in selected nodes directly explains most of the Recall@5 improvement (0.09 → 0.21). The system was truncating traversal catastrophically early.

---

## Next Steps

1. **Run full 7,405-query evaluation** with PHI_CONF_INIT=0.95 and current path-product σ
2. **Report σ calibration** — expect positive ρ(σ, EM) with bug fixed; if still negative, implement geometric mean σ
3. **Add β=0 ablation variant** to account for HotpotQA's missing timestamps
4. **Clarify §4.1 guarantee scope** — distinguish f(S,q) coverage guarantee from F(S,q) selection objective
5. **Verify semantic_only gap** at n=7,405 — if confirmed, update §8 Limitations with benchmark-specific weight calibration note
