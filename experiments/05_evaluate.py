"""
PATHFINDER Experiment — Step 5: Full Evaluation Pipeline
=========================================================
Runs all systems, computes all 8 metrics from §7.5, prints a results table,
and saves per-query results + summary to results/results.json.

Metrics implemented:
  §7.5.1  Recall@5
  §7.5.2  EM / F1
  §7.5.3  Node Expansion Rate (NER)
  §7.5.4  Latency (traversal only, p50 / p95)
  §7.5.5  σ Calibration (Spearman ρ, ECE, bucket analysis, three-tier validation)
  §7.5.7  Anchor quality rank (median rank, top-1/3/5 %)
  §7.5.8  Weight ablation (5 variants)
  §7.5.9  Coverage ratio vs S*_frontier on small graphs

Usage:
    export GROQ_API_KEY=your_key_here
    python 05_evaluate.py --graphs data/hotpotqa_graphs.pkl --max_samples 100

For a fast smoke test (no LLM):
    python 05_evaluate.py --graphs data/hotpotqa_graphs.pkl --max_samples 20 --no_llm
"""

import os
import sys
import json
import time
import pickle
import argparse
import itertools
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
from run_pathfinder  import (run_pathfinder, compute_F, TraversalResult,
                              ALPHA, BETA, GAMMA, DELTA, EPSILON, K_TOK)
from run_baselines   import naive_rag, bfs_2hop, spreading_activation
from generate_answers import generate_answer, exact_match, f1_score


# ── Gold node identification ───────────────────────────────────────────────────
def get_gold_nodes(record: dict) -> list[int]:
    """
    Map HotpotQA supporting_facts {title, sent_id} to node indices in graph.
    Returns list of matching node indices (may be 0 if sentence not found).
    """
    sf_titles   = record["supporting_facts"]["title"]
    sf_sent_ids = record["supporting_facts"]["sent_id"]
    gold_set    = set(zip(sf_titles, sf_sent_ids))

    gold_idxs = []
    for i, node in enumerate(record["graph"]["nodes"]):
        if (node["doc_title"], node["sent_idx"]) in gold_set:
            gold_idxs.append(i)
    return gold_idxs


# ── §7.5.1 Recall@k ───────────────────────────────────────────────────────────
def recall_at_k(retrieved: list[int], gold: list[int], k: int = 5) -> int:
    """1 if all gold nodes appear in retrieved[:k], else 0."""
    if not gold:
        return 1
    return int(all(g in set(retrieved[:k]) for g in gold))


# ── §7.5.5 σ Calibration helpers ──────────────────────────────────────────────
SIGMA_BUCKETS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]

def sigma_calibration(sigmas: list[float], ems: list[int]) -> dict:
    """
    Compute:
      - Spearman ρ between σ and EM
      - Bucket analysis (mean EM per σ bucket)
      - ECE (Expected Calibration Error)
      - Three-tier validation: EM(proceed) vs EM(hedge) vs EM(re-traverse)
    """
    if len(sigmas) < 5:
        return {}

    rho, pval = spearmanr(sigmas, ems)

    bucket_stats = {}
    for lo, hi in SIGMA_BUCKETS:
        pairs = [(s, e) for s, e in zip(sigmas, ems) if lo <= s < hi]
        if pairs:
            s_vals, e_vals = zip(*pairs)
            mean_em   = float(np.mean(e_vals))
            mean_conf = float(np.mean(s_vals))
            bucket_stats[f"[{lo:.1f},{hi:.1f})"] = {
                "n":        len(pairs),
                "mean_em":  round(mean_em, 4),
                "mean_sigma": round(mean_conf, 4),
                "cal_error":  round(abs(mean_em - mean_conf), 4),
            }

    # ECE
    ece = sum(
        (v["n"] / len(sigmas)) * v["cal_error"]
        for v in bucket_stats.values()
    )

    # Three-tier: proceed (≥0.5), hedge ([0.3,0.5)), re-traverse (<0.3)
    tier_results: dict[str, list[int]] = {"proceed": [], "hedge": [], "retrav": []}
    for s, e in zip(sigmas, ems):
        if s >= 0.5:
            tier_results["proceed"].append(e)
        elif s >= 0.3:
            tier_results["hedge"].append(e)
        else:
            tier_results["retrav"].append(e)

    tier_em = {
        tier: round(float(np.mean(vals)), 4) if vals else None
        for tier, vals in tier_results.items()
    }

    return {
        "spearman_rho":   round(float(rho), 4),
        "spearman_pval":  round(float(pval), 6),
        "ece":            round(float(ece), 4),
        "bucket_analysis": bucket_stats,
        "tier_em":        tier_em,
    }


# ── §7.5.7 Anchor quality rank ─────────────────────────────────────────────────
def anchor_quality(records: list[dict], gold_nodes_list: list[list[int]]) -> dict:
    """
    For each query, find rank of nearest gold node in cosine similarity ordering.
    """
    ranks = []
    for rec, gold_nodes in zip(records, gold_nodes_list):
        gd = rec["graph"]
        if not gold_nodes or gd is None:
            continue
        phi_sem_q = gd["embeddings"] @ gd["q_emb"]
        ranked = np.argsort(phi_sem_q)[::-1].tolist()
        # Take the best-ranked gold node
        best_rank = min(
            (ranked.index(g) + 1 for g in gold_nodes if g in ranked),
            default=None
        )
        if best_rank is not None:
            ranks.append(best_rank)

    if not ranks:
        return {}
    return {
        "n":          len(ranks),
        "median":     float(np.median(ranks)),
        "mean":       round(float(np.mean(ranks)), 2),
        "top1_pct":   round(float(np.mean([r == 1 for r in ranks])), 4),
        "top3_pct":   round(float(np.mean([r <= 3 for r in ranks])), 4),
        "top5_pct":   round(float(np.mean([r <= 5 for r in ranks])), 4),
    }


# ── §7.5.8 Weight ablation ─────────────────────────────────────────────────────
ABLATION_VARIANTS = {
    "semantic_only":  (1.00, 0.00, 0.00, 0.00, 0.00),
    "plus_temporal":  (0.77, 0.23, 0.00, 0.00, 0.00),
    "plus_structural":(0.63, 0.18, 0.19, 0.00, 0.00),
    "plus_domain":    (0.56, 0.16, 0.17, 0.11, 0.00),
    "full_default":   (ALPHA, BETA, GAMMA, DELTA, EPSILON),
}

def weight_ablation(records: list[dict], gold_nodes_list: list[list[int]]) -> dict:
    """Run PATHFINDER with each ablation weight set; report Recall@5."""
    results = {}
    for name, w in ABLATION_VARIANTS.items():
        r5_list = []
        for rec, gold_nodes in zip(records, gold_nodes_list):
            gd  = rec["graph"]
            res = run_pathfinder(gd, weights=w)
            r5_list.append(recall_at_k(res.S, gold_nodes, k=5))
        results[name] = {
            "weights":   w,
            "recall@5":  round(float(np.mean(r5_list)), 4),
            "n":         len(r5_list),
        }
        print(f"    Ablation [{name}]: Recall@5 = {results[name]['recall@5']:.4f}")
    return results


# ── §7.5.9 Coverage ratio vs S*_frontier ──────────────────────────────────────
def _enumerate_subtrees(G: nx.DiGraph, v0: int, k: int,
                        max_nodes: int = 25) -> list[list[int]]:
    """
    Enumerate all connected subtrees of size ≤ k rooted at v0 via DFS.
    Only feasible for |V| ≤ max_nodes.  Returns list of node lists.
    """
    subtrees: list[list[int]] = []

    def dfs(current_tree: list[int], frontier_set: set[int]):
        subtrees.append(list(current_tree))
        if len(current_tree) >= k:
            return
        for v in sorted(frontier_set):  # deterministic order
            new_tree = current_tree + [v]
            # New frontier: all successors of v not already in tree or frontier
            new_frontier = frontier_set - {v}
            for u in G.successors(v):
                if u not in set(new_tree) and u not in new_frontier:
                    new_frontier.add(u)
            dfs(new_tree, new_frontier)

    init_frontier = {u for u in G.successors(v0)}
    dfs([v0], init_frontier)
    return subtrees


def coverage_ratio(records: list[dict], max_samples: int = 200,
                   max_v: int = 25, k: int = 5) -> dict:
    """
    For graphs with |V| ≤ max_v: enumerate S*_frontier, compare to PATHFINDER.
    Returns ratio distribution statistics.
    """
    ratios = []
    fc_violations = 0   # instances where ratio < (1-1/e)

    eligible = [r for r in records if r["graph"]["N"] <= max_v][:max_samples]
    if not eligible:
        return {"n": 0, "note": f"No graphs with |V| ≤ {max_v}"}

    for rec in tqdm(eligible, desc="Coverage ratio (exhaustive)"):
        gd = rec["graph"]
        G  = gd["G"]
        N  = gd["N"]
        phi_sem_q = gd["embeddings"] @ gd["q_emb"]
        v0 = int(np.argmax(phi_sem_q))

        # Enumerate all connected subtrees of size ≤ k
        subtrees = _enumerate_subtrees(G, v0, k, max_nodes=max_v)

        # Compute F for each subtree
        w = (ALPHA, BETA, GAMMA, DELTA, EPSILON)
        best_F = 0.0
        for st in subtrees:
            F_st = compute_F(st, G, gd["q_dom"], phi_sem_q, w)
            if F_st > best_F:
                best_F = F_st

        # PATHFINDER F
        pf_res = run_pathfinder(gd, k_tok=k * 50)   # token budget ~ k nodes
        pf_F   = pf_res.F

        if best_F > 1e-6:
            ratio = pf_F / best_F
            ratios.append(ratio)
            if ratio < (1 - 1 / np.e) - 0.01:   # small tolerance
                fc_violations += 1

    if not ratios:
        return {"n": 0}
    return {
        "n":                   len(ratios),
        "mean_ratio":          round(float(np.mean(ratios)), 4),
        "min_ratio":           round(float(np.min(ratios)), 4),
        "median_ratio":        round(float(np.median(ratios)), 4),
        "pct_above_632":       round(float(np.mean([r >= 0.632 for r in ratios])), 4),
        "pct_above_80":        round(float(np.mean([r >= 0.80 for r in ratios])), 4),
        "pct_above_90":        round(float(np.mean([r >= 0.90 for r in ratios])), 4),
        "fc_violations":       fc_violations,       # ratio < (1-1/e): Condition FC may have failed
        "fc_violation_pct":    round(fc_violations / len(ratios), 4),
    }


# ── Main evaluation loop ───────────────────────────────────────────────────────
def run_evaluation(records: list[dict],
                   groq_api_key: str = None,
                   output_path: str = "results/results.json",
                   run_ablation: bool = True,
                   run_coverage_ratio: bool = True) -> dict:

    Path("results").mkdir(exist_ok=True)
    n = len(records)
    print(f"\n{'='*60}")
    print(f"PATHFINDER Evaluation  —  {n} queries")
    print(f"{'='*60}\n")

    # Pre-compute gold nodes for all records
    gold_nodes_list = [get_gold_nodes(r) for r in records]

    # ── Per-query results ──────────────────────────────────────────────────────
    per_query = []
    latencies = []
    groq_client = None
    if groq_api_key:
        from groq import Groq
        groq_client = Groq(api_key=groq_api_key)

    for rec, gold_nodes in tqdm(zip(records, gold_nodes_list),
                                total=n, desc="Main loop"):
        gd = rec["graph"]
        q  = rec["question"]
        ga = rec["answer"]

        # ── PATHFINDER ──────────────────────────────────────────────────────
        t0  = time.perf_counter()
        pf  = run_pathfinder(gd)
        lat = time.perf_counter() - t0
        latencies.append(lat)

        r5_pf = recall_at_k(pf.S, gold_nodes, k=5)

        em_pf, f1_pf, pred_pf = 0, 0.0, ""
        if groq_client:
            pred_pf = generate_answer(q, pf.S, gd["G"], groq_client)
            em_pf   = exact_match(pred_pf, ga)
            f1_pf   = f1_score(pred_pf, ga)

        # ── Baselines ────────────────────────────────────────────────────────
        naive_nodes = naive_rag(gd, k=5)
        bfs_nodes   = bfs_2hop(gd)
        sa_nodes    = spreading_activation(gd)

        r5_naive = recall_at_k(naive_nodes, gold_nodes, k=5)
        r5_bfs   = recall_at_k(bfs_nodes,   gold_nodes, k=5)
        r5_sa    = recall_at_k(sa_nodes,     gold_nodes, k=5)

        per_query.append({
            "id":       rec["id"],
            "question": q,
            "gold_answer": ga,
            "gold_nodes":  gold_nodes,
            "pathfinder": {
                "S":         pf.S,
                "F":         round(pf.F, 6),
                "sigma":     round(pf.sigma, 6),
                "flag":      pf.confidence_flag,
                "retries":   pf.retries,
                "n_nodes":   len(pf.S),
                "recall@5":  r5_pf,
                "em":        em_pf,
                "f1":        round(f1_pf, 4),
                "prediction":pred_pf,
                "latency_s": round(lat, 5),
            },
            "naive_rag":            {"S": naive_nodes, "recall@5": r5_naive},
            "bfs_2hop":             {"S": bfs_nodes,   "recall@5": r5_bfs},
            "spreading_activation": {"S": sa_nodes,    "recall@5": r5_sa},
        })

    # ── Aggregate main metrics ─────────────────────────────────────────────────
    pf_r5    = float(np.mean([q["pathfinder"]["recall@5"] for q in per_query]))
    naive_r5 = float(np.mean([q["naive_rag"]["recall@5"]  for q in per_query]))
    bfs_r5   = float(np.mean([q["bfs_2hop"]["recall@5"]   for q in per_query]))
    sa_r5    = float(np.mean([q["spreading_activation"]["recall@5"] for q in per_query]))

    pf_em  = float(np.mean([q["pathfinder"]["em"]  for q in per_query]))
    pf_f1  = float(np.mean([q["pathfinder"]["f1"]  for q in per_query]))
    pf_ner = float(np.mean([q["pathfinder"]["n_nodes"] for q in per_query]))

    lat_arr = np.array(latencies)
    lat_p50 = float(np.percentile(lat_arr, 50))
    lat_p95 = float(np.percentile(lat_arr, 95))

    # ── §7.5.5 σ Calibration ────────────────────────────────────────────────
    sigmas = [q["pathfinder"]["sigma"] for q in per_query]
    ems    = [q["pathfinder"]["em"]    for q in per_query]
    sig_cal = sigma_calibration(sigmas, ems) if groq_client else {"note": "skipped — no LLM"}

    # ── §7.5.7 Anchor quality ─────────────────────────────────────────────────
    anch = anchor_quality(records, gold_nodes_list)

    # ── §7.5.8 Weight ablation ────────────────────────────────────────────────
    print("\n§7.5.8 Weight ablation...")
    abl = weight_ablation(records, gold_nodes_list) if run_ablation else {}

    # ── §7.5.9 Coverage ratio ─────────────────────────────────────────────────
    print("\n§7.5.9 Coverage ratio (small graphs)...")
    cov = coverage_ratio(records) if run_coverage_ratio else {}

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "n_queries": n,
        "recall@5": {
            "pathfinder":          round(pf_r5,    4),
            "naive_rag":           round(naive_r5, 4),
            "bfs_2hop":            round(bfs_r5,   4),
            "spreading_activation":round(sa_r5,    4),
        },
        "em_f1": {
            "em":  round(pf_em, 4),
            "f1":  round(pf_f1, 4),
        },
        "node_expansion_rate": {
            "mean": round(pf_ner, 2),
        },
        "latency_seconds": {
            "p50": round(lat_p50, 5),
            "p95": round(lat_p95, 5),
        },
        "sigma_calibration":  sig_cal,
        "anchor_quality":     anch,
        "weight_ablation":    abl,
        "coverage_ratio":     cov,
    }

    # ── Print results table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS TABLE")
    print(f"{'='*60}")
    print(f"{'System':<25} {'Recall@5':>10}")
    print("-" * 37)
    for sys_name, r5 in summary["recall@5"].items():
        marker = " <-" if sys_name == "pathfinder" else ""
        print(f"{'  '+sys_name:<25} {r5:>10.4f}{marker}")
    if groq_client:
        print(f"\nPATHFINDER EM={pf_em:.4f}  F1={pf_f1:.4f}")
    print(f"Node Expansion Rate  : {pf_ner:.2f} nodes/query")
    print(f"Latency p50 / p95    : {lat_p50*1000:.2f} ms / {lat_p95*1000:.2f} ms")

    if sig_cal.get("spearman_rho") is not None:
        print(f"\nσ Calibration:")
        print(f"  Spearman ρ  = {sig_cal['spearman_rho']:.4f}  (p={sig_cal['spearman_pval']:.4f})")
        print(f"  ECE         = {sig_cal['ece']:.4f}")
        te = sig_cal.get("tier_em", {})
        print(f"  Proceed EM  = {te.get('proceed', 'n/a')}  "
              f"Hedge EM = {te.get('hedge', 'n/a')}  "
              f"Re-trav EM = {te.get('retrav', 'n/a')}")

    if anch:
        print(f"\nAnchor Quality:")
        print(f"  Median rank = {anch['median']:.1f}  "
              f"Top-1 = {anch['top1_pct']:.2%}  "
              f"Top-3 = {anch['top3_pct']:.2%}  "
              f"Top-5 = {anch['top5_pct']:.2%}")

    if cov:
        print(f"\nCoverage Ratio (n={cov.get('n', 0)}):")
        print(f"  Mean ratio  = {cov.get('mean_ratio', 'n/a')}")
        pct_632 = float(cov.get('pct_above_632', 0.0))
        pct_80  = float(cov.get('pct_above_80',  0.0))
        print(f"  ≥ 63.2%     = {pct_632:.2%}  "
              f"  ≥ 80%  = {pct_80:.2%}")
        print(f"  FC violations = {cov.get('fc_violations', 0)} "
              f"({cov.get('fc_violation_pct', 0):.2%})")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {"summary": summary, "per_query": per_query}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved → {output_path}")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full PATHFINDER evaluation")
    parser.add_argument("--graphs",       default="data/hotpotqa_graphs.pkl")
    parser.add_argument("--max_samples",  type=int, default=None,
                        help="Limit evaluation to N queries (None = all)")
    parser.add_argument("--output",       default="results/results.json")
    parser.add_argument("--no_llm",       action="store_true",
                        help="Skip LLM generation (Recall@5 only)")
    parser.add_argument("--no_ablation",  action="store_true")
    parser.add_argument("--no_coverage",  action="store_true")
    args = parser.parse_args()

    # Load graphs
    if not Path(args.graphs).exists():
        print(f"Graph cache not found: {args.graphs}")
        print("Run:  python 01_build_kg.py  first")
        sys.exit(1)

    print(f"Loading graphs from {args.graphs}...")
    with open(args.graphs, "rb") as f:
        records = pickle.load(f)

    if args.max_samples:
        records = records[:args.max_samples]

    groq_key = None if args.no_llm else os.getenv("GROQ_API_KEY")
    if not args.no_llm and not groq_key:
        print("Warning: GROQ_API_KEY not set. Running without LLM (Recall@5 only).")
        groq_key = None

    run_evaluation(
        records,
        groq_api_key=groq_key,
        output_path=args.output,
        run_ablation=not args.no_ablation,
        run_coverage_ratio=not args.no_coverage,
    )
