"""
PATHFINDER Experiment — Phase 12, Task 12.3: Latency Profiling
===============================================================
Profile pathfinder execution with fine-grained timing breakdown.

Usage:
    python 20_latency_profiling.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/raw/latency_profiling.json
"""
import sys, json, pickle, argparse, time, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_pathfinder import run_pathfinder, tok_count, delta_full_fast, compute_sigma, K_TOK


def get_gold_nodes(record):
    gold = set()
    nodes = record["graph"]["nodes"]
    sf = record.get("supporting_facts", {})
    if isinstance(sf, dict) and "title" in sf:
        gold_set = set(zip(sf["title"], sf["sent_id"]))
        for i, node in enumerate(nodes):
            if (node["doc_title"], node["sent_idx"]) in gold_set:
                gold.add(i)
    if not gold:
        for i, node in enumerate(nodes):
            if node.get("is_supporting", False):
                gold.add(i)
    return gold


def profiled_traverse(gd, k_tok=K_TOK):
    """Run pathfinder with fine-grained timing."""
    G = gd["G"]
    q_emb = gd["q_emb"]
    q_dom = gd["q_dom"]
    embs = gd["embeddings"]
    N = gd["N"]

    timings = {"entry_selection": 0, "frontier_expansion": 0, "marginal_gain": 0,
               "sigma_computation": 0, "total": 0}

    t_total_start = time.perf_counter()

    if N == 0 or np.linalg.norm(q_emb) < 1e-8:
        timings["total"] = (time.perf_counter() - t_total_start) * 1000
        return [], timings

    # Entry node selection
    t0 = time.perf_counter()
    phi_sem_q = embs @ q_emb
    v0 = int(np.argmax(phi_sem_q))
    timings["entry_selection"] = (time.perf_counter() - t0) * 1000

    if tok_count(v0, G) > k_tok:
        timings["total"] = (time.perf_counter() - t_total_start) * 1000
        return [], timings

    S = [v0]
    S_set = {v0}
    parent = {v0: None}
    tok = tok_count(v0, G)
    product_factor = 1.0 - max(0.0, float(phi_sem_q[v0]))
    weights = (1.0, 0.0, 0.0, 0.0, 0.0)

    t_frontier_start = time.perf_counter()
    frontier = {}
    for u in G.successors(v0):
        if u not in S_set:
            frontier[u] = v0
    timings["frontier_expansion"] = (time.perf_counter() - t_frontier_start) * 1000

    marginal_total = 0
    while frontier and tok < k_tok:
        rem = k_tok - tok
        feasible = {v: p for v, p in frontier.items() if tok_count(v, G) <= rem}
        if not feasible:
            break

        t_mg = time.perf_counter()
        best_v, best_gain = None, -1.0
        for v in feasible:
            g = delta_full_fast(v, G, q_dom, phi_sem_q, weights, product_factor)
            if g > best_gain:
                best_gain, best_v = g, v
        marginal_total += (time.perf_counter() - t_mg) * 1000

        if best_v is None:
            break

        S.append(best_v)
        S_set.add(best_v)
        parent[best_v] = feasible[best_v]
        product_factor *= (1.0 - max(0.0, float(phi_sem_q[best_v])))
        tok += tok_count(best_v, G)

        del frontier[best_v]
        t_fe = time.perf_counter()
        for u in G.successors(best_v):
            if u not in S_set and u not in frontier:
                frontier[u] = best_v
        timings["frontier_expansion"] += (time.perf_counter() - t_fe) * 1000

    timings["marginal_gain"] = marginal_total

    # Sigma computation
    t_sigma = time.perf_counter()
    sigma = compute_sigma(S, parent, G)
    timings["sigma_computation"] = (time.perf_counter() - t_sigma) * 1000

    timings["total"] = (time.perf_counter() - t_total_start) * 1000

    return S, timings


def main():
    parser = argparse.ArgumentParser(description="Latency Profiling")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--output", default="results/raw/latency_profiling.json")
    args = parser.parse_args()

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]

    per_query = []

    for rec in tqdm(records, desc="Latency profiling"):
        gd = rec["graph"]
        S, timings = profiled_traverse(gd)
        per_query.append({
            "n_nodes": gd["N"],
            "n_selected": len(S),
            **{k: round(v, 4) for k, v in timings.items()},
        })

    # Aggregate
    components = ["entry_selection", "frontier_expansion", "marginal_gain", "sigma_computation", "total"]
    summary = {}
    for comp in components:
        vals = [q[comp] for q in per_query]
        summary[comp] = {
            "mean_ms": round(float(np.mean(vals)), 4),
            "p50_ms": round(float(np.percentile(vals, 50)), 4),
            "p95_ms": round(float(np.percentile(vals, 95)), 4),
            "p99_ms": round(float(np.percentile(vals, 99)), 4),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nLatency Profiling (ms):")
    print(f"{'Component':<25} {'Mean':>8} {'p50':>8} {'p95':>8} {'p99':>8}")
    print("-" * 55)
    for comp in components:
        s = summary[comp]
        print(f"{comp:<25} {s['mean_ms']:>8.3f} {s['p50_ms']:>8.3f} {s['p95_ms']:>8.3f} {s['p99_ms']:>8.3f}")
    total = summary["total"]["mean_ms"]
    print(f"\nComponent breakdown (% of total):")
    for comp in components[:-1]:
        pct = summary[comp]["mean_ms"] / total * 100 if total > 0 else 0
        print(f"  {comp}: {pct:.1f}%")


if __name__ == "__main__":
    main()
