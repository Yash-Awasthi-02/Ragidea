"""
PATHFINDER Experiment — Phase 5, Task 5.2: Teleportation Parameter Sensitivity
===============================================================================
Sweep θ_teleport, TopK, MAX_TELEPORTS to find optimal teleportation parameters.

Usage:
    python 05c_teleportation_sensitivity.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --output results/raw/teleportation_sensitivity.json
"""
import sys, json, pickle, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import run_pathfinder as rp
from run_pathfinder import run_pathfinder


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


def recall_at_k(retrieved, gold, k=5):
    if not gold:
        return 1
    return int(all(g in set(retrieved[:k]) for g in gold))


def main():
    parser = argparse.ArgumentParser(description="Teleportation Sensitivity Sweep")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", default="results/raw/teleportation_sensitivity.json")
    args = parser.parse_args()

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]
    gold_sets = [get_gold_nodes(r) for r in records]

    # Save originals
    orig_theta = rp.THETA_TELEPORT
    orig_topk = rp.TELEPORT_TOPK
    orig_max = rp.MAX_TELEPORTS

    results = []

    # One-at-a-time sweeps
    theta_vals = [0.001, 0.005, 0.01, 0.05, 0.10]
    topk_vals = [3, 5, 10, 15]
    max_teleport_vals = [1, 2, 3, 5, 10]

    sweeps = []
    for v in theta_vals:
        sweeps.append(("theta", v, {v: [("THETA_TELEPORT", v)]}))
    for v in topk_vals:
        sweeps.append(("topk", v, {v: [("TELEPORT_TOPK", v)]}))
    for v in max_teleport_vals:
        sweeps.append(("max_teleports", v, {v: [("MAX_TELEPORTS", v)]}))

    print(f"Sweeping teleportation parameters on {len(records)} queries...")

    for param_name, param_val, _ in tqdm(sweeps, desc="Parameter sweep"):
        # Set parameter
        if param_name == "theta":
            rp.THETA_TELEPORT = param_val
            rp.TELEPORT_TOPK = orig_topk
            rp.MAX_TELEPORTS = orig_max
        elif param_name == "topk":
            rp.THETA_TELEPORT = orig_theta
            rp.TELEPORT_TOPK = param_val
            rp.MAX_TELEPORTS = orig_max
        elif param_name == "max_teleports":
            rp.THETA_TELEPORT = orig_theta
            rp.TELEPORT_TOPK = orig_topk
            rp.MAX_TELEPORTS = param_val

        recalls = []
        for i, rec in enumerate(records):
            try:
                res = run_pathfinder(rec["graph"], enable_teleport=True)
                recalls.append(recall_at_k(res.S, gold_sets[i], 5))
            except Exception:
                recalls.append(0)

        results.append({
            "parameter": param_name,
            "value": param_val,
            "mean_recall_at_5": round(float(np.mean(recalls)), 4),
            "std_recall_at_5": round(float(np.std(recalls)), 4),
        })

    # Restore
    rp.THETA_TELEPORT = orig_theta
    rp.TELEPORT_TOPK = orig_topk
    rp.MAX_TELEPORTS = orig_max

    results.sort(key=lambda x: x["mean_recall_at_5"], reverse=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "n_queries": len(records)}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nTop 5 configurations:")
    for r in results[:5]:
        print(f"  {r['parameter']}={r['value']} → R@5={r['mean_recall_at_5']:.4f} ± {r['std_recall_at_5']:.4f}")


if __name__ == "__main__":
    main()
