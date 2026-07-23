"""
PATHFINDER Experiment — Phase 5, Task 5.4: Teleportation Impact Analysis
=========================================================================
Per-query analysis: which queries did teleportation help vs hurt?

Usage:
    python 05d_teleportation_impact.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/raw/teleportation_impact.json
"""
import sys, json, pickle, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
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
    parser = argparse.ArgumentParser(description="Teleportation Impact Analysis")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--output", default="results/raw/teleportation_impact.json")
    args = parser.parse_args()

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]

    per_query = []
    n_changed = 0
    n_helped = 0
    n_hurt = 0
    improvements = []

    for rec in tqdm(records, desc="Impact analysis"):
        gd = rec["graph"]
        gold = get_gold_nodes(rec)

        res_off = run_pathfinder(gd, enable_teleport=False)
        res_on = run_pathfinder(gd, enable_teleport=True)

        r5_off = recall_at_k(res_off.S, gold, 5)
        r5_on = recall_at_k(res_on.S, gold, 5)

        changed = set(res_on.S) != set(res_off.S)
        teleport_nodes = [v for v in res_on.S if res_on.parent.get(v) is None and v != res_on.S[0]]
        gold_teleport = [v for v in teleport_nodes if v in gold]

        delta = r5_on - r5_off
        if changed:
            n_changed += 1
        if delta > 0:
            n_helped += 1
        elif delta < 0:
            n_hurt += 1
        improvements.append(delta)

        per_query.append({
            "question": rec["question"][:80],
            "r5_off": r5_off,
            "r5_on": r5_on,
            "delta": delta,
            "changed": changed,
            "n_teleport_nodes": len(teleport_nodes),
            "n_gold_teleport": len(gold_teleport),
            "n_nodes_off": len(res_off.S),
            "n_nodes_on": len(res_on.S),
        })

    summary = {
        "n_queries": len(records),
        "n_changed": n_changed,
        "n_helped": n_helped,
        "n_hurt": n_hurt,
        "frac_changed": round(n_changed / len(records), 4),
        "frac_helped": round(n_helped / len(records), 4),
        "frac_hurt": round(n_hurt / len(records), 4),
        "mean_improvement": round(float(np.mean(improvements)), 4),
        "mean_r5_off": round(float(np.mean([q["r5_off"] for q in per_query])), 4),
        "mean_r5_on": round(float(np.mean([q["r5_on"] for q in per_query])), 4),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nTeleportation Impact Summary:")
    print(f"  Queries changed:  {n_changed}/{len(records)} ({summary['frac_changed']:.1%})")
    print(f"  Helped:           {n_helped} ({summary['frac_helped']:.1%})")
    print(f"  Hurt:             {n_hurt} ({summary['frac_hurt']:.1%})")
    print(f"  Mean R@5 (off):   {summary['mean_r5_off']:.4f}")
    print(f"  Mean R@5 (on):    {summary['mean_r5_on']:.4f}")
    print(f"  Mean improvement: {summary['mean_improvement']:.4f}")


if __name__ == "__main__":
    main()
