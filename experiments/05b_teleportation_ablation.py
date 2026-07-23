"""
PATHFINDER Experiment — Phase 5, Task 5.1: Teleportation Ablation
===================================================================
Compare Pure Graph (no teleport) vs Teleportation Hybrid vs Naive RAG.

Usage:
    python 05b_teleportation_ablation.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/raw/teleportation_ablation.json
"""
import sys, json, pickle, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_pathfinder import run_pathfinder
from run_baselines import naive_rag


def get_gold_nodes(record):
    gold = set()
    nodes = record["graph"]["nodes"]
    sf = record.get("supporting_facts", {})
    if isinstance(sf, dict) and "title" in sf:
        gold_set = set(zip(sf["title"], sf["sent_id"]))
        for i, node in enumerate(nodes):
            if (node["doc_title"], node["sent_idx"]) in gold_set:
                gold.add(i)
    elif isinstance(sf, list):
        for pair in sf:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                for i, node in enumerate(nodes):
                    if node["doc_title"] == pair[0] and node["sent_idx"] == pair[1]:
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


def paragraph_recall_at_k(retrieved, gold, nodes, k=5):
    if not gold:
        return 0.0
    gold_titles = {nodes[v]["doc_title"] for v in gold}
    ret_titles = {nodes[v]["doc_title"] for v in retrieved[:k]}
    return len(gold_titles & ret_titles) / len(gold_titles) if gold_titles else 0.0


def main():
    parser = argparse.ArgumentParser(description="Teleportation Ablation")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--output", default="results/raw/teleportation_ablation.json")
    args = parser.parse_args()

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]

    configs = ["pure_graph", "teleport_hybrid", "naive_rag"]
    per_query = []

    for rec in tqdm(records, desc="Ablation"):
        gd = rec["graph"]
        nodes = gd["nodes"]
        gold = get_gold_nodes(rec)

        # Pure graph (teleport OFF)
        try:
            res_pg = run_pathfinder(gd, enable_teleport=False)
            pg = {
                "recall@5": recall_at_k(res_pg.S, gold, 5),
                "recall@10": recall_at_k(res_pg.S, gold, 10),
                "recall@20": recall_at_k(res_pg.S, gold, 20),
                "paragraph_recall@5": paragraph_recall_at_k(res_pg.S, gold, nodes, 5),
                "n_nodes": len(res_pg.S),
            }
        except Exception:
            pg = {"recall@5": 0, "recall@10": 0, "recall@20": 0, "paragraph_recall@5": 0, "n_nodes": 0}

        # Teleport hybrid
        try:
            res_tp = run_pathfinder(gd, enable_teleport=True)
            tp = {
                "recall@5": recall_at_k(res_tp.S, gold, 5),
                "recall@10": recall_at_k(res_tp.S, gold, 10),
                "recall@20": recall_at_k(res_tp.S, gold, 20),
                "paragraph_recall@5": paragraph_recall_at_k(res_tp.S, gold, nodes, 5),
                "n_nodes": len(res_tp.S),
            }
        except Exception:
            tp = {"recall@5": 0, "recall@10": 0, "recall@20": 0, "paragraph_recall@5": 0, "n_nodes": 0}

        # Naive RAG
        nr_nodes = naive_rag(gd, k=20)
        nr = {
            "recall@5": recall_at_k(nr_nodes, gold, 5),
            "recall@10": recall_at_k(nr_nodes, gold, 10),
            "recall@20": recall_at_k(nr_nodes, gold, 20),
            "paragraph_recall@5": paragraph_recall_at_k(nr_nodes, gold, nodes, 5),
            "n_nodes": len(nr_nodes),
        }

        per_query.append({"question": rec["question"][:80], "pure_graph": pg, "teleport_hybrid": tp, "naive_rag": nr})

    # Aggregate
    summary = {}
    for cfg in configs:
        summary[cfg] = {
            "recall@5": round(float(np.mean([q[cfg]["recall@5"] for q in per_query])), 4),
            "recall@10": round(float(np.mean([q[cfg]["recall@10"] for q in per_query])), 4),
            "recall@20": round(float(np.mean([q[cfg]["recall@20"] for q in per_query])), 4),
            "paragraph_recall@5": round(float(np.mean([q[cfg]["paragraph_recall@5"] for q in per_query])), 4),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query, "n_queries": len(per_query)}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\n{'Config':<20} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'ParaR@5':>8}")
    print("-" * 56)
    for cfg in configs:
        s = summary[cfg]
        print(f"{cfg:<20} {s['recall@5']:>8.4f} {s['recall@10']:>8.4f} {s['recall@20']:>8.4f} {s['paragraph_recall@5']:>8.4f}")


if __name__ == "__main__":
    main()
