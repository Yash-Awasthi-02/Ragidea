"""
PATHFINDER Experiment — Phase 8, Task 8.4: Dynamic Edge Synthesis
=================================================================
When teleportation reveals disconnected clusters, synthesize edges between them.

Usage:
    python 13_dynamic_edge_synthesis.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --output results/raw/dynamic_edge_synthesis.json
"""
import sys, json, pickle, argparse, numpy as np, copy
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
    parser = argparse.ArgumentParser(description="Dynamic Edge Synthesis")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", default="results/raw/dynamic_edge_synthesis.json")
    args = parser.parse_args()

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]

    per_query = []
    total_edges_synthesized = 0
    queries_helped = 0

    for rec in tqdm(records, desc="Edge synthesis"):
        gd = rec["graph"]
        G = gd["G"]
        gold = get_gold_nodes(rec)

        # Original run
        res_orig = run_pathfinder(gd, enable_teleport=True)
        r5_orig = recall_at_k(res_orig.S, gold, 5)

        # Identify teleportation nodes (parent=None, non-root)
        teleport_nodes = [v for v in res_orig.S if res_orig.parent.get(v) is None and v != res_orig.S[0]]

        # Synthesize edges: connect root cluster to teleportation targets
        edges_added = 0
        if teleport_nodes:
            import networkx as nx
            G_modified = G.copy()
            root = res_orig.S[0]
            embs = gd["embeddings"]

            for tp_node in teleport_nodes:
                # Check if already reachable from root
                try:
                    if nx.has_path(G_modified, root, tp_node):
                        continue
                except Exception:
                    pass
                # Synthesize edge from root to teleportation target
                w = float(max(0.0, np.dot(embs[root], embs[tp_node])))
                if w > 0.3:  # only add if semantically meaningful
                    G_modified.add_edge(root, tp_node, weight=w)
                    edges_added += 1

            if edges_added > 0:
                # Re-run with modified graph (teleport OFF since we added real edges)
                gd_modified = dict(gd)
                gd_modified["G"] = G_modified
                res_modified = run_pathfinder(gd_modified, enable_teleport=False)
                r5_modified = recall_at_k(res_modified.S, gold, 5)
            else:
                r5_modified = r5_orig
        else:
            r5_modified = r5_orig

        total_edges_synthesized += edges_added
        if r5_modified > r5_orig:
            queries_helped += 1

        per_query.append({
            "question": rec["question"][:80],
            "r5_original": r5_orig,
            "r5_with_synthesis": r5_modified,
            "delta": r5_modified - r5_orig,
            "edges_synthesized": edges_added,
            "n_teleport_nodes": len(teleport_nodes),
        })

    summary = {
        "n_queries": len(per_query),
        "total_edges_synthesized": total_edges_synthesized,
        "queries_helped": queries_helped,
        "mean_r5_original": round(float(np.mean([q["r5_original"] for q in per_query])), 4),
        "mean_r5_synthesis": round(float(np.mean([q["r5_with_synthesis"] for q in per_query])), 4),
        "mean_delta": round(float(np.mean([q["delta"] for q in per_query])), 4),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nDynamic Edge Synthesis Summary:")
    print(f"  Total edges synthesized: {total_edges_synthesized}")
    print(f"  Queries helped:          {queries_helped}/{len(per_query)}")
    print(f"  Mean R@5 (original):     {summary['mean_r5_original']:.4f}")
    print(f"  Mean R@5 (synthesis):    {summary['mean_r5_synthesis']:.4f}")
    print(f"  Mean improvement:        {summary['mean_delta']:.4f}")


if __name__ == "__main__":
    main()
