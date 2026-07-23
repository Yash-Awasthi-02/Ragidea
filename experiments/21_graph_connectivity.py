"""
PATHFINDER Experiment — Phase 8, Task 8.5: Graph Connectivity Analysis
======================================================================
Correlate graph connectivity with R@5 gap (Naive RAG vs PATHFINDER).

Usage:
    python 21_graph_connectivity.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/raw/graph_connectivity.json
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
    parser = argparse.ArgumentParser(description="Graph Connectivity Analysis")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--output", default="results/raw/graph_connectivity.json")
    args = parser.parse_args()

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]

    per_query = []

    for rec in tqdm(records, desc="Connectivity analysis"):
        gd = rec["graph"]
        G = gd["G"]
        N = gd["N"]
        gold = get_gold_nodes(rec)

        # Connectivity metrics
        import networkx as nx
        undirected = G.to_undirected()
        components = list(nx.connected_components(undirected))
        n_components = len(components)
        component_sizes = [len(c) for c in components]
        avg_component_size = float(np.mean(component_sizes)) if component_sizes else 0
        max_component_size = max(component_sizes) if component_sizes else 0

        # Edge density
        max_possible_edges = N * (N - 1) if N > 1 else 1
        edge_density = G.number_of_edges() / max_possible_edges if max_possible_edges > 0 else 0

        # Inter-component edge density (fraction of edges between different components)
        node_to_comp = {}
        for i, comp in enumerate(components):
            for node in comp:
                node_to_comp[node] = i
        inter_component_edges = sum(1 for u, v in G.edges() if node_to_comp.get(u, -1) != node_to_comp.get(v, -2))
        inter_component_density = inter_component_edges / max(G.number_of_edges(), 1)

        # Run pathfinder and naive rag
        res_pf = run_pathfinder(gd, enable_teleport=True)
        nr_nodes = naive_rag(gd, k=5)
        r5_pf = recall_at_k(res_pf.S, gold, 5)
        r5_nr = recall_at_k(nr_nodes, gold, 5)
        r5_gap = r5_nr - r5_pf  # positive = Naive RAG advantage

        per_query.append({
            "n_nodes": N,
            "n_edges": G.number_of_edges(),
            "n_components": n_components,
            "avg_component_size": round(avg_component_size, 2),
            "max_component_size": max_component_size,
            "edge_density": round(edge_density, 4),
            "inter_component_density": round(inter_component_density, 4),
            "r5_pathfinder": r5_pf,
            "r5_naive_rag": r5_nr,
            "r5_gap": r5_gap,
        })

    # Correlation analysis
    gaps = [q["r5_gap"] for q in per_query]
    correlations = {}
    for metric in ["n_components", "avg_component_size", "edge_density", "inter_component_density"]:
        vals = [q[metric] for q in per_query]
        if len(set(vals)) > 1 and len(set(gaps)) > 1:
            from scipy.stats import spearmanr, pearsonr
            rho, p_rho = spearmanr(vals, gaps)
            r, p_r = pearsonr(vals, gaps)
            correlations[metric] = {
                "spearman_rho": round(float(rho), 4),
                "spearman_p": round(float(p_rho), 4),
                "pearson_r": round(float(r), 4),
                "pearson_p": round(float(p_r), 4),
            }

    summary = {
        "n_queries": len(per_query),
        "mean_n_components": round(float(np.mean([q["n_components"] for q in per_query])), 2),
        "mean_edge_density": round(float(np.mean([q["edge_density"] for q in per_query])), 4),
        "mean_r5_gap": round(float(np.mean(gaps)), 4),
        "correlations": correlations,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nGraph Connectivity Analysis:")
    print(f"  Mean components:    {summary['mean_n_components']:.1f}")
    print(f"  Mean edge density:  {summary['mean_edge_density']:.4f}")
    print(f"  Mean R@5 gap (NR-PF): {summary['mean_r5_gap']:.4f}")
    print(f"\nCorrelations with R@5 gap (Naive RAG advantage):")
    for metric, corr in correlations.items():
        print(f"  {metric}: Spearman ρ={corr['spearman_rho']:.3f} (p={corr['spearman_p']:.3f}), Pearson r={corr['pearson_r']:.3f}")
    print(f"\nInterpretation:")
    if correlations.get("n_components", {}).get("spearman_rho", 0) > 0.1:
        print("  → More components → larger Naive RAG advantage (graph is fragmented)")
    if correlations.get("edge_density", {}).get("spearman_rho", 0) < -0.1:
        print("  → Higher edge density → smaller Naive RAG advantage (graph is well-connected)")
    if correlations.get("inter_component_density", {}).get("spearman_rho", 0) < -0.1:
        print("  → More inter-component edges → smaller Naive RAG advantage")


if __name__ == "__main__":
    main()
