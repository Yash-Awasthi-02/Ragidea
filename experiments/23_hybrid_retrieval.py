"""
PATHFINDER Experiment — Phase B: Hybrid Retrieval Strategies
=============================================================
Three hybrid retrieval modes that combine dense + graph traversal:

  1. Dense-Anchor Hybrid: Top-k dense nodes as anchors, graph fills rest
  2. Dense-Graph Interleaving: Always consider dense + frontier candidates
  3. Two-Stage: Dense top-20 → PATHFINDER submodular selection of best k

Usage:
    python 23_hybrid_retrieval.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/raw/hybrid_retrieval.json
    python 23_hybrid_retrieval.py --graphs data/hotpotqa_graphs_passage.pkl --max_samples 500 --output results/raw/hybrid_passage.json
"""
import sys, json, pickle, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_pathfinder import (
    run_pathfinder, _single_pass, tok_count, delta_full_fast,
    compute_sigma, compute_F, K_TOK, ALPHA, BETA, GAMMA, DELTA, EPSILON,
    TraversalResult,
)
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


def paragraph_recall_at_k(retrieved, gold, nodes, k=5):
    if not gold:
        return 0.0
    gold_titles = {nodes[v]["doc_title"] for v in gold}
    ret_titles = {nodes[v]["doc_title"] for v in retrieved[:k]}
    return len(gold_titles & ret_titles) / len(gold_titles) if gold_titles else 0.0


# ── Strategy 1: Dense-Anchor Hybrid ──────────────────────────────────────────
def run_dense_anchor_hybrid(graph_data, n_anchors=3, k_tok=K_TOK):
    """
    Take top-n_anchors dense nodes as primary anchors.
    Use PATHFINDER's frontier expansion from each anchor to fill remaining budget.
    Merge all anchor traversals, deduplicate, return best by F(S,q).
    """
    G = graph_data["G"]
    q_emb = graph_data["q_emb"]
    q_dom = graph_data["q_dom"]
    embs = graph_data["embeddings"]
    N = graph_data["N"]
    phi_sem_q = embs @ q_emb

    if N == 0:
        return []

    # Get top-n dense nodes
    ranked = np.argsort(phi_sem_q)[::-1]
    anchors = ranked[:n_anchors].tolist()

    # For each anchor, run a mini-traversal
    all_selected = set()
    weights = (ALPHA, BETA, GAMMA, DELTA, EPSILON)

    # Attach phi_dom
    phi_dom_matrix = graph_data["phi_dom_matrix"]
    for i in range(N):
        G.nodes[i]["phi_dom"] = phi_dom_matrix[i]

    # Start with all anchors in S (bypass frontier constraint)
    S = list(anchors)
    S_set = set(anchors)
    parent = {a: None for a in anchors}

    tok = sum(tok_count(a, G) for a in anchors)
    product_factor = 1.0
    for a in anchors:
        product_factor *= (1.0 - max(0.0, float(phi_sem_q[a])))

    # Build frontier from all anchors' neighbors
    frontier = {}
    for a in anchors:
        for u in G.successors(a):
            if u not in S_set and u not in frontier:
                frontier[u] = a

    # Greedy expansion
    while frontier and tok < k_tok:
        rem = k_tok - tok
        feasible = {v: p for v, p in frontier.items() if tok_count(v, G) <= rem}
        if not feasible:
            break

        best_v, best_gain = None, -1.0
        for v in feasible:
            g = delta_full_fast(v, G, q_dom, phi_sem_q, weights, product_factor)
            if g > best_gain:
                best_gain, best_v = g, v
        if best_v is None:
            break

        S.append(best_v)
        S_set.add(best_v)
        parent[best_v] = feasible[best_v]
        product_factor *= (1.0 - max(0.0, float(phi_sem_q[best_v])))
        tok += tok_count(best_v, G)

        del frontier[best_v]
        for u in G.successors(best_v):
            if u not in S_set and u not in frontier:
                frontier[u] = best_v

    return S


# ── Strategy 2: Dense-Graph Interleaving ─────────────────────────────────────
def run_dense_graph_interleave(graph_data, dense_topk=5, k_tok=K_TOK):
    """
    At each greedy step, consider BOTH frontier nodes AND top-k global dense nodes.
    Select by Δ_full regardless of source. This makes teleportation the default.
    """
    G = graph_data["G"]
    q_emb = graph_data["q_emb"]
    q_dom = graph_data["q_dom"]
    embs = graph_data["embeddings"]
    N = graph_data["N"]
    phi_sem_q = embs @ q_emb

    if N == 0:
        return []

    phi_dom_matrix = graph_data["phi_dom_matrix"]
    for i in range(N):
        G.nodes[i]["phi_dom"] = phi_dom_matrix[i]

    # Entry node
    v0 = int(np.argmax(phi_sem_q))
    if tok_count(v0, G) > k_tok:
        return []

    S = [v0]
    S_set = {v0}
    parent = {v0: None}
    tok = tok_count(v0, G)
    product_factor = 1.0 - max(0.0, float(phi_sem_q[v0]))
    weights = (ALPHA, BETA, GAMMA, DELTA, EPSILON)

    # Precompute global dense ranking
    global_ranked = np.argsort(phi_sem_q)[::-1]
    dense_ptr = 0  # pointer into global_ranked for nodes not yet in S

    frontier = {}
    for u in G.successors(v0):
        if u not in S_set:
            frontier[u] = v0

    while frontier and tok < k_tok and dense_ptr < N:
        rem = k_tok - tok

        # Build candidate pool: frontier + next dense nodes not in S
        candidates = {}
        for v, p in frontier.items():
            if tok_count(v, G) <= rem:
                candidates[v] = p  # graph parent

        # Add dense candidates
        added_dense = 0
        while dense_ptr < N and added_dense < dense_topk:
            v = int(global_ranked[dense_ptr])
            dense_ptr += 1
            if v not in S_set and tok_count(v, G) <= rem:
                candidates[v] = None  # None = dense (teleportation)
                added_dense += 1

        if not candidates:
            break

        # Select best by Δ_full
        best_v, best_gain = None, -1.0
        for v in candidates:
            g = delta_full_fast(v, G, q_dom, phi_sem_q, weights, product_factor)
            if g > best_gain:
                best_gain, best_v = g, v
        if best_v is None:
            break

        disc_parent = candidates[best_v]
        S.append(best_v)
        S_set.add(best_v)
        parent[best_v] = disc_parent
        product_factor *= (1.0 - max(0.0, float(phi_sem_q[best_v])))
        tok += tok_count(best_v, G)

        # Expand frontier only for graph-connected nodes
        if disc_parent is not None:
            del frontier[best_v]
            for u in G.successors(best_v):
                if u not in S_set and u not in frontier:
                    frontier[u] = best_v
        else:
            # Dense node — add its neighbors to frontier too
            for u in G.successors(best_v):
                if u not in S_set and u not in frontier:
                    frontier[u] = best_v

    return S


# ── Strategy 3: Two-Stage (Dense → Submodular Selection) ─────────────────────
def run_two_stage(graph_data, dense_k=20, final_k=5, k_tok=K_TOK):
    """
    Stage 1: Dense retrieval gets top-dense_k candidates.
    Stage 2: PATHFINDER's submodular selection picks the best final_k from those.
    """
    G = graph_data["G"]
    q_emb = graph_data["q_emb"]
    q_dom = graph_data["q_dom"]
    embs = graph_data["embeddings"]
    N = graph_data["N"]
    phi_sem_q = embs @ q_emb

    if N == 0:
        return []

    # Stage 1: Dense top-k
    dense_ranked = np.argsort(phi_sem_q)[::-1][:dense_k]
    dense_set = set(dense_ranked.tolist())

    # Stage 2: Submodular selection from dense candidates
    # Build a subgraph with only dense nodes and edges between them
    phi_dom_matrix = graph_data["phi_dom_matrix"]
    for i in range(N):
        G.nodes[i]["phi_dom"] = phi_dom_matrix[i]

    # Greedy submodular selection from dense candidates
    weights = (ALPHA, BETA, GAMMA, DELTA, EPSILON)
    S = []
    S_set = set()
    product_factor = 1.0
    tok = 0

    # Entry: best dense node
    v0 = int(dense_ranked[0])
    S.append(v0)
    S_set.add(v0)
    product_factor *= (1.0 - max(0.0, float(phi_sem_q[v0])))
    tok += tok_count(v0, G)

    # Greedy: select from remaining dense candidates by Δ_full
    remaining = [v for v in dense_ranked.tolist() if v not in S_set]

    while remaining and tok < k_tok and len(S) < final_k * 2:
        rem = k_tok - tok
        best_v, best_gain = None, -1.0
        for v in remaining:
            if tok_count(v, G) > rem:
                continue
            g = delta_full_fast(v, G, q_dom, phi_sem_q, weights, product_factor)
            if g > best_gain:
                best_gain, best_v = g, v

        if best_v is None:
            break

        S.append(best_v)
        S_set.add(best_v)
        product_factor *= (1.0 - max(0.0, float(phi_sem_q[best_v])))
        tok += tok_count(best_v, G)
        remaining.remove(best_v)

    return S


# ── Main evaluation ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hybrid Retrieval Evaluation")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--output", default="results/raw/hybrid_retrieval.json")
    args = parser.parse_args()

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]

    configs = {
        "pathfinder_original": lambda gd: run_pathfinder(gd, enable_teleport=True).S,
        "dense_anchor_3": lambda gd: run_dense_anchor_hybrid(gd, n_anchors=3),
        "dense_anchor_5": lambda gd: run_dense_anchor_hybrid(gd, n_anchors=5),
        "dense_interleave_5": lambda gd: run_dense_graph_interleave(gd, dense_topk=5),
        "two_stage_20_5": lambda gd: run_two_stage(gd, dense_k=20, final_k=5),
        "naive_rag": lambda gd: naive_rag(gd, k=20),
    }

    per_query = []

    for rec in tqdm(records, desc="Hybrid retrieval"):
        gd = rec["graph"]
        nodes = gd["nodes"]
        gold = get_gold_nodes(rec)

        query_results = {"question": rec["question"][:80]}

        for config_name, run_fn in configs.items():
            try:
                S = run_fn(gd)
                query_results[config_name] = {
                    "recall@5": recall_at_k(S, gold, 5),
                    "recall@10": recall_at_k(S, gold, 10),
                    "recall@20": recall_at_k(S, gold, 20),
                    "paragraph_recall@5": paragraph_recall_at_k(S, gold, nodes, 5),
                    "n_nodes": len(S),
                }
            except Exception as e:
                query_results[config_name] = {
                    "recall@5": 0, "recall@10": 0, "recall@20": 0,
                    "paragraph_recall@5": 0, "n_nodes": 0, "error": str(e)[:100]
                }

        per_query.append(query_results)

    # Aggregate
    summary = {}
    for config_name in configs:
        vals = [q[config_name] for q in per_query if config_name in q]
        summary[config_name] = {
            "recall@5": round(float(np.mean([v["recall@5"] for v in vals])), 4),
            "recall@10": round(float(np.mean([v["recall@10"] for v in vals])), 4),
            "recall@20": round(float(np.mean([v["recall@20"] for v in vals])), 4),
            "paragraph_recall@5": round(float(np.mean([v["paragraph_recall@5"] for v in vals])), 4),
            "mean_nodes": round(float(np.mean([v["n_nodes"] for v in vals])), 2),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query, "n_queries": len(per_query)}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\n{'Config':<25} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'ParaR@5':>8} {'Nodes':>6}")
    print("-" * 67)
    for name, s in summary.items():
        print(f"{name:<25} {s['recall@5']:>8.4f} {s['recall@10']:>8.4f} {s['recall@20']:>8.4f} {s['paragraph_recall@5']:>8.4f} {s['mean_nodes']:>6.1f}")


if __name__ == "__main__":
    main()
