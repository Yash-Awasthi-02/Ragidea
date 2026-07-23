"""
PATHFINDER Experiment — Phase 6, Task 6.3: Confidence Model Selection
=====================================================================
Train a classifier to select the best σ model per query based on query features.

Usage:
    python 07_confidence_model_selection.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --output results/raw/confidence_model_selection.json
"""
import sys, json, pickle, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_pathfinder import run_pathfinder, compute_sigma, compute_sigma_min, compute_sigma_product


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
    parser = argparse.ArgumentParser(description="Confidence Model Selection")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", default="results/raw/confidence_model_selection.json")
    args = parser.parse_args()

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]

    per_query = []
    features = []
    labels = []

    for rec in tqdm(records, desc="Feature extraction"):
        gd = rec["graph"]
        G = gd["G"]
        gold = get_gold_nodes(rec)

        try:
            res = run_pathfinder(gd, enable_teleport=True)
        except Exception:
            continue

        s_prod = compute_sigma_product(res.S, res.parent, G)
        s_geom = compute_sigma(res.S, res.parent, G, use_geometric_mean=True)
        s_min = compute_sigma_min(res.S, res.parent, G)
        r5 = recall_at_k(res.S, gold, 5)

        # Features
        n_nodes = gd["N"]
        n_edges = G.number_of_edges()
        edge_weights = [G[u][v].get("weight", 0) for u, v in G.edges()]
        mean_ew = float(np.mean(edge_weights)) if edge_weights else 0.0
        q_len = len(rec["question"].split())
        phi_sem_q = gd["embeddings"] @ gd["q_emb"]
        mean_sim = float(np.mean(phi_sem_q))
        n_gold = len(gold)
        n_selected = len(res.S)

        feat = [n_nodes, n_edges, mean_ew, q_len, mean_sim, n_gold, n_selected]
        features.append(feat)
        labels.append(1 if s_geom > s_prod else 0)

        per_query.append({
            "question": rec["question"][:80],
            "sigma_product": round(s_prod, 6),
            "sigma_geometric_mean": round(s_geom, 6),
            "sigma_bottleneck": round(s_min, 6),
            "recall@5": r5,
            "features": {"n_nodes": n_nodes, "n_edges": n_edges, "mean_edge_weight": round(mean_ew, 4),
                         "query_length": q_len, "mean_sim_to_query": round(mean_sim, 4),
                         "n_gold": n_gold, "n_selected": n_selected},
        })

    # Train logistic regression
    feature_names = ["n_nodes", "n_edges", "mean_edge_weight", "query_length", "mean_sim_to_query", "n_gold", "n_selected"]
    model_coefs = {}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        X = np.array(features)
        y = np.array(labels)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X_scaled, y)
        model_coefs = {name: round(float(coef), 4) for name, coef in zip(feature_names, clf.coef_[0])}
        model_coefs["intercept"] = round(float(clf.intercept_[0]), 4)
        model_coefs["train_accuracy"] = round(float(clf.score(X_scaled, y)), 4)
    except Exception as e:
        model_coefs = {"error": str(e)}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"per_query": per_query, "model_coefficients": model_coefs, "n_queries": len(per_query)}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nModel coefficients (predicting sigma_geom > sigma_prod):")
    for k, v in model_coefs.items():
        print(f"  {k}: {v}")
    print(f"\nFraction where sigma_geom > sigma_prod: {float(np.mean(labels)):.2%}")


if __name__ == "__main__":
    main()
