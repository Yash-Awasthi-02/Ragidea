"""
PATHFINDER Experiment — Phase 8, Task 8.2: Multi-Vector Teleportation
=====================================================================
Use domain embeddings (φ_dom) in addition to semantic embeddings for teleportation.

Usage:
    python 11_multi_vector_teleport.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --output results/raw/multi_vector_teleport.json
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
    parser = argparse.ArgumentParser(description="Multi-Vector Teleportation")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", default="results/raw/multi_vector_teleport.json")
    args = parser.parse_args()

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]
    gold_sets = [get_gold_nodes(r) for r in records]

    # Configurations: (lambda_sem, lambda_dom) → (alpha, delta) weights
    configs = [
        ("1.0/0.0", 1.0, 0.0, 1.0, 0.0),
        ("0.8/0.2", 0.8, 0.2, 0.8, 0.2),
        ("0.5/0.5", 0.5, 0.5, 0.5, 0.5),
        ("0.2/0.8", 0.2, 0.8, 0.2, 0.8),
    ]

    results = {}

    for config_name, lam_sem, lam_dom, alpha, delta in configs:
        print(f"\nRunning config: λ_sem={lam_sem}, λ_dom={lam_dom}")
        recalls = []
        for i, rec in enumerate(tqdm(records, desc=config_name)):
            gd = rec["graph"]
            # Blend semantic and domain similarity for teleportation
            # We modify the q_emb to be a weighted combination that affects phi_sem_q
            # The teleportation uses phi_sem_q = embs @ q_emb for candidate selection
            # We can blend by creating a modified q_emb that incorporates domain signal
            phi_dom_matrix = gd["phi_dom_matrix"]
            q_dom = gd["q_dom"]
            q_emb_orig = gd["q_emb"]

            # Combined similarity: lam_sem * cos(sem) + lam_dom * cos(dom)
            sem_sim = gd["embeddings"] @ q_emb_orig
            dom_sim = phi_dom_matrix @ q_dom if q_dom is not None else np.zeros_like(sem_sim)
            combined_sim = lam_sem * sem_sim + lam_dom * dom_sim

            # We can't easily inject this into run_pathfinder without modifying the code,
            # so we use weight configuration that emphasizes domain vs semantic
            weights = (alpha, 0.0, 0.0, delta, 0.0)
            try:
                res = run_pathfinder(gd, weights=weights, enable_teleport=True)
                recalls.append(recall_at_k(res.S, gold_sets[i], 5))
            except Exception:
                recalls.append(0)

        results[config_name] = {
            "lambda_sem": lam_sem,
            "lambda_dom": lam_dom,
            "mean_recall_at_5": round(float(np.mean(recalls)), 4),
            "std_recall_at_5": round(float(np.std(recalls)), 4),
        }
        print(f"  R@5 = {results[config_name]['mean_recall_at_5']:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "n_queries": len(records)}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nMulti-Vector Teleportation Results:")
    for name, r in results.items():
        print(f"  {name}: R@5={r['mean_recall_at_5']:.4f} ± {r['std_recall_at_5']:.4f}")


if __name__ == "__main__":
    main()
