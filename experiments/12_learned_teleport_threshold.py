"""
PATHFINDER Experiment — Phase 8, Task 8.3: Learned Teleportation Threshold
==========================================================================
Extract per-query features and train a classifier to predict when teleportation helps.

Usage:
    python 12_learned_teleport_threshold.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --output results/raw/learned_teleport_threshold.json
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
    parser = argparse.ArgumentParser(description="Learned Teleportation Threshold")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", default="results/raw/learned_teleport_threshold.json")
    args = parser.parse_args()

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]

    features = []
    labels = []
    per_query = []

    for rec in tqdm(records, desc="Feature extraction"):
        gd = rec["graph"]
        gold = get_gold_nodes(rec)
        phi_sem_q = gd["embeddings"] @ gd["q_emb"]

        # Run with teleport ON and OFF
        res_on = run_pathfinder(gd, enable_teleport=True)
        res_off = run_pathfinder(gd, enable_teleport=False)
        r5_on = recall_at_k(res_on.S, gold, 5)
        r5_off = recall_at_k(res_off.S, gold, 5)

        # Features
        feat = [
            gd["N"],                                          # n_nodes
            gd["G"].number_of_edges(),                        # n_edges
            float(np.mean(phi_sem_q)),                        # mean_sim_to_query
            float(np.std(phi_sem_q)),                         # std_sim_to_query
            float(np.max(phi_sem_q)),                         # max_sim
            float(np.min(phi_sem_q)),                         # min_sim
            len(rec["question"].split()),                     # query_length
            len(res_off.S),                                   # n_selected_off
            float(np.mean([gd["G"].nodes[v].get("phi_conf", 0.7) for v in res_off.S])) if res_off.S else 0.0,  # mean_conf
        ]
        features.append(feat)
        label = 1 if r5_on > r5_off else (0 if r5_on < r5_off else -1)  # helped=1, hurt=0, neutral=-1
        labels.append(label)

        per_query.append({
            "question": rec["question"][:80],
            "r5_off": r5_off, "r5_on": r5_on, "delta": r5_on - r5_off,
            "n_nodes": gd["N"], "n_edges": gd["G"].number_of_edges(),
        })

    feature_names = ["n_nodes", "n_edges", "mean_sim", "std_sim", "max_sim", "min_sim", "query_len", "n_selected_off", "mean_conf"]

    # Train classifier (only on helped vs hurt, exclude neutral)
    model_info = {}
    helped = [i for i, l in enumerate(labels) if l == 1]
    hurt = [i for i, l in enumerate(labels) if l == 0]
    neutral = [i for i, l in enumerate(labels) if l == -1]

    model_info["n_helped"] = len(helped)
    model_info["n_hurt"] = len(hurt)
    model_info["n_neutral"] = len(neutral)

    if len(helped) > 2 and len(hurt) > 2:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        train_idx = helped + hurt
        X = np.array([features[i] for i in train_idx])
        y = np.array([labels[i] for i in train_idx])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X_scaled, y)
        model_info["coefficients"] = {n: round(float(c), 4) for n, c in zip(feature_names, clf.coef_[0])}
        model_info["intercept"] = round(float(clf.intercept_[0]), 4)
        model_info["train_accuracy"] = round(float(clf.score(X_scaled, y)), 4)
    else:
        model_info["error"] = "Not enough helped/hurt samples for training"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model_info": model_info, "per_query": per_query, "feature_names": feature_names}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nTeleportation Impact:")
    print(f"  Helped:  {model_info['n_helped']}")
    print(f"  Hurt:    {model_info['n_hurt']}")
    print(f"  Neutral: {model_info['n_neutral']}")
    if "coefficients" in model_info:
        print(f"\nFeature importance (predicting teleport helps):")
        coefs = model_info["coefficients"]
        for name, coef in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {name}: {coef:+.4f}")


if __name__ == "__main__":
    main()
