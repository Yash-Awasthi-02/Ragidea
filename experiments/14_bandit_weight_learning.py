"""
PATHFINDER Experiment — Phase 9, Task 9.3: Bandit Weight Learning
=================================================================
Thompson Sampling for online weight configuration exploration.

Usage:
    python 14_bandit_weight_learning.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/raw/bandit_weight_learning.json
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


# Arm definitions: (alpha, beta, gamma, delta, epsilon)
ARMS = [
    {"name": "semantic_only", "weights": (1.0, 0.0, 0.0, 0.0, 0.0)},
    {"name": "paper_defaults", "weights": (0.5, 0.0, 0.15, 0.0, 0.1)},
    {"name": "high_semantic", "weights": (0.7, 0.0, 0.1, 0.0, 0.05)},
    {"name": "very_high_sem", "weights": (0.9, 0.0, 0.05, 0.0, 0.0)},
    {"name": "confidence_heavy", "weights": (0.5, 0.0, 0.0, 0.0, 0.15)},
    {"name": "structural_heavy", "weights": (0.8, 0.0, 0.2, 0.0, 0.0)},
]


def main():
    parser = argparse.ArgumentParser(description="Bandit Weight Learning")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--output", default="results/raw/bandit_weight_learning.json")
    args = parser.parse_args()

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]
    gold_sets = [get_gold_nodes(r) for r in records]

    # Thompson Sampling: Beta(alpha=1, beta=1) per arm
    posteriors = {arm["name"]: {"alpha": 1.0, "beta": 1.0} for arm in ARMS}

    per_query = []
    arm_counts = {arm["name"]: 0 for arm in ARMS}
    cumulative_recall = []

    for i, rec in enumerate(tqdm(records, desc="Bandit learning")):
        # Sample from each arm's Beta posterior
        samples = {}
        for arm in ARMS:
            a = posteriors[arm["name"]]["alpha"]
            b = posteriors[arm["name"]]["beta"]
            samples[arm["name"]] = np.random.beta(a, b)

        # Select arm with highest sample
        selected_arm = max(samples, key=samples.get)
        arm_weights = next(arm["weights"] for arm in ARMS if arm["name"] == selected_arm)
        arm_counts[selected_arm] += 1

        # Run pathfinder with selected weights
        try:
            res = run_pathfinder(rec["graph"], weights=arm_weights, enable_teleport=True)
            reward = recall_at_k(res.S, gold_sets[i], 5)
        except Exception:
            reward = 0

        # Update posterior
        posteriors[selected_arm]["alpha"] += reward
        posteriors[selected_arm]["beta"] += (1 - reward)

        cumulative_recall.append(reward)
        per_query.append({
            "query_idx": i,
            "selected_arm": selected_arm,
            "reward": reward,
            "cumulative_mean_recall": round(float(np.mean(cumulative_recall)), 4),
        })

    # Final arm rankings
    arm_rankings = []
    for arm in ARMS:
        name = arm["name"]
        a = posteriors[name]["alpha"]
        b = posteriors[name]["beta"]
        arm_rankings.append({
            "name": name,
            "weights": arm["weights"],
            "n_selected": arm_counts[name],
            "posterior_mean": round(a / (a + b), 4),
            "alpha": round(a, 2),
            "beta": round(b, 2),
        })
    arm_rankings.sort(key=lambda x: x["posterior_mean"], reverse=True)

    # Convergence analysis
    n = len(per_query)
    early = float(np.mean(cumulative_recall[:n//5])) if n >= 5 else 0
    late = float(np.mean(cumulative_recall[-n//5:])) if n >= 5 else 0

    summary = {
        "n_queries": n,
        "final_mean_recall": round(float(np.mean(cumulative_recall)), 4),
        "early_recall": round(early, 4),
        "late_recall": round(late, 4),
        "convergence": round(late - early, 4),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "arm_rankings": arm_rankings, "per_query": per_query}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nBandit Weight Learning Summary:")
    print(f"  Final mean R@5: {summary['final_mean_recall']:.4f}")
    print(f"  Early R@5:      {summary['early_recall']:.4f}")
    print(f"  Late R@5:       {summary['late_recall']:.4f}")
    print(f"  Convergence:    {summary['convergence']:+.4f}")
    print(f"\nArm Rankings:")
    for r in arm_rankings:
        print(f"  {r['name']:<20} R@5={r['posterior_mean']:.4f}  n={r['n_selected']:>4d}  weights={r['weights']}")


if __name__ == "__main__":
    main()
