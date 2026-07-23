"""
PATHFINDER Experiment — Phase 7, Task 7.5: Bayesian Hyperparameter Optimization
===============================================================================
Use Optuna TPE sampler to optimize (α, β, γ, δ, ε) for max Recall@5.

Usage:
    pip install optuna
    python 09_bayesian_optimization.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --trials 100 --output results/raw/bayesian_opt.json
"""
import sys, json, pickle, argparse, numpy as np
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Bayesian Hyperparameter Optimization")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output", default="results/raw/bayesian_opt.json")
    args = parser.parse_args()

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Please install optuna: pip install optuna")
        sys.exit(1)

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]
    gold_sets = [get_gold_nodes(r) for r in records]

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.3, 1.0)
        beta = trial.suggest_float("beta", 0.0, 0.3)
        gamma = trial.suggest_float("gamma", 0.0, 0.3)
        delta = trial.suggest_float("delta", 0.0, 0.2)
        epsilon = trial.suggest_float("epsilon", 0.0, 0.2)
        weights = (alpha, beta, gamma, delta, epsilon)

        recalls = []
        for i, rec in enumerate(records):
            try:
                res = run_pathfinder(rec["graph"], weights=weights, enable_teleport=True)
                recalls.append(recall_at_k(res.S, gold_sets[i], 5))
            except Exception:
                recalls.append(0)
        return float(np.mean(recalls))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_params
    best_value = study.best_value

    trials_data = []
    for t in study.trials:
        trials_data.append({
            "number": t.number,
            "params": t.params,
            "value": round(t.value, 4) if t.value is not None else None,
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "best_params": best,
            "best_recall_at_5": round(best_value, 4),
            "n_trials": args.trials,
            "n_queries": len(records),
            "trials": trials_data,
        }, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nBest configuration (R@5={best_value:.4f}):")
    for k, v in best.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
