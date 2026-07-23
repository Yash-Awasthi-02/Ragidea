"""
PATHFINDER Experiment — Step 3: Multidimensional Hyperparameter Grid Search
=============================================================================
Phase 2, Task 2.2: Grid search over (α, γ, ε) on HotpotQA & 2Wiki subsets.

Grid:
  α (Semantic Coverage):    [0.5, 0.7, 0.9, 1.0]
  γ (Structural Importance): [0.0, 0.05, 0.10, 0.20]
  ε (Epistemic Confidence):  [0.0, 0.05, 0.10]

β and δ are held at 0.0 (no temporal/domain signal in these datasets).
Total combinations: 4 × 4 × 3 = 48 per dataset.

Usage:
    python 03_grid_search.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/raw/grid_search_hotpotqa.json
    python 03_grid_search.py --graphs data/2wiki_graphs.pkl --max_samples 500 --output results/raw/grid_search_2wiki.json
"""

import os
import sys
import json
import time
import pickle
import argparse
import itertools
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_pathfinder import run_pathfinder
from run_baselines import naive_rag


# ── Grid definitions (Phase 2, Task 2.2) ─────────────────────────────────────
ALPHA_GRID  = [0.5, 0.7, 0.9, 1.0]
GAMMA_GRID  = [0.0, 0.05, 0.10, 0.20]
EPSILON_GRID = [0.0, 0.05, 0.10]

BETA_FIXED  = 0.0   # no temporal signal in HotpotQA/2Wiki
DELTA_FIXED = 0.0   # no domain signal


def get_gold_nodes(record: dict) -> set[int]:
    """
    Map supporting facts to graph node indices.
    Supports both HotpotQA/2Wiki (sentence-level) and MuSiQue (paragraph-level).
    """
    gold = set()
    G = record["graph"]["G"]
    supporting_facts = record.get("supporting_facts", [])

    # HotpotQA / 2Wiki: list of [title, sent_idx] pairs
    for sf in supporting_facts:
        if isinstance(sf, (list, tuple)) and len(sf) == 2:
            title, sent_idx = sf
            for node in G.nodes:
                if (G.nodes[node].get("doc_title") == title and
                        G.nodes[node].get("sent_idx") == sent_idx):
                    gold.add(node)

    # MuSiQue: paragraph-level is_supporting flag
    if not gold:
        for node in G.nodes:
            if G.nodes[node].get("is_supporting", False):
                gold.add(node)

    return gold


def recall_at_k(selected: list[int], gold: set[int], k: int = 5) -> float:
    """Recall@k: fraction of gold nodes in top-k selected."""
    if not gold:
        return 0.0
    top_k = set(selected[:k])
    return len(top_k & gold) / len(gold)


def run_grid_search(records: list[dict], max_samples: int = 500,
                    k: int = 5) -> list[dict]:
    """
    Run grid search over (α, γ, ε) weight combinations.
    Returns list of result dicts sorted by mean Recall@5 descending.
    """
    records = records[:max_samples]
    results = []

    # Precompute gold nodes per record
    gold_sets = []
    for rec in records:
        gold = get_gold_nodes(rec)
        gold_sets.append(gold)

    # Generate all weight combinations
    combos = list(itertools.product(ALPHA_GRID, GAMMA_GRID, EPSILON_GRID))
    total = len(combos)
    print(f"Grid search: {total} combinations × {len(records)} samples")

    for alpha, gamma, epsilon in tqdm(combos, desc="Grid combinations"):
        weights = (alpha, BETA_FIXED, gamma, DELTA_FIXED, epsilon)
        recalls = []

        for i, rec in enumerate(records):
            gd = rec["graph"]
            try:
                res = run_pathfinder(gd, weights=weights, enable_teleport=True)
                r = recall_at_k(res.S, gold_sets[i], k=k)
                recalls.append(r)
            except Exception:
                recalls.append(0.0)

        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        std_recall = float(np.std(recalls)) if recalls else 0.0

        results.append({
            "alpha": alpha,
            "beta": BETA_FIXED,
            "gamma": gamma,
            "delta": DELTA_FIXED,
            "epsilon": epsilon,
            "mean_recall_at_5": round(mean_recall, 4),
            "std_recall_at_5": round(std_recall, 4),
            "n_samples": len(records),
        })

    # Sort by mean Recall@5 descending
    results.sort(key=lambda x: x["mean_recall_at_5"], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="PATHFINDER Grid Search")
    parser.add_argument("--graphs", required=True, help="Path to graph pickle file")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max samples to evaluate (default: 500)")
    parser.add_argument("--output", default="results/raw/grid_search.json",
                        help="Output JSON path")
    parser.add_argument("--k", type=int, default=5, help="Recall@k (default: 5)")
    args = parser.parse_args()

    if not Path(args.graphs).exists():
        print(f"Graph cache not found: {args.graphs}")
        print("Run: python 01_build_kg.py first")
        sys.exit(1)

    print(f"Loading graphs from {args.graphs}...")
    with open(args.graphs, "rb") as f:
        records = pickle.load(f)

    results = run_grid_search(records, max_samples=args.max_samples, k=args.k)

    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump({
            "dataset": args.graphs,
            "max_samples": args.max_samples,
            "k": args.k,
            "grid": {
                "alpha": ALPHA_GRID,
                "gamma": GAMMA_GRID,
                "epsilon": EPSILON_GRID,
                "beta_fixed": BETA_FIXED,
                "delta_fixed": DELTA_FIXED,
            },
            "results": results,
        }, f, indent=2)

    print(f"\nGrid search complete. Results saved to {out_path}")
    print(f"\nTop 5 configurations (Recall@{args.k}):")
    for r in results[:5]:
        print(f"  α={r['alpha']:.2f} γ={r['gamma']:.2f} ε={r['epsilon']:.2f}"
              f"  →  R@{args.k}={r['mean_recall_at_5']:.4f} ± {r['std_recall_at_5']:.4f}")


if __name__ == "__main__":
    main()
