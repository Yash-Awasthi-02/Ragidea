"""
PATHFINDER Experiment — Step 3b: Confidence Calibration Comparison
===================================================================
Phase 2, Task 2.3: Compare 3 confidence aggregation models:

  1. Product Confidence:      σ_prod(S) = min_{v∈S} Π W(e)·φ_conf(u)
  2. Geometric Mean Confidence: σ_geom(S) = min_{v∈S} (Π W(e)·φ_conf(u))^{1/L}
  3. Bottleneck Confidence:   σ_min(S) = min_{v∈S} min_{e,u on path} {W(e), φ_conf(u)}

This script runs PATHFINDER on a dataset, computes all 3 σ models per query,
and outputs calibration data for plotting (σ vs actual answer EM/F1).

Usage:
    python 04_confidence_calibration.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --output results/raw/confidence_calibration.json

    # With LLM answers (requires GROQ_API_KEY):
    export GROQ_API_KEY=your_key
    python 04_confidence_calibration.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --output results/raw/confidence_calibration.json --with_llm
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_pathfinder import (
    run_pathfinder, compute_sigma, compute_sigma_min, compute_sigma_product,
    TraversalResult,
)


def get_gold_nodes(record: dict) -> set[int]:
    """Map supporting facts to graph node indices."""
    gold = set()
    G = record["graph"]["G"]
    supporting_facts = record.get("supporting_facts", [])

    for sf in supporting_facts:
        if isinstance(sf, (list, tuple)) and len(sf) == 2:
            title, sent_idx = sf
            for node in G.nodes:
                if (G.nodes[node].get("doc_title") == title and
                        G.nodes[node].get("sent_idx") == sent_idx):
                    gold.add(node)

    if not gold:
        for node in G.nodes:
            if G.nodes[node].get("is_supporting", False):
                gold.add(node)

    return gold


def recall_at_k(selected: list[int], gold: set[int], k: int = 5) -> float:
    if not gold:
        return 0.0
    top_k = set(selected[:k])
    return len(top_k & gold) / len(gold)


def run_confidence_comparison(records: list[dict], max_samples: int = 200,
                              k: int = 5, groq_api_key: str = None) -> dict:
    """
    Run PATHFINDER on each record, compute all 3 confidence models,
    and optionally generate LLM answers for EM/F1 calibration.
    """
    records = records[:max_samples]
    per_query = []

    # LLM setup
    llm_client = None
    if groq_api_key:
        try:
            from groq import Groq
            llm_client = Groq(api_key=groq_api_key)
        except ImportError:
            print("Warning: groq package not installed. Skipping LLM answers.")

    from generate_answers import generate_answer, exact_match, f1_score

    for rec in tqdm(records, desc="Confidence calibration"):
        gd = rec["graph"]
        gold = get_gold_nodes(rec)

        try:
            res = run_pathfinder(gd, enable_teleport=True)
        except Exception as e:
            continue

        # Compute all 3 confidence models
        G = gd["G"]
        sigma_prod = compute_sigma_product(res.S, res.parent, G)
        sigma_geom = compute_sigma(res.S, res.parent, G, use_geometric_mean=True)
        sigma_min = compute_sigma_min(res.S, res.parent, G)

        # Recall@k
        r_at_5 = recall_at_k(res.S, gold, k=k)
        r_at_10 = recall_at_k(res.S, gold, k=10)
        r_at_20 = recall_at_k(res.S, gold, k=20)

        entry = {
            "question": rec["question"][:100],
            "sigma_product": round(sigma_prod, 6),
            "sigma_geometric_mean": round(sigma_geom, 6),
            "sigma_bottleneck": round(sigma_min, 6),
            "recall_at_5": round(r_at_5, 4),
            "recall_at_10": round(r_at_10, 4),
            "recall_at_20": round(r_at_20, 4),
            "n_selected": len(res.S),
            "F": round(res.F, 4),
            "confidence_flag": res.confidence_flag,
        }

        # LLM answer for EM/F1 calibration
        if llm_client:
            try:
                pred = generate_answer(rec["question"], res.S, G, llm_client)
                gold_answer = rec.get("answer", "")
                entry["em"] = exact_match(pred, gold_answer)
                entry["f1"] = round(f1_score(pred, gold_answer), 4)
                entry["pred_answer"] = pred[:100]
            except Exception:
                entry["em"] = None
                entry["f1"] = None

        per_query.append(entry)

    # Summary statistics
    summary = {
        "n_queries": len(per_query),
        "sigma_product": {
            "mean": float(np.mean([q["sigma_product"] for q in per_query])),
            "std": float(np.std([q["sigma_product"] for q in per_query])),
            "min": float(np.min([q["sigma_product"] for q in per_query])),
            "max": float(np.max([q["sigma_product"] for q in per_query])),
        },
        "sigma_geometric_mean": {
            "mean": float(np.mean([q["sigma_geometric_mean"] for q in per_query])),
            "std": float(np.std([q["sigma_geometric_mean"] for q in per_query])),
            "min": float(np.min([q["sigma_geometric_mean"] for q in per_query])),
            "max": float(np.max([q["sigma_geometric_mean"] for q in per_query])),
        },
        "sigma_bottleneck": {
            "mean": float(np.mean([q["sigma_bottleneck"] for q in per_query])),
            "std": float(np.std([q["sigma_bottleneck"] for q in per_query])),
            "min": float(np.min([q["sigma_bottleneck"] for q in per_query])),
            "max": float(np.max([q["sigma_bottleneck"] for q in per_query])),
        },
    }

    # Spearman correlation (if we have EM data)
    if per_query and per_query[0].get("em") is not None:
        from scipy.stats import spearmanr
        sigmas_prod = [q["sigma_product"] for q in per_query]
        sigmas_geom = [q["sigma_geometric_mean"] for q in per_query]
        sigmas_min = [q["sigma_bottleneck"] for q in per_query]
        ems = [q["em"] for q in per_query if q["em"] is not None]

        if len(ems) == len(per_query):
            rho_prod, _ = spearmanr(sigmas_prod, ems)
            rho_geom, _ = spearmanr(sigmas_geom, ems)
            rho_min, _ = spearmanr(sigmas_min, ems)
            summary["spearman_rho_vs_em"] = {
                "product": round(float(rho_prod), 4),
                "geometric_mean": round(float(rho_geom), 4),
                "bottleneck": round(float(rho_min), 4),
            }

    return {"per_query": per_query, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="Confidence Calibration Comparison")
    parser.add_argument("--graphs", required=True, help="Path to graph pickle file")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", default="results/raw/confidence_calibration.json")
    parser.add_argument("--with_llm", action="store_true", help="Generate LLM answers for EM/F1")
    args = parser.parse_args()

    if not Path(args.graphs).exists():
        print(f"Graph cache not found: {args.graphs}")
        sys.exit(1)

    print(f"Loading graphs from {args.graphs}...")
    with open(args.graphs, "rb") as f:
        records = pickle.load(f)

    groq_key = os.getenv("GROQ_API_KEY") if args.with_llm else None

    result = run_confidence_comparison(records, args.max_samples, groq_api_key=groq_key)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nSummary:")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
