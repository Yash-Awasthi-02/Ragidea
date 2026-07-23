"""
PATHFINDER Experiment — Phase 10, Task 10.3: NLI Path Verification
==================================================================
Verify root-to-leaf paths using NLI entailment. Flag low-entailment paths.

Usage:
    python 17_nli_path_verification.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --output results/raw/nli_path_verification.json
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


def extract_paths(S, parent):
    """Extract all root-to-leaf paths from the parent tree."""
    S_set = set(S)
    leaves = [v for v in S if not any(parent.get(u) == v for u in S)]
    paths = []
    for leaf in leaves:
        path = []
        cur = leaf
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="NLI Path Verification")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", default="results/raw/nli_path_verification.json")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-base")
    args = parser.parse_args()

    nli_pipeline = None
    try:
        from transformers import pipeline
        nli_pipeline = pipeline("text-classification", model=args.nli_model)
        print(f"Loaded NLI model: {args.nli_model}")
    except ImportError:
        print("transformers not installed — using lexical overlap heuristic")

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]

    per_query = []

    for rec in tqdm(records, desc="NLI path verification"):
        gd = rec["graph"]
        G = gd["G"]
        gold = get_gold_nodes(rec)
        question = rec["question"]

        res = run_pathfinder(gd, enable_teleport=True)
        r5 = recall_at_k(res.S, gold, 5)

        # Extract paths
        paths = extract_paths(res.S, res.parent)

        # Verify each path with NLI
        path_scores = []
        flagged_paths = 0
        for path in paths:
            path_text = " ".join([G.nodes[v].get("text", "")[:100] for v in path])
            entailment_score = 0.0

            if nli_pipeline:
                try:
                    result = nli_pipeline(f"{path_text} [SEP] {question}")
                    if isinstance(result, list):
                        for r in result:
                            if r.get("label", "").lower() == "entailment":
                                entailment_score = r["score"]
                                break
                except Exception:
                    pass
            else:
                # Lexical overlap fallback
                path_tokens = set(path_text.lower().split())
                q_tokens = set(question.lower().split())
                entailment_score = len(path_tokens & q_tokens) / max(len(q_tokens), 1)

            path_scores.append(round(entailment_score, 4))
            if entailment_score < 0.5:
                flagged_paths += 1

        per_query.append({
            "question": question[:80],
            "recall@5": r5,
            "n_paths": len(paths),
            "n_flagged": flagged_paths,
            "path_scores": path_scores,
            "min_path_score": round(min(path_scores), 4) if path_scores else 0,
            "mean_path_score": round(float(np.mean(path_scores)), 4) if path_scores else 0,
        })

    summary = {
        "n_queries": len(per_query),
        "mean_r5": round(float(np.mean([q["recall@5"] for q in per_query])), 4),
        "mean_paths": round(float(np.mean([q["n_paths"] for q in per_query])), 2),
        "mean_flagged": round(float(np.mean([q["n_flagged"] for q in per_query])), 2),
        "mean_min_score": round(float(np.mean([q["min_path_score"] for q in per_query])), 4),
        "mean_mean_score": round(float(np.mean([q["mean_path_score"] for q in per_query])), 4),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nNLI Path Verification:")
    print(f"  Mean R@5:          {summary['mean_r5']:.4f}")
    print(f"  Mean paths/query:  {summary['mean_paths']:.1f}")
    print(f"  Mean flagged:      {summary['mean_flagged']:.1f}")
    print(f"  Mean min score:    {summary['mean_min_score']:.4f}")
    print(f"  Mean mean score:   {summary['mean_mean_score']:.4f}")


if __name__ == "__main__":
    main()
