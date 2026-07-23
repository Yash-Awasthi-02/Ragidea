"""
PATHFINDER Experiment — Phase 6, Task 6.5: NLI-Based Sufficiency Check
======================================================================
Replace heuristic sufficiency with NLI entailment verification.
Optionally uses transformers NLI model; falls back to lexical overlap.

Usage:
    python 08_nli_sufficiency.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --output results/raw/nli_sufficiency.json
"""
import sys, json, pickle, argparse, numpy as np, time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_pathfinder import run_pathfinder, SUFFICIENCY_THRESHOLD


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


def lexical_overlap_sufficiency(S, G, question, threshold=0.3):
    """Fallback: check if answer tokens overlap with context tokens."""
    context_tokens = set()
    for v in S:
        context_tokens.update(G.nodes[v].get("text", "").lower().split())
    q_tokens = set(question.lower().split())
    if not q_tokens:
        return False
    overlap = len(context_tokens & q_tokens) / len(q_tokens)
    return overlap >= threshold


def main():
    parser = argparse.ArgumentParser(description="NLI Sufficiency Check")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", default="results/raw/nli_sufficiency.json")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-base")
    args = parser.parse_args()

    # Try loading NLI model
    nli_pipeline = None
    try:
        from transformers import pipeline
        nli_pipeline = pipeline("text-classification", model=args.nli_model)
        print(f"Loaded NLI model: {args.nli_model}")
    except ImportError:
        print("transformers not installed — using lexical overlap fallback")

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]

    per_query = []

    for rec in tqdm(records, desc="NLI sufficiency"):
        gd = rec["graph"]
        G = gd["G"]
        gold = get_gold_nodes(rec)
        question = rec["question"]

        # Run pathfinder
        res = run_pathfinder(gd, enable_teleport=True)
        r5 = recall_at_k(res.S, gold, 5)

        # Heuristic sufficiency
        product = 1.0
        for v in res.S:
            sim = max(0.0, float((gd["embeddings"] @ gd["q_emb"])[v]))
            product *= (1.0 - sim)
        heuristic_sufficient = (1.0 - product) >= SUFFICIENCY_THRESHOLD

        # NLI sufficiency
        nli_sufficient = False
        nli_score = 0.0
        if nli_pipeline:
            context = " ".join([G.nodes[v].get("text", "") for v in res.S[:10]])
            try:
                result = nli_pipeline(f"{context} [SEP] {question}")
                # result is list of dicts with 'label' and 'score'
                if isinstance(result, list):
                    for r in result:
                        if r.get("label", "").lower() == "entailment":
                            nli_score = r["score"]
                            nli_sufficient = nli_score > 0.5
                            break
            except Exception:
                nli_sufficient = heuristic_sufficient
        else:
            nli_sufficient = lexical_overlap_sufficiency(res.S, G, question)
            nli_score = -1.0  # indicates fallback

        per_query.append({
            "question": question[:80],
            "recall@5": r5,
            "n_nodes": len(res.S),
            "heuristic_sufficient": heuristic_sufficient,
            "nli_sufficient": nli_sufficient,
            "nli_score": round(nli_score, 4),
            "agreement": heuristic_sufficient == nli_sufficient,
        })

    agreement_rate = float(np.mean([q["agreement"] for q in per_query]))
    summary = {
        "n_queries": len(per_query),
        "agreement_rate": round(agreement_rate, 4),
        "heuristic_sufficient_rate": round(float(np.mean([q["heuristic_sufficient"] for q in per_query])), 4),
        "nli_sufficient_rate": round(float(np.mean([q["nli_sufficient"] for q in per_query])), 4),
        "mean_r5": round(float(np.mean([q["recall@5"] for q in per_query])), 4),
        "mean_nodes": round(float(np.mean([q["n_nodes"] for q in per_query])), 2),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nNLI vs Heuristic Sufficiency:")
    print(f"  Agreement rate:      {summary['agreement_rate']:.1%}")
    print(f"  Heuristic suff rate: {summary['heuristic_sufficient_rate']:.1%}")
    print(f"  NLI suff rate:       {summary['nli_sufficient_rate']:.1%}")
    print(f"  Mean R@5:            {summary['mean_r5']:.4f}")
    print(f"  Mean nodes/query:    {summary['mean_nodes']:.1f}")


if __name__ == "__main__":
    main()
