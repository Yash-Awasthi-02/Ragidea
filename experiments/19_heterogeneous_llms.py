"""
PATHFINDER Experiment — Phase 12, Task 12.2: Heterogeneous LLM Evaluation
==========================================================================
Evaluate PATHFINDER with different generator LLMs via Groq.

Usage:
    set GROQ_API_KEY=your_key
    python 19_heterogeneous_llms.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --model llama-3.3-70b-versatile --output results/raw/heterogeneous_llm.json
    python 19_heterogeneous_llms.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --model llama3-8b-8192 --output results/raw/heterogeneous_llm_8b.json
"""
import sys, json, pickle, argparse, os, time, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_pathfinder import run_pathfinder
from generate_answers import generate_answer, exact_match, f1_score


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
    parser = argparse.ArgumentParser(description="Heterogeneous LLM Evaluation")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set")
        sys.exit(1)

    from groq import Groq
    client = Groq(api_key=api_key)

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]

    output_path = args.output or f"results/raw/heterogeneous_llm_{args.model.replace('.', '_')}.json"

    per_query = []

    for rec in tqdm(records, desc=f"LLM eval ({args.model})"):
        gd = rec["graph"]
        G = gd["G"]
        gold = get_gold_nodes(rec)
        question = rec["question"]
        gold_answer = rec.get("answer", "")

        # Run pathfinder
        res = run_pathfinder(gd, enable_teleport=True)
        r5 = recall_at_k(res.S, gold, 5)

        # Generate answer with specified model
        try:
            pred = generate_answer(question, res.S, G, client, model=args.model)
            em = exact_match(pred, gold_answer)
            f1 = f1_score(pred, gold_answer)
        except Exception as e:
            pred, em, f1 = f"error: {e}", 0, 0.0
            time.sleep(1)

        per_query.append({
            "question": question[:80],
            "gold_answer": gold_answer[:50],
            "prediction": pred[:50] if isinstance(pred, str) else str(pred)[:50],
            "em": em,
            "f1": round(f1, 4),
            "recall@5": r5,
        })

    summary = {
        "model": args.model,
        "n_queries": len(per_query),
        "mean_em": round(float(np.mean([q["em"] for q in per_query])), 4),
        "mean_f1": round(float(np.mean([q["f1"] for q in per_query])), 4),
        "mean_r5": round(float(np.mean([q["recall@5"] for q in per_query])), 4),
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nHeterogeneous LLM Results ({args.model}):")
    print(f"  EM:      {summary['mean_em']:.4f}")
    print(f"  F1:      {summary['mean_f1']:.4f}")
    print(f"  R@5:     {summary['mean_r5']:.4f}")


if __name__ == "__main__":
    main()
