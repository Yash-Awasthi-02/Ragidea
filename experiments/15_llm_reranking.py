"""
PATHFINDER Experiment — Phase 10, Task 10.1: LLM Reranking
===========================================================
Rerank PATHFINDER's candidate set using an LLM for multi-hop reasoning quality.

Usage:
    set GROQ_API_KEY=your_key
    python 15_llm_reranking.py --graphs data/hotpotqa_graphs.pkl --max_samples 50 --output results/raw/llm_reranking.json
"""
import sys, json, pickle, argparse, os, time, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_pathfinder import run_pathfinder
from generate_answers import build_context, exact_match, f1_score


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


def llm_rerank(question, S, G, client, model="llama-3.3-70b-versatile"):
    """Ask LLM to rerank candidate nodes by relevance to the question."""
    if not S:
        return S

    # Build numbered context passages
    passages = []
    for i, v in enumerate(S):
        text = G.nodes[v].get("text", "")[:200]
        passages.append(f"[{i}] {text}")

    prompt = f"""Question: {question}

Context passages (in PATHFINDER selection order):
{chr(10).join(passages)}

Rank these passages by relevance to answering the question. Return ONLY a comma-separated list of indices in order of most relevant to least relevant. Example: 3,0,1,2,4

Ranking:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        ranking_str = response.choices[0].message.content.strip()
        # Parse comma-separated indices
        indices = [int(x.strip()) for x in ranking_str.split(",") if x.strip().isdigit()]
        # Map back to original node indices
        reranked = [S[i] for i in indices if i < len(S)]
        # Append any missing nodes
        for v in S:
            if v not in reranked:
                reranked.append(v)
        return reranked
    except Exception as e:
        print(f"  LLM rerank error: {e}")
        return S


def main():
    parser = argparse.ArgumentParser(description="LLM Reranking")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output", default="results/raw/llm_reranking.json")
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
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

    per_query = []

    for rec in tqdm(records, desc="LLM reranking"):
        gd = rec["graph"]
        G = gd["G"]
        gold = get_gold_nodes(rec)
        question = rec["question"]

        # Run pathfinder
        res = run_pathfinder(gd, enable_teleport=True)
        r5_original = recall_at_k(res.S, gold, 5)

        # LLM rerank
        reranked_S = llm_rerank(question, res.S, G, client, args.model)
        r5_reranked = recall_at_k(reranked_S, gold, 5)

        # Rate limit
        time.sleep(0.5)

        per_query.append({
            "question": question[:80],
            "r5_original": r5_original,
            "r5_reranked": r5_reranked,
            "delta": r5_reranked - r5_original,
            "n_nodes": len(res.S),
        })

    summary = {
        "n_queries": len(per_query),
        "mean_r5_original": round(float(np.mean([q["r5_original"] for q in per_query])), 4),
        "mean_r5_reranked": round(float(np.mean([q["r5_reranked"] for q in per_query])), 4),
        "mean_delta": round(float(np.mean([q["delta"] for q in per_query])), 4),
        "n_improved": sum(1 for q in per_query if q["delta"] > 0),
        "n_hurt": sum(1 for q in per_query if q["delta"] < 0),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nLLM Reranking Results:")
    print(f"  Original R@5:  {summary['mean_r5_original']:.4f}")
    print(f"  Reranked R@5:  {summary['mean_r5_reranked']:.4f}")
    print(f"  Delta:         {summary['mean_delta']:+.4f}")
    print(f"  Improved:      {summary['n_improved']}")
    print(f"  Hurt:          {summary['n_hurt']}")


if __name__ == "__main__":
    main()
