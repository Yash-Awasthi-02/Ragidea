"""
PATHFINDER Experiment — Phase C: LLM Reranking Integrated Evaluation
=====================================================================
Run full evaluation with LLM reranking as a post-processing step.
Compares: PATHFINDER original vs PATHFINDER + LLM rerank vs Naive RAG.

Usage:
    set GROQ_API_KEY=your_key_here
    python 25_rerank_eval.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/raw/rerank_eval.json
    python 25_rerank_eval.py --graphs data/hotpotqa_graphs_passage.pkl --max_samples 500 --output results/raw/rerank_passage.json
"""
import sys, json, pickle, argparse, os, time, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_pathfinder import run_pathfinder
from run_baselines import naive_rag
from generate_answers import generate_answer, exact_match, f1_score, build_context


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
    if not S or len(S) <= 1:
        return S

    # Build numbered context passages (truncate to first 15 nodes)
    S_trunc = S[:15]
    passages = []
    for i, v in enumerate(S_trunc):
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
        indices = [int(x.strip()) for x in ranking_str.split(",") if x.strip().isdigit()]
        reranked = [S_trunc[i] for i in indices if i < len(S_trunc)]
        # Append any missing nodes
        for v in S:
            if v not in reranked:
                reranked.append(v)
        return reranked
    except Exception as e:
        print(f"  LLM rerank error: {e}")
        return S


def main():
    parser = argparse.ArgumentParser(description="LLM Reranking Evaluation")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--output", default="results/raw/rerank_eval.json")
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--no_llm", action="store_true", help="Skip LLM reranking (baseline only)")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    client = None
    if not args.no_llm and api_key:
        from groq import Groq
        client = Groq(api_key=api_key)
        print(f"LLM reranking enabled: {args.model}")
    else:
        print("LLM reranking disabled (no GROQ_API_KEY or --no_llm)")

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]

    per_query = []

    for rec in tqdm(records, desc="Rerank eval"):
        gd = rec["graph"]
        G = gd["G"]
        nodes = gd["nodes"]
        gold = get_gold_nodes(rec)
        question = rec["question"]
        gold_answer = rec.get("answer", "")

        # PATHFINDER original
        res_pf = run_pathfinder(gd, enable_teleport=True)
        r5_pf = recall_at_k(res_pf.S, gold, 5)
        r10_pf = recall_at_k(res_pf.S, gold, 10)

        # PATHFINDER + LLM rerank
        r5_rerank = r5_pf
        r10_rerank = r10_pf
        em_pf = 0
        f1_pf = 0.0
        em_rerank = 0
        f1_rerank = 0.0

        if client:
            reranked_S = llm_rerank(question, res_pf.S, G, client, args.model)
            r5_rerank = recall_at_k(reranked_S, gold, 5)
            r10_rerank = recall_at_k(reranked_S, gold, 10)

            # Generate answers
            try:
                pred_pf = generate_answer(question, res_pf.S, G, client, model=args.model)
                em_pf = exact_match(pred_pf, gold_answer)
                f1_pf = f1_score(pred_pf, gold_answer)
            except Exception:
                pass

            try:
                pred_rerank = generate_answer(question, reranked_S, G, client, model=args.model)
                em_rerank = exact_match(pred_rerank, gold_answer)
                f1_rerank = f1_score(pred_rerank, gold_answer)
            except Exception:
                pass

            time.sleep(0.5)  # Rate limit

        # Naive RAG
        nr_nodes = naive_rag(gd, k=20)
        r5_nr = recall_at_k(nr_nodes, gold, 5)
        r10_nr = recall_at_k(nr_nodes, gold, 10)

        per_query.append({
            "question": question[:80],
            "pathfinder": {"r5": r5_pf, "r10": r10_pf, "em": em_pf, "f1": round(f1_pf, 4), "n_nodes": len(res_pf.S)},
            "pathfinder_rerank": {"r5": r5_rerank, "r10": r10_rerank, "em": em_rerank, "f1": round(f1_rerank, 4)},
            "naive_rag": {"r5": r5_nr, "r10": r10_nr},
        })

    # Aggregate
    summary = {
        "n_queries": len(per_query),
        "pathfinder": {
            "r5": round(float(np.mean([q["pathfinder"]["r5"] for q in per_query])), 4),
            "r10": round(float(np.mean([q["pathfinder"]["r10"] for q in per_query])), 4),
            "em": round(float(np.mean([q["pathfinder"]["em"] for q in per_query])), 4),
            "f1": round(float(np.mean([q["pathfinder"]["f1"] for q in per_query])), 4),
        },
        "pathfinder_rerank": {
            "r5": round(float(np.mean([q["pathfinder_rerank"]["r5"] for q in per_query])), 4),
            "r10": round(float(np.mean([q["pathfinder_rerank"]["r10"] for q in per_query])), 4),
            "em": round(float(np.mean([q["pathfinder_rerank"]["em"] for q in per_query])), 4),
            "f1": round(float(np.mean([q["pathfinder_rerank"]["f1"] for q in per_query])), 4),
        },
        "naive_rag": {
            "r5": round(float(np.mean([q["naive_rag"]["r5"] for q in per_query])), 4),
            "r10": round(float(np.mean([q["naive_rag"]["r10"] for q in per_query])), 4),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\n{'Config':<25} {'R@5':>8} {'R@10':>8} {'EM':>8} {'F1':>8}")
    print("-" * 55)
    for name in ["pathfinder", "pathfinder_rerank", "naive_rag"]:
        s = summary[name]
        print(f"{name:<25} {s['r5']:>8.4f} {s.get('r10',0):>8.4f} {s.get('em',0):>8.4f} {s.get('f1',0):>8.4f}")


if __name__ == "__main__":
    main()
