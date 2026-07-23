"""
PATHFINDER Experiment — Phase 10, Task 10.2: LLM-Guided Traversal
==================================================================
LLM selects the next frontier node at each traversal step.

Usage:
    set GROQ_API_KEY=your_key
    python 16_llm_guided_traversal.py --graphs data/hotpotqa_graphs.pkl --max_samples 20 --output results/raw/llm_guided_traversal.json
"""
import sys, json, pickle, argparse, os, time, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_pathfinder import run_pathfinder, tok_count, K_TOK


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


def llm_guided_traverse(gd, question, client, model="llama-3.3-70b-versatile", k_tok=K_TOK):
    """LLM-guided traversal: LLM picks next frontier node at each step."""
    G = gd["G"]
    embs = gd["embeddings"]
    q_emb = gd["q_emb"]
    N = gd["N"]

    if N == 0:
        return []

    phi_sem_q = embs @ q_emb
    v0 = int(np.argmax(phi_sem_q))

    S = [v0]
    S_set = {v0}
    tok = tok_count(v0, G)

    frontier = {}
    for u in G.successors(v0):
        if u not in S_set:
            frontier[u] = v0

    while frontier and tok < k_tok:
        rem = k_tok - tok
        feasible = {v: p for v, p in frontier.items() if tok_count(v, G) <= rem}
        if not feasible:
            break

        # Build prompt with frontier candidates
        context = " ".join([G.nodes[v].get("text", "")[:100] for v in S])
        candidates = []
        for i, v in enumerate(sorted(feasible.keys())):
            text = G.nodes[v].get("text", "")[:150]
            candidates.append(f"[{i}] {text}")

        prompt = f"""Question: {question}
Currently selected context: {context[:500]}
Frontier candidates:
{chr(10).join(candidates[:15])}

Which candidate should be added next to best support answering the question? Return ONLY the index number (0-{len(candidates)-1}). If the current context is sufficient, return -1.

Index:"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            answer = response.choices[0].message.content.strip()
            idx = int(answer) if answer.lstrip("-").isdigit() else -1
        except Exception:
            idx = -1

        if idx == -1 or idx >= len(candidates):
            break

        feasible_list = sorted(feasible.keys())
        v_star = feasible_list[idx]
        S.append(v_star)
        S_set.add(v_star)
        tok += tok_count(v_star, G)

        del frontier[v_star]
        for u in G.successors(v_star):
            if u not in S_set and u not in frontier:
                frontier[u] = v_star

        time.sleep(0.3)  # Rate limit

    return S


def main():
    parser = argparse.ArgumentParser(description="LLM-Guided Traversal")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--output", default="results/raw/llm_guided_traversal.json")
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

    for rec in tqdm(records, desc="LLM-guided traversal"):
        gd = rec["graph"]
        gold = get_gold_nodes(rec)
        question = rec["question"]

        # Greedy pathfinder
        res_greedy = run_pathfinder(gd, enable_teleport=True)
        r5_greedy = recall_at_k(res_greedy.S, gold, 5)

        # LLM-guided
        llm_S = llm_guided_traverse(gd, question, client, args.model)
        r5_llm = recall_at_k(llm_S, gold, 5)

        per_query.append({
            "question": question[:80],
            "r5_greedy": r5_greedy,
            "r5_llm_guided": r5_llm,
            "delta": r5_llm - r5_greedy,
            "n_nodes_greedy": len(res_greedy.S),
            "n_nodes_llm": len(llm_S),
        })

    summary = {
        "n_queries": len(per_query),
        "mean_r5_greedy": round(float(np.mean([q["r5_greedy"] for q in per_query])), 4),
        "mean_r5_llm": round(float(np.mean([q["r5_llm_guided"] for q in per_query])), 4),
        "mean_delta": round(float(np.mean([q["delta"] for q in per_query])), 4),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nLLM-Guided vs Greedy Traversal:")
    print(f"  Greedy R@5:     {summary['mean_r5_greedy']:.4f}")
    print(f"  LLM-guided R@5: {summary['mean_r5_llm']:.4f}")
    print(f"  Delta:          {summary['mean_delta']:+.4f}")


if __name__ == "__main__":
    main()
