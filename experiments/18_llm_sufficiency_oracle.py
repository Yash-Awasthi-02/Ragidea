"""
PATHFINDER Experiment — Phase 10, Task 10.4: LLM Sufficiency Oracle
====================================================================
Compare heuristic vs LLM vs hybrid sufficiency check.

Usage:
    set GROQ_API_KEY=your_key
    python 18_llm_sufficiency_oracle.py --graphs data/hotpotqa_graphs.pkl --max_samples 20 --output results/raw/llm_sufficiency_oracle.json
"""
import sys, json, pickle, argparse, os, time, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_pathfinder import run_pathfinder, tok_count, delta_full_fast, K_TOK, SUFFICIENCY_THRESHOLD
from run_baselines import naive_rag


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


def llm_sufficiency_check(question, S, G, client, model="llama-3.3-70b-versatile"):
    """Ask LLM if current context is sufficient to answer the question."""
    context = " ".join([G.nodes[v].get("text", "")[:150] for v in S[:10]])
    prompt = f"""Question: {question}
Context: {context[:2000]}

Is there sufficient information in the context to answer the question? Reply ONLY "Yes" or "No".

Answer:"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer
    except Exception:
        return True  # Default to sufficient on error


def heuristic_sufficiency(S, phi_sem_q, threshold=SUFFICIENCY_THRESHOLD):
    product = 1.0
    for v in S:
        product *= (1.0 - max(0.0, float(phi_sem_q[v])))
    return (1.0 - product) >= threshold


def hybrid_traverse(gd, question, client, model, k_tok=K_TOK, check_interval=2, heuristic_threshold=0.7):
    """Hybrid: heuristic for fast pass, LLM when uncertain."""
    G = gd["G"]
    q_emb = gd["q_emb"]
    q_dom = gd["q_dom"]
    embs = gd["embeddings"]
    N = gd["N"]
    phi_sem_q = embs @ q_emb

    if N == 0 or np.linalg.norm(q_emb) < 1e-8:
        return []

    v0 = int(np.argmax(phi_sem_q))
    if tok_count(v0, G) > k_tok:
        return []

    S = [v0]
    S_set = {v0}
    parent = {v0: None}
    tok = tok_count(v0, G)
    product_factor = 1.0 - max(0.0, float(phi_sem_q[v0]))
    weights = (1.0, 0.0, 0.0, 0.0, 0.0)

    frontier = {}
    for u in G.successors(v0):
        if u not in S_set:
            frontier[u] = v0

    step = 0
    while frontier and tok < k_tok:
        rem = k_tok - tok
        feasible = {v: p for v, p in frontier.items() if tok_count(v, G) <= rem}
        if not feasible:
            break

        best_v, best_gain = None, -1.0
        for v in feasible:
            g = delta_full_fast(v, G, q_dom, phi_sem_q, weights, product_factor)
            if g > best_gain:
                best_gain, best_v = g, v
        if best_v is None:
            break

        S.append(best_v)
        S_set.add(best_v)
        parent[best_v] = feasible[best_v]
        product_factor *= (1.0 - max(0.0, float(phi_sem_q[best_v])))
        tok += tok_count(best_v, G)

        # Check sufficiency every check_interval steps
        step += 1
        if step % check_interval == 0:
            heuristic_sufficient = (1.0 - product_factor) >= heuristic_threshold
            if heuristic_threshold <= (1.0 - product_factor) <= 0.9:
                # Uncertain zone — ask LLM
                if llm_sufficiency_check(question, S, G, client, model):
                    break
            elif heuristic_sufficient:
                break

        del frontier[best_v]
        for u in G.successors(best_v):
            if u not in S_set and u not in frontier:
                frontier[u] = best_v
        time.sleep(0.2)

    return S


def main():
    parser = argparse.ArgumentParser(description="LLM Sufficiency Oracle")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--output", default="results/raw/llm_sufficiency_oracle.json")
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

    for rec in tqdm(records, desc="LLM sufficiency oracle"):
        gd = rec["graph"]
        gold = get_gold_nodes(rec)
        question = rec["question"]

        # Heuristic (normal pathfinder)
        res_heuristic = run_pathfinder(gd, enable_teleport=True)
        r5_heuristic = recall_at_k(res_heuristic.S, gold, 5)
        n_heuristic = len(res_heuristic.S)

        # Hybrid (heuristic + LLM when uncertain)
        hybrid_S = hybrid_traverse(gd, question, client, args.model)
        r5_hybrid = recall_at_k(hybrid_S, gold, 5)
        n_hybrid = len(hybrid_S)

        per_query.append({
            "question": question[:80],
            "r5_heuristic": r5_heuristic,
            "r5_hybrid": r5_hybrid,
            "n_nodes_heuristic": n_heuristic,
            "n_nodes_hybrid": n_hybrid,
            "delta_r5": r5_hybrid - r5_heuristic,
            "delta_nodes": n_hybrid - n_heuristic,
        })

    summary = {
        "n_queries": len(per_query),
        "mean_r5_heuristic": round(float(np.mean([q["r5_heuristic"] for q in per_query])), 4),
        "mean_r5_hybrid": round(float(np.mean([q["r5_hybrid"] for q in per_query])), 4),
        "mean_nodes_heuristic": round(float(np.mean([q["n_nodes_heuristic"] for q in per_query])), 2),
        "mean_nodes_hybrid": round(float(np.mean([q["n_nodes_hybrid"] for q in per_query])), 2),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nHeuristic vs Hybrid LLM Sufficiency:")
    print(f"  Heuristic R@5:  {summary['mean_r5_heuristic']:.4f}  nodes: {summary['mean_nodes_heuristic']:.1f}")
    print(f"  Hybrid R@5:     {summary['mean_r5_hybrid']:.4f}  nodes: {summary['mean_nodes_hybrid']:.1f}")


if __name__ == "__main__":
    main()
