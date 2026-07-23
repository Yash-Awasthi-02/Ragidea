"""
PATHFINDER Experiment — Phase 8, Task 8.1: FAISS/HNSW Teleportation
====================================================================
Compare numpy argsort vs FAISS HNSW for teleportation candidate selection.

Usage:
    pip install faiss-cpu
    python 10_faiss_teleport.py --graphs data/hotpotqa_graphs.pkl --max_samples 200 --output results/raw/faiss_teleport.json
"""
import sys, json, pickle, argparse, numpy as np, time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_pathfinder import run_pathfinder, TELEPORT_TOPK


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
    parser = argparse.ArgumentParser(description="FAISS Teleportation Benchmark")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", default="results/raw/faiss_teleport.json")
    args = parser.parse_args()

    try:
        import faiss
        has_faiss = True
        print("FAISS available — comparing ANN vs brute-force")
    except ImportError:
        has_faiss = False
        print("FAISS not installed — using numpy brute-force only (pip install faiss-cpu for ANN)")

    with open(args.graphs, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_samples]

    per_query = []

    for rec in tqdm(records, desc="FAISS benchmark"):
        gd = rec["graph"]
        gold = get_gold_nodes(rec)
        embs = gd["embeddings"].astype(np.float32)
        q_emb = gd["q_emb"].astype(np.float32)
        N = gd["N"]

        # Numpy brute-force timing
        t0 = time.perf_counter()
        phi_sem_q = embs @ q_emb
        numpy_ranked = np.argsort(phi_sem_q)[::-1][:TELEPORT_TOPK]
        numpy_time = (time.perf_counter() - t0) * 1000

        # FAISS timing
        faiss_time = 0.0
        faiss_ranked = numpy_ranked  # default same
        if has_faiss and N > 1:
            index = faiss.IndexHNSWFlat(embs.shape[1], 16)
            index.add(embs)
            t0 = time.perf_counter()
            _, faiss_indices = index.search(q_emb.reshape(1, -1), TELEPORT_TOPK)
            faiss_time = (time.perf_counter() - t0) * 1000
            faiss_ranked = faiss_indices[0]

        # Run pathfinder
        res = run_pathfinder(gd, enable_teleport=True)
        r5 = recall_at_k(res.S, gold, 5)

        # Agreement between numpy and faiss
        agreement = len(set(numpy_ranked.tolist()) & set(faiss_ranked.tolist())) / TELEPORT_TOPK

        per_query.append({
            "n_nodes": N,
            "numpy_time_ms": round(numpy_time, 4),
            "faiss_time_ms": round(faiss_time, 4),
            "candidate_agreement": round(agreement, 4),
            "recall@5": r5,
        })

    summary = {
        "n_queries": len(per_query),
        "mean_numpy_time_ms": round(float(np.mean([q["numpy_time_ms"] for q in per_query])), 4),
        "mean_faiss_time_ms": round(float(np.mean([q["faiss_time_ms"] for q in per_query])), 4),
        "mean_agreement": round(float(np.mean([q["candidate_agreement"] for q in per_query])), 4),
        "mean_recall@5": round(float(np.mean([q["recall@5"] for q in per_query])), 4),
        "has_faiss": has_faiss,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nFAISS vs Numpy Teleportation Lookup:")
    print(f"  Numpy time: {summary['mean_numpy_time_ms']:.3f} ms")
    print(f"  FAISS time: {summary['mean_faiss_time_ms']:.3f} ms")
    print(f"  Agreement:  {summary['mean_agreement']:.1%}")
    print(f"  R@5:        {summary['mean_recall@5']:.4f}")


if __name__ == "__main__":
    main()
