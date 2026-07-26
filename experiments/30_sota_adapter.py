"""
PATHFINDER — SOTA Comparison Adapter (PLAN §5.1, CODE ONLY — DO NOT RUN)
=========================================================================
Uniform scoring interface so ANY retrieval system can be compared against
PATHFINDER on identical metrics (EM, F1, R@5, R@10) using OUR normalisation
(HotpotQA official). The adapter takes a system's (question, retrieved_passages)
and computes metrics with the exact-match / F1 from generate_answers.py.

This module is pure scoring + plumbing. It never calls an LLM itself and never
loads a dataset. The actual SOTA runs happen in PLAN §6 (user-driven).

Interface contract
------------------
A "system" is any callable:
    retrieve(question: str, qid: str) -> list[str]   # ordered passages, top-first
It returns the passages the system would feed its generator, most-relevant first.

Scoring:
    score_system(name, retrieve, records, gold_fn) -> dict
        EM / F1 need a prediction string per query (from the system's generator,
        supplied via `predict_fn`), and R@k needs gold passage ids per query.

Fair-comparison protocol (PLAN §5.3):
  * Same 500-query subset for every system.
  * Same generator (Llama-3.3-70B via Groq) unless a system ships its own.
  * Note where a SOTA system uses GPT-4 / stronger embeddings than our default.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional
import json

# Reuse OUR normalisation for apples-to-apples EM/F1.
from generate_answers import exact_match, f1_score, RATE_LIMITED


# ── Metric helpers ────────────────────────────────────────────────────────────
def recall_at_k(retrieved_ids: list, gold_ids: set, k: int) -> int:
    """1 iff ALL gold passages appear within the top-k retrieved. Binary (HotpotQA)."""
    if not gold_ids:
        return 1
    top = set(retrieved_ids[:k])
    return int(all(g in top for g in gold_ids))


def precision_at_k(retrieved_ids: list, gold_ids: set, k: int) -> float:
    if not gold_ids or k == 0:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & set(gold_ids)) / float(k)


# ── System adapter ────────────────────────────────────────────────────────────
def score_system(
    name: str,
    retrieve: Callable[[str, str], list],
    records: Iterable[dict],
    gold_fn: Callable[[dict], set],
    id_fn: Optional[Callable[[dict, int], object]] = None,
    predict_fn: Optional[Callable[[str, list], str]] = None,
    ks: tuple = (5, 10),
) -> dict:
    """
    Score one system over a set of query records.

    Args:
        name       : system label for reporting.
        retrieve   : fn(question, qid) -> ordered list of retrieved items
                     (passage texts OR ids; must match what gold_fn/id_fn expect).
        records    : iterable of {"id", "question", "answer", ...} query dicts.
        gold_fn    : fn(record) -> set of gold passage ids for R@k.
        id_fn      : fn(record, rank) -> id for the retrieved item at `rank`.
                     If None, retrieved items are assumed to already be ids.
        predict_fn : fn(question, retrieved_passages) -> prediction string.
                     If None, EM/F1 are skipped (retrieval-only scoring).

    Returns:
        {"system", "n", "em", "f1", "recall@k", "precision@k", "per_query"}
    """
    per_query = []
    ems, f1s = [], []
    r_at = {k: [] for k in ks}
    p_at = {k: [] for k in ks}
    n_rate_limited = 0

    for rec in records:
        qid = rec.get("id", "")
        question = rec["question"]
        gold_answer = rec.get("answer", "")
        retrieved = retrieve(question, qid)
        retrieved_ids = [id_fn(rec, i) if id_fn else it
                         for i, it in enumerate(retrieved)]

        gold = set(gold_fn(rec))
        row = {"id": qid, "question": question[:80]}

        for k in ks:
            r_at[k].append(recall_at_k(retrieved_ids, gold, k))
            p_at[k].append(precision_at_k(retrieved_ids, gold, k))
            row[f"recall@{k}"] = r_at[k][-1]

        if predict_fn is not None:
            pred = predict_fn(question, retrieved)
            if pred == RATE_LIMITED:
                n_rate_limited += 1
                row["em"] = row["f1"] = None
            else:
                em = exact_match(pred, gold_answer)
                f1 = f1_score(pred, gold_answer)
                ems.append(em); f1s.append(f1)
                row["em"] = em; row["f1"] = round(f1, 4)

        per_query.append(row)

    n = len(per_query)
    summary = {
        "system": name,
        "n": n,
        "n_rate_limited": n_rate_limited,
        "quota_exhausted": n_rate_limited > 0,
    }
    if predict_fn is not None:
        summary["em"] = round(sum(ems) / len(ems), 4) if ems else 0.0
        summary["f1"] = round(sum(f1s) / len(f1s), 4) if f1s else 0.0
    for k in ks:
        summary[f"recall@{k}"] = round(sum(r_at[k]) / n, 4) if n else 0.0
        summary[f"precision@{k}"] = round(sum(p_at[k]) / n, 4) if n else 0.0

    return {"summary": summary, "per_query": per_query}


# ── Registry of comparison systems (filled in PLAN §6) ───────────────────────
# Each entry documents how to build that system's `retrieve` callable.
SYSTEMS = {
    "pathfinder_coherent": "run_pathfinder(gd) — graph-coherent traversal",
    "pathfinder_flat":     "run_pathfinder(gd, always_dense=True) — PATHFINDER-Flat",
    "pathfinder_hybrid":   "run_pathfinder_hybrid(gd) — dense-anchor hybrid",
    "naive_rag":           "run_baselines.naive_rag(gd, k=K)",
    "ircot":               "IRCoT adapter — PENDING §6 (interleaved retrieval+CoT)",
    "subgraphrag":         "SubgraphRAG adapter — PENDING §6",
    "hipporag2":           "HippoRAG 2 adapter — PENDING §6",
    "lightrag":            "LightRAG adapter — PENDING §6",
}


def list_pending() -> list:
    """Return which registered systems still need a §6 adapter implementation."""
    return [k for k, v in SYSTEMS.items() if "PENDING" in v]


if __name__ == "__main__":
    # Smoke test on a toy in-memory example (no dataset, no LLM, no API).
    print("30_sota_adapter smoke test (toy data, no dataset/LLM)\n")

    recs = [
        {"id": "q1", "question": "Who wrote X?", "answer": "author a"},
        {"id": "q2", "question": "Where is Y?", "answer": "place b"},
    ]
    gold = {"q1": {"p1", "p2"}, "q2": {"p3"}}

    def toy_retrieve(question, qid):
        return ["p1", "p2", "p9", "p3", "p8"] if qid == "q1" else ["p3", "p7"]

    def toy_predict(question, passages):
        return "author a" if "Who" in question else "place b"

    out = score_system(
        "toy_system", toy_retrieve, recs,
        gold_fn=lambda r: gold[r["id"]],
        predict_fn=toy_predict,
    )
    print(json.dumps(out["summary"], indent=2))
    assert out["summary"]["recall@5"] == 1.0
    assert out["summary"]["recall@10"] == 1.0
    assert out["summary"]["em"] == 1.0
    print("\nSmoke test passed. Pending SOTA adapters:", list_pending())
